#!/usr/bin/env python3
"""Remove unambiguous coordinated solvent from generated MOF CIFs in place."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sqlite3
import stat
import time
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from mofbb.decompose.constants import is_metal

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = WORKSPACE_ROOT / "2m_mofs_v2"
DEFAULT_DATABASE = DEFAULT_DATASET_ROOT / "generation.sqlite3"
DEFAULT_MOFS_ROOT = DEFAULT_DATASET_ROOT / "mofs"
DEFAULT_NODE_DIR = WORKSPACE_ROOT / "lib" / "node"

SPLIT_OXYGEN_SITE_MAX_DISTANCE = 1.05
MARKER = b"# Coordinated solvent removed in place by mofbb-clean-solvent-v1.\n"
NODE_BLOCK_PATTERN = re.compile(rb"BuildingBlock:\s+(m_node\d+)(?:,|\s)")

_WORKER_DIRTY_COUNTS: dict[str, int] = {}
_WORKER_DRY_RUN = True


@dataclass(frozen=True)
class CifTransform:
    """Validated result of transforming one generated CIF."""

    content: bytes | None
    status: str
    removed_atoms: int = 0
    removed_bonds: int = 0
    bytes_saved: int = 0


@dataclass(frozen=True)
class BatchResult:
    """Compact aggregate returned by one worker batch."""

    processed: int
    cleaned: int
    already_clean: int
    missing: int
    removed_atoms: int
    removed_bonds: int
    bytes_saved: int
    errors: int
    error_examples: tuple[str, ...]


def _candidate_atoms(
    symbols: dict[str, str],
    adjacency: dict[str, set[str]],
    bond_lengths: dict[tuple[str, str], float],
) -> set[str]:
    """Return explicit terminal aqua and split/disordered oxygen atoms."""

    solvent: set[str] = set()
    for oxygen, symbol in symbols.items():
        if symbol != "O":
            continue
        neighbors = adjacency[oxygen]
        metals = {
            atom
            for atom in neighbors
            if symbols[atom] != "X" and is_metal(symbols[atom])
        }
        hydrogens = {atom for atom in neighbors if symbols[atom] == "H"}
        other = neighbors - metals - hydrogens
        if (
            len(metals) == 1
            and len(hydrogens) == 2
            and not other
            and all(
                adjacency[hydrogen] <= metals | {oxygen} for hydrogen in hydrogens
            )
        ):
            solvent.update({oxygen, *hydrogens})

    for (left, right), distance in bond_lengths.items():
        if (
            symbols[left] != "O"
            or symbols[right] != "O"
            or distance > SPLIT_OXYGEN_SITE_MAX_DISTANCE
        ):
            continue
        left_neighbors = adjacency[left]
        right_neighbors = adjacency[right]
        left_metals = {
            atom
            for atom in left_neighbors
            if symbols[atom] != "X" and is_metal(symbols[atom])
        }
        right_metals = {
            atom
            for atom in right_neighbors
            if symbols[atom] != "X" and is_metal(symbols[atom])
        }
        if len(left_metals) != 1 or left_metals != right_metals:
            continue

        allowed = left_metals | {left, right}
        if any(
            symbols[atom] != "H" and atom not in allowed
            for atom in left_neighbors | right_neighbors
        ):
            continue
        hydrogens = {
            atom
            for atom in left_neighbors | right_neighbors
            if symbols[atom] == "H"
        }
        if any(
            adjacency[hydrogen] - left_metals - {left, right}
            for hydrogen in hydrogens
        ):
            continue
        solvent.update({left, right, *hydrogens})

    return solvent


def _read_node_candidate_count(path: Path) -> tuple[str, int] | None:
    lines = path.read_text(encoding="utf-8").splitlines()
    atom_count = int(lines[0])
    atom_rows = [line.split() for line in lines[2 : atom_count + 2]]
    if len(atom_rows) != atom_count:
        raise ValueError(f"truncated atom table: {path}")

    labels = [str(index) for index in range(atom_count)]
    symbols = {
        label: row[0] for label, row in zip(labels, atom_rows, strict=True)
    }
    positions = {
        label: tuple(float(value) for value in row[1:4])
        for label, row in zip(labels, atom_rows, strict=True)
    }
    adjacency = {label: set() for label in labels}
    bond_lengths: dict[tuple[str, str], float] = {}
    for line in lines[atom_count + 2 :]:
        fields = line.split()
        if len(fields) < 2:
            continue
        left, right = fields[:2]
        if left not in symbols or right not in symbols:
            raise ValueError(f"invalid bond endpoint in {path}: {line!r}")
        adjacency[left].add(right)
        adjacency[right].add(left)
        vector = tuple(
            positions[right][axis] - positions[left][axis] for axis in range(3)
        )
        bond_lengths[tuple(sorted((left, right)))] = math.sqrt(
            sum(value * value for value in vector)
        )

    candidates = _candidate_atoms(symbols, adjacency, bond_lengths)
    if not candidates:
        return None
    return path.stem, len(candidates)


def discover_dirty_nodes(node_dir: Path, workers: int) -> dict[str, int]:
    """Map contaminated metal-node IDs to removable atoms per instance."""

    paths = sorted(node_dir.glob("m_node*.xyz"))
    if not paths:
        raise FileNotFoundError(f"no metal-node XYZ files found in {node_dir}")
    with ThreadPoolExecutor(max_workers=min(workers, 64)) as executor:
        rows = executor.map(_read_node_candidate_count, paths)
        return {name: count for row in rows if row for name, count in (row,)}


def _readonly_database_uri(path: Path) -> str:
    return f"{path.resolve().as_uri()}?mode=ro&immutable=1"


def affected_cif_paths(
    database: Path,
    mofs_root: Path,
    node_dir: Path,
    dirty_nodes: dict[str, int],
    limit: int,
) -> list[Path]:
    """Use the generation index to select only CIFs using dirty nodes."""

    connection = sqlite3.connect(_readonly_database_uri(database), uri=True)
    try:
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.execute("CREATE TEMP TABLE dirty(path TEXT PRIMARY KEY)")
        node_paths = [
            str((node_dir / f"{name}.xyz").resolve()) for name in dirty_nodes
        ]
        connection.executemany(
            "INSERT INTO dirty(path) VALUES (?)",
            ((path,) for path in node_paths),
        )
        sql = """
            SELECT samples.cif_path
            FROM samples
            WHERE EXISTS (
                SELECT 1
                FROM json_each(samples.nodes_json) AS node
                JOIN dirty
                  ON json_extract(node.value, '$.source_path') = dirty.path
            )
            ORDER BY samples.id
        """
        parameters: tuple[int, ...] = ()
        if limit:
            sql += " LIMIT ?"
            parameters = (limit,)
        relative_paths = [row[0] for row in connection.execute(sql, parameters)]
    finally:
        connection.close()

    dataset_root = database.resolve().parent
    expected_prefix = f"{mofs_root.resolve().name}/"
    paths = []
    for relative in relative_paths:
        if not relative.startswith(expected_prefix):
            raise ValueError(f"indexed CIF is outside {mofs_root}: {relative}")
        paths.append(dataset_root / relative)
    return paths


def _find_line(lines: list[bytes], token: bytes, start: int = 0) -> int:
    for index in range(start, len(lines)):
        if lines[index].strip() == token:
            return index
    raise ValueError(f"missing CIF field {token.decode('ascii')}")


def transform_cif(content: bytes, dirty_counts: dict[str, int]) -> CifTransform:
    """Validate and remove coordinated-solvent atom and bond rows."""

    if MARKER in content[:4096]:
        return CifTransform(None, "already_clean")

    block_ids = [match.decode("ascii") for match in NODE_BLOCK_PATTERN.findall(content)]
    expected_atoms = sum(dirty_counts.get(block_id, 0) for block_id in block_ids)
    if expected_atoms == 0:
        raise ValueError("CIF header contains no expected contaminated node instance")

    lines = content.splitlines(keepends=True)
    atom_header = _find_line(lines, b"_atom_type_partial_charge")
    atom_start = atom_header + 1
    atom_end = _find_line(lines, b"loop_", atom_start)
    bond_header = _find_line(lines, b"_ccdc_geom_bond_type", atom_end)
    bond_start = bond_header + 1

    symbols: dict[str, str] = {}
    atom_line_by_label: dict[str, int] = {}
    for line_index in range(atom_start, atom_end):
        fields = lines[line_index].split()
        if not fields:
            continue
        if len(fields) != 6:
            raise ValueError(f"invalid atom row at line {line_index + 1}")
        label = fields[0].decode("ascii")
        if label in symbols:
            raise ValueError(f"duplicate atom label {label}")
        symbols[label] = fields[1].decode("ascii")
        atom_line_by_label[label] = line_index
    if not symbols:
        raise ValueError("empty atom table")

    adjacency = {label: set() for label in symbols}
    bond_lengths: dict[tuple[str, str], float] = {}
    bond_rows: list[tuple[int, str, str]] = []
    for line_index in range(bond_start, len(lines)):
        fields = lines[line_index].split()
        if not fields:
            continue
        if len(fields) != 5:
            raise ValueError(f"invalid bond row at line {line_index + 1}")
        left = fields[0].decode("ascii")
        right = fields[1].decode("ascii")
        if left not in symbols or right not in symbols:
            raise ValueError(f"unknown bond endpoint at line {line_index + 1}")
        adjacency[left].add(right)
        adjacency[right].add(left)
        pair = tuple(sorted((left, right)))
        bond_lengths[pair] = float(fields[2])
        bond_rows.append((line_index, left, right))

    candidates = _candidate_atoms(symbols, adjacency, bond_lengths)
    if len(candidates) != expected_atoms:
        raise ValueError(
            "candidate count does not match declared building-block instances: "
            f"detected={len(candidates)}, expected={expected_atoms}, "
            f"blocks={block_ids}"
        )

    removed_lines = {atom_line_by_label[label] for label in candidates}
    removed_bonds = 0
    for line_index, left, right in bond_rows:
        if left in candidates or right in candidates:
            removed_lines.add(line_index)
            removed_bonds += 1

    first_line_end = content.find(b"\n")
    if first_line_end < 0:
        raise ValueError("CIF contains no newline")
    new_lines = [
        line for line_index, line in enumerate(lines) if line_index not in removed_lines
    ]
    new_lines.insert(1, MARKER)
    transformed = b"".join(new_lines)
    return CifTransform(
        transformed,
        "would_clean",
        removed_atoms=len(candidates),
        removed_bonds=removed_bonds,
        bytes_saved=len(content) - len(transformed),
    )


def _atomic_replace(path: Path, content: bytes) -> None:
    original_mode = stat.S_IMODE(path.stat().st_mode)
    temporary = path.with_name(f".{path.name}.solvent-clean-{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
        os.chmod(temporary, original_mode)
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _initialize_worker(dirty_counts: dict[str, int], dry_run: bool) -> None:
    global _WORKER_DIRTY_COUNTS, _WORKER_DRY_RUN
    _WORKER_DIRTY_COUNTS = dirty_counts
    _WORKER_DRY_RUN = dry_run


def _process_batch(raw_paths: tuple[str, ...]) -> BatchResult:
    cleaned = 0
    already_clean = 0
    missing = 0
    removed_atoms = 0
    removed_bonds = 0
    bytes_saved = 0
    errors = 0
    examples = []
    for raw_path in raw_paths:
        path = Path(raw_path)
        try:
            transform = transform_cif(path.read_bytes(), _WORKER_DIRTY_COUNTS)
            if transform.status == "already_clean":
                already_clean += 1
                continue
            if transform.content is None:
                raise RuntimeError("cleaning transform unexpectedly has no content")
            if not _WORKER_DRY_RUN:
                _atomic_replace(path, transform.content)
            cleaned += 1
            removed_atoms += transform.removed_atoms
            removed_bonds += transform.removed_bonds
            bytes_saved += transform.bytes_saved
        except FileNotFoundError:
            missing += 1
        except Exception as error:  # noqa: BLE001 - isolate malformed CIFs
            errors += 1
            if len(examples) < 5:
                examples.append(f"{path}: {type(error).__name__}: {error}")
    return BatchResult(
        processed=len(raw_paths),
        cleaned=cleaned,
        already_clean=already_clean,
        missing=missing,
        removed_atoms=removed_atoms,
        removed_bonds=removed_bonds,
        bytes_saved=bytes_saved,
        errors=errors,
        error_examples=tuple(examples),
    )


def _batches(values: Iterable[Path], size: int) -> Iterator[tuple[str, ...]]:
    batch = []
    for value in values:
        batch.append(str(value))
        if len(batch) == size:
            yield tuple(batch)
            batch.clear()
    if batch:
        yield tuple(batch)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--mofs-root", type=Path, default=DEFAULT_MOFS_ROOT)
    parser.add_argument("--node-dir", type=Path, default=DEFAULT_NODE_DIR)
    parser.add_argument("--workers", type=int, default=min(64, os.cpu_count() or 1))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--progress-every", type=int, default=10_000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path in (args.database, args.mofs_root, args.node_dir):
        if not path.exists():
            raise FileNotFoundError(path)
    if min(args.workers, args.batch_size, args.progress_every) < 1:
        raise ValueError("workers, batch size, and progress interval must be positive")
    if args.limit < 0:
        raise ValueError("limit cannot be negative")

    started = time.monotonic()
    dirty_nodes = discover_dirty_nodes(args.node_dir, args.workers)
    print(f"dirty_nodes={len(dirty_nodes)}", flush=True)
    paths = affected_cif_paths(
        args.database,
        args.mofs_root,
        args.node_dir,
        dirty_nodes,
        args.limit,
    )
    print(
        f"selected_cifs={len(paths)} dry_run={args.dry_run} workers={args.workers}",
        flush=True,
    )

    totals = {
        "processed": 0,
        "cleaned": 0,
        "already_clean": 0,
        "missing": 0,
        "removed_atoms": 0,
        "removed_bonds": 0,
        "bytes_saved": 0,
        "errors": 0,
    }
    error_examples: list[str] = []
    next_progress = args.progress_every
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_initialize_worker,
        initargs=(dirty_nodes, args.dry_run),
    ) as executor:
        results = executor.map(
            _process_batch,
            _batches(paths, args.batch_size),
            chunksize=1,
        )
        for result in results:
            for key in totals:
                totals[key] += getattr(result, key)
            if len(error_examples) < 100:
                error_examples.extend(
                    result.error_examples[: 100 - len(error_examples)]
                )
            if totals["processed"] >= next_progress:
                elapsed = time.monotonic() - started
                print(
                    f"processed={totals['processed']}/{len(paths)} "
                    f"cleaned={totals['cleaned']} "
                    f"already_clean={totals['already_clean']} "
                    f"missing={totals['missing']} "
                    f"errors={totals['errors']} rate={totals['processed']/elapsed:.1f}/s",
                    flush=True,
                )
                next_progress += args.progress_every

    elapsed = time.monotonic() - started
    summary = {
        **totals,
        "selected_cifs": len(paths),
        "dirty_nodes": len(dirty_nodes),
        "dry_run": args.dry_run,
        "elapsed_seconds": round(elapsed, 3),
        "rate_cifs_per_second": round(totals["processed"] / elapsed, 3),
        "gib_saved": round(totals["bytes_saved"] / (1024**3), 6),
        "error_examples": error_examples,
    }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 1 if totals["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

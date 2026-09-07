#!/usr/bin/env python3
"""Merge independent MOF runs into one deduplicated, resumable output."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import time


SHARD_SIZE = 10_000
COLUMNS = (
    "signature,cif_sha256,topology,nodes_json,edges_json,quality_json,"
    "detected_space_group_number,generation_tier,generation_phase,created_at"
)


def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--target", type=positive, default=2_000_000)
    parser.add_argument("--all", dest="import_all", action="store_true")
    parser.add_argument("--commit-every", type=positive, default=1_000)
    parser.add_argument("--file-workers", type=positive, default=16)
    return parser.parse_args()


def database(root: Path) -> Path:
    path = root / "generation.sqlite3"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def running_generator(root: Path) -> int | None:
    marker = os.fsencode(str(root))
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            command = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if b"generate_500k_single_linker_mofs.py" in command and marker in command:
            return int(entry.name)
    return None


def source_file(root: Path, relative: str) -> Path:
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise FileNotFoundError(f"invalid source CIF: {relative}")
    return root / relative_path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect(source: Path, destination: Path, expected_hash: str) -> bool:
    try:
        os.link(source, destination)
    except OSError as error:
        if error.errno == errno.EEXIST:
            if file_sha256(destination) != expected_hash:
                raise FileExistsError(f"conflicting destination CIF: {destination}")
            return False
        if error.errno not in {errno.EXDEV, errno.EPERM, errno.EOPNOTSUPP}:
            raise
        shutil.copy2(source, destination)
    return True


def collect_task(task: tuple[Path, Path, str]) -> tuple[Path, bool, Exception | None]:
    source, destination, expected_hash = task
    try:
        return destination, collect(source, destination, expected_hash), None
    except Exception as error:
        return destination, False, error


def set_meta(connection: sqlite3.Connection, key: str, value: object) -> None:
    connection.execute(
        "INSERT INTO meta(key,value) VALUES(?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, str(value)),
    )


def merge_source(
    connection: sqlite3.Connection,
    destination: Path,
    source: Path,
    target: int | None,
    commit_every: int,
    file_workers: int,
) -> tuple[int, int]:
    source_connection = sqlite3.connect(f"file:{database(source)}?mode=ro", uri=True)
    source_connection.row_factory = sqlite3.Row
    seen_signatures, seen_hashes = set(), set()
    for signature, cif_hash in connection.execute(
        "SELECT signature,cif_sha256 FROM samples"
    ):
        seen_signatures.add(signature)
        seen_hashes.add(cif_hash)
    print(f"merge_index_loaded={len(seen_signatures)}", flush=True)

    imported = duplicates = 0
    next_id = int(connection.execute("SELECT COALESCE(MAX(id),0)+1 FROM samples").fetchone()[0])
    records, tasks, directories = [], [], set()

    def commit_batch(executor: ThreadPoolExecutor) -> None:
        nonlocal imported
        if not records:
            return
        results = list(executor.map(collect_task, tasks))
        errors = [error for _, _, error in results if error]
        if errors:
            for path, created, _ in results:
                if created:
                    path.unlink(missing_ok=True)
            raise errors[0]
        try:
            connection.executemany(
                f"INSERT INTO samples(id,sample_id,cif_path,cif_size,{COLUMNS}) "
                f"VALUES(?,?,?, ?,{','.join('?' for _ in COLUMNS.split(','))})",
                records,
            )
            connection.commit()
        except BaseException:
            connection.rollback()
            for path, created, _ in results:
                if created:
                    path.unlink(missing_ok=True)
            raise
        imported += len(records)
        records.clear()
        tasks.clear()
        print(
            f"merge_progress={next_id - 1}/{target if target is not None else 'all'} "
            f"source={source.name} duplicates={duplicates}",
            flush=True,
        )

    try:
        with ThreadPoolExecutor(max_workers=file_workers) as executor:
            rows = source_connection.execute(
                f"SELECT cif_path,cif_size,{COLUMNS} FROM samples ORDER BY id"
            )
            for row in rows:
                if target is not None and next_id > target:
                    break
                if (
                    row["signature"] in seen_signatures
                    or row["cif_sha256"] in seen_hashes
                ):
                    duplicates += 1
                    continue

                sample_id = f"single_linker_mof_{next_id:07d}_{row['topology']}"
                shard = (next_id - 1) // SHARD_SIZE
                relative = Path("mofs") / f"{shard:05d}" / f"{sample_id}.cif"
                cif = destination / relative
                if cif.parent not in directories:
                    cif.parent.mkdir(parents=True, exist_ok=True)
                    directories.add(cif.parent)
                tasks.append((source_file(source, row["cif_path"]), cif, row["cif_sha256"]))
                records.append(
                    (
                        next_id,
                        sample_id,
                        str(relative),
                        row["cif_size"],
                        *(row[column] for column in COLUMNS.split(",")),
                    )
                )
                seen_signatures.add(row["signature"])
                seen_hashes.add(row["cif_sha256"])
                next_id += 1
                if len(records) == commit_every:
                    commit_batch(executor)
            commit_batch(executor)
    finally:
        source_connection.close()
    return imported, duplicates


def main() -> None:
    args = arguments()
    destination = args.destination.expanduser().resolve()
    sources = [path.expanduser().resolve() for path in args.source]
    if destination in sources:
        raise ValueError("destination cannot also be a source")
    for root in (destination, *sources):
        pid = running_generator(root)
        if pid:
            raise RuntimeError(f"generation is still running for {root} (PID {pid})")

    connection = sqlite3.connect(database(destination), timeout=120)
    connection.row_factory = sqlite3.Row
    count, maximum = connection.execute(
        "SELECT COUNT(*),COALESCE(MAX(id),0) FROM samples"
    ).fetchone()
    if count != maximum:
        raise RuntimeError("destination sample IDs must be contiguous")
    target = None if args.import_all else args.target
    if target is not None and count > target:
        raise RuntimeError(f"destination already contains {count} samples")

    started = time.monotonic()
    results = {}
    for source in sources:
        imported, duplicates = merge_source(
            connection,
            destination,
            source,
            target,
            args.commit_every,
            args.file_workers,
        )
        results[str(source)] = {"imported": imported, "duplicates": duplicates}
        if target is not None and connection.execute("SELECT COUNT(*) FROM samples").fetchone()[0] >= target:
            break

    final = int(connection.execute("SELECT COUNT(*) FROM samples").fetchone()[0])
    p1 = int(
        connection.execute(
            "SELECT COUNT(*) FROM samples WHERE detected_space_group_number=1"
        ).fetchone()[0]
    )
    summary = {
        "status": "complete" if target is None or final == target else "shortfall",
        "target": target,
        "mode": "all_unique" if target is None else "target",
        "starting_count": count,
        "final_count": final,
        "p1_count": p1,
        "p1_fraction": p1 / final if final else 0.0,
        "sources": results,
        "elapsed_seconds": time.monotonic() - started,
        "updated_at": time.time(),
    }
    set_meta(connection, "target_count", final if target is None else target)
    set_meta(connection, "last_merge", json.dumps(summary, separators=(",", ":")))
    connection.commit()
    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    connection.close()
    (destination / "merge_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"merge_status={summary['status']} final={final}/{target if target is not None else 'all'} "
        f"p1_fraction={summary['p1_fraction']:.4f}",
        flush=True,
    )
    if target is not None and final != target:
        raise RuntimeError(f"only {final} unique MOFs are available; generate more")


if __name__ == "__main__":
    main()

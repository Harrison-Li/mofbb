#!/usr/bin/env python3
"""Randomly assemble diverse MOFs from the CoRE-MOF BB LMDB and PORMAKE nets."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import logging
import lmdb
import os
import pickle
import random
import shutil
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import ase.neighborlist
import numpy as np
from pymatgen.core import Structure

from mofbb.compose import Builder, BuildingBlock, Topology
from mofbb.compose.log import logger


LMDB_PATH = Path("/nanolab/libo/mof_builder/dataset/core_mofs_bbs_asr.lmdb")
TOPOLOGY_DIR = Path(
    "/nanolab/libo/mof_builder/PORMAKE/src/pormake/database/topologies"
)
DEFAULT_OUTPUT = Path("/nanolab/libo/mof_builder/mof_recover")
DEFAULT_COUNT = 100
DEFAULT_SEED = 42
MAX_TOPOLOGY_SLOTS = 30
MAX_ATTEMPTS_PER_TOPOLOGY = 12
MAX_NODE_ATOMS = 120
MAX_EDGE_ATOMS = 80
MAX_RMSD = 0.60
MIN_ATOM_DISTANCE = 0.50
ACCURACY = 4


@dataclass(frozen=True)
class BlockRecord:
    key: str
    source_key: str
    structure_id: str
    bb_name: str
    kind: str
    xyz: str
    num_atoms: int
    coordination: int
    sha256: str


@dataclass(frozen=True)
class TopologyRecord:
    name: str
    cgd_path: Path
    spacegroup: str
    num_nodes: int
    num_edges: int
    node_type_cns: tuple[tuple[int, int], ...]
    edge_types: tuple[tuple[int, int], ...]

    @property
    def num_slots(self) -> int:
        return self.num_nodes + self.num_edges


def configure_builder_logging() -> None:
    """Deduplicate the handlers shared by MOFBB and PORMAKE imports."""

    seen_handlers = set()
    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler):
            identity = ("file", os.path.realpath(handler.baseFilename))
        else:
            identity = ("stream", id(getattr(handler, "stream", None)))
        if identity in seen_handlers:
            logger.removeHandler(handler)
            handler.close()
            continue
        seen_handlers.add(identity)
        handler.setLevel(logging.WARNING)
    logger.setLevel(logging.WARNING)


def load_block_pools() -> tuple[dict[tuple[str, int], list[BlockRecord]], int]:
    pools: dict[tuple[str, int], list[BlockRecord]] = defaultdict(list)
    skipped = 0
    environment = lmdb.open(
        str(LMDB_PATH),
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        max_readers=1,
    )
    try:
        with environment.begin(buffers=False) as transaction:
            for raw_key, compressed in transaction.cursor():
                if raw_key == b"__metadata__":
                    continue
                try:
                    data = json.loads(gzip.decompress(compressed))
                    xyz = data["bb_xyz"]
                    lines = xyz.splitlines()
                    num_atoms = int(lines[0])
                    atom_lines = lines[2 : 2 + num_atoms]
                    coordination = sum(
                        line.split()[0] == "X" for line in atom_lines
                    )
                    kind = data["bb_kind"]
                    atom_limit = MAX_EDGE_ATOMS if kind == "edge" else MAX_NODE_ATOMS
                    if kind == "edge" and coordination != 2:
                        skipped += 1
                        continue
                    if kind in {"metal_node", "organic_node"} and coordination < 3:
                        skipped += 1
                        continue
                    if kind not in {"edge", "metal_node", "organic_node"}:
                        skipped += 1
                        continue
                    if num_atoms > atom_limit:
                        skipped += 1
                        continue
                    record = BlockRecord(
                        key=raw_key.decode("utf-8"),
                        source_key=data["source_key"],
                        structure_id=data["structure_id"],
                        bb_name=data["bb_name"],
                        kind=kind,
                        xyz=xyz,
                        num_atoms=num_atoms,
                        coordination=coordination,
                        sha256=hashlib.sha256(xyz.encode("utf-8")).hexdigest(),
                    )
                    pools[(kind, coordination)].append(record)
                except (KeyError, ValueError, IndexError, gzip.BadGzipFile):
                    skipped += 1
    finally:
        environment.close()
    return pools, skipped


def load_topologies(
    pools: dict[tuple[str, int], list[BlockRecord]],
    max_slots: int,
) -> tuple[list[TopologyRecord], int]:
    records = []
    skipped = 0
    for pickle_path in sorted(TOPOLOGY_DIR.glob("*.pickle")):
        cgd_path = pickle_path.with_suffix(".cgd")
        if not cgd_path.is_file():
            skipped += 1
            continue
        try:
            with pickle_path.open("rb") as handle:
                topology = pickle.load(handle)
            node_type_cns = tuple(
                (int(node_type), int(cn))
                for node_type, cn in zip(
                    topology.unique_node_types, topology.unique_cn
                )
            )
            edge_types = tuple(
                tuple(int(value) for value in edge_type)
                for edge_type in topology.unique_edge_types
            )
            record = TopologyRecord(
                name=pickle_path.stem,
                cgd_path=cgd_path,
                spacegroup=str(topology.spacegroup),
                num_nodes=int(topology.n_nodes),
                num_edges=int(topology.n_edges),
                node_type_cns=node_type_cns,
                edge_types=edge_types,
            )
        except Exception:
            skipped += 1
            continue

        cns = [cn for _, cn in node_type_cns]
        has_metal = any(pools.get(("metal_node", cn)) for cn in cns)
        has_all_nodes = all(
            pools.get(("metal_node", cn)) or pools.get(("organic_node", cn))
            for cn in cns
        )
        if (
            record.num_slots > max_slots
            or len(node_type_cns) > 2
            or not edge_types
            or not has_metal
            or not has_all_nodes
        ):
            skipped += 1
            continue
        records.append(record)
    return records, skipped


def choose_block(
    rng: random.Random,
    candidates: list[BlockRecord],
    unavailable_sources: set[str],
    unavailable_hashes: set[str],
) -> BlockRecord:
    available = [
        record
        for record in candidates
        if record.structure_id not in unavailable_sources
        and record.sha256 not in unavailable_hashes
    ]
    if not available:
        raise ValueError("no unused compatible building block remains")
    return rng.choice(available)


def select_blocks(
    rng: random.Random,
    topology: TopologyRecord,
    pools: dict[tuple[str, int], list[BlockRecord]],
    used_sources: set[str],
    used_hashes: set[str],
) -> tuple[dict[int, BlockRecord], dict[tuple[int, int], BlockRecord]]:
    selected_sources = set(used_sources)
    selected_hashes = set(used_hashes)
    metal_types = [
        node_type
        for node_type, cn in topology.node_type_cns
        if pools.get(("metal_node", cn))
    ]
    required_metal_type = rng.choice(metal_types)
    node_records = {}

    for node_type, cn in topology.node_type_cns:
        if node_type == required_metal_type:
            candidates = pools[("metal_node", cn)]
        else:
            preferred_kind = rng.choice(("organic_node", "metal_node"))
            candidates = pools.get((preferred_kind, cn), [])
            if not candidates:
                fallback = "metal_node" if preferred_kind == "organic_node" else "organic_node"
                candidates = pools.get((fallback, cn), [])
        record = choose_block(rng, candidates, selected_sources, selected_hashes)
        node_records[node_type] = record
        selected_sources.add(record.structure_id)
        selected_hashes.add(record.sha256)

    edge_records = {}
    for edge_type in topology.edge_types:
        record = choose_block(
            rng,
            pools[("edge", 2)],
            selected_sources,
            selected_hashes,
        )
        edge_records[edge_type] = record
        selected_sources.add(record.structure_id)
        selected_hashes.add(record.sha256)
    return node_records, edge_records


def block_metadata(record: BlockRecord, saved_path: Path) -> dict:
    return {
        "lmdb_key": record.key,
        "source_key": record.source_key,
        "structure_id": record.structure_id,
        "bb_name": record.bb_name,
        "kind": record.kind,
        "coordination": record.coordination,
        "num_atoms_with_x": record.num_atoms,
        "bb_sha256": record.sha256,
        "saved_xyz": str(saved_path),
    }


def write_selected_blocks(
    directory: Path,
    node_records: dict[int, BlockRecord],
    edge_records: dict[tuple[int, int], BlockRecord],
) -> tuple[dict[int, Path], dict[tuple[int, int], Path]]:
    directory.mkdir(parents=True)
    node_paths = {}
    for node_type, record in node_records.items():
        path = directory / f"node_type_{node_type}.xyz"
        path.write_text(record.xyz, encoding="utf-8")
        node_paths[node_type] = path
    edge_paths = {}
    for edge_type, record in edge_records.items():
        label = "_".join(str(value) for value in edge_type)
        path = directory / f"edge_type_{label}.xyz"
        path.write_text(record.xyz, encoding="utf-8")
        edge_paths[edge_type] = path
    return node_paths, edge_paths


def build_one(
    topology_record: TopologyRecord,
    node_paths: dict[int, Path],
    edge_paths: dict[tuple[int, int], Path],
    cif_path: Path,
) -> tuple[object, Structure, float]:
    topology = Topology(topology_record.cgd_path)
    actual_type_cns = tuple(
        (int(node_type), int(cn))
        for node_type, cn in zip(topology.unique_node_types, topology.unique_cn)
    )
    if actual_type_cns != topology_record.node_type_cns:
        raise ValueError("CGD and cached topology coordination metadata differ")

    framework = Builder().build_by_type(
        topology=topology,
        node_bbs={key: BuildingBlock(path) for key, path in node_paths.items()},
        edge_bbs={key: BuildingBlock(path) for key, path in edge_paths.items()},
        accuracy=ACCURACY,
    )
    if float(framework.info["max_rmsd"]) > MAX_RMSD:
        raise ValueError(
            f"node-placement RMSD {framework.info['max_rmsd']:.3f} exceeds {MAX_RMSD:.3f}"
        )
    if "X" in framework.atoms.get_chemical_symbols():
        raise ValueError("assembled framework still contains X sites")
    if not np.isfinite(framework.atoms.positions).all() or framework.atoms.cell.volume <= 0:
        raise ValueError("assembled framework has invalid coordinates or cell")

    distances = ase.neighborlist.neighbor_list("d", framework.atoms, 3.0)
    minimum_distance = float(np.min(distances)) if len(distances) else float("inf")
    if minimum_distance < MIN_ATOM_DISTANCE:
        raise ValueError(
            f"minimum atom distance {minimum_distance:.3f} Å is below "
            f"{MIN_ATOM_DISTANCE:.3f} Å"
        )

    framework.write_cif(cif_path)
    if not cif_path.is_file():
        raise ValueError("framework CIF writer did not produce a file")
    structure = Structure.from_file(cif_path, primitive=False)
    expected_composition = Counter(framework.atoms.get_chemical_symbols())
    written_composition = Counter(site.specie.symbol for site in structure)
    if len(structure) != len(framework.atoms) or written_composition != expected_composition:
        raise ValueError("written CIF does not preserve atom count and composition")
    return framework, structure, minimum_distance


def write_manifests(output: Path, rows: list[dict], summary: dict) -> None:
    manifest_json = output / "manifest.json"
    temporary_json = manifest_json.with_suffix(".json.tmp")
    temporary_json.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_json, manifest_json)

    if rows:
        manifest_csv = output / "manifest.csv"
        temporary_csv = manifest_csv.with_suffix(".csv.tmp")
        fields = [
            "sample_id", "cif", "topology", "spacegroup", "coordination_numbers",
            "num_atoms", "formula", "max_rmsd", "mean_rmsd", "relax_obj",
            "minimum_atom_distance", "node_lmdb_keys", "edge_lmdb_keys", "cif_sha256",
        ]
        with temporary_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "sample_id": row["sample_id"],
                        "cif": row["cif"],
                        "topology": row["topology"]["name"],
                        "spacegroup": row["topology"]["spacegroup"],
                        "coordination_numbers": json.dumps(row["topology"]["node_type_cns"]),
                        "num_atoms": row["quality"]["num_atoms"],
                        "formula": row["quality"]["formula"],
                        "max_rmsd": row["quality"]["max_rmsd"],
                        "mean_rmsd": row["quality"]["mean_rmsd"],
                        "relax_obj": row["quality"]["relax_obj"],
                        "minimum_atom_distance": row["quality"]["minimum_atom_distance"],
                        "node_lmdb_keys": json.dumps(
                            [item["block"]["lmdb_key"] for item in row["node_assignments"]]
                        ),
                        "edge_lmdb_keys": json.dumps(
                            [item["block"]["lmdb_key"] for item in row["edge_assignments"]]
                        ),
                        "cif_sha256": row["quality"]["cif_sha256"],
                    }
                )
        os.replace(temporary_csv, manifest_csv)

    summary_path = output / "generation_summary.json"
    temporary_summary = summary_path.with_suffix(".json.tmp")
    temporary_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_summary, summary_path)


def generate(output: Path, count: int, seed: int, max_slots: int) -> None:
    mofs_dir = output / "mofs"
    blocks_dir = output / "building_blocks"
    staging_dir = output / ".random_reconstruct_staging"
    protected = [mofs_dir, blocks_dir, output / "manifest.json", output / "manifest.csv"]
    existing = [path for path in protected if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing results: {existing}")

    output.mkdir(parents=True, exist_ok=True)
    mofs_dir.mkdir()
    blocks_dir.mkdir()
    staging_dir.mkdir()
    rng = random.Random(seed)
    rows: list[dict] = []
    failures: list[dict] = []
    used_sources: set[str] = set()
    used_hashes: set[str] = set()

    pools, skipped_blocks = load_block_pools()
    topologies, skipped_topologies = load_topologies(pools, max_slots)
    configure_builder_logging()
    rng.shuffle(topologies)
    print(
        f"eligible_topologies={len(topologies)} skipped_blocks={skipped_blocks} "
        f"target={count} seed={seed}",
        flush=True,
    )

    try:
        for topology_record in topologies:
            if len(rows) >= count:
                break
            attempted_combinations: set[tuple[str, ...]] = set()
            for attempt in range(1, MAX_ATTEMPTS_PER_TOPOLOGY + 1):
                attempt_dir = staging_dir / f"{topology_record.name}-{attempt}"
                try:
                    node_records, edge_records = select_blocks(
                        rng, topology_record, pools, used_sources, used_hashes
                    )
                    combination = tuple(
                        sorted(
                            record.sha256
                            for record in (*node_records.values(), *edge_records.values())
                        )
                    )
                    if combination in attempted_combinations:
                        continue
                    attempted_combinations.add(combination)
                    node_paths, edge_paths = write_selected_blocks(
                        attempt_dir / "building_blocks", node_records, edge_records
                    )
                    staged_cif = attempt_dir / "structure.cif"
                    framework, structure, minimum_distance = build_one(
                        topology_record, node_paths, edge_paths, staged_cif
                    )

                    sample_id = f"random_mof_{len(rows) + 1:03d}_{topology_record.name}"
                    final_cif = mofs_dir / f"{sample_id}.cif"
                    final_blocks = blocks_dir / sample_id
                    os.replace(staged_cif, final_cif)
                    os.replace(attempt_dir / "building_blocks", final_blocks)
                    selected = [*node_records.values(), *edge_records.values()]
                    used_sources.update(record.structure_id for record in selected)
                    used_hashes.update(record.sha256 for record in selected)
                    cell = [float(value) for value in structure.lattice.parameters]
                    row = {
                        "sample_id": sample_id,
                        "cif": str(final_cif.relative_to(output)),
                        "topology": {
                            "name": topology_record.name,
                            "cgd_path": str(topology_record.cgd_path),
                            "spacegroup": topology_record.spacegroup,
                            "num_nodes": topology_record.num_nodes,
                            "num_edges": topology_record.num_edges,
                            "node_type_cns": [list(item) for item in topology_record.node_type_cns],
                            "edge_types": [list(item) for item in topology_record.edge_types],
                        },
                        "node_assignments": [
                            {
                                "node_type": node_type,
                                "block": block_metadata(
                                    record,
                                    final_blocks.relative_to(output) / node_paths[node_type].name,
                                ),
                            }
                            for node_type, record in node_records.items()
                        ],
                        "edge_assignments": [
                            {
                                "edge_type": list(edge_type),
                                "block": block_metadata(
                                    record,
                                    final_blocks.relative_to(output) / edge_paths[edge_type].name,
                                ),
                            }
                            for edge_type, record in edge_records.items()
                        ],
                        "quality": {
                            "num_atoms": len(structure),
                            "formula": structure.composition.formula,
                            "cell_parameters": cell,
                            "max_rmsd": float(framework.info["max_rmsd"]),
                            "mean_rmsd": float(framework.info["mean_rmsd"]),
                            "relax_obj": float(framework.info["geometry_optimization_objective"]),
                            "minimum_atom_distance": minimum_distance,
                            "cif_sha256": hashlib.sha256(final_cif.read_bytes()).hexdigest(),
                        },
                    }
                    rows.append(row)
                    print(
                        f"completed={len(rows)}/{count} topology={topology_record.name} "
                        f"atoms={len(structure)} attempt={attempt}",
                        flush=True,
                    )
                    break
                except Exception as exc:
                    failures.append(
                        {
                            "topology": topology_record.name,
                            "attempt": attempt,
                            "error_type": type(exc).__name__,
                            "error": str(exc)[:1000],
                        }
                    )
                finally:
                    if attempt_dir.exists():
                        shutil.rmtree(attempt_dir)

            summary = {
                "requested": count,
                "completed": len(rows),
                "seed": seed,
                "source_lmdb": str(LMDB_PATH),
                "topology_directory": str(TOPOLOGY_DIR),
                "eligible_topologies": len(topologies),
                "skipped_topologies": skipped_topologies,
                "skipped_blocks": skipped_blocks,
                "failed_attempts": len(failures),
                "failure_types": dict(Counter(item["error_type"] for item in failures)),
                "settings": {
                    "max_topology_slots": max_slots,
                    "max_attempts_per_topology": MAX_ATTEMPTS_PER_TOPOLOGY,
                    "max_node_atoms_with_x": MAX_NODE_ATOMS,
                    "max_edge_atoms_with_x": MAX_EDGE_ATOMS,
                    "max_rmsd": MAX_RMSD,
                    "minimum_atom_distance_angstrom": MIN_ATOM_DISTANCE,
                    "accuracy": ACCURACY,
                    "unique_topologies": True,
                    "globally_unique_bb_hashes": True,
                    "globally_unique_bb_source_structures": True,
                },
                "failures": failures,
            }
            write_manifests(output, rows, summary)
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)

    if len(rows) != count:
        raise RuntimeError(f"generated {len(rows)} of {count} requested MOFs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-topology-slots", type=int, default=MAX_TOPOLOGY_SLOTS)
    args = parser.parse_args()
    if args.count < 1 or args.max_topology_slots < 1:
        parser.error("--count and --max-topology-slots must be positive")
    return args


def main() -> None:
    warnings.filterwarnings("ignore", message="Optimal rotation is not uniquely.*")
    logger.setLevel(logging.WARNING)
    args = parse_args()
    generate(args.output, args.count, args.seed, args.max_topology_slots)


if __name__ == "__main__":
    main()

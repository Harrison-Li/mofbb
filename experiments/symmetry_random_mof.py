#!/usr/bin/env python3
"""Build random MOF candidates from symmetry-matched blocks and PORMAKE nets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import pickle
import random
import shutil
import warnings
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import ase.neighborlist
import numpy as np
from pymatgen.core import Molecule, Structure
from pymatgen.symmetry.analyzer import PointGroupAnalyzer

from mofbb.compose import Builder, BuildingBlock, Topology
from mofbb.compose.log import logger
from mofbb.decompose.bonding import read_bonded_structure


MOF_BUILDER_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    MOF_BUILDER_ROOT / "dataset/core_mofs_bbs_symmetry_manifest.csv"
)
DEFAULT_TOPOLOGY_DIR = (
    MOF_BUILDER_ROOT / "PORMAKE/src/pormake/database/topologies"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent

POINT_GROUP_TOLERANCE = 0.3
POINT_GROUP_EIGEN_TOLERANCE = 0.01
POINT_GROUP_MATRIX_TOLERANCE = 0.1
EDGE_POINT_GROUP = "D*h"


@dataclass(frozen=True)
class GenerationConfig:
    manifest: Path
    topology_dir: Path
    output: Path
    count: int = 10
    seed: int = 42
    max_topology_slots: int = 30
    max_node_types: int = 2
    max_attempts_per_topology: int = 12
    max_node_atoms: int = 120
    max_edge_atoms: int = 80
    max_rmsd: float = 0.60
    minimum_atom_distance: float = 0.50
    accuracy: int = 4
    requested_space_groups: tuple[str, ...] = ()


@dataclass(frozen=True)
class BlockRecord:
    path: Path
    source_key: str
    structure_id: str
    kind: str
    num_atoms: int
    coordination: int
    port_point_group: str
    molecular_point_group: str
    chirality: str
    linker_end_equivalence: str
    sha256: str


@dataclass(frozen=True)
class NodeSite:
    node_type: int
    coordination: int
    point_group: str


@dataclass(frozen=True)
class TopologyRecord:
    name: str
    cgd_path: Path
    space_group: str
    num_nodes: int
    num_edges: int
    node_sites: tuple[NodeSite, ...]
    edge_types: tuple[tuple[int, int], ...]

    @property
    def num_slots(self) -> int:
        return self.num_nodes + self.num_edges


def configure_quiet_builder_logging() -> None:
    """Silence duplicate MOFBB/PORMAKE progress handlers in batch runs."""

    seen: set[tuple[str, object]] = set()
    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler):
            identity = ("file", os.path.realpath(handler.baseFilename))
        else:
            identity = ("stream", id(getattr(handler, "stream", None)))
        if identity in seen:
            logger.removeHandler(handler)
            handler.close()
        else:
            seen.add(identity)
    logger.disabled = True


def load_block_pools(
    config: GenerationConfig,
) -> tuple[dict[tuple[str, int, str], list[BlockRecord]], Counter[str]]:
    """Load blocks whose port and molecular symmetries strictly agree."""

    pools: dict[tuple[str, int, str], list[BlockRecord]] = defaultdict(list)
    skipped: Counter[str] = Counter()
    with config.manifest.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                kind = row["bb_kind"]
                coordination = int(row["coordination_number"])
                num_atoms = int(row["num_atoms"])
                path = Path(row["output_path"])
                point_group = row["port_point_group"]
                molecular_group = row["molecular_point_group"]
            except (KeyError, TypeError, ValueError):
                skipped["malformed_manifest_row"] += 1
                continue

            if row.get("classification_status") != "complete":
                skipped["incomplete_symmetry_classification"] += 1
                continue
            if kind not in {"metal_node", "organic_node", "edge"}:
                skipped["unsupported_kind"] += 1
                continue
            if not path.is_file():
                skipped["missing_xyz"] += 1
                continue
            atom_limit = config.max_edge_atoms if kind == "edge" else config.max_node_atoms
            if num_atoms > atom_limit:
                skipped["atom_limit"] += 1
                continue
            if (
                point_group in {"C1", "unclassified"}
                or molecular_group in {"C1", "unclassified"}
            ):
                skipped["trivial_or_unclassified_symmetry"] += 1
                continue
            if kind == "edge" and (
                coordination != 2
                or point_group != EDGE_POINT_GROUP
                or molecular_group != EDGE_POINT_GROUP
                or row.get("chirality") != "achiral"
                or row.get("linker_end_equivalence") != "equivalent"
            ):
                skipped["edge_symmetry"] += 1
                continue
            if kind != "edge" and coordination < 3:
                skipped["node_coordination"] += 1
                continue
            if kind != "edge" and point_group != molecular_group:
                skipped["node_symmetry_mismatch"] += 1
                continue

            record = BlockRecord(
                path=path,
                source_key=row["source_key"],
                structure_id=row["structure_id"],
                kind=kind,
                num_atoms=num_atoms,
                coordination=coordination,
                port_point_group=point_group,
                molecular_point_group=molecular_group,
                chirality=row["chirality"],
                linker_end_equivalence=row["linker_end_equivalence"],
                sha256=row["xyz_sha256"],
            )
            pools[(kind, coordination, point_group)].append(record)
    return dict(pools), skipped


def point_group_from_directions(directions: np.ndarray) -> str:
    analyzer = PointGroupAnalyzer(
        Molecule(["He"] * len(directions), directions),
        tolerance=POINT_GROUP_TOLERANCE,
        eigen_tolerance=POINT_GROUP_EIGEN_TOLERANCE,
        matrix_tolerance=POINT_GROUP_MATRIX_TOLERANCE,
    )
    return analyzer.sch_symbol


def topology_record_from_object(topology: object, path: Path) -> TopologyRecord:
    sites: list[NodeSite] = []
    for node_type, coordination in zip(
        topology.unique_node_types, topology.unique_cn, strict=True
    ):
        node_type = int(node_type)
        representative = next(
            int(index)
            for index in topology.node_indices
            if int(topology.get_node_type(index)) == node_type
        )
        directions = topology.local_structure(representative).positions
        sites.append(
            NodeSite(
                node_type=node_type,
                coordination=int(coordination),
                point_group=point_group_from_directions(directions),
            )
        )

    return TopologyRecord(
        name=str(topology.name),
        cgd_path=path.with_suffix(".cgd"),
        space_group=str(topology.spacegroup),
        num_nodes=int(topology.n_nodes),
        num_edges=int(topology.n_edges),
        node_sites=tuple(sites),
        edge_types=tuple(
            tuple(int(value) for value in edge_type)
            for edge_type in topology.unique_edge_types
        ),
    )


def load_eligible_topologies(
    config: GenerationConfig,
    pools: dict[tuple[str, int, str], list[BlockRecord]],
) -> tuple[list[TopologyRecord], Counter[str]]:
    records: list[TopologyRecord] = []
    skipped: Counter[str] = Counter()
    requested_groups = set(config.requested_space_groups)

    pickle_paths = sorted(config.topology_dir.glob("*.pickle"))
    for pickle_path in pickle_paths:
        if not pickle_path.with_suffix(".cgd").is_file():
            skipped["missing_cgd"] += 1
            continue
        try:
            # These are trusted PORMAKE database pickles, not user-supplied files.
            with pickle_path.open("rb") as handle:
                topology = pickle.load(handle)
        except Exception:
            skipped["unreadable_topology"] += 1
            continue
        topology_space_group = str(topology.spacegroup)
        if requested_groups and topology_space_group not in requested_groups:
            skipped["space_group_filter"] += 1
            continue
        if int(topology.n_slots) > config.max_topology_slots:
            skipped["slot_limit"] += 1
            continue
        if (
            len(topology.unique_node_types) > config.max_node_types
            or not len(topology.unique_edge_types)
        ):
            skipped["type_limit"] += 1
            continue
        try:
            record = topology_record_from_object(topology, pickle_path)
        except Exception:
            skipped["point_group_classification"] += 1
            continue
        if not all(
            pools.get(("metal_node", site.coordination, site.point_group))
            or pools.get(("organic_node", site.coordination, site.point_group))
            for site in record.node_sites
        ):
            skipped["missing_exact_node_match"] += 1
            continue
        if not any(
            pools.get(("metal_node", site.coordination, site.point_group))
            for site in record.node_sites
        ):
            skipped["no_metal_node_match"] += 1
            continue
        if not pools.get(("edge", 2, EDGE_POINT_GROUP)):
            skipped["no_edge_match"] += 1
            continue
        records.append(record)
    return records, skipped


def choose_block(
    rng: random.Random,
    candidates: list[BlockRecord],
    selected_sources: set[str],
    selected_hashes: set[str],
    globally_used_hashes: set[str],
) -> BlockRecord:
    locally_unique = [
        block
        for block in candidates
        if block.structure_id not in selected_sources and block.sha256 not in selected_hashes
    ]
    if not locally_unique:
        raise ValueError("no locally unique symmetry-compatible block remains")
    globally_new = [
        block for block in locally_unique if block.sha256 not in globally_used_hashes
    ]
    return rng.choice(globally_new or locally_unique)


def select_blocks(
    rng: random.Random,
    topology: TopologyRecord,
    pools: dict[tuple[str, int, str], list[BlockRecord]],
    globally_used_hashes: set[str],
) -> tuple[dict[int, BlockRecord], dict[tuple[int, int], BlockRecord]]:
    metal_sites = [
        site
        for site in topology.node_sites
        if pools.get(("metal_node", site.coordination, site.point_group))
    ]
    required_metal_type = rng.choice(metal_sites).node_type
    selected_sources: set[str] = set()
    selected_hashes: set[str] = set()
    nodes: dict[int, BlockRecord] = {}

    for site in topology.node_sites:
        if site.node_type == required_metal_type:
            kind = "metal_node"
        else:
            kinds = [
                kind
                for kind in ("organic_node", "metal_node")
                if pools.get((kind, site.coordination, site.point_group))
            ]
            kind = rng.choice(kinds)
        block = choose_block(
            rng,
            pools[(kind, site.coordination, site.point_group)],
            selected_sources,
            selected_hashes,
            globally_used_hashes,
        )
        nodes[site.node_type] = block
        selected_sources.add(block.structure_id)
        selected_hashes.add(block.sha256)

    edges: dict[tuple[int, int], BlockRecord] = {}
    edge_candidates = pools[("edge", 2, EDGE_POINT_GROUP)]
    for edge_type in topology.edge_types:
        block = choose_block(
            rng,
            edge_candidates,
            selected_sources,
            selected_hashes,
            globally_used_hashes,
        )
        edges[edge_type] = block
        selected_sources.add(block.structure_id)
        selected_hashes.add(block.sha256)
    return nodes, edges


def copy_selected_blocks(
    directory: Path,
    nodes: dict[int, BlockRecord],
    edges: dict[tuple[int, int], BlockRecord],
) -> tuple[dict[int, Path], dict[tuple[int, int], Path]]:
    directory.mkdir(parents=True)
    node_paths: dict[int, Path] = {}
    for node_type, record in nodes.items():
        target = directory / f"node_type_{node_type}.xyz"
        shutil.copy2(record.path, target)
        node_paths[node_type] = target
    edge_paths: dict[tuple[int, int], Path] = {}
    for edge_type, record in edges.items():
        label = "_".join(str(value) for value in edge_type)
        target = directory / f"edge_type_{label}.xyz"
        shutil.copy2(record.path, target)
        edge_paths[edge_type] = target
    return node_paths, edge_paths


def minimum_distance(atoms: object) -> float | None:
    distances = ase.neighborlist.neighbor_list("d", atoms, 3.0)
    return float(np.min(distances)) if len(distances) else None


def build_and_validate(
    config: GenerationConfig,
    topology_record: TopologyRecord,
    node_paths: dict[int, Path],
    edge_paths: dict[tuple[int, int], Path],
    cif_path: Path,
) -> dict[str, object]:
    topology = Topology(topology_record.cgd_path)
    framework = Builder().build_by_type(
        topology=topology,
        node_bbs={key: BuildingBlock(path) for key, path in node_paths.items()},
        edge_bbs={key: BuildingBlock(path) for key, path in edge_paths.items()},
        accuracy=config.accuracy,
    )
    max_rmsd = float(framework.info["max_rmsd"])
    if max_rmsd > config.max_rmsd:
        raise ValueError(
            f"node-placement RMSD {max_rmsd:.3f} exceeds {config.max_rmsd:.3f}"
        )
    if "X" in framework.atoms.get_chemical_symbols():
        raise ValueError("assembled framework still contains X connection sites")
    if not np.isfinite(framework.atoms.positions).all() or framework.atoms.cell.volume <= 0:
        raise ValueError("assembled framework has invalid coordinates or cell")
    closest = minimum_distance(framework.atoms)
    if closest is not None and closest < config.minimum_atom_distance:
        raise ValueError(
            f"minimum atom distance {closest:.3f} A is below "
            f"{config.minimum_atom_distance:.3f} A"
        )

    framework.write_cif(cif_path)
    structure = Structure.from_file(cif_path, primitive=False)
    expected = Counter(framework.atoms.get_chemical_symbols())
    written = Counter(site.specie.symbol for site in structure)
    if len(structure) != len(framework.atoms) or written != expected:
        raise ValueError("MOFBB CIF changed atom count or composition")

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="crystal system 'triclinic' is not interpreted.*",
        )
        _, written_bonds = read_bonded_structure(cif_path)

    expected_pairs = {
        tuple(sorted((int(i), int(j))))
        for i, j in framework.bonds
    }
    written_pairs = set(written_bonds.pairs)
    if written_pairs != expected_pairs:
        raise ValueError(
            "MOFBB CIF changed bond connectivity: "
            f"expected {len(expected_pairs)} unique bonds, "
            f"found {len(written_pairs)}"
        )

    for i, j in sorted(expected_pairs):
        expected_distance = float(
            framework.atoms.get_distance(i, j, mic=True)
        )
        written_distance = float(
            np.linalg.norm(written_bonds.vector(i, j))
        )
        if not np.isclose(
            written_distance,
            expected_distance,
            atol=0.01,
            rtol=0.0,
        ):
            raise ValueError(
                f"MOFBB CIF changed bond {i}-{j} from "
                f"{expected_distance:.3f} A to "
                f"{written_distance:.3f} A during serialization"
            )

    return {
        "num_atoms": len(structure),
        "formula": structure.composition.formula,
        "cell_parameters": [float(value) for value in structure.lattice.parameters],
        "max_rmsd": max_rmsd,
        "mean_rmsd": float(framework.info["mean_rmsd"]),
        "relax_obj": float(framework.info["relax_obj"]),
        "minimum_atom_distance_angstrom": closest,
    }


def block_manifest(record: BlockRecord, saved_path: Path) -> dict[str, object]:
    data = asdict(record)
    data["path"] = str(record.path)
    data["saved_xyz"] = str(saved_path)
    return data


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def write_results(output: Path, rows: list[dict], summary: dict) -> None:
    atomic_json(output / "manifest.json", rows)
    atomic_json(output / "generation_summary.json", summary)
    temporary = output / "manifest.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "sample_id",
            "topology",
            "topology_space_group",
            "formula",
            "num_atoms",
            "max_rmsd",
            "minimum_atom_distance_angstrom",
            "cif",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "topology": row["topology"]["name"],
                    "topology_space_group": row["topology"]["space_group"],
                    "formula": row["quality"]["formula"],
                    "num_atoms": row["quality"]["num_atoms"],
                    "max_rmsd": row["quality"]["max_rmsd"],
                    "minimum_atom_distance_angstrom": row["quality"][
                        "minimum_atom_distance_angstrom"
                    ],
                    "cif": row["cif"],
                }
            )
    os.replace(temporary, output / "manifest.csv")


def generate(config: GenerationConfig) -> list[dict]:
    mofs_dir = config.output / "mofs"
    blocks_dir = config.output / "building_blocks"
    staging_dir = config.output / ".random_reconstruct_staging"
    protected = [
        mofs_dir,
        blocks_dir,
        config.output / "manifest.json",
        config.output / "manifest.csv",
    ]
    existing = [path for path in protected if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing results: {existing}")

    pools, skipped_blocks = load_block_pools(config)
    topologies, skipped_topologies = load_eligible_topologies(config, pools)
    configure_quiet_builder_logging()
    rng = random.Random(config.seed)
    rng.shuffle(topologies)
    print(
        f"eligible_topologies={len(topologies)} "
        f"skipped_blocks={sum(skipped_blocks.values())} "
        f"target={config.count} seed={config.seed}",
        flush=True,
    )
    config.output.mkdir(parents=True, exist_ok=True)
    for directory in (mofs_dir, blocks_dir, staging_dir):
        directory.mkdir()

    rows: list[dict] = []
    failures: list[dict[str, object]] = []
    globally_used_hashes: set[str] = set()
    try:
        for topology in topologies:
            if len(rows) >= config.count:
                break
            attempted_combinations: set[tuple[str, ...]] = set()
            for attempt in range(1, config.max_attempts_per_topology + 1):
                attempt_dir = staging_dir / f"{topology.name}-{attempt}"
                try:
                    nodes, edges = select_blocks(
                        rng, topology, pools, globally_used_hashes
                    )
                    combination = tuple(
                        sorted(
                            block.sha256 for block in (*nodes.values(), *edges.values())
                        )
                    )
                    if combination in attempted_combinations:
                        continue
                    attempted_combinations.add(combination)
                    node_paths, edge_paths = copy_selected_blocks(
                        attempt_dir / "building_blocks", nodes, edges
                    )
                    staged_cif = attempt_dir / "mofbb_structure.cif"
                    quality = build_and_validate(
                        config,
                        topology,
                        node_paths,
                        edge_paths,
                        staged_cif,
                    )

                    sample_id = f"random_mof_{len(rows) + 1:03d}_{topology.name}"
                    final_cif = mofs_dir / f"{sample_id}.cif"
                    final_blocks = blocks_dir / sample_id
                    os.replace(staged_cif, final_cif)
                    os.replace(attempt_dir / "building_blocks", final_blocks)
                    selected = [*nodes.values(), *edges.values()]
                    globally_used_hashes.update(block.sha256 for block in selected)

                    row = {
                        "sample_id": sample_id,
                        "cif": str(final_cif.relative_to(config.output)),
                        "topology": {
                            "name": topology.name,
                            "cgd_path": str(topology.cgd_path),
                            "space_group": topology.space_group,
                            "num_nodes": topology.num_nodes,
                            "num_edges": topology.num_edges,
                            "node_sites": [asdict(site) for site in topology.node_sites],
                            "edge_types": [list(edge_type) for edge_type in topology.edge_types],
                        },
                        "node_assignments": [
                            {
                                "node_type": node_type,
                                "block": block_manifest(
                                    block,
                                    final_blocks.relative_to(config.output)
                                    / node_paths[node_type].name,
                                ),
                            }
                            for node_type, block in nodes.items()
                        ],
                        "edge_assignments": [
                            {
                                "edge_type": list(edge_type),
                                "block": block_manifest(
                                    block,
                                    final_blocks.relative_to(config.output)
                                    / edge_paths[edge_type].name,
                                ),
                            }
                            for edge_type, block in edges.items()
                        ],
                        "quality": quality,
                    }
                    row["quality"]["cif_sha256"] = hashlib.sha256(
                        final_cif.read_bytes()
                    ).hexdigest()
                    rows.append(row)
                    print(
                        f"completed={len(rows)}/{config.count} topology={topology.name} "
                        f"atoms={quality['num_atoms']} attempt={attempt}",
                        flush=True,
                    )
                    break
                except Exception as error:
                    failures.append(
                        {
                            "topology": topology.name,
                            "attempt": attempt,
                            "error_type": type(error).__name__,
                            "error": str(error)[:1000],
                        }
                    )
                finally:
                    if attempt_dir.exists():
                        shutil.rmtree(attempt_dir)
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        for directory in (mofs_dir, blocks_dir):
            try:
                directory.rmdir()
            except OSError:
                pass
        try:
            config.output.rmdir()
        except OSError:
            pass

    summary = {
        "requested": config.count,
        "completed": len(rows),
        "seed": config.seed,
        "source_manifest": str(config.manifest),
        "topology_directory": str(config.topology_dir),
        "eligible_topologies": len(topologies),
        "skipped_blocks": dict(skipped_blocks),
        "skipped_topologies": dict(skipped_topologies),
        "failed_attempts": len(failures),
        "failure_types": dict(Counter(item["error_type"] for item in failures)),
        "settings": {
            **asdict(config),
            "manifest": str(config.manifest),
            "topology_dir": str(config.topology_dir),
            "output": str(config.output),
            "selection_rule": (
                "node CN, port point group, and molecular point group must exactly "
                "match the topology site; linkers must have D*h port and molecular "
                "symmetry, achirality, and equivalent ends"
            ),
        },
        "failures": failures,
    }
    write_results(config.output, rows, summary)
    if len(rows) != config.count:
        raise RuntimeError(
            f"generated {len(rows)} of {config.count} requested candidates; "
            f"see {config.output / 'generation_summary.json'}"
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--topology-dir", type=Path, default=DEFAULT_TOPOLOGY_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-topology-slots", type=int, default=30)
    parser.add_argument(
        "--space-group",
        action="append",
        default=[],
        help="restrict topology H-M symbols; repeat for multiple groups",
    )
    args = parser.parse_args()
    if args.count < 1 or args.max_topology_slots < 1:
        parser.error("--count and --max-topology-slots must be positive")
    if not args.manifest.is_file():
        parser.error(f"manifest not found: {args.manifest}")
    if not args.topology_dir.is_dir():
        parser.error(f"topology directory not found: {args.topology_dir}")
    return args


def main() -> None:
    warnings.filterwarnings("ignore", message="Optimal rotation is not uniquely.*")
    warnings.filterwarnings(
        "ignore",
        message=(
            "Full symbol not available, falling back to short Hermann Mauguin "
            "symbol.*"
        ),
    )
    args = parse_args()
    config = GenerationConfig(
        manifest=args.manifest,
        topology_dir=args.topology_dir,
        output=args.output,
        count=args.count,
        seed=args.seed,
        max_topology_slots=args.max_topology_slots,
        requested_space_groups=tuple(args.space_group),
    )
    generate(config)


if __name__ == "__main__":
    main()

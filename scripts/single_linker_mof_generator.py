#!/usr/bin/env python3
"""Topology-first generation of metal/organic-node MOFs without replacement."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from functools import lru_cache
import hashlib
from itertools import permutations, product
import json
import logging
import math
import os
import pickle
import random
import resource
import shutil
import statistics
import warnings
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
import time

import ase.neighborlist
from ase.io import read as read_ase
import numpy as np
from pymatgen.core import Molecule
from pymatgen.symmetry.analyzer import PointGroupAnalyzer
from pymatgen.symmetry.groups import SpaceGroup
from scipy.spatial.transform import Rotation
import spglib
from sympy.combinatorics import Permutation, PermutationGroup
from threadpoolctl import threadpool_limits

from mofbb.compose import Builder, BuildingBlock, Locator, Topology
from mofbb.compose.log import logger
from mofbb.compose.locator import (
    PortAssignmentError,
    find_best_orientation,
    find_best_permutation,
)
from mofbb.compose.pbc import PBC
from mofbb.compose.placement.symmetry import SymmetryPlacement


DEFAULT_GENERATION_WORKERS = min(16, os.cpu_count() or 1)

POINT_GROUP_TOLERANCE = 0.3
POINT_GROUP_EIGEN_TOLERANCE = 0.01
POINT_GROUP_MATRIX_TOLERANCE = 0.1
EDGE_POINT_GROUP = "D*h"
ALL_NODE_POINT_GROUP = "*"
GRAPH_SYMMETRY_BASIS = "element_and_bond_labeled_graph_automorphism"
GRAPH_SYMMETRY_FIELDS = (
    "symmetry_basis",
    "graph_port_automorphism_order",
    "graph_port_automorphism_generators",
    "graph_supports_port_point_group",
)
TOPOLOGY_CACHE_MTIME_TOLERANCE_NS = 5_000_000_000
RESUME_CHECKPOINT_VERSION = 3


@dataclass(frozen=True)
class GenerationConfig:
    manifest: Path
    topology_dir: Path
    output: Path
    count: int = 10
    seed: int = 42
    max_topology_slots: int = 30
    max_node_atoms: int = 120
    max_edge_atoms: int = 80
    edge_conformer_filter: str = "all"
    max_rmsd: float = 0.60
    minimum_atom_distance: float = 0.50
    accuracy: int = 4
    symmetry_tolerance: float = 0.1
    strict_symmetry: bool = False
    strict_prefilter: bool = True
    max_candidates_per_topology: int = 2_500
    direct_candidate_cap: int = 0
    candidate_scan_multiplier: int = 20
    attempt_batch_size: int = 250
    adaptive_min_attempts: int = 750
    adaptive_minimum_yield: float = 0.02
    adaptive_confidence_z: float = 1.96
    worker_memory_limit_gib: int = 8
    resume_revalidate: bool = False
    requested_space_groups: tuple[str, ...] = ()


@dataclass(frozen=True)
class BlockRecord:
    path: Path
    source_key: str
    structure_id: str
    kind: str
    contains_carbon: bool
    num_atoms: int
    coordination: int
    port_point_group: str
    molecular_point_group: str
    chirality: str
    linker_end_equivalence: str
    symmetry_basis: str | None
    graph_port_automorphism_order: int | None
    graph_supports_port_point_group: str | None
    sha256: str
    connection_anchor_symbols: tuple[str, ...]
    connection_anchor_lengths: tuple[float, ...]
    port_shape: tuple[float, ...]
    port_radius_span: float
    port_directions: tuple[tuple[float, float, float], ...]
    graph_port_automorphism_generators: (
        tuple[tuple[int, ...], ...] | None
    ) = None


@dataclass(frozen=True)
class NodeSite:
    node_type: int
    coordination: int
    point_group: str
    port_shape: tuple[float, ...]
    port_directions: tuple[tuple[float, float, float], ...]
    incident_edge_types: tuple[tuple[int, int], ...]
    stabilizer_order: int = 1
    stabilizer_port_permutations: tuple[tuple[int, ...], ...] = ()


@dataclass(frozen=True)
class EdgeSite:
    """One crystallographic edge orbit and its endpoint action."""

    edge_type: tuple[int, int]
    stabilizer_order: int
    stabilizer_port_permutations: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class TopologyRecord:
    name: str
    cgd_path: Path
    space_group: str
    num_nodes: int
    num_edges: int
    node_sites: tuple[NodeSite, ...]
    edge_types: tuple[tuple[int, int], ...]
    edge_sites: tuple[EdgeSite, ...] = ()

    @property
    def num_slots(self) -> int:
        return self.num_nodes + self.num_edges


@dataclass(frozen=True)
class CandidateAssignment:
    """One unique, pre-screened assignment for a single topology."""

    signature: str
    nodes: tuple[tuple[int, BlockRecord], ...]
    edges: tuple[tuple[tuple[int, int], BlockRecord], ...]
    score: tuple[float, float, float]


@dataclass(frozen=True)
class FittedBlock:
    """A block whose inexpensive port fit has already been evaluated."""

    block: BlockRecord
    rmsd: float


_CANDIDATE_WORKER_CONFIG: GenerationConfig | None = None
_CANDIDATE_WORKER_POOLS: (
    dict[tuple[str, int, str], list[BlockRecord]] | None
) = None
_GENERATION_WORKER_BUILDER: Builder | None = None


@lru_cache(maxsize=4_096)
def cached_building_block(path: str) -> BuildingBlock:
    """Load each XYZ template once per long-lived generation worker."""

    return BuildingBlock(Path(path))


@lru_cache(maxsize=64)
def cached_topology(path: str) -> Topology:
    """Load each topology once per long-lived generation worker."""

    return Topology(Path(path))


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


def configure_generation_worker(memory_limit_gib: int) -> None:
    """Bound one builder process so a pathological batch cannot exhaust RAM."""

    global _GENERATION_WORKER_BUILDER
    configure_quiet_builder_logging()
    _GENERATION_WORKER_BUILDER = Builder()
    limit = memory_limit_gib * 1024**3
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))


def configure_candidate_worker(
    config: GenerationConfig,
    pools: dict[tuple[str, int, str], list[BlockRecord]],
) -> None:
    """Initialize one parallel topology-candidate worker."""

    global _CANDIDATE_WORKER_CONFIG, _CANDIDATE_WORKER_POOLS
    _CANDIDATE_WORKER_CONFIG = config
    _CANDIDATE_WORKER_POOLS = pools
    configure_quiet_builder_logging()
    limit = config.worker_memory_limit_gib * 1024**3
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))


def enumerate_topology_candidates_job(
    topology: TopologyRecord,
) -> tuple[str, tuple[CandidateAssignment, ...], dict[str, object]]:
    """Enumerate one bounded topology queue in a worker process."""

    if _CANDIDATE_WORKER_CONFIG is None or _CANDIDATE_WORKER_POOLS is None:
        raise RuntimeError("candidate worker was not initialized")
    started = time.monotonic()
    node_stats: dict[str, int] = {}
    edge_stats: dict[str, int] = {}
    # Small 3x3 alignment problems become dramatically slower when every
    # process starts a large OpenBLAS pool. Match the already-thread-limited
    # build stage and let process-level parallelism provide concurrency.
    with threadpool_limits(limits=1, user_api="blas"):
        pools = _CANDIDATE_WORKER_POOLS
        if (
            _CANDIDATE_WORKER_CONFIG.strict_symmetry
            and _CANDIDATE_WORKER_CONFIG.strict_prefilter
        ):
            pools, node_stats = strict_node_compatible_pools(
                _CANDIDATE_WORKER_CONFIG,
                topology,
                pools,
            )
            pools, edge_stats = strict_edge_compatible_pools(topology, pools)
        candidates = enumerate_topology_candidates(
            _CANDIDATE_WORKER_CONFIG,
            topology,
            pools,
        )
        exact_symmetry_stats: dict[str, int | float] = {}
        if (
            _CANDIDATE_WORKER_CONFIG.strict_symmetry
            and _CANDIDATE_WORKER_CONFIG.strict_prefilter
        ):
            candidates, exact_symmetry_stats = prefilter_candidate_queue(
                _CANDIDATE_WORKER_CONFIG,
                topology,
                candidates,
            )
    return topology.name, candidates, {
        "elapsed_seconds": time.monotonic() - started,
        "nodes": node_stats,
        "edges": edge_stats,
        "exact_symmetry": exact_symmetry_stats,
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def block_kind_from_path(path: Path) -> str | None:
    if path.name.startswith("m_node"):
        return "metal_node"
    if path.name.startswith("o_node"):
        return "organic_node"
    if path.name.startswith("edge"):
        return "edge"
    return None


def edge_conformer_is_selected(
    row: dict[str, str],
    edge_conformer_filter: str,
) -> bool:
    """Return whether an edge row belongs to the requested conformer subset."""

    if edge_conformer_filter == "all":
        return True
    point_group = row.get("conformer_point_group") or row.get(
        "molecular_point_group", ""
    )
    return point_group not in {"", "C1", "unclassified"}


def resolve_manifest_block_paths(
    config: GenerationConfig,
    rows: list[dict[str, str]],
) -> tuple[dict[int, Path], int, dict[int, str]]:
    """Resolve renamed library blocks only when their recorded bytes match."""

    resolved: dict[int, Path] = {}
    unresolved: dict[tuple[str, str], list[int]] = defaultdict(list)
    unresolved_reasons: dict[int, str] = {}
    library_roots = {config.topology_dir.parent}

    for index, row in enumerate(rows):
        raw_path = row.get("output_path", "")
        expected_sha256 = row.get("xyz_sha256", "")
        kind = row.get("bb_kind", "")
        path = Path(raw_path)
        if raw_path:
            library_roots.add(path.parent.parent)
        if not expected_sha256:
            unresolved_reasons[index] = "malformed_manifest_row"
            continue
        try:
            if path.is_file() and file_sha256(path) == expected_sha256:
                resolved[index] = path
                continue
            reason = "xyz_hash_mismatch" if path.is_file() else "missing_xyz"
        except OSError:
            reason = "unreadable_xyz"
        unresolved[(kind, expected_sha256)].append(index)
        unresolved_reasons[index] = reason

    if not unresolved:
        return resolved, 0, {}

    search_directories: set[Path] = set()
    for root in library_roots:
        search_directories.update(
            {
                root / "node",
                root / "nodes",
                root / "edge",
                root / "bbs" / "node",
                root / "bbs" / "nodes",
                root / "bbs" / "edge",
            }
        )

    relocated = 0
    seen_paths: set[Path] = set()
    for directory in sorted(search_directories):
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.xyz")):
            if path in seen_paths:
                continue
            seen_paths.add(path)
            kind = block_kind_from_path(path)
            if kind is None:
                continue
            try:
                key = (kind, file_sha256(path))
            except OSError:
                continue
            indices = unresolved.pop(key, ())
            for index in indices:
                resolved[index] = path
                unresolved_reasons.pop(index, None)
                relocated += 1

    return resolved, relocated, unresolved_reasons


def port_shape_signature(
    directions: np.ndarray,
) -> tuple[float, ...]:
    """Describe connection directions independently of rotation and ordering."""

    directions = np.asarray(directions, dtype=float)
    pairwise = np.linalg.norm(
        directions[:, np.newaxis, :] - directions[np.newaxis, :, :],
        axis=2,
    )
    upper = pairwise[np.triu_indices(len(directions), k=1)]
    return tuple(float(value) for value in np.sort(upper))


def port_shape_error(
    left: tuple[float, ...],
    right: tuple[float, ...],
) -> float:
    """Return the RMS difference between two invariant port signatures."""

    if len(left) != len(right) or not left:
        return float("inf")
    return float(
        np.sqrt(
            np.mean(
                (np.asarray(left, dtype=float) - np.asarray(right, dtype=float))
                ** 2
            )
        )
    )


@lru_cache(maxsize=None)
def euler_rotation_matrices(num_slices: int) -> tuple[np.ndarray, ...]:
    """Return the same bounded rotation grid used by the MOFBB locator."""

    angles = np.asarray(
        list(
            product(
                np.linspace(0.0, 360.0, num_slices),
                np.linspace(0.0, 180.0, num_slices),
                np.linspace(0.0, 360.0, num_slices),
            )
        )
    )
    return tuple(Rotation.from_euler("zxz", -angles, degrees=True).as_matrix())


@lru_cache(maxsize=65536)
def port_fit_rmsd(
    target_directions: tuple[tuple[float, float, float], ...],
    block_directions: tuple[tuple[float, float, float], ...],
    accuracy: int,
    allowed: tuple[tuple[bool, ...], ...] = (),
) -> float:
    """Mirror the builder's port permutation search without assembling a MOF."""

    target = np.asarray(target_directions, dtype=float)
    block = np.asarray(block_directions, dtype=float)
    if target.shape != block.shape or target.ndim != 2 or target.shape[1] != 3:
        return float("inf")
    allowed_array = np.asarray(allowed, dtype=bool) if allowed else None
    if allowed_array is not None and allowed_array.shape != (
        len(target),
        len(block),
    ):
        return float("inf")

    num_ports = len(target)
    best_rmsd = float("inf")
    if num_ports <= 5:
        candidate_permutations = (
            np.asarray(values, dtype=int)
            for values in permutations(range(num_ports))
            if allowed_array is None
            or np.all(allowed_array[np.arange(num_ports), values])
        )
    else:
        canonical = block
        if num_ports <= 15:
            weights = np.arange(1, num_ports + 1, dtype=float)
            centered = block - np.average(block, axis=0, weights=weights)
            _, axes = np.linalg.eigh(
                (centered * weights[:, None]).T @ centered
            )
            pivots = np.argmax(np.abs(centered @ axes), axis=0)
            signs = np.sign((centered @ axes)[pivots, np.arange(3)])
            axes *= np.where(signs == 0, 1, signs)
            if np.linalg.det(axes) < 0:
                axes[:, 0] *= -1
            canonical = block @ axes

        def bounded_permutations():
            for rotation in euler_rotation_matrices(max(4, accuracy)):
                rotated = canonical @ rotation.T
                try:
                    yield find_best_permutation(
                        target,
                        rotated,
                        allowed=allowed_array,
                    )
                except PortAssignmentError:
                    continue

        candidate_permutations = bounded_permutations()

    for permutation in candidate_permutations:
        _, rmsd = find_best_orientation(target, block[permutation])
        best_rmsd = min(best_rmsd, float(rmsd))
        if best_rmsd < 1.0e-4:
            break
    return best_rmsd


def screened_port_fit_rmsd(
    site: NodeSite,
    block: BlockRecord,
    maximum_rmsd: float,
    accuracy: int,
) -> float:
    """Reject shape mismatches, then run the builder-equivalent port fit."""

    shape_error = port_shape_error(site.port_shape, block.port_shape)
    if shape_error > maximum_rmsd:
        return float("inf")
    return port_fit_rmsd(
        site.port_directions,
        block.port_directions,
        accuracy,
    )


def load_block_pools(
    config: GenerationConfig,
) -> tuple[dict[tuple[str, int, str], list[BlockRecord]], Counter[str]]:
    """Load symmetry-eligible blocks, bucketed by geometric port point group."""

    pools: dict[tuple[str, int, str], list[BlockRecord]] = defaultdict(list)
    skipped: Counter[str] = Counter()
    with config.manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    resolved_paths, relocated, unresolved_paths = resolve_manifest_block_paths(
        config,
        rows,
    )
    if relocated or unresolved_paths:
        print(
            f"manifest_paths_relocated_by_hash={relocated} "
            f"unresolved_manifest_blocks={len(unresolved_paths)}",
            flush=True,
        )

    for index, row in enumerate(rows):
        if index in unresolved_paths:
            skipped[unresolved_paths[index]] += 1
            continue
        try:
            kind = row["bb_kind"]
            coordination = int(row["coordination_number"])
            num_atoms = int(row["num_atoms"])
            path = resolved_paths[index]
            point_group = row["port_point_group"]
        except (KeyError, TypeError, ValueError):
            skipped["malformed_manifest_row"] += 1
            continue

        graph_fields_present = any(field in row for field in GRAPH_SYMMETRY_FIELDS)
        graph_metadata: (
            tuple[str, int, str, tuple[tuple[int, ...], ...] | None] | None
        ) = None
        if graph_fields_present:
            try:
                basis = row["symmetry_basis"].strip()
                graph_order = int(row["graph_port_automorphism_order"])
                raw_generators = row.get("graph_port_automorphism_generators")
                graph_generators = (
                    tuple(
                        tuple(int(value) for value in permutation)
                        for permutation in json.loads(raw_generators)
                    )
                    if raw_generators not in {None, ""}
                    else None
                )
                supports_value = (
                    row["graph_supports_port_point_group"].strip().lower()
                )
                if supports_value not in {"true", "false", "unclassified"}:
                    raise ValueError
                graph_metadata = (
                    basis,
                    graph_order,
                    supports_value,
                    graph_generators,
                )
            except (
                json.JSONDecodeError,
                KeyError,
                AttributeError,
                TypeError,
                ValueError,
            ):
                skipped["malformed_graph_symmetry"] += 1
                continue

        if row.get("classification_status") != "complete":
            skipped["incomplete_symmetry_classification"] += 1
            continue
        if kind not in {"metal_node", "organic_node", "edge"}:
            skipped["unsupported_kind"] += 1
            continue
        if (
            kind == "edge"
            and not edge_conformer_is_selected(row, config.edge_conformer_filter)
        ):
            skipped["edge_c1_conformer"] += 1
            continue
        atom_limit = (
            config.max_edge_atoms if kind == "edge" else config.max_node_atoms
        )
        if num_atoms > atom_limit:
            skipped["atom_limit"] += 1
            continue
        if graph_metadata is not None:
            (
                basis,
                graph_order,
                graph_supports_port_group,
                graph_generators,
            ) = graph_metadata
            if basis != GRAPH_SYMMETRY_BASIS:
                skipped["unsupported_symmetry_basis"] += 1
                continue
        else:
            # Legacy cache manifests contain only conformer-derived fields.
            # Their accepted rows were already symmetry-filtered by the cache
            # producer, so retain them without re-imposing molecular equality.
            basis = None
            graph_order = None
            graph_supports_port_group = None
            graph_generators = None
            if point_group in {"C1", "unclassified"}:
                skipped["legacy_trivial_or_unclassified_port_symmetry"] += 1
                continue
        if kind == "edge" and (
            coordination != 2
            or point_group != EDGE_POINT_GROUP
        ):
            skipped["edge_symmetry"] += 1
            continue
        if kind != "edge" and coordination < 3:
            skipped["node_coordination"] += 1
            continue

        try:
            block = BuildingBlock(path)
            contains_carbon = "C" in block.atoms.get_chemical_symbols()
            anchor_symbols = tuple(block.connection_point_anchor_symbols)
            anchor_lengths = tuple(
                float(value) for value in block.connection_point_anchor_lengths
            )
            port_directions = tuple(
                tuple(float(value) for value in direction)
                for direction in block.local_structure().positions
            )
            port_radii = np.linalg.norm(
                block.connection_points
                - np.mean(block.connection_points, axis=0),
                axis=1,
            )
        except Exception:
            skipped["invalid_block_metadata"] += 1
            continue
        if block.n_connection_points != coordination:
            skipped["connection_point_count_mismatch"] += 1
            continue
        if (
            len(anchor_symbols) != coordination
            or len(anchor_lengths) != coordination
            or not all(anchor_symbols)
        ):
            skipped["invalid_connection_anchor_symbols"] += 1
            continue

        record = BlockRecord(
            path=path,
            source_key=row["source_key"],
            structure_id=row["structure_id"],
            kind=kind,
            contains_carbon=contains_carbon,
            num_atoms=num_atoms,
            coordination=coordination,
            port_point_group=point_group,
            molecular_point_group=(
                row.get("conformer_point_group")
                or row.get("molecular_point_group", "unclassified")
            ),
            chirality=row.get("chirality", "unclassified"),
            linker_end_equivalence=row.get(
                "linker_end_equivalence", "unclassified"
            ),
            symmetry_basis=basis,
            graph_port_automorphism_order=graph_order,
            graph_supports_port_point_group=graph_supports_port_group,
            sha256=row["xyz_sha256"],
            connection_anchor_symbols=anchor_symbols,
            connection_anchor_lengths=anchor_lengths,
            port_shape=port_shape_signature(np.asarray(port_directions)),
            port_radius_span=float(np.max(port_radii) - np.min(port_radii)),
            port_directions=port_directions,
            graph_port_automorphism_generators=graph_generators,
        )
        pools[(kind, coordination, point_group)].append(record)
        if kind != "edge":
            pools[(kind, coordination, ALL_NODE_POINT_GROUP)].append(record)
    return dict(pools), skipped


def point_group_from_directions(directions: np.ndarray) -> str:
    analyzer = PointGroupAnalyzer(
        Molecule(["He"] * len(directions), directions),
        tolerance=POINT_GROUP_TOLERANCE,
        eigen_tolerance=POINT_GROUP_EIGEN_TOLERANCE,
        matrix_tolerance=POINT_GROUP_MATRIX_TOLERANCE,
    )
    return analyzer.sch_symbol


def topology_record_from_object(
    topology: object,
    path: Path,
) -> TopologyRecord:
    symmetry = SymmetryPlacement.cached(topology)
    node_orbit_by_type = {
        int(topology.node_types[orbit.representative]): orbit
        for orbit in symmetry.node_orbits
    }
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
        directions = np.asarray(
            topology.local_structure(representative).positions,
            dtype=float,
        )
        port_directions = tuple(
            tuple(float(value) for value in direction)
            for direction in directions
        )
        incident_edge_types = tuple(
            tuple(
                int(value)
                for value in topology.edge_types[int(neighbor.index)]
            )
            for neighbor in topology.neighbor_list[representative]
        )
        orbit = node_orbit_by_type[node_type]
        stabilizer_port_permutations = tuple(
            sorted(
                {
                    tuple(
                        int(value)
                        for value in symmetry.offset_map(
                            representative,
                            representative,
                            int(operation),
                        )
                    )
                    for operation in orbit.stabilizers
                }
            )
        )
        sites.append(
            NodeSite(
                node_type=node_type,
                coordination=int(coordination),
                point_group=point_group_from_directions(directions),
                port_shape=port_shape_signature(directions),
                port_directions=port_directions,
                incident_edge_types=incident_edge_types,
                stabilizer_order=len(orbit.stabilizers),
                stabilizer_port_permutations=stabilizer_port_permutations,
            )
        )

    edge_sites = []
    for orbit in symmetry.edge_orbits:
        representative = orbit.representative
        edge_type = tuple(
            int(value) for value in topology.edge_types[representative]
        )
        port_permutations = tuple(
            sorted(
                {
                    tuple(
                        int(value)
                        for value in symmetry.offset_map(
                            representative,
                            representative,
                            int(operation),
                        )
                    )
                    for operation in orbit.stabilizers
                }
            )
        )
        edge_sites.append(
            EdgeSite(
                edge_type=edge_type,
                stabilizer_order=len(orbit.stabilizers),
                stabilizer_port_permutations=port_permutations,
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
        edge_sites=tuple(edge_sites),
    )


def compatible_topology_cache(topology: object) -> bool:
    """Accept current topologies and the legacy PORMAKE database cache."""

    topology_type = type(topology)
    return isinstance(topology, Topology) or (
        topology_type.__module__ == "pormake.topology"
        and topology_type.__name__ == "Topology"
    )


def topology_cache_is_stale(cgd_path: Path, pickle_path: Path) -> bool:
    """Allow small timestamp skew introduced while copying cache files."""

    return (
        cgd_path.stat().st_mtime_ns
        > pickle_path.stat().st_mtime_ns + TOPOLOGY_CACHE_MTIME_TOLERANCE_NS
    )


def node_pool_key(
    kind: str,
    site: NodeSite,
) -> tuple[str, int, str]:
    """Route by port geometry; molecular symmetry is checked separately."""

    point_group = (
        ALL_NODE_POINT_GROUP if site.stabilizer_order == 1 else site.point_group
    )
    return kind, site.coordination, point_group


def node_records_for_site(
    pools: dict[tuple, list],
    kind: str,
    site: NodeSite,
) -> list:
    """Return the general- or special-position node pool for one site."""

    return pools.get(node_pool_key(kind, site), [])


def edge_sites_for_type(
    topology: TopologyRecord,
    edge_type: tuple[int, int],
) -> tuple[EdgeSite, ...]:
    """Return crystallographic requirements for one decorated edge type."""

    sites = tuple(site for site in topology.edge_sites if site.edge_type == edge_type)
    if sites:
        return sites
    # Synthetic and legacy TopologyRecord instances predate explicit edge-site
    # metadata. Treat them as general positions and let the Builder remain the
    # authoritative validator.
    return (EdgeSite(edge_type, 1, ((0, 1),)),)


def unique_library_blocks(pools: dict[tuple, list]) -> tuple[BlockRecord, ...]:
    """Return each base library record once despite broad node-pool aliases."""

    unique: dict[tuple[str, str], BlockRecord] = {}
    for values in pools.values():
        for value in values:
            if isinstance(value, BlockRecord):
                unique[(value.kind, value.sha256)] = value
    return tuple(unique.values())


def load_eligible_topologies(
    config: GenerationConfig,
    pools: dict[tuple[str, int, str], list[BlockRecord]],
) -> tuple[list[TopologyRecord], Counter[str]]:
    records: list[TopologyRecord] = []
    skipped: Counter[str] = Counter()
    requested_groups = set(config.requested_space_groups)

    cgd_paths = sorted(config.topology_dir.glob("*.cgd"))
    pickle_paths = sorted(config.topology_dir.glob("*.pickle"))
    if pickle_paths:
        cached_names = {path.stem for path in pickle_paths}
        skipped["uncached_topology"] += sum(
            cgd_path.stem not in cached_names for cgd_path in cgd_paths
        )
        topology_sources: list[tuple[Path, Path | None]] = [
            (pickle_path.with_suffix(".cgd"), pickle_path)
            for pickle_path in pickle_paths
        ]
    else:
        topology_sources = [(cgd_path, None) for cgd_path in cgd_paths]

    print(
        f"screening_topologies={len(topology_sources)} "
        f"cached={len(pickle_paths)}",
        flush=True,
    )
    for cgd_path, pickle_path in topology_sources:
        if not cgd_path.is_file():
            skipped["missing_cgd"] += 1
            continue
        try:
            if pickle_path is not None:
                try:
                    # These are trusted database pickles, not user-supplied files.
                    with pickle_path.open("rb") as handle:
                        topology = pickle.load(handle)
                    if not compatible_topology_cache(topology):
                        raise TypeError("topology pickle uses a different class")
                    if topology_cache_is_stale(cgd_path, pickle_path):
                        raise ValueError("topology pickle is older than its CGD")
                except Exception:
                    topology = Topology(cgd_path)
            else:
                topology = Topology(cgd_path)
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
        if len(topology.unique_node_types) != 2:
            skipped["type_limit"] += 1
            continue
        try:
            record = topology_record_from_object(topology, cgd_path)
        except Exception:
            skipped["point_group_classification"] += 1
            continue
        if not all(
            node_records_for_site(pools, "metal_node", site)
            or node_records_for_site(pools, "organic_node", site)
            for site in record.node_sites
        ):
            skipped["missing_site_compatible_node_pool"] += 1
            continue
        if not compatible_metal_organic_pairs(record, pools):
            skipped["no_metal_organic_assignment"] += 1
            continue
        records.append(record)
    return records, skipped


def compatible_metal_organic_pairs(
    topology: TopologyRecord,
    pools: dict[tuple[str, int, str], list[BlockRecord]],
) -> list[tuple[int, int]]:
    """Return distinct node-type pairs assignable as metal and organic."""

    return [
        (metal_site.node_type, organic_site.node_type)
        for metal_site in topology.node_sites
        for organic_site in topology.node_sites
        if metal_site.node_type != organic_site.node_type
        and node_records_for_site(pools, "metal_node", metal_site)
        and node_records_for_site(pools, "organic_node", organic_site)
    ]


def strict_node_compatible_pools(
    config: GenerationConfig,
    topology_record: TopologyRecord,
    pools: dict[tuple, list],
) -> tuple[
    dict[tuple, list],
    dict[str, int],
]:
    """Build cheap, site-aware node pools without loading atomic conformers.

    General positions receive every coordination-compatible block. Special
    positions retain the geometric point-group bucket, while obvious C1 and
    trivial-port-action mismatches are discarded using manifest metadata.
    Full atomic stabilizer checks are deferred until a candidate is consumed.
    """

    filtered = dict(pools)
    checked = accepted = 0
    general_position_candidates = 0
    c1_special_rejections = 0
    port_action_rejections = 0

    for site in topology_record.node_sites:
        for kind in ("metal_node", "organic_node"):
            compatible: list[FittedBlock] = []
            for record in node_records_for_site(pools, kind, site):
                checked += 1
                if site.stabilizer_order > 1 and record.molecular_point_group == "C1":
                    c1_special_rejections += 1
                    continue
                nonidentity_action = any(
                    permutation != tuple(range(site.coordination))
                    for permutation in site.stabilizer_port_permutations
                )
                if (
                    nonidentity_action
                    and record.graph_port_automorphism_order is not None
                    and record.graph_port_automorphism_order <= 1
                ):
                    port_action_rejections += 1
                    continue
                rmsd = screened_port_fit_rmsd(
                    site,
                    record,
                    config.max_rmsd,
                    config.accuracy,
                )
                if rmsd > config.max_rmsd:
                    continue
                compatible.append(FittedBlock(record, float(rmsd)))
                accepted += 1
                if site.stabilizer_order == 1:
                    general_position_candidates += 1
            filtered[
                (kind, site.node_type, site.coordination, site.point_group)
            ] = compatible

    return filtered, {
        "checked": checked,
        "accepted": accepted,
        "rejected": checked - accepted,
        "general_position_candidates": general_position_candidates,
        "c1_special_rejections": c1_special_rejections,
        "port_action_rejections": port_action_rejections,
    }


def strict_edge_conformer_pool(
    pools: dict[tuple[str, int, str], list[BlockRecord]],
) -> list[BlockRecord]:
    """Return all connected two-port conformers; the site decides reversibility."""

    return list(pools.get(("edge", 2, EDGE_POINT_GROUP), ()))


def strict_edge_compatible_pools(
    topology_record: TopologyRecord,
    pools: dict[tuple, list],
) -> tuple[dict[tuple, list], dict[str, int]]:
    """Build inexpensive edge pools independently for each decorated type."""

    filtered = dict(pools)
    candidates = strict_edge_conformer_pool(pools)
    checked = accepted = 0
    general_position_candidates = 0
    endpoint_exchange_rejections = 0
    c1_special_rejections = 0

    for edge_type in topology_record.edge_types:
        sites = edge_sites_for_type(topology_record, edge_type)
        special = any(site.stabilizer_order > 1 for site in sites)
        endpoint_exchange = any(
            (1, 0) in site.stabilizer_port_permutations for site in sites
        )
        compatible = []
        for record in candidates:
            checked += 1
            if endpoint_exchange and record.linker_end_equivalence != "equivalent":
                endpoint_exchange_rejections += 1
                continue
            if special and record.molecular_point_group == "C1":
                c1_special_rejections += 1
                continue
            compatible.append(record)
            accepted += 1
            if not special:
                general_position_candidates += 1
        filtered[("edge", edge_type, 2, EDGE_POINT_GROUP)] = compatible

    return filtered, {
        "checked": checked,
        "accepted": accepted,
        "rejected": checked - accepted,
        "general_position_candidates": general_position_candidates,
        "endpoint_exchange_rejections": endpoint_exchange_rejections,
        "c1_special_rejections": c1_special_rejections,
    }


@lru_cache(maxsize=65_536)
def graph_permutation_supported(
    degree: int,
    generators: tuple[tuple[int, ...], ...] | None,
    candidate: tuple[int, ...],
) -> bool:
    """Test membership in a cached induced port-permutation group."""

    identity = tuple(range(degree))
    if candidate == identity:
        return True
    # A legacy manifest without generators was admitted under the older,
    # stronger full-point-group gate, so retaining it is backward compatible.
    if generators is None:
        return True
    if not generators:
        return False
    group = PermutationGroup(
        *[
            Permutation(list(generator), size=degree)
            for generator in generators
        ]
    )
    permutation = Permutation(list(candidate), size=degree)
    return bool(group.contains(permutation, strict=True))


def block_supports_stabilizer_action(
    record: BlockRecord,
    site_permutations: tuple[tuple[int, ...], ...],
    placement: np.ndarray,
) -> bool:
    """Check the site action after conjugating it into BB port order."""

    placement = np.asarray(placement, dtype=int)
    degree = len(placement)
    for site_permutation in site_permutations:
        site_permutation = np.asarray(site_permutation, dtype=int)
        induced = np.empty(degree, dtype=int)
        induced[placement] = placement[site_permutation]
        if not graph_permutation_supported(
            degree,
            record.graph_port_automorphism_generators,
            tuple(int(value) for value in induced),
        ):
            return False
    return True


def strict_node_record_fit(
    config: GenerationConfig,
    topology_record: TopologyRecord,
    node_type: int,
    record: BlockRecord,
) -> float | None:
    """Perform one exact node/site compatibility calculation."""

    site = next(
        site for site in topology_record.node_sites if site.node_type == node_type
    )
    rmsd = screened_port_fit_rmsd(
        site,
        record,
        config.max_rmsd,
        config.accuracy,
    )
    if rmsd > config.max_rmsd:
        return None
    if site.stabilizer_order == 1:
        return float(rmsd)
    if record.molecular_point_group == "C1":
        return None

    topology = cached_topology(str(topology_record.cgd_path))
    symmetry = SymmetryPlacement.cached(topology)
    orbit = next(
        orbit
        for orbit in symmetry.node_orbits
        if int(topology.node_types[orbit.representative]) == node_type
    )
    representative = orbit.representative
    target = topology.local_structure(representative)
    target_positions = np.asarray(target.atoms.positions, dtype=float)
    block = cached_building_block(str(record.path))
    source_positions = np.asarray(block.local_structure().positions, dtype=float)
    locator = Locator()
    best = float("inf")
    for permutation in locator.candidate_permutations(
        target,
        block,
        max_n_slices=max(4, config.accuracy),
    ):
        rotation, candidate_rmsd = find_best_orientation(
            target_positions,
            source_positions[permutation],
        )
        candidate_rmsd = float(candidate_rmsd)
        if candidate_rmsd > config.max_rmsd or candidate_rmsd >= best:
            continue
        if not block_supports_stabilizer_action(
            record,
            site.stabilizer_port_permutations,
            permutation,
        ):
            continue
        located = block.copy()
        located.atoms.set_positions(
            (block.atoms.positions - block.centroid) @ rotation + block.centroid
        )
        located.set_centroid(topology.atoms.positions[representative])
        diagnostic = symmetry.diagnose(
            orbit,
            located,
            permutation,
            np.asarray(topology.atoms.cell, dtype=float),
            config.symmetry_tolerance,
        )
        if diagnostic["compatible"]:
            best = candidate_rmsd
            if best < 1.0e-4:
                break
    return None if not math.isfinite(best) else best


def strict_edge_record_compatible(
    config: GenerationConfig,
    topology_record: TopologyRecord,
    edge_type: tuple[int, int],
    record: BlockRecord,
) -> bool:
    """Test one linker only on the edge type it will occupy."""

    sites = edge_sites_for_type(topology_record, edge_type)
    if all(site.stabilizer_order == 1 for site in sites):
        return True
    if record.molecular_point_group == "C1":
        return False
    if any(
        (1, 0) in site.stabilizer_port_permutations for site in sites
    ) and record.linker_end_equivalence != "equivalent":
        return False

    topology = cached_topology(str(topology_record.cgd_path))
    symmetry = SymmetryPlacement.cached(topology)
    locator = Locator()
    cell = np.asarray(topology.atoms.cell, dtype=float)
    relevant_orbits = tuple(
        orbit
        for orbit in symmetry.edge_orbits
        if tuple(int(value) for value in topology.edge_types[orbit.representative])
        == edge_type
    )
    if not relevant_orbits:
        return False
    block = cached_building_block(str(record.path))
    for orbit in relevant_orbits:
        if len(orbit.stabilizers) == 1:
            continue
        representative = orbit.representative
        target_points = topology.atoms.positions[representative] + np.asarray(
            [
                neighbor.distance_vector
                for neighbor in topology.neighbor_list[representative]
            ]
        )
        axis = target_points[1] - target_points[0]
        axis /= np.linalg.norm(axis)
        axes = np.eye(3)
        transverse = axes[np.argmin(np.abs(axes @ axis))]
        transverse -= np.dot(transverse, axis) * axis
        transverse /= np.linalg.norm(transverse)
        compatible = False
        for permutation in (np.asarray([0, 1]), np.asarray([1, 0])):
            try:
                located, _, _ = locator.locate_edge_with_frame(
                    target_points,
                    block,
                    permutation,
                    transverse,
                )
                located.set_centroid(np.mean(target_points, axis=0))
                _, diagnostic = symmetry.fit_torsion(
                    orbit,
                    located,
                    permutation,
                    cell,
                    config.symmetry_tolerance,
                )
            except Exception:
                continue
            if diagnostic["compatible"]:
                compatible = True
                break
        if not compatible:
            return False
    return True


def prefilter_candidate_queue(
    config: GenerationConfig,
    topology_record: TopologyRecord,
    candidates: tuple[CandidateAssignment, ...],
) -> tuple[tuple[CandidateAssignment, ...], dict[str, int | float]]:
    """Exact-screen only unique block/site pairs present in a bounded queue.

    Node pairs are checked first so candidates with an invalid node cannot
    trigger unnecessary linker torsion checks.  Each unique pair is evaluated
    once in this topology worker, then every candidate containing a rejected
    pair is removed before quota scheduling or build-attempt accounting.
    """

    started = time.monotonic()
    node_pairs = {
        (node_type, record)
        for candidate in candidates
        for node_type, record in candidate.nodes
    }
    node_compatible = {
        pair: strict_node_record_fit(
            config,
            topology_record,
            pair[0],
            pair[1],
        )
        is not None
        for pair in node_pairs
    }
    node_filtered = tuple(
        candidate
        for candidate in candidates
        if all(node_compatible[pair] for pair in candidate.nodes)
    )

    edge_pairs = {
        (edge_type, record)
        for candidate in node_filtered
        for edge_type, record in candidate.edges
    }
    edge_compatible = {
        pair: strict_edge_record_compatible(
            config,
            topology_record,
            pair[0],
            pair[1],
        )
        for pair in edge_pairs
    }
    accepted = tuple(
        candidate
        for candidate in node_filtered
        if all(edge_compatible[pair] for pair in candidate.edges)
    )

    general_node_types = {
        site.node_type
        for site in topology_record.node_sites
        if site.stabilizer_order == 1
    }
    general_edge_types = {
        edge_type
        for edge_type in topology_record.edge_types
        if all(
            site.stabilizer_order == 1
            for site in edge_sites_for_type(topology_record, edge_type)
        )
    }
    return accepted, {
        "input_candidates": len(candidates),
        "accepted_candidates": len(accepted),
        "rejected_candidates": len(candidates) - len(accepted),
        "node_rejected_candidates": len(candidates) - len(node_filtered),
        "edge_rejected_candidates": len(node_filtered) - len(accepted),
        "unique_node_site_block_pairs": len(node_pairs),
        "accepted_node_site_block_pairs": sum(node_compatible.values()),
        "rejected_node_site_block_pairs": len(node_pairs)
        - sum(node_compatible.values()),
        "general_node_site_block_pairs": sum(
            node_type in general_node_types for node_type, _ in node_pairs
        ),
        "unique_edge_site_block_pairs": len(edge_pairs),
        "accepted_edge_site_block_pairs": sum(edge_compatible.values()),
        "rejected_edge_site_block_pairs": len(edge_pairs)
        - sum(edge_compatible.values()),
        "general_edge_site_block_pairs": sum(
            edge_type in general_edge_types for edge_type, _ in edge_pairs
        ),
        "elapsed_seconds": time.monotonic() - started,
    }


@lru_cache(maxsize=65536)
def connection_port_compatibility(
    left: BlockRecord,
    right: BlockRecord,
) -> tuple[tuple[bool, ...], ...]:
    """Return the full left-port/right-port fusion feasibility matrix."""

    rows = []
    for symbol_i, length_i in zip(
        left.connection_anchor_symbols,
        left.connection_anchor_lengths,
        strict=True,
    ):
        row = []
        for symbol_j, length_j in zip(
            right.connection_anchor_symbols,
            right.connection_anchor_lengths,
            strict=True,
        ):
            row.append(
                not PBC.is_carbon_metal_pair(symbol_i, symbol_j)
                and length_i + length_j
                <= PBC.max_bond_len(symbol_i, symbol_j)
            )
        rows.append(tuple(row))
    return tuple(rows)


def other_node_type(edge_type: tuple[int, int], node_type: int) -> int:
    """Return the type across an edge from ``node_type``."""

    left, right = edge_type
    if node_type == left:
        return right
    if node_type == right:
        return left
    raise ValueError(f"node type {node_type} is not incident to {edge_type}")


def node_port_feasibility(
    site: NodeSite,
    nodes: dict[int, BlockRecord],
    edges: dict[tuple[int, int], BlockRecord],
    flipped_edge_types: frozenset[tuple[int, int]],
) -> tuple[tuple[bool, ...], ...]:
    """Build the target-edge/source-node port mask for one topology site."""

    node = nodes[site.node_type]
    rows = []
    for edge_type in site.incident_edge_types:
        edge = edges.get(edge_type)
        if edge is None:
            partner = nodes[other_node_type(edge_type, site.node_type)]
            compatibility = connection_port_compatibility(node, partner)
            rows.append(tuple(any(row) for row in compatibility))
            continue

        compatibility = connection_port_compatibility(node, edge)
        if edge_type[0] == edge_type[1]:
            rows.append(tuple(any(row) for row in compatibility))
            continue

        endpoint = int(site.node_type == edge_type[1])
        if edge_type in flipped_edge_types:
            endpoint = 1 - endpoint
        rows.append(tuple(row[endpoint] for row in compatibility))
    return tuple(rows)


def assignment_prebuild_risk(
    topology: TopologyRecord,
    nodes: dict[int, BlockRecord],
    edges: dict[tuple[int, int], BlockRecord],
    *,
    maximum_rmsd: float = 0.60,
    accuracy: int = 4,
) -> float | None:
    """Require a feasible, geometry-aware bijection at every node type.

    The previous gate retained an assignment whenever *one* port pair was
    compatible.  That allowed the same rare compatible source port to be
    implicitly reused for every incident edge.  This gate mirrors the
    builder's one-port-per-neighbor assignment and jointly tests the two
    orientations of each heterotypic inserted edge.
    """

    oriented_edges = tuple(
        edge_type
        for edge_type in edges
        if edge_type[0] != edge_type[1]
    )
    best_risk = float("inf")
    for flip_values in product((False, True), repeat=len(oriented_edges)):
        flipped = frozenset(
            edge_type
            for edge_type, flip in zip(
                oriented_edges,
                flip_values,
                strict=True,
            )
            if flip
        )
        risk = 0.0
        feasible = True
        for site in topology.node_sites:
            allowed = node_port_feasibility(site, nodes, edges, flipped)
            if (
                len(allowed) != site.coordination
                or any(len(row) != site.coordination for row in allowed)
                or any(not any(row) for row in allowed)
            ):
                feasible = False
                break
            fit = port_fit_rmsd(
                site.port_directions,
                nodes[site.node_type].port_directions,
                accuracy,
                allowed,
            )
            if fit > maximum_rmsd:
                feasible = False
                break
            risk += sum(
                1.0 - sum(row) / len(row)
                for row in allowed
            )
        if feasible:
            best_risk = min(best_risk, risk)
    return None if not math.isfinite(best_risk) else best_risk


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


@lru_cache(maxsize=256)
def canonical_space_group_number(symbol: str) -> int:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return int(SpaceGroup(symbol).int_number)


def detect_crystal_symmetry(
    atoms: object,
    symprec: float,
    angle_tolerance: float = -1.0,
) -> dict[str, object]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dataset = spglib.get_symmetry_dataset(
            (
                np.asarray(atoms.cell.array, dtype=float),
                np.asarray(atoms.get_scaled_positions(wrap=False), dtype=float),
                np.asarray(atoms.numbers, dtype=np.intc),
            ),
            symprec=float(symprec),
            angle_tolerance=float(angle_tolerance),
        )
    if dataset is None:
        raise ValueError("spglib returned no post-build symmetry dataset")
    return {
        "detected_space_group_number": int(dataset.number),
        "detected_space_group_symbol": str(dataset.international),
        "detected_point_group": str(dataset.pointgroup),
        "detected_symmetry_operations": int(len(dataset.rotations)),
        "detected_symprec_angstrom": float(symprec),
        "detected_angle_tolerance_degree": float(angle_tolerance),
    }


def build_and_validate(
    config: GenerationConfig,
    topology_record: TopologyRecord,
    node_paths: dict[int, Path],
    edge_paths: dict[tuple[int, int], Path],
    cif_path: Path,
    topology: Topology | None = None,
    crystal_symmetry_symprec: float | None = None,
    crystal_symmetry_angle_tolerance: float = -1.0,
) -> dict[str, object]:
    if topology is None:
        topology = cached_topology(str(topology_record.cgd_path))
    builder = _GENERATION_WORKER_BUILDER or Builder()
    framework = builder.build_by_type(
        topology=topology,
        node_bbs={
            key: cached_building_block(str(path))
            for key, path in node_paths.items()
        },
        edge_bbs={
            key: cached_building_block(str(path))
            for key, path in edge_paths.items()
        },
        accuracy=config.accuracy,
        symmetry_tolerance=config.symmetry_tolerance,
        strict_symmetry=config.strict_symmetry,
    )
    max_rmsd = float(framework.info["max_rmsd"])
    if max_rmsd > config.max_rmsd:
        raise ValueError(
            f"node-placement RMSD {max_rmsd:.3f} exceeds {config.max_rmsd:.3f}"
        )
    if "X" in framework.atoms.get_chemical_symbols():
        raise ValueError("assembled framework still contains X connection sites")
    if "C" not in framework.atoms.get_chemical_symbols():
        raise ValueError("assembled framework contains no carbon")
    if not np.isfinite(framework.atoms.positions).all() or framework.atoms.cell.volume <= 0:
        raise ValueError("assembled framework has invalid coordinates or cell")
    closest = minimum_distance(framework.atoms)
    if closest is not None and closest < config.minimum_atom_distance:
        raise ValueError(
            f"minimum atom distance {closest:.3f} A is below "
            f"{config.minimum_atom_distance:.3f} A"
        )

    framework.write_cif(cif_path)
    if not cif_path.is_file() or cif_path.stat().st_size == 0:
        raise ValueError("MOFBB did not write a nonempty CIF")

    diagnostics = framework.info.get("symmetry_placement_diagnostics", {})
    fallback_orbits = sum(
        not item.get(
            "symmetry_propagated",
            item.get("covariant", True),
        )
        for kind in ("nodes", "edges")
        for item in diagnostics.get(kind, ())
    )

    quality = {
        "num_atoms": len(framework.atoms),
        "formula": framework.atoms.get_chemical_formula(mode="metal"),
        "cell_parameters": [
            float(value) for value in framework.atoms.cell.cellpar()
        ],
        "max_rmsd": max_rmsd,
        "mean_rmsd": float(framework.info["mean_rmsd"]),
        "geometry_optimization_objective": float(
            framework.info["geometry_optimization_objective"]
        ),
        "relax_obj": float(framework.info["relax_obj"]),
        "minimum_atom_distance_angstrom": closest,
        "cif_space_group_symbol": "P1",
        "cif_space_group_number": 1,
        "placement_mode": (
            "strict_symmetry" if config.strict_symmetry else "allow_p1_fallback"
        ),
        "symmetry_fallback_orbit_count": int(fallback_orbits),
    }
    if crystal_symmetry_symprec is not None:
        quality.update(
            detect_crystal_symmetry(
                framework.atoms,
                crystal_symmetry_symprec,
                crystal_symmetry_angle_tolerance,
            )
        )
        quality["detected_matches_source_space_group"] = bool(
            quality["detected_space_group_number"]
            == canonical_space_group_number(topology_record.space_group)
        )
    return quality


def validate_staged_cif(
    config: GenerationConfig,
    cif_path: Path,
) -> dict[str, object]:
    """Revalidate and recover quality metadata for an interrupted run."""

    lines = cif_path.read_text(encoding="utf-8").splitlines()

    def comment_value(prefix: str) -> float | None:
        for line in lines:
            if line.startswith(prefix):
                return float(line.split(":", 1)[1].strip())
        return None

    atoms = read_ase(cif_path)
    symbols = atoms.get_chemical_symbols()
    if "X" in symbols or "C" not in symbols:
        raise ValueError("staged CIF fails carbon/connection-site validation")
    if (
        not np.isfinite(atoms.positions).all()
        or not np.isfinite(atoms.get_scaled_positions(wrap=False)).all()
        or atoms.cell.volume <= 0
    ):
        raise ValueError("staged CIF has invalid coordinates or cell")
    closest = minimum_distance(atoms)
    if closest is not None and closest < config.minimum_atom_distance:
        raise ValueError("staged CIF fails minimum-distance validation")
    max_rmsd = comment_value("# Max RMSD:")
    if max_rmsd is None or max_rmsd > config.max_rmsd:
        raise ValueError("staged CIF fails RMSD validation")
    geometry_objective = comment_value("# Geometry optimization objective:")
    if geometry_objective is None:
        geometry_objective = comment_value("# Relax obj value:")
    return {
        "num_atoms": len(atoms),
        "formula": atoms.get_chemical_formula(mode="metal"),
        "cell_parameters": [float(value) for value in atoms.cell.cellpar()],
        "max_rmsd": max_rmsd,
        "mean_rmsd": comment_value("# Mean RMSD:"),
        "geometry_optimization_objective": geometry_objective,
        "relax_obj": geometry_objective,
        "minimum_atom_distance_angstrom": closest,
        "cif_space_group_symbol": "P1",
        "cif_space_group_number": 1,
        "placement_mode": (
            "strict_symmetry" if config.strict_symmetry else "allow_p1_fallback"
        ),
        "symmetry_fallback_orbit_count": None,
        "cif_sha256": file_sha256(cif_path),
        "resumed_from_staging": True,
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


def validation_fingerprint(config: GenerationConfig) -> str:
    """Identify the validation rules under which a staged CIF was accepted."""

    rules = {
        "version": RESUME_CHECKPOINT_VERSION,
        "max_rmsd": config.max_rmsd,
        "minimum_atom_distance": config.minimum_atom_distance,
        "accuracy": config.accuracy,
        "symmetry_tolerance": config.symmetry_tolerance,
        "strict_symmetry": config.strict_symmetry,
    }
    return hashlib.sha256(
        json.dumps(rules, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def append_resume_checkpoint(
    path: Path,
    config: GenerationConfig,
    topology: TopologyRecord,
    queue_fingerprint: str,
    result: dict[str, object],
    accepted: list[dict[str, object]],
) -> None:
    """Persist one completed quota job before its progress line is printed."""

    topology_stat = topology.cgd_path.stat()
    record = {
        "version": RESUME_CHECKPOINT_VERSION,
        "validation_fingerprint": validation_fingerprint(config),
        "topology": topology.name,
        "topology_size": topology_stat.st_size,
        "topology_mtime_ns": topology_stat.st_mtime_ns,
        "queue_fingerprint": queue_fingerprint,
        "pass_index": int(result["pass_index"]),
        "next_cursor": int(result["next_cursor"]),
        "attempted": int(result["attempted"]),
        "successes": [],
    }
    for success in accepted:
        cif_path = Path(success["cif_path"])
        cif_stat = cif_path.stat()
        record["successes"].append(
            {
                "signature": success["candidate"].signature,
                "candidate_index": int(success["candidate_index"]),
                "cif_name": cif_path.name,
                "cif_size": cif_stat.st_size,
                "cif_mtime_ns": cif_stat.st_mtime_ns,
                "quality": success["quality"],
            }
        )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_resume_checkpoint(path: Path) -> list[dict[str, object]]:
    """Read complete JSONL records, tolerating only a truncated final write."""

    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    records = []
    for index, line in enumerate(lines):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
    return records


def candidate_queue_fingerprint(
    candidates: tuple[CandidateAssignment, ...],
) -> str:
    """Hash candidate order so a checkpoint cursor cannot skip a new queue."""

    digest = hashlib.sha256()
    for candidate in candidates:
        digest.update(candidate.signature.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def write_results(output: Path, rows: list[dict], summary: dict) -> None:
    atomic_json(output / "manifest.json", rows)
    atomic_json(output / "generation_summary.json", summary)
    temporary = output / "manifest.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "sample_id",
            "topology",
            "topology_space_group",
            "cif_space_group",
            "cif_space_group_number",
            "placement_mode",
            "symmetry_fallback_orbit_count",
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
                    "cif_space_group": row["quality"][
                        "cif_space_group_symbol"
                    ],
                    "cif_space_group_number": row["quality"][
                        "cif_space_group_number"
                    ],
                    "placement_mode": row["quality"]["placement_mode"],
                    "symmetry_fallback_orbit_count": row["quality"][
                        "symmetry_fallback_orbit_count"
                    ],
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


def candidate_assignment_signature(
    topology_name: str,
    nodes: dict[int, BlockRecord],
    edges: dict[tuple[int, int], BlockRecord],
) -> str:
    """Hash one topology/type assignment independently of output naming."""

    value = {
        "topology": topology_name,
        "nodes": sorted(
            (int(node_type), block.sha256)
            for node_type, block in nodes.items()
        ),
        "edges": sorted(
            (tuple(int(value) for value in edge_type), block.sha256)
            for edge_type, block in edges.items()
        ),
    }
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def enumerate_topology_candidates(
    config: GenerationConfig,
    topology: TopologyRecord,
    pools: dict[tuple[str, int, str], list[BlockRecord]],
) -> tuple[CandidateAssignment, ...]:
    """Build a bounded deterministic queue without materializing its product."""

    site_candidates: dict[
        tuple[int, str],
        list[tuple[BlockRecord, float]],
    ] = {}
    for site in topology.node_sites:
        for kind in ("metal_node", "organic_node"):
            if config.strict_symmetry and config.strict_prefilter:
                fitted = pools.get(
                    (kind, site.node_type, site.coordination, site.point_group),
                    [],
                )
                site_candidates[(site.node_type, kind)] = [
                    (item.block, item.rmsd) for item in fitted
                ]
                continue
            compatible = []
            for block in node_records_for_site(pools, kind, site):
                fit = screened_port_fit_rmsd(
                    site,
                    block,
                    config.max_rmsd,
                    config.accuracy,
                )
                if fit <= config.max_rmsd:
                    compatible.append((block, fit))
            site_candidates[(site.node_type, kind)] = compatible

    orientations = []
    for metal_site in topology.node_sites:
        for organic_site in topology.node_sites:
            if metal_site.node_type == organic_site.node_type:
                continue
            metal_blocks = tuple(
                site_candidates[(metal_site.node_type, "metal_node")]
            )
            organic_blocks = tuple(
                site_candidates[(organic_site.node_type, "organic_node")]
            )
            if metal_blocks and organic_blocks:
                orientations.append(
                    (
                        metal_site,
                        organic_site,
                        metal_blocks,
                        organic_blocks,
                    )
                )

    def permuted_indices(size: int, label: str):
        """Traverse a Cartesian index space bijectively in seeded order."""

        if size <= 0:
            return
        digest = hashlib.sha256(
            f"{config.seed}:{topology.name}:{label}".encode()
        ).digest()
        offset = int.from_bytes(digest[:16], "big") % size
        if size == 1:
            yield 0
            return
        step = int.from_bytes(digest[16:], "big") % size or 1
        while math.gcd(step, size) != 1:
            step += 1
            if step >= size:
                step = 1
        scan_limit = min(
            size,
            config.max_candidates_per_topology
            * config.candidate_scan_multiplier,
        )
        for index in range(scan_limit):
            yield (offset + index * step) % size

    def locate_orientation(
        global_index: int,
        sizes: list[int],
    ) -> tuple[
        tuple[
            NodeSite,
            NodeSite,
            tuple[tuple[BlockRecord, float], ...],
            tuple[tuple[BlockRecord, float], ...],
        ],
        int,
    ]:
        for orientation, size in zip(orientations, sizes, strict=True):
            if global_index < size:
                return orientation, global_index
            global_index -= size
        raise IndexError("candidate Cartesian index exceeds its spaces")

    if config.strict_symmetry and config.strict_prefilter:
        edge_options = tuple(
            (edge_type, block)
            for edge_type in topology.edge_types
            for block in pools.get(
                ("edge", edge_type, 2, EDGE_POINT_GROUP),
                [],
            )
        )
    else:
        edge_options = tuple(
            (edge_type, block)
            for edge_type in topology.edge_types
            for block in pools.get(("edge", 2, EDGE_POINT_GROUP), [])
        )
    direct: dict[str, CandidateAssignment] = {}
    inserted: dict[str, CandidateAssignment] = {}

    def order_key(candidate: CandidateAssignment) -> tuple[object, ...]:
        tie_breaker = hashlib.sha256(
            f"{config.seed}:{topology.name}:{candidate.signature}".encode()
        ).hexdigest()
        return (*candidate.score, tie_breaker)

    direct_sizes = [
        len(metal_blocks) * len(organic_blocks)
        for _, _, metal_blocks, organic_blocks in orientations
    ]
    for global_index in permuted_indices(sum(direct_sizes), "direct"):
        direct_cap = (
            config.direct_candidate_cap
            if config.direct_candidate_cap > 0
            else config.max_candidates_per_topology
        )
        if len(direct) >= direct_cap:
            break
        orientation, local_index = locate_orientation(
            global_index,
            direct_sizes,
        )
        metal_site, organic_site, metal_blocks, organic_blocks = orientation
        organic_index = local_index % len(organic_blocks)
        metal_index = local_index // len(organic_blocks)
        metal_block, metal_fit = metal_blocks[metal_index]
        organic_block, organic_fit = organic_blocks[organic_index]
        if (
            not organic_block.contains_carbon
            or metal_block.sha256 == organic_block.sha256
            or metal_block.structure_id == organic_block.structure_id
        ):
            continue
        nodes = {
            metal_site.node_type: metal_block,
            organic_site.node_type: organic_block,
        }
        risk = assignment_prebuild_risk(
            topology,
            nodes,
            {},
            maximum_rmsd=config.max_rmsd,
            accuracy=config.accuracy,
        )
        if risk is None:
            continue
        signature = candidate_assignment_signature(topology.name, nodes, {})
        candidate = CandidateAssignment(
            signature=signature,
            nodes=tuple(sorted(nodes.items())),
            edges=(),
            score=(
                risk,
                max(metal_fit, organic_fit),
                max(block.port_radius_span for block in nodes.values()),
            ),
        )
        if signature not in direct or order_key(candidate) < order_key(
            direct[signature]
        ):
            direct[signature] = candidate

    inserted_sizes = [
        len(metal_blocks)
        * len(organic_blocks)
        * len(edge_options)
        for _, _, metal_blocks, organic_blocks in orientations
    ]
    for global_index in permuted_indices(sum(inserted_sizes), "inserted"):
        if len(inserted) >= config.max_candidates_per_topology:
            break
        orientation, local_index = locate_orientation(
            global_index,
            inserted_sizes,
        )
        metal_site, organic_site, metal_blocks, organic_blocks = orientation
        edge_option_index = local_index % len(edge_options)
        local_index //= len(edge_options)
        organic_index = local_index % len(organic_blocks)
        metal_index = local_index // len(organic_blocks)
        metal_block, metal_fit = metal_blocks[metal_index]
        organic_block, organic_fit = organic_blocks[organic_index]
        edge_type, edge_block = edge_options[edge_option_index]
        if (
            metal_block.sha256 == organic_block.sha256
            or metal_block.structure_id == organic_block.structure_id
            or edge_block.sha256
            in {metal_block.sha256, organic_block.sha256}
            or edge_block.structure_id
            in {metal_block.structure_id, organic_block.structure_id}
            or (
                not organic_block.contains_carbon
                and not edge_block.contains_carbon
            )
        ):
            continue
        nodes = {
            metal_site.node_type: metal_block,
            organic_site.node_type: organic_block,
        }
        edges = {edge_type: edge_block}
        risk = assignment_prebuild_risk(
            topology,
            nodes,
            edges,
            maximum_rmsd=config.max_rmsd,
            accuracy=config.accuracy,
        )
        if risk is None:
            continue
        signature = candidate_assignment_signature(topology.name, nodes, edges)
        candidate = CandidateAssignment(
            signature=signature,
            nodes=tuple(sorted(nodes.items())),
            edges=tuple(sorted(edges.items())),
            score=(
                risk,
                max(metal_fit, organic_fit),
                max(block.port_radius_span for block in nodes.values()),
            ),
        )
        if signature not in inserted or order_key(candidate) < order_key(
            inserted[signature]
        ):
            inserted[signature] = candidate

    queues = [
        sorted(direct.values(), key=order_key),
        sorted(inserted.values(), key=order_key),
    ]
    if config.strict_symmetry and config.strict_prefilter:
        return tuple(sorted((*queues[0], *queues[1]), key=order_key))
    ordered: list[CandidateAssignment] = []
    positions = [0, 0]
    turn = 0
    while (
        len(ordered) < config.max_candidates_per_topology
        and any(position < len(queue) for position, queue in zip(positions, queues))
    ):
        queue_index = turn % 2
        if positions[queue_index] >= len(queues[queue_index]):
            queue_index = 1 - queue_index
        ordered.append(queues[queue_index][positions[queue_index]])
        positions[queue_index] += 1
        turn += 1
    return tuple(ordered)


def build_topology_quota(job: dict[str, object]) -> dict[str, object]:
    """Consume one topology's candidate queue without replacement."""

    config = job["config"]
    topology_record = job["topology"]
    candidates = job["candidates"]
    start = int(job["start"])
    target = int(job["target"])
    attempt_limit = int(job["attempt_limit"])
    pass_index = int(job["pass_index"])
    staging = Path(job["staging"])
    staging.mkdir(parents=True, exist_ok=True)
    configure_quiet_builder_logging()
    successes = []
    failure_types: Counter[str] = Counter()
    failure_examples = []
    attempted = 0
    cursor = start

    with threadpool_limits(limits=1, user_api="blas"):
        try:
            topology = cached_topology(str(topology_record.cgd_path))
        except Exception as error:
            return {
                "topology": topology_record.name,
                "pass_index": pass_index,
                "start": start,
                "next_cursor": start,
                "attempted": 0,
                "requested": target,
                "successes": [],
                "failure_types": {type(error).__name__: 1},
                "failure_examples": [
                    {
                        "candidate_index": None,
                        "error_type": type(error).__name__,
                        "error": str(error)[:1000],
                    }
                ],
                "topology_error": True,
            }

        stop = min(len(candidates), start + attempt_limit)
        for candidate_index in range(start, stop):
            if len(successes) >= target:
                break
            candidate = candidates[candidate_index]
            attempted += 1
            cursor = candidate_index + 1
            cif_path = staging / f"{candidate.signature}.cif"
            nodes = dict(candidate.nodes)
            edges = dict(candidate.edges)
            try:
                quality = build_and_validate(
                    config,
                    topology_record,
                    {key: block.path for key, block in nodes.items()},
                    {key: block.path for key, block in edges.items()},
                    cif_path,
                    topology=topology,
                )
                quality["cif_sha256"] = file_sha256(cif_path)
                successes.append(
                    {
                        "candidate_index": candidate_index,
                        "candidate": candidate,
                        "cif_path": str(cif_path),
                        "quality": quality,
                    }
                )
            except Exception as error:
                cif_path.unlink(missing_ok=True)
                error_type = type(error).__name__
                failure_types[error_type] += 1
                if len(failure_examples) < 20:
                    failure_examples.append(
                        {
                            "candidate_index": candidate_index,
                            "error_type": error_type,
                            "error": str(error)[:1000],
                        }
                    )

    return {
        "topology": topology_record.name,
        "pass_index": pass_index,
        "start": start,
        "next_cursor": cursor,
        "attempted": attempted,
        "requested": target,
        "successes": successes,
        "failure_types": dict(failure_types),
        "failure_examples": failure_examples,
        "topology_error": False,
    }


def balanced_extra_targets(
    current_counts: dict[str, int],
    active_names: list[str],
    needed: int,
) -> dict[str, int]:
    """Water-fill extra quotas so productive topologies stay balanced."""

    targets: Counter[str] = Counter()
    projected = dict(current_counts)
    for _ in range(needed):
        name = min(
            active_names,
            key=lambda value: (projected[value], value),
        )
        targets[name] += 1
        projected[name] += 1
    return dict(targets)


def wilson_upper_bound(successes: int, attempts: int, z: float = 1.96) -> float:
    """Return a one-sided upper confidence bound for a binomial yield."""

    if attempts <= 0:
        return 1.0
    proportion = min(max(successes / attempts, 0.0), 1.0)
    z_squared = z * z
    denominator = 1.0 + z_squared / attempts
    center = proportion + z_squared / (2.0 * attempts)
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / attempts
        + z_squared / (4.0 * attempts * attempts)
    )
    return min(1.0, (center + radius) / denominator)


def low_yield_retirement(
    successes: int,
    attempts: int,
    *,
    minimum_attempts: int,
    minimum_yield: float,
    confidence_z: float,
) -> dict[str, float | int] | None:
    """Describe a queue proven too weak to merit normal quota probing."""

    if minimum_yield <= 0.0 or attempts < minimum_attempts:
        return None
    upper_bound = wilson_upper_bound(successes, attempts, confidence_z)
    if upper_bound >= minimum_yield:
        return None
    return {
        "successes": successes,
        "attempts": attempts,
        "observed_yield": successes / attempts,
        "yield_upper_bound": upper_bound,
        "minimum_yield": minimum_yield,
        "confidence_z": confidence_z,
    }


def round_robin_successes(
    successes: dict[str, list[dict[str, object]]],
) -> list[tuple[str, dict[str, object]]]:
    """Interleave topology outputs instead of grouping one net at a time."""

    ordered = []
    names = sorted(successes)
    index = 0
    while True:
        added = False
        for name in names:
            if index < len(successes[name]):
                ordered.append((name, successes[name][index]))
                added = True
        if not added:
            break
        index += 1
    return ordered


def generate_topology_balanced(
    config: GenerationConfig,
    *,
    workers: int = DEFAULT_GENERATION_WORKERS,
    max_passes: int = 30,
    resume: bool = False,
) -> list[dict]:
    """Generate a balanced, without-replacement set topology by topology."""

    if (
        config.count < 1
        or workers < 1
        or max_passes < 1
        or config.max_candidates_per_topology < 1
        or config.edge_conformer_filter not in {"all", "non_c1"}
        or config.candidate_scan_multiplier < 1
        or config.attempt_batch_size < 1
        or config.adaptive_min_attempts < 1
        or not 0.0 <= config.adaptive_minimum_yield <= 1.0
        or config.adaptive_confidence_z <= 0.0
        or config.worker_memory_limit_gib < 1
        or config.symmetry_tolerance <= 0.0
    ):
        raise ValueError(
            "generation counts, workers, and limits must be positive; "
            "edge_conformer_filter must be 'all' or 'non_c1'; "
            "adaptive_minimum_yield must be between zero and one"
        )
    output = config.output
    protected = [
        output / "mofs",
        output / "building_blocks",
        output / "manifest.json",
        output / "manifest.csv",
        output / "generation_summary.json",
    ]
    existing = [path for path in protected if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing results: {existing}")

    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    staging = output / ".topology_quota_staging"
    if staging.exists() and not resume:
        raise FileExistsError(f"refusing to overwrite staging: {staging}")
    if resume and not staging.is_dir():
        raise FileNotFoundError(f"resume staging directory not found: {staging}")
    staging.mkdir(exist_ok=True)
    configure_quiet_builder_logging()
    # Point-group and small alignment calculations do not benefit from a large
    # BLAS team.  Limiting this serial phase also prevents it from contending
    # with the process-parallel candidate phase that follows.
    with threadpool_limits(limits=1, user_api="blas"):
        pools, skipped_blocks = load_block_pools(config)
        selected_block_counts = Counter(
            record.kind for record in unique_library_blocks(pools)
        )
        print(
            f"selected_blocks={dict(sorted(selected_block_counts.items()))} "
            f"edge_conformer_filter={config.edge_conformer_filter}",
            flush=True,
        )
        if not selected_block_counts["edge"]:
            raise RuntimeError(
                "the selected symmetry cache contains no usable edges"
            )
        if config.strict_symmetry and config.strict_prefilter:
            strict_edges = strict_edge_conformer_pool(pools)
            print(
                f"site_routed_edge_pool={len(strict_edges)} "
                f"including_c1_and_directional=true",
                flush=True,
            )
        topologies, skipped_topologies = load_eligible_topologies(config, pools)
    topologies = sorted(topologies, key=lambda item: item.name)
    if not topologies:
        raise RuntimeError("no eligible two-node topologies were found")

    candidate_queues: dict[str, tuple[CandidateAssignment, ...]] = {}
    candidate_preparation_stats: dict[str, dict[str, object]] = {}
    topology_by_name = {topology.name: topology for topology in topologies}
    with ProcessPoolExecutor(
        max_workers=min(workers, len(topologies)),
        initializer=configure_candidate_worker,
        initargs=(config, pools),
    ) as executor:
        futures = {
            executor.submit(enumerate_topology_candidates_job, topology): topology
            for topology in topologies
        }
        for index, future in enumerate(as_completed(futures), start=1):
            topology = futures[future]
            try:
                name, candidates, preparation_stats = future.result()
            except Exception as error:
                name = topology.name
                candidates = ()
                preparation_stats = {}
                skipped_topologies[
                    f"candidate_enumeration_{type(error).__name__}"
                ] += 1
                print(
                    f"candidate_topology_error name={name} "
                    f"error={type(error).__name__}:{str(error)[:500]}",
                    flush=True,
                )
            candidate_queues[name] = candidates
            candidate_preparation_stats[name] = preparation_stats
            exact_stats = preparation_stats.get("exact_symmetry", {})
            print(
                f"candidate_topology={index}/{len(topologies)} "
                f"name={name} unique_assignments={len(candidates)} "
                f"symmetry_pruned={exact_stats.get('rejected_candidates', 0)}",
                flush=True,
            )

    preparation_elapsed_seconds = time.monotonic() - started
    build_phase_started = time.monotonic()

    base_quota, remainder = divmod(config.count, len(topologies))
    quota_order = [topology.name for topology in topologies]
    random.Random(config.seed).shuffle(quota_order)
    initial_quotas = {
        name: base_quota + int(index < remainder)
        for index, name in enumerate(quota_order)
    }
    print(
        f"topology_first_plan={len(topologies)} target={config.count} "
        f"base_quota={base_quota} remainder={remainder} "
        f"mean_quota={config.count / len(topologies):.6f}",
        flush=True,
    )
    queue_fingerprints = {
        name: candidate_queue_fingerprint(candidates)
        for name, candidates in candidate_queues.items()
    }
    checkpoint_path = staging / "resume_checkpoint.jsonl"

    cursors = {name: 0 for name in topology_by_name}
    successes: dict[str, list[dict[str, object]]] = {
        name: [] for name in topology_by_name
    }
    seen_signatures: set[str] = set()
    seen_cif_hashes: set[str] = set()
    attempted_counts: Counter[str] = Counter()
    failure_types: Counter[str] = Counter()
    failure_examples: list[dict[str, object]] = []
    duplicates = Counter()
    quarantined_topologies: set[str] = set()
    retired_low_yield: dict[str, dict[str, float | int]] = {}
    low_yield_fallbacks: list[dict[str, object]] = []
    pass_summaries = []
    resumed_count = 0
    resume_start_pass = 1

    def update_low_yield_status(name: str, pass_index: int) -> bool:
        retirement = low_yield_retirement(
            len(successes[name]),
            attempted_counts[name],
            minimum_attempts=config.adaptive_min_attempts,
            minimum_yield=config.adaptive_minimum_yield,
            confidence_z=config.adaptive_confidence_z,
        )
        if retirement is None:
            return False
        if name not in retired_low_yield:
            retired_low_yield[name] = {
                **retirement,
                "retired_after_pass": pass_index,
                "cursor": cursors[name],
                "queue_size": len(candidate_queues[name]),
            }
            print(
                f"topology_queue_retired name={name} pass={pass_index} "
                f"successes={len(successes[name])} "
                f"attempts={attempted_counts[name]} "
                f"observed_yield={retirement['observed_yield']:.6f} "
                f"upper_bound={retirement['yield_upper_bound']:.6f} "
                f"minimum_yield={config.adaptive_minimum_yield:.6f}",
                flush=True,
            )
        return True

    def productive_topology_names(pass_index: int) -> list[str]:
        normal = [
            name
            for name in topology_by_name
            if successes[name]
            and name not in quarantined_topologies
            and name not in retired_low_yield
            and cursors[name] < len(candidate_queues[name])
        ]
        if normal:
            productive_cap = max(32, workers * 4)
            normal.sort(
                key=lambda name: (
                    len(successes[name]) / max(1, attempted_counts[name]),
                    len(successes[name]),
                    name,
                ),
                reverse=True,
            )
            return normal[:productive_cap]

        # Retired queues are a reserve, not a hard exclusion: if they are the
        # only source left, probe only the statistically strongest ones so a
        # strict yield cutoff cannot make an otherwise achievable target fail.
        fallback = [
            name
            for name in retired_low_yield
            if successes[name]
            and name not in quarantined_topologies
            and cursors[name] < len(candidate_queues[name])
        ]
        fallback.sort(
            key=lambda name: (
                wilson_upper_bound(
                    len(successes[name]),
                    attempted_counts[name],
                    config.adaptive_confidence_z,
                ),
                len(successes[name]),
                name,
            ),
            reverse=True,
        )
        selected = fallback[:workers]
        if selected:
            event = {
                "pass_index": pass_index,
                "topologies": selected,
                "reason": "all_nonretired_productive_queues_exhausted",
            }
            low_yield_fallbacks.append(event)
            print(
                f"low_yield_fallback pass={pass_index} "
                f"topologies={','.join(selected)}",
                flush=True,
            )
        return selected

    if resume:
        checkpoint_records = read_resume_checkpoint(checkpoint_path)
        valid_jobs: dict[tuple[int, str], dict[str, object]] = {}
        checkpoint_successes: dict[
            tuple[str, str], dict[str, object]
        ] = {}
        expected_validation = validation_fingerprint(config)
        for record in checkpoint_records:
            try:
                name = str(record["topology"])
                topology = topology_by_name[name]
                topology_stat = topology.cgd_path.stat()
                valid = (
                    int(record["version"]) == RESUME_CHECKPOINT_VERSION
                    and record["validation_fingerprint"] == expected_validation
                    and record["queue_fingerprint"] == queue_fingerprints[name]
                    and int(record["topology_size"]) == topology_stat.st_size
                    and int(record["topology_mtime_ns"])
                    == topology_stat.st_mtime_ns
                )
            except (KeyError, TypeError, ValueError):
                continue
            if not valid:
                continue
            pass_index = int(record["pass_index"])
            valid_jobs[(pass_index, name)] = record
            for saved in record.get("successes", []):
                checkpoint_successes[(name, str(saved["signature"]))] = saved

        for (pass_index, name), record in sorted(valid_jobs.items()):
            attempted_counts[name] += int(record["attempted"])
            cursors[name] = max(cursors[name], int(record["next_cursor"]))
            resume_start_pass = max(resume_start_pass, pass_index + 1)

        recovered_by_name: dict[
            str,
            list[tuple[int, dict[str, object]]],
        ] = defaultdict(list)
        recovery_jobs = []
        staged_count = 0
        checkpoint_loaded = 0

        def add_recovered(
            name: str,
            candidate_index: int,
            candidate: CandidateAssignment,
            cif_path: Path,
            quality: dict[str, object],
        ) -> None:
            if candidate.signature in seen_signatures:
                raise ValueError(
                    f"duplicate staged assignment signature: {candidate.signature}"
                )
            cif_hash = str(quality["cif_sha256"])
            if cif_hash in seen_cif_hashes:
                raise ValueError(f"duplicate staged CIF hash: {cif_hash}")
            seen_signatures.add(candidate.signature)
            seen_cif_hashes.add(cif_hash)
            quality = {**quality, "resumed_from_staging": True}
            recovered_by_name[name].append(
                (
                    candidate_index,
                    {
                        "candidate_index": candidate_index,
                        "candidate": candidate,
                        "cif_path": str(cif_path),
                        "quality": quality,
                    },
                )
            )

        for name, candidates in candidate_queues.items():
            by_signature = {
                candidate.signature: (index, candidate)
                for index, candidate in enumerate(candidates)
            }
            for cif_path in sorted((staging / name).glob("*.cif")):
                candidate_entry = by_signature.get(cif_path.stem)
                if candidate_entry is None:
                    raise ValueError(
                        f"staged signature is absent from regenerated queue: "
                        f"{cif_path}"
                    )
                candidate_index, candidate = candidate_entry
                staged_count += 1
                saved = checkpoint_successes.get((name, candidate.signature))
                trusted = False
                if saved is not None and not config.resume_revalidate:
                    try:
                        cif_stat = cif_path.stat()
                        quality = dict(saved["quality"])
                        trusted = (
                            int(saved["candidate_index"]) == candidate_index
                            and saved["cif_name"] == cif_path.name
                            and int(saved["cif_size"]) == cif_stat.st_size
                            and int(saved["cif_mtime_ns"])
                            == cif_stat.st_mtime_ns
                            and len(str(quality["cif_sha256"])) == 64
                        )
                    except (KeyError, TypeError, ValueError):
                        trusted = False
                if trusted:
                    add_recovered(
                        name,
                        candidate_index,
                        candidate,
                        cif_path,
                        quality,
                    )
                    checkpoint_loaded += 1
                else:
                    recovery_jobs.append(
                        (name, candidate_index, candidate, cif_path)
                    )

        if not staged_count:
            raise RuntimeError("resume staging contains no CIF files")
        print(
            f"resume_recovery_plan={staged_count} "
            f"checkpoint_loaded={checkpoint_loaded} "
            f"revalidate={len(recovery_jobs)}",
            flush=True,
        )
        if recovery_jobs:
            with ProcessPoolExecutor(
                max_workers=min(workers, len(recovery_jobs)),
                initializer=configure_generation_worker,
                initargs=(config.worker_memory_limit_gib,),
            ) as executor:
                validated = 0
                batch_size = max(32, workers * 4)
                for batch_start in range(0, len(recovery_jobs), batch_size):
                    batch = recovery_jobs[batch_start : batch_start + batch_size]
                    futures = {
                        executor.submit(validate_staged_cif, config, cif_path): (
                            name,
                            candidate_index,
                            candidate,
                            cif_path,
                        )
                        for name, candidate_index, candidate, cif_path in batch
                    }
                    for future in as_completed(futures):
                        name, candidate_index, candidate, cif_path = futures[future]
                        add_recovered(
                            name,
                            candidate_index,
                            candidate,
                            cif_path,
                            future.result(),
                        )
                        validated += 1
                        if validated % 250 == 0 or validated == len(recovery_jobs):
                            print(
                                f"resume_validated={validated}/{len(recovery_jobs)}",
                                flush=True,
                            )

        for name in candidate_queues:
            recovered = recovered_by_name[name]
            recovered.sort(key=lambda value: value[0])
            successes[name].extend(value for _, value in recovered)
            if recovered:
                cursors[name] = max(cursors[name], recovered[-1][0] + 1)

        resumed_count = sum(len(values) for values in successes.values())
        prior_failures = max(0, sum(attempted_counts.values()) - resumed_count)
        failure_types["prior_run_rejected_candidate"] += prior_failures
        print(
            f"resume_loaded={resumed_count}/{config.count} "
            f"resume_start_pass={resume_start_pass} "
            f"prior_attempted={sum(attempted_counts.values())}",
            flush=True,
        )
        for name in topology_by_name:
            update_low_yield_status(name, max(0, resume_start_pass - 1))

    def accept_result(result: dict[str, object]) -> int:
        name = str(result["topology"])
        cursors[name] = max(cursors[name], int(result["next_cursor"]))
        attempted_counts[name] += int(result["attempted"])
        failure_types.update(result["failure_types"])
        for failure in result["failure_examples"]:
            if len(failure_examples) < 1000:
                failure_examples.append(
                    {
                        **failure,
                        "topology": name,
                        "pass_index": result["pass_index"],
                    }
                )
        accepted_successes = []
        for success in result["successes"]:
            candidate = success["candidate"]
            cif_path = Path(success["cif_path"])
            if candidate.signature in seen_signatures:
                duplicates["assignment"] += 1
                cif_path.unlink(missing_ok=True)
                continue
            cif_hash = success["quality"]["cif_sha256"]
            if cif_hash in seen_cif_hashes:
                duplicates["cif"] += 1
                cif_path.unlink(missing_ok=True)
                continue
            seen_signatures.add(candidate.signature)
            seen_cif_hashes.add(cif_hash)
            successes[name].append(success)
            accepted_successes.append(success)
        append_resume_checkpoint(
            checkpoint_path,
            config,
            topology_by_name[name],
            queue_fingerprints[name],
            result,
            accepted_successes,
        )
        return len(accepted_successes)

    if resumed_count:
        active_names = productive_topology_names(resume_start_pass)
        if resumed_count < config.count and not active_names:
            raise RuntimeError("no productive candidate queues remain for resume")
        targets = balanced_extra_targets(
            {name: len(successes[name]) for name in topology_by_name},
            active_names,
            config.count - resumed_count,
        )
    else:
        targets = initial_quotas
    generation_executor = ProcessPoolExecutor(
        max_workers=min(workers, len(topology_by_name)),
        initializer=configure_generation_worker,
        initargs=(config.worker_memory_limit_gib,),
    )
    for pass_index in range(resume_start_pass, max_passes + 1):
        jobs = []
        for name, target in sorted(targets.items()):
            remaining_candidates = len(candidate_queues[name]) - cursors[name]
            if target <= 0 or remaining_candidates <= 0:
                continue
            jobs.append(
                {
                    "config": config,
                    "topology": topology_by_name[name],
                    "candidates": candidate_queues[name],
                    "start": cursors[name],
                    "target": target,
                    "attempt_limit": min(
                        remaining_candidates,
                        config.attempt_batch_size,
                    ),
                    "pass_index": pass_index,
                    "staging": str(staging / name),
                }
            )
        if not jobs:
            break

        pass_attempted = 0
        pass_accepted = 0
        futures = {
            generation_executor.submit(build_topology_quota, job): job
            for job in jobs
        }
        for future in as_completed(futures):
            job = futures[future]
            name = str(job["topology"].name)
            try:
                result = future.result()
            except Exception as error:
                quarantined_topologies.add(name)
                result = {
                    "topology": name,
                    "pass_index": pass_index,
                    "next_cursor": len(candidate_queues[name]),
                    "attempted": 0,
                    "successes": [],
                    "failure_types": {type(error).__name__: 1},
                    "failure_examples": [
                        {
                            "candidate_index": None,
                            "error_type": type(error).__name__,
                            "error": str(error)[:1000],
                        }
                    ],
                }
            accepted = accept_result(result)
            update_low_yield_status(name, pass_index)
            pass_attempted += int(result["attempted"])
            pass_accepted += accepted
            print(
                f"quota_pass={pass_index} topology={name} "
                f"accepted={accepted}/{job['target']} "
                f"total={len(successes[name])} "
                f"attempted={result['attempted']} "
                f"cursor={cursors[name]}/{len(candidate_queues[name])}",
                flush=True,
            )

        total = sum(len(values) for values in successes.values())
        pass_summaries.append(
            {
                "pass_index": pass_index,
                "jobs": len(jobs),
                "attempted": pass_attempted,
                "accepted": pass_accepted,
                "cumulative": total,
                "retired_low_yield_topologies": len(retired_low_yield),
            }
        )
        print(
            f"quota_pass_complete={pass_index} attempted={pass_attempted} "
            f"accepted={pass_accepted} total={total}/{config.count}",
            flush=True,
        )
        if total >= config.count:
            break

        productive = productive_topology_names(pass_index + 1)
        if not productive:
            break
        needed = config.count - total
        targets = balanced_extra_targets(
            {name: len(successes[name]) for name in topology_by_name},
            productive,
            needed,
        )
    generation_executor.shutdown()
    build_elapsed_seconds = time.monotonic() - build_phase_started

    finalization_started = time.monotonic()
    ordered_successes = round_robin_successes(successes)[: config.count]
    mofs_dir = output / "mofs"
    blocks_dir = output / "building_blocks"
    topology_dir = output / "topologies"
    for directory in (mofs_dir, blocks_dir, topology_dir):
        directory.mkdir()
    used_topologies = sorted({name for name, _ in ordered_successes})
    for name in used_topologies:
        shutil.copy2(
            topology_by_name[name].cgd_path,
            topology_dir / f"{name}.cgd",
        )

    rows = []
    for index, (name, success) in enumerate(ordered_successes, start=1):
        topology = topology_by_name[name]
        candidate = success["candidate"]
        sample_id = f"single_linker_mof_{index:05d}_{name}"
        final_cif = mofs_dir / f"{sample_id}.cif"
        os.replace(success["cif_path"], final_cif)
        final_blocks = blocks_dir / sample_id
        nodes = dict(candidate.nodes)
        edges = dict(candidate.edges)
        node_paths, edge_paths = copy_selected_blocks(
            final_blocks,
            nodes,
            edges,
        )
        row = {
            "sample_id": sample_id,
            "cif": str(final_cif.relative_to(output)),
            "assignment_signature": candidate.signature,
            "generation_seed": config.seed,
            "topology": {
                "name": topology.name,
                "cgd_path": str(topology.cgd_path),
                "saved_cgd": str(
                    (topology_dir / f"{name}.cgd").relative_to(output)
                ),
                "space_group": topology.space_group,
                "num_nodes": topology.num_nodes,
                "num_edges": topology.num_edges,
                "node_sites": [asdict(site) for site in topology.node_sites],
                "edge_types": [list(value) for value in topology.edge_types],
            },
            "node_assignments": [
                {
                    "node_type": node_type,
                    "block": block_manifest(
                        block,
                        final_blocks.relative_to(output)
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
                        final_blocks.relative_to(output)
                        / edge_paths[edge_type].name,
                    ),
                }
                for edge_type, block in edges.items()
            ],
            "quality": success["quality"],
        }
        rows.append(row)

    final_counts = {
        name: sum(row["topology"]["name"] == name for row in rows)
        for name in topology_by_name
    }
    count_values = list(final_counts.values())
    finalization_elapsed_seconds = time.monotonic() - finalization_started
    preparation_totals = {
        section: dict(
            sum(
                (
                    Counter(stats.get(section, {}))
                    for stats in candidate_preparation_stats.values()
                ),
                Counter(),
            )
        )
        for section in ("nodes", "edges", "exact_symmetry")
    }
    summary = {
        "requested": config.count,
        "completed": len(rows),
        "seed": config.seed,
        "mode": "topology_first_without_replacement",
        "resumed_from_staging": resumed_count,
        "elapsed_seconds": time.monotonic() - started,
        "phase_elapsed_seconds": {
            "block_loading_topology_screening_candidate_enumeration": (
                preparation_elapsed_seconds
            ),
            "building_and_testing": build_elapsed_seconds,
            "finalization": finalization_elapsed_seconds,
        },
        "candidate_preparation_totals": preparation_totals,
        "source_manifest": str(config.manifest),
        "topology_directory": str(config.topology_dir),
        "eligible_topologies": len(topologies),
        "productive_topologies": sum(value > 0 for value in count_values),
        "initial_mean_quota": config.count / len(topologies),
        "initial_quota_by_topology": dict(sorted(initial_quotas.items())),
        "candidate_count_by_topology": {
            name: len(candidate_queues[name]) for name in sorted(candidate_queues)
        },
        "attempted_count_by_topology": dict(sorted(attempted_counts.items())),
        "final_count_by_topology": dict(sorted(final_counts.items())),
        "final_count_statistics_all_eligible": {
            "mean": statistics.fmean(count_values),
            "median": statistics.median(count_values),
            "minimum": min(count_values),
            "maximum": max(count_values),
        },
        "unique_assignment_signatures": len(
            {row["assignment_signature"] for row in rows}
        ),
        "unique_cif_hashes": len(
            {row["quality"]["cif_sha256"] for row in rows}
        ),
        "duplicates_removed": dict(duplicates),
        "quarantined_topologies": sorted(quarantined_topologies),
        "retired_low_yield_topologies": dict(sorted(retired_low_yield.items())),
        "low_yield_fallbacks": low_yield_fallbacks,
        "failed_attempts": sum(failure_types.values()),
        "failure_types": dict(failure_types),
        "failure_examples": failure_examples,
        "skipped_blocks": dict(skipped_blocks),
        "skipped_topologies": dict(skipped_topologies),
        "quota_passes": pass_summaries,
        "settings": {
            **asdict(config),
            "manifest": str(config.manifest),
            "topology_dir": str(config.topology_dir),
            "output": str(config.output),
            "workers": workers,
            "max_passes": max_passes,
            "selection_rule": (
                "retain every connected library block and route nodes by local port "
                "geometry without requiring molecular symmetry; by default allow "
                "independent orbit placement, including P1 outcomes, without "
                "post-build space-group analysis; when strict symmetry is explicitly "
                "requested, route "
                "blocks by the crystallographic site stabilizer, exact-check each "
                "unique block/site pair appearing in a bounded queue once, and "
                "prune incompatible candidates before quota scheduling; traverse "
                "each CN/site/port-shape-compatible Cartesian space in a seeded "
                "affine order without replacement and retain a bounded, geometry-"
                "fitted queue per topology; require a chemically and "
                "bond-length-feasible one-to-one node-port assignment, including "
                "joint orientations for heterotypic inserted edges; use "
                "one-metal-node + one-organic-node + optional-one-edge assignment "
                "with CN2 and D*h port geometry for linkers; filter linkers by the "
                "configured conformer rule only when explicitly requested; when "
                "strict symmetry is enabled, reject any node or edge whose complete "
                "atomic conformer and mapped ports fail a crystallographic-site "
                "stabilizer or orbit-covariance check, without independent-placement "
                "fallback; otherwise retain that independent-placement P1 candidate; "
                "chirality is diagnostic only; "
                "enumerate topology queues in parallel, deduplicate per topology "
                "before building, and consume each topology "
                "queue without replacement to its requested-count/N quota; "
                "redistribute shortfalls by water-filling productive topologies; "
                "retire queues whose Wilson yield upper bound is below the "
                "configured minimum after the pilot-attempt floor, retaining "
                "them only as an exhaustion fallback"
            ),
        },
    }
    write_results(output, rows, summary)
    shutil.rmtree(staging)
    if len(rows) != config.count:
        raise RuntimeError(
            f"generated {len(rows)} of {config.count} requested unique MOFs; "
            f"see {output / 'generation_summary.json'}"
        )
    return rows

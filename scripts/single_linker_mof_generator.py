#!/usr/bin/env python3
"""Shared records, geometry, and validation for single-linker MOF generators.

Library loading lives in single_linker_catalog; the balanced 10k workflow lives
in single_linker_quota. Their public entry points below load those modules on
demand. Record classes stay here to preserve stored candidate/checkpoint types.
"""

from __future__ import annotations

import csv
from functools import lru_cache
import hashlib
from itertools import permutations, product
import json
import logging
import math
import os
import resource
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import time

import ase.neighborlist
from ase import Atoms
from ase.io import read as read_ase
import numpy as np
from pymatgen.symmetry.groups import SpaceGroup
from scipy.spatial.transform import Rotation
import spglib
from sympy.combinatorics import Permutation, PermutationGroup

from mofbb.compose import Builder, BuildingBlock, Locator, Topology
from mofbb.compose.connectivity import CONNECTIVITY_POLICY_VERSION, validate_3d_connectivity
from mofbb.compose.linkers import SINGLE_LINKER_POLICY_VERSION, validate_single_linker
from mofbb.compose.log import logger
from mofbb.compose.locator import (
    PortAssignmentError,
    find_best_orientation,
    find_best_permutation,
)
from mofbb.compose.pbc import PBC
from mofbb.compose.placement.symmetry import SymmetryPlacement


DEFAULT_GENERATION_WORKERS = min(16, os.cpu_count() or 1)

EDGE_POINT_GROUP = "D*h"
ALL_NODE_POINT_GROUP = "*"
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
    max_output_atoms: int = 0
    edge_conformer_filter: str = "all"
    max_rmsd: float = 0.60
    minimum_atom_distance: float = 0.50
    require_3d_connectivity: bool = True
    accuracy: int = 4
    symmetry_tolerance: float = 0.1
    strict_symmetry: bool = False
    strict_prefilter: bool = True
    relaxed_all_node_shapes: bool = False
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    *,
    kinds: tuple[str, ...] = ("metal_node", "organic_node"),
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
        for kind in kinds:
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
                not PBC.is_forbidden_fused_pair(symbol_i, symbol_j)
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

    Mirror the builder's one-port-per-neighbor assignment and jointly test both
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


def _validate_structure(
    config: GenerationConfig,
    atoms: Atoms,
    *,
    staged: bool = False,
) -> dict[str, object]:
    """Apply shared structure checks and collect quality metadata."""

    source = "staged CIF" if staged else "assembled framework"
    symbols = atoms.get_chemical_symbols()
    if staged and ("X" in symbols or "C" not in symbols):
        raise ValueError("staged CIF fails carbon/connection-site validation")
    if "X" in symbols:
        raise ValueError("assembled framework still contains X connection sites")
    if "C" not in symbols:
        raise ValueError("assembled framework contains no carbon")
    if (
        not np.isfinite(atoms.positions).all()
        or atoms.cell.volume <= 0
        or (staged and not np.isfinite(atoms.get_scaled_positions(wrap=False)).all())
    ):
        raise ValueError(f"{source} has invalid coordinates or cell")
    if config.max_output_atoms and len(atoms) > config.max_output_atoms:
        raise ValueError(
            f"{source} has {len(atoms)} atoms, exceeding the "
            f"configured limit of {config.max_output_atoms}"
        )
    connectivity = {}
    if config.require_3d_connectivity:
        connectivity = validate_3d_connectivity(atoms, minimum_distance_cutoff=3.0)
        closest = connectivity["minimum_atom_distance_angstrom"]
    else:
        distances = ase.neighborlist.neighbor_list("d", atoms, 3.0)
        closest = float(distances.min()) if len(distances) else None
    if closest is not None and closest < config.minimum_atom_distance:
        if staged:
            raise ValueError("staged CIF fails minimum-distance validation")
        raise ValueError(
            f"minimum atom distance {closest:.3f} A is below "
            f"{config.minimum_atom_distance:.3f} A"
        )

    return {
        **connectivity,
        **validate_single_linker(atoms),
        "num_atoms": len(atoms),
        "formula": atoms.get_chemical_formula(mode="metal"),
        "cell_parameters": [float(value) for value in atoms.cell.cellpar()],
        "minimum_atom_distance_angstrom": closest,
        "cif_space_group_symbol": "P1",
        "cif_space_group_number": 1,
        "placement_mode": (
            "strict_symmetry" if config.strict_symmetry else "allow_p1_fallback"
        ),
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
    atoms = framework.atoms
    structure_quality = _validate_structure(config, atoms)
    framework.write_cif(cif_path)
    if not cif_path.is_file() or cif_path.stat().st_size == 0:
        raise ValueError("MOFBB did not write a nonempty CIF")

    diagnostics = framework.info.get("symmetry_placement_diagnostics", {})
    fallback_orbits = sum(
        not item["symmetry_propagated"]
        for kind in ("nodes", "edges")
        for item in diagnostics.get(kind, ())
    )

    quality = {
        **structure_quality,
        "max_rmsd": max_rmsd,
        "mean_rmsd": float(framework.info["mean_rmsd"]),
        "geometry_optimization_objective": float(
            framework.info["geometry_optimization_objective"]
        ),
        "relax_obj": float(framework.info["geometry_optimization_objective"]),
        "symmetry_fallback_orbit_count": int(fallback_orbits),
    }
    if crystal_symmetry_symprec is not None:
        quality.update(
            detect_crystal_symmetry(
                atoms,
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
    structure_quality = _validate_structure(config, atoms, staged=True)
    max_rmsd = comment_value("# Max RMSD:")
    if max_rmsd is None or max_rmsd > config.max_rmsd:
        raise ValueError("staged CIF fails RMSD validation")
    geometry_objective = comment_value("# Geometry optimization objective:")
    if geometry_objective is None:
        geometry_objective = comment_value("# Relax obj value:")
    return {
        **structure_quality,
        "max_rmsd": max_rmsd,
        "mean_rmsd": comment_value("# Mean RMSD:"),
        "geometry_optimization_objective": geometry_objective,
        "relax_obj": geometry_objective,
        "symmetry_fallback_orbit_count": None,
        "cif_sha256": file_sha256(cif_path),
        "resumed_from_staging": True,
    }


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
        "single_linker_policy": SINGLE_LINKER_POLICY_VERSION,
        "max_rmsd": config.max_rmsd,
        "minimum_atom_distance": config.minimum_atom_distance,
        "accuracy": config.accuracy,
        "symmetry_tolerance": config.symmetry_tolerance,
        "strict_symmetry": config.strict_symmetry,
        "max_output_atoms": config.max_output_atoms,
    }
    if config.require_3d_connectivity:
        rules["connectivity_policy"] = CONNECTIVITY_POLICY_VERSION
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
        quality_fields = [
            "cif_space_group_number",
            "placement_mode",
            "symmetry_fallback_orbit_count",
            "formula",
            "num_atoms",
            "max_rmsd",
            "minimum_atom_distance_angstrom",
        ]
        fields = [
            "sample_id", "topology", "topology_space_group", "cif_space_group",
            *quality_fields, "cif",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "topology": row["topology"]["name"],
                    "topology_space_group": row["topology"]["space_group"],
                    "cif_space_group": row["quality"]["cif_space_group_symbol"],
                    **{field: row["quality"][field] for field in quality_fields},
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




def load_block_pools(
    config: GenerationConfig,
) -> tuple[dict[tuple[str, int, str], list[BlockRecord]], Counter[str]]:
    """Load the building-block catalog used by all generation workflows."""
    from single_linker_catalog import load_block_pools as load

    return load(config)


def topology_record_from_object(topology: object, path: Path) -> TopologyRecord:
    """Extract the topology's node and edge site records."""
    from single_linker_catalog import topology_record_from_object as convert

    return convert(topology, path)


def load_eligible_topologies(
    config: GenerationConfig,
    pools: dict[tuple[str, int, str], list[BlockRecord]],
    *,
    topology_mode: str = "binodal",
    symmetry_mode: str = "strict",
) -> tuple[list[TopologyRecord], Counter[str]]:
    """Load topologies supported by the requested generation formulation."""
    from single_linker_catalog import load_eligible_topologies as load

    return load(config, pools, topology_mode=topology_mode, symmetry_mode=symmetry_mode)


def generate_topology_balanced(
    config: GenerationConfig,
    *,
    workers: int = DEFAULT_GENERATION_WORKERS,
    max_passes: int = 30,
    resume: bool = False,
) -> list[dict]:
    """Run the balanced 10k workflow, including checkpoint recovery."""
    from single_linker_quota import generate_topology_balanced as generate

    return generate(config, workers=workers, max_passes=max_passes, resume=resume)

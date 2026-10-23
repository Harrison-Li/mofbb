"""Building-block manifest loading and topology eligibility for MOF generation.

The core module owns record types, geometry checks, and worker caches. Public
loading entry points remain there for existing generation scripts.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import pickle

import numpy as np
from pymatgen.core import Molecule
from pymatgen.symmetry.analyzer import PointGroupAnalyzer

import single_linker_mof_generator as core


POINT_GROUP_TOLERANCE = 0.3
POINT_GROUP_EIGEN_TOLERANCE = 0.01
POINT_GROUP_MATRIX_TOLERANCE = 0.1
GRAPH_SYMMETRY_BASIS = "element_and_bond_labeled_graph_automorphism"
GRAPH_SYMMETRY_FIELDS = (
    "symmetry_basis",
    "graph_port_automorphism_order",
    "graph_port_automorphism_generators",
    "graph_supports_port_point_group",
)
TOPOLOGY_CACHE_MTIME_TOLERANCE_NS = 5_000_000_000


def block_kind_from_path(path: Path) -> str | None:
    if path.name.startswith("m_node"):
        return "metal_node"
    if path.name.startswith("o_node"):
        return "organic_node"
    if path.name.startswith("edge"):
        return "edge"
    return None


def resolve_manifest_block_paths(
    config: core.GenerationConfig,
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
            if path.is_file() and core.file_sha256(path) == expected_sha256:
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
                key = (kind, core.file_sha256(path))
            except OSError:
                continue
            indices = unresolved.pop(key, ())
            for index in indices:
                resolved[index] = path
                unresolved_reasons.pop(index, None)
                relocated += 1

    return resolved, relocated, unresolved_reasons


def load_block_pools(
    config: core.GenerationConfig,
) -> tuple[dict[tuple[str, int, str], list[core.BlockRecord]], Counter[str]]:
    """Load symmetry-eligible blocks, bucketed by geometric port point group."""

    pools: dict[tuple[str, int, str], list[core.BlockRecord]] = defaultdict(list)
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
                graph_supports_port_group = (
                    row["graph_supports_port_point_group"].strip().lower()
                )
                if graph_supports_port_group not in {"true", "false", "unclassified"}:
                    raise ValueError
            except (KeyError, AttributeError, TypeError, ValueError):
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
            and not core.edge_conformer_is_selected(row, config.edge_conformer_filter)
        ):
            skipped["edge_c1_conformer"] += 1
            continue
        atom_limit = (
            config.max_edge_atoms if kind == "edge" else config.max_node_atoms
        )
        if num_atoms > atom_limit:
            skipped["atom_limit"] += 1
            continue
        if graph_fields_present:
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
            or point_group != core.EDGE_POINT_GROUP
        ):
            skipped["edge_symmetry"] += 1
            continue
        if kind != "edge" and coordination < 3:
            skipped["node_coordination"] += 1
            continue

        try:
            block = core.BuildingBlock(path)
            block.validate_connectivity()
            contains_carbon = "C" in block.atoms.get_chemical_symbols()
            conn_symbols = tuple(block.conn_symbols)
            conn_dists = tuple(
                float(value) for value in block.conn_dists
            )
            port_directions = tuple(
                tuple(float(value) for value in direction)
                for direction in block.local_structure().positions
            )
            port_radii = np.linalg.norm(
                block.port_pos
                - np.mean(block.port_pos, axis=0),
                axis=1,
            )
        except Exception:
            skipped["invalid_block_metadata"] += 1
            continue
        if block.n_ports != coordination:
            skipped["connection_point_count_mismatch"] += 1
            continue
        if (
            len(conn_symbols) != coordination
            or len(conn_dists) != coordination
            or not all(conn_symbols)
        ):
            skipped["invalid_connection_anchor_symbols"] += 1
            continue

        record = core.BlockRecord(
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
            connection_anchor_symbols=conn_symbols,
            connection_anchor_lengths=conn_dists,
            port_shape=core.port_shape_signature(np.asarray(port_directions)),
            port_radius_span=float(np.max(port_radii) - np.min(port_radii)),
            port_directions=port_directions,
            graph_port_automorphism_generators=graph_generators,
        )
        pools[(kind, coordination, point_group)].append(record)
        if kind != "edge":
            pools[(kind, coordination, core.ALL_NODE_POINT_GROUP)].append(record)
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
) -> core.TopologyRecord:
    symmetry = core.SymPlacement.cached(topology)

    def port_permutations(orbit) -> tuple[tuple[int, ...], ...]:
        return tuple(sorted({
            tuple(int(value) for value in symmetry.offset_map(
                orbit.representative, orbit.representative, int(operation)
            ))
            for operation in orbit.stabilizers
        }))

    node_orbit_by_type = {
        int(topology.node_types[orbit.representative]): orbit
        for orbit in symmetry.node_orbits
    }
    sites: list[core.NodeSite] = []
    for node_type, coordination in zip(
        topology.unique_node_types, topology.unique_cn, strict=True
    ):
        node_type = int(node_type)
        representative = next(
            int(index)
            for index in topology.node_idx
            if int(topology.node_types[index]) == node_type
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
        sites.append(
            core.NodeSite(
                node_type=node_type,
                coordination=int(coordination),
                point_group=point_group_from_directions(directions),
                port_shape=core.port_shape_signature(directions),
                port_directions=port_directions,
                incident_edge_types=incident_edge_types,
                stabilizer_order=len(orbit.stabilizers),
                stabilizer_port_permutations=port_permutations(orbit),
            )
        )

    edge_sites = []
    for orbit in symmetry.edge_orbits:
        representative = orbit.representative
        edge_type = tuple(
            int(value) for value in topology.edge_types[representative]
        )
        edge_sites.append(
            core.EdgeSite(
                edge_type=edge_type,
                stabilizer_order=len(orbit.stabilizers),
                stabilizer_port_permutations=port_permutations(orbit),
            )
        )

    return core.TopologyRecord(
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
    return isinstance(topology, core.Topology) or (
        topology_type.__module__ == "pormake.topology"
        and topology_type.__name__ == "Topology"
    )


def topology_cache_is_stale(cgd_path: Path, pickle_path: Path) -> bool:
    """Allow small timestamp skew introduced while copying cache files."""

    return (
        cgd_path.stat().st_mtime_ns
        > pickle_path.stat().st_mtime_ns + TOPOLOGY_CACHE_MTIME_TOLERANCE_NS
    )


def load_eligible_topologies(
    config: core.GenerationConfig,
    pools: dict[tuple[str, int, str], list[core.BlockRecord]],
    *,
    topology_mode: str = "binodal",
    symmetry_mode: str = "strict",
) -> tuple[list[core.TopologyRecord], Counter[str]]:
    """Load topologies supported by the requested single-linker formulation.

    ``binodal`` retains the historical formulation of one metal node and one
    organic node, with an optional edge block.  ``uninodal`` uses one metal
    node and requires an organic linker on the topology's self-edge.
    ``both`` includes both topology families and all three constructions.
    """

    if topology_mode == "both":
        uninodal, skipped_uninodal = load_eligible_topologies(
            config, pools, topology_mode="uninodal", symmetry_mode=symmetry_mode
        )
        binodal, skipped_binodal = load_eligible_topologies(
            config, pools, topology_mode="binodal", symmetry_mode=symmetry_mode
        )
        skipped = Counter({f"uninodal:{k}": v for k, v in skipped_uninodal.items()})
        skipped.update({f"binodal:{k}": v for k, v in skipped_binodal.items()})
        return uninodal + binodal, skipped
    if topology_mode not in {"binodal", "uninodal"}:
        raise ValueError(f"unknown topology mode: {topology_mode}")
    if symmetry_mode not in {"strict", "relaxed"}:
        raise ValueError(f"unknown symmetry mode: {symmetry_mode}")

    records: list[core.TopologyRecord] = []
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
                    topology = core.Topology(cgd_path)
            else:
                topology = core.Topology(cgd_path)
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
        expected_node_types = 2 if topology_mode == "binodal" else 1
        if len(topology.unique_node_types) != expected_node_types:
            skipped["type_limit"] += 1
            continue
        try:
            # Read legacy PORMAKE topologies from CGD to apply the current rules.
            if not hasattr(topology, "validate_3d"):
                topology = core.Topology(cgd_path)
            topology.validate_3d()
        except ValueError:
            skipped["non_3d_topology"] += 1
            continue
        try:
            record = core.topology_record_from_object(topology, cgd_path)
        except Exception:
            skipped["point_group_classification"] += 1
            continue
        if topology_mode == "binodal":
            if symmetry_mode == "relaxed" and config.relaxed_all_node_shapes:
                # P1 fallback need not reproduce a special site's point group.
                # Coordination is the eligibility gate; make_space still applies
                # the normal port-fit RMSD and the Builder validates assembly.
                if not any(
                    metal.node_type != organic.node_type
                    and pools.get(("metal_node", metal.coordination, core.ALL_NODE_POINT_GROUP))
                    and pools.get(("organic_node", organic.coordination, core.ALL_NODE_POINT_GROUP))
                    for metal in record.node_sites
                    for organic in record.node_sites
                ):
                    skipped["no_relaxed_coordination_assignment"] += 1
                    continue
                records.append(record)
                continue
            if not all(
                core.node_records_for_site(pools, "metal_node", site)
                or core.node_records_for_site(pools, "organic_node", site)
                for site in record.node_sites
            ):
                skipped["missing_site_compatible_node_pool"] += 1
                continue
            if not core.compatible_metal_organic_pairs(record, pools):
                skipped["no_metal_organic_assignment"] += 1
                continue
        else:
            site = record.node_sites[0]
            if symmetry_mode == "strict":
                if not core.node_records_for_site(pools, "metal_node", site):
                    skipped["missing_site_compatible_metal_pool"] += 1
                    continue
            elif not pools.get(
                ("metal_node", site.coordination, core.ALL_NODE_POINT_GROUP), ()
            ):
                # Relaxed placement does not require a block to reproduce the
                # crystallographic site stabilizer, but coordination must still
                # be supported.  The geometric RMSD gate is applied when the
                # candidate space is prepared.
                skipped["missing_relaxed_coordination_metal_pool"] += 1
                continue
            self_edge = (site.node_type, site.node_type)
            if not record.edge_types or any(
                edge_type != self_edge for edge_type in record.edge_types
            ):
                skipped["missing_required_self_edge"] += 1
                continue
            if symmetry_mode == "strict":
                edge_pools, _ = core.strict_edge_compatible_pools(record, pools)
                if any(
                    not edge_pools.get(
                        ("edge", edge_type, 2, core.EDGE_POINT_GROUP),
                        (),
                    )
                    for edge_type in record.edge_types
                ):
                    skipped["missing_site_compatible_edge_pool"] += 1
                    continue
            elif not pools.get(("edge", 2, core.EDGE_POINT_GROUP), ()):
                skipped["missing_relaxed_edge_pool"] += 1
                continue
        records.append(record)
    return records, skipped

"""Read-only geometric candidate screening for an abstract topology."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from mofbb.compose.building_block import Edge, Node
from mofbb.compose.locator import Locator, find_best_orientation


NOTICE = (
    "Geometric candidates only: role, X-port count and normalized port geometry "
    "are screened. Site symmetry, partner chemistry, connectivity, lengths and "
    "atomic clashes still require assembly validation. Edges are optional."
)


def find_avail_bbs(topology, library, *, max_rmsd=0.1, n_angles=4):
    '''
    Find library blocks whose X-port geometry fits a topology.

    A block is included once if it fits every slot of at least one type.
    Folder membership sets the node or edge role; the atoms set node chemistry.
    Blocks with at most five ports use all permutations. Larger blocks use
    sampling, which can miss a fit. Assembly must still check site symmetry,
    partner chemistry, connectivity, bond lengths, and atomic overlaps.
    No manifest or cache is read.

    Args:
        topology (Topology):
            Net to compare with the library blocks.
        library (str or Path):
            Library directory; scans node/**/*.xyz and edge/**/*.xyz.
        max_rmsd (float):
            Largest allowed RMSD between unit port directions; dimensionless.
        n_angles (int):
            Samples per Euler-angle axis for blocks with more than five ports.

    Returns:
        A dictionary of absolute XYZ paths under m_nodes, o_nodes, and edges.
    '''
    library = Path(library).expanduser().resolve()
    groups = {"node": [], "edge": []}
    for node_type in topology.unique_node_types:
        groups["node"].append([
            int(i) for i in topology.node_idx if topology.node_types[i] == node_type
        ])
    for edge_type in topology.unique_edge_types:
        groups["edge"].append([
            int(i) for i in topology.edge_idx
            if np.array_equal(topology.edge_types[i], edge_type)
        ])
    targets = {i: topology.connection_directions(i)
               for role_groups in groups.values() for slots in role_groups for i in slots}
    locator = Locator()
    result = {"m_nodes": [], "o_nodes": [], "edges": []}
    keys = {"metal_node": "m_nodes", "organic_node": "o_nodes", "edge": "edges"}
    for role, block_class in (("node", Node), ("edge", Edge)):
        for path in sorted((library / role).rglob("*.xyz")):
            block = block_class.read(path)
            relevant = [slots for slots in groups[role]
                        if all(int(topology.cn[i]) == block.n_ports for i in slots)]
            if not relevant:
                continue
            source = block.connection_directions()
            for slots in relevant:
                worst = 0.0
                for slot in slots:
                    target = targets[slot]
                    best = float("inf")
                    for perm in locator.candidate_permutations(target, block, n_angles=n_angles):
                        _, rmsd = find_best_orientation(target, source[perm])
                        best = min(best, float(rmsd))
                        if best < 1e-6:
                            break
                    worst = max(worst, best)
                    if worst > max_rmsd:
                        break
                if worst <= max_rmsd:
                    result[keys[block.kind]].append(str(path.resolve()))
                    break
    return result


def format_report(result, *, limit=10):
    '''
    Turn candidate paths into a short, readable report.

    Args:
        result (dict[str, list[str]]):
            Candidate paths grouped by block kind.
        limit (int):
            Maximum number of paths to display for each kind.

    Returns:
        A multiline string with screening limits, candidate counts, and paths.
    '''
    lines = [NOTICE]
    for kind, paths in result.items():
        lines.append(f"{kind}: {len(paths)} candidates")
        lines.extend(f"  {path}" for path in paths[:limit])
        if len(paths) > limit:
            lines.append(f"  ... {len(paths) - limit} more (use --compatible-json for all)")
    return "\n".join(lines)

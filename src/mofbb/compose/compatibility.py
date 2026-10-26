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
    """Return {"m_nodes": [...], "o_nodes": [...], "edges": [...]} XYZ paths.

    Scan ``library/node/**/*.xyz`` and ``library/edge/**/*.xyz`` directly.
    Folder membership specifies node versus edge role; actual atoms determine
    node chemistry, X-port count and geometry. No manifest/cache is read.
    A block is included once if it fits every slot of at least one type.
    RMSD uses unit directions (dimensionless). More than five ports use sampled
    permutations, so this is geometric screening, not a feasibility proof.
    """
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
    targets = {i: topology.local_structure(i)
               for role_groups in groups.values() for slots in role_groups for i in slots}
    locator = Locator()
    result = {"m_nodes": [], "o_nodes": [], "edges": []}
    keys = {"metal_node": "m_nodes", "organic_node": "o_nodes", "edge": "edges"}
    for role, block_class in (("node", Node), ("edge", Edge)):
        for path in sorted((library / role).rglob("*.xyz")):
            block = block_class(path)
            relevant = [slots for slots in groups[role]
                        if all(int(topology.cn[i]) == block.n_ports for i in slots)]
            if not relevant:
                continue
            source = block.local_structure().positions
            for slots in relevant:
                worst = 0.0
                for slot in slots:
                    target = targets[slot]
                    best = float("inf")
                    for perm in locator.candidate_permutations(target, block, n_angles=n_angles):
                        _, rmsd = find_best_orientation(target.positions, source[perm])
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
    """Format a compact report without truncating the underlying results."""
    lines = [NOTICE]
    for kind, paths in result.items():
        lines.append(f"{kind}: {len(paths)} candidates")
        lines.extend(f"  {path}" for path in paths[:limit])
        if len(paths) > limit:
            lines.append(f"  ... {len(paths) - limit} more (use --compatible-json for all)")
    return "\n".join(lines)

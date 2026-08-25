"""Framework composition pipeline."""

import numpy as np

from mofbb.compose.edge import Edge
from mofbb.compose.locator import Locator
from mofbb.compose.pbc import PBC
from mofbb.compose.framework import FrameworkAssembler
from mofbb.compose.placement.blocks import BlockPlacement
from mofbb.compose.placement.closure import CellClosure
from mofbb.compose.placement.slots import SlotAssignment
from mofbb.compose.scaler import Scaler


class Builder:
    """Assemble building blocks on a topology."""

    def __init__(self, locator=None, scaler=None):
        """Store the locator and scaler, creating defaults if needed."""
        self.locator = locator if locator is not None else Locator()
        self.scaler = scaler if scaler is not None else Scaler()

    def make_bbs_by_type(self, topology, node_bbs, edge_bbs=None):
        """Expand type-keyed blocks into independent per-slot copies."""
        bbs = [None] * topology.n_slots

        for i in topology.node_indices:
            t = topology.node_types[i]
            bbs[i] = node_bbs[t].copy()

        edge_bbs = edge_bbs or {}

        for i in topology.edge_indices:
            t = tuple(topology.edge_types[i])
            edge_block = edge_bbs.get(t)
            if edge_block is not None:
                bbs[i] = edge_block.copy()

        return bbs

    def build_by_type(
        self,
        topology,
        node_bbs,
        edge_bbs=None,
        permutations=None,
        *,
        accuracy=6,
        wrap=True,
        tolerance=PBC.tolerance,
        scale=PBC.scale,
        overlap_scale=PBC.overlap_scale,
        symmetry_tolerance=0.1,
        strict_symmetry=False,
    ):
        """Build from node- and edge-type maps."""
        bbs = self.make_bbs_by_type(topology, node_bbs, edge_bbs)
        return self.build(
            topology,
            bbs,
            permutations,
            accuracy=accuracy,
            wrap=wrap,
            tolerance=tolerance,
            scale=scale,
            overlap_scale=overlap_scale,
            symmetry_tolerance=symmetry_tolerance,
            strict_symmetry=strict_symmetry,
        )

    def build(
        self,
        topology,
        bbs,
        permutations=None,
        *,
        accuracy=6,
        wrap=True,
        tolerance=PBC.tolerance,
        scale=PBC.scale,
        overlap_scale=PBC.overlap_scale,
        symmetry_tolerance=0.1,
        strict_symmetry=False,
    ):
        """Run assignment, closure, placement, scaling, and assembly."""
        overlap_scale = float(overlap_scale)
        symmetry_tolerance = float(symmetry_tolerance)
        if permutations is None:
            permutations = topology.atoms.info.get("port_mappings")
        assignment = SlotAssignment(
            topology,
            bbs,
            self.locator,
            PBC(),
            permutations,
            accuracy=accuracy,
            sym_tol=symmetry_tolerance,
            strict_symmetry=strict_symmetry,
        ).run()
        closure = CellClosure(
            assignment,
            Edge,
            tolerance=tolerance,
            bond_scale=scale,
        ).run()
        assignment.validate_connections(closure.blocks, closure.permutations, scale)
        placement = BlockPlacement(
            assignment,
            closure,
            self.scaler,
            Edge,
        ).run()
        symmetry_diagnostics = assignment.diag
        symmetry_diagnostics["nodes"].sort(
            key=lambda item: item["representative"]
        )
        symmetry_diagnostics["edges"].sort(
            key=lambda item: item["representative"]
        )
        info = {
            "topology": placement.topology,
            "located_bbs": placement.blocks,
            "permutations": placement.permutations,
            "relax_obj": placement.scaling_result.fun,
            "max_rmsd": np.max(placement.rmsds),
            "mean_rmsd": np.mean(placement.rmsds),
            "tolerance": tolerance,
            "scale": scale,
            "overlap_scale": overlap_scale,
            "fixed_cell_closed": closure.closed,
            "edge_frame_diagnostics": placement.edge_frame_diagnostics,
            "symmetry_placement_diagnostics": symmetry_diagnostics,
            "strict_symmetry": bool(strict_symmetry),
        }

        return FrameworkAssembler(assignment, placement, overlap_scale).build(
            info, wrap=wrap
        )

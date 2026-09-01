"""Framework composition pipeline."""

import numpy as np

from mofbb.compose.edge import Edge
from mofbb.compose.locator import Locator
from mofbb.compose.pbc import PBC
from mofbb.compose.framework import FrameworkAssembler
from mofbb.compose.placement.blocks import BlockPlacement
from mofbb.compose.node_lattice import NodeLatticeOptimizer
from mofbb.compose.placement.fixed_cell import FixedCellFrameworkFitter
from mofbb.compose.placement.slots import SlotAssignment


class Builder:
    """Assemble building blocks on a topology."""

    def __init__(self, locator=None, geometry_optimizer=None, *, scaler=None):
        """Store placement helpers, creating defaults when needed."""
        if geometry_optimizer is not None and scaler is not None:
            raise ValueError(
                "Pass geometry_optimizer or the legacy scaler argument, not both."
            )
        self.locator = locator if locator is not None else Locator()
        if geometry_optimizer is None:
            geometry_optimizer = scaler
        self.geometry_optimizer = (
            geometry_optimizer
            if geometry_optimizer is not None
            else NodeLatticeOptimizer()
        )

    @property
    def scaler(self):
        """Compatibility alias for :attr:`geometry_optimizer`."""
        return self.geometry_optimizer

    @scaler.setter
    def scaler(self, value):
        self.geometry_optimizer = value

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
        """Run assignment, fixed-cell fitting, geometry optimization, and assembly."""
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
        fixed_cell_fit = FixedCellFrameworkFitter(
            assignment,
            Edge,
            tolerance=tolerance,
            bond_scale=scale,
        ).run()
        assignment.validate_connections(
            fixed_cell_fit.blocks,
            fixed_cell_fit.permutations,
            scale,
        )
        placement = BlockPlacement(
            assignment,
            fixed_cell_fit,
            self.geometry_optimizer,
            Edge,
        ).run()
        symmetry_diagnostics = assignment.diag
        symmetry_diagnostics["nodes"].sort(
            key=lambda item: item["representative"]
        )
        symmetry_diagnostics["edges"].sort(
            key=lambda item: item["representative"]
        )
        # Preserve the old diagnostic key at the external result boundary.
        for category in ("nodes", "edges"):
            for diagnostic in symmetry_diagnostics[category]:
                if "symmetry_propagated" in diagnostic:
                    diagnostic["covariant"] = diagnostic["symmetry_propagated"]

        info = {
            "topology": placement.topology,
            "located_bbs": placement.blocks,
            "permutations": placement.permutations,
            "geometry_optimization_objective": (
                placement.geometry_optimization_result.fun
            ),
            "relax_obj": placement.geometry_optimization_result.fun,
            "max_rmsd": np.max(placement.rmsds),
            "mean_rmsd": np.mean(placement.rmsds),
            "tolerance": tolerance,
            "scale": scale,
            "overlap_scale": overlap_scale,
            "fixed_cell_fitted": fixed_cell_fit.fitted,
            "fixed_cell_closed": fixed_cell_fit.fitted,
            "edge_frame_diagnostics": placement.edge_frame_diagnostics,
            "symmetry_placement_diagnostics": symmetry_diagnostics,
            "strict_symmetry": bool(strict_symmetry),
        }

        return FrameworkAssembler(assignment, placement, overlap_scale).build(
            info, wrap=wrap
        )

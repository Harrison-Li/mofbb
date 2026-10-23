"""Framework composition pipeline."""

import numpy as np

from mofbb.compose.edge import Edge
from mofbb.compose.locator import Locator
from mofbb.compose.pbc import PBC
from mofbb.compose.framework import FrameworkAssembler
from mofbb.compose.placement.blocks import BlockPlacement
from mofbb.compose.scaler import FixedCellFrameworkFitter, Scaler
from mofbb.compose.placement.slots import SlotAssignment


class Builder:
    """Assemble building blocks on a topology."""

    def __init__(self, locator=None, geometry_optimizer=None):
        """Store placement helpers, creating defaults when needed."""
        self.locator = locator if locator is not None else Locator()
        self.geometry_optimizer = (
            geometry_optimizer
            if geometry_optimizer is not None
            else Scaler()
        )

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
        require_3d=True,
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
            require_3d=require_3d,
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
        require_3d=True,
    ):
        """Assign ports, jointly fit rigid geometry, and assemble the framework."""
        if require_3d:
            topology.validate_3d()
        if len(bbs) != topology.n_slots:
            raise ValueError("Provide one building-block entry per topology slot.")
        for slot, block in enumerate(bbs):
            if block is None:
                if slot in topology.node_indices:
                    raise ValueError(f"Topology node {slot} requires a building block.")
                continue
            if block.n_connection_points != topology.cn[slot]:
                raise ValueError(f"Building block at slot {slot} has the wrong number of X ports.")
            if require_3d:
                block.validate_connectivity(scale)
            else:
                block.validate_chemistry()
        if permutations is None:
            permutations = topology.atoms.info.get("port_mappings")
        for slot, permutation in (permutations or {}).items():
            if not np.array_equal(np.sort(permutation), np.arange(topology.cn[slot])):
                raise ValueError(f"Port assignment at slot {slot} must use each X port once.")
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
        ).run(fit_geometry=(
            not hasattr(self.geometry_optimizer, "optimize_placement")
            or "symmetry_operations" in topology.atoms.info
        ))
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
        symmetry_diagnostics = placement.symmetry_diagnostics
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
            "geometry_optimization_objective": (
                placement.geometry_optimization_result.fun
            ),
            "max_rmsd": np.max(placement.rmsds),
            "mean_rmsd": np.mean(placement.rmsds),
            "tolerance": tolerance,
            "scale": scale,
            "overlap_scale": overlap_scale,
            "fixed_cell_fitted": fixed_cell_fit.fitted,
            "edge_frame_diagnostics": placement.edge_frame_diagnostics,
            "symmetry_placement_diagnostics": symmetry_diagnostics,
            "strict_symmetry": bool(strict_symmetry),
            "require_3d": bool(require_3d),
        }

        return FrameworkAssembler(assignment, placement, overlap_scale).build(
            info, wrap=wrap
        )

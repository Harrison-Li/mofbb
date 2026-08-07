"""
Framework composition pipeline, adapted from
https://github.com/Sangwon91/PORMAKE/blob/main/src/pormake/builder.py
"""

import numpy as np

from mofbb.compose.edge import Edge
from mofbb.compose.locator import Locator
from mofbb.compose.pbc import PBC
from mofbb.compose.placement.assembly import FrameworkAssembler
from mofbb.compose.placement.blocks import BlockPlacement
from mofbb.compose.placement.closure import CellClosure
from mofbb.compose.placement.slots import SlotAssignment
from mofbb.compose.scaler import Scaler


class Builder:
    """Orchestrate assembly of a framework from blocks and a topology.

    Holds the two pluggable strategy objects the assembly depends on --
    a :class:`~mofbb.compose.locator.Locator` for orienting building blocks
    onto slots and a :class:`~mofbb.compose.scaler.Scaler` for resizing the
    topology cell -- and exposes :meth:`build` (and the per-type convenience
    wrapper) to run the full pipeline. Keeping locator and scaler injectable
    lets callers swap matching or scaling behavior without changing the
    orchestration logic.

    Parameters
    ----------
    locator : Locator, optional
        Strategy for finding the best orientation and permutation of a
        building block on a slot. A default :class:`Locator` is created when
        omitted.
    scaler : Scaler, optional
        Strategy for rescaling the topology unit cell to match real
        building-block sizes. A default :class:`Scaler` is created when
        omitted.

    Attributes
    ----------
    locator : Locator
        The orientation and permutation matcher used during the build.
    scaler : Scaler
        The cell-rescaling strategy used during the build.
    """

    def __init__(self, locator=None, scaler=None):
        """Store the locator and scaler, creating defaults if needed."""
        self.locator = locator if locator is not None else Locator()
        self.scaler = scaler if scaler is not None else Scaler()

    def make_bbs_by_type(self, topology, node_bbs, edge_bbs=None):
        """Build the per-slot building-block list from type maps.

        Expands compact, type-keyed dictionaries into the dense slot-indexed
        list that :meth:`build` expects, copying each building block so every
        slot owns an independent instance. Edge types absent from ``edge_bbs``
        are left empty (``None``), producing direct node-to-node bonds.

        Parameters
        ----------
        topology : Topology
            Topology whose slots are being populated.
        node_bbs : dict
            Maps each node type to the building block placed on that type.
        edge_bbs : dict, optional
            Maps each edge type to its linker block. ``None`` values leave the
            corresponding edge slots empty.

        Returns
        -------
        list
            Independent building blocks indexed by topology slot.
        """
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
    ):
        """Assemble a framework from type-keyed building-block maps.

        This is the usual public entry point for callers that assign one block
        to every node or edge type. The maps are expanded with
        :meth:`make_bbs_by_type`, then the explicit build options are forwarded
        to :meth:`build`.

        Parameters
        ----------
        topology : Topology
            Net on which the framework is assembled.
        node_bbs : dict
            Node-type to building-block mapping.
        edge_bbs : dict, optional
            Edge-type to linker mapping; omitted types become direct bonds.
        permutations : dict, optional
            Preselected connection-point permutations keyed by slot index.
        accuracy : int, optional
            Search resolution for slots with more than five ports.
        wrap : bool, optional
            Whether to wrap the completed framework into its unit cell.
        tolerance, scale, overlap_scale : float, optional
            Geometric thresholds for X-pair closure, fused bonds, and atomic
            overlap respectively.
        symmetry_tolerance : float, optional
            Cartesian tolerance used for site-stabilizer diagnostics.

        Returns
        -------
        Framework
            The assembled periodic framework.
        """
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
    ):
        """Assemble a framework from a block assigned to each topology slot.

        The pipeline chooses chemically compatible port permutations, places
        symmetry-related blocks, attempts fixed-cell closure where applicable,
        scales the topology otherwise, and finally fuses dummy ``X`` connection
        points into real bonds. Site-stabilizer incompatibilities are recorded
        in the framework metadata and fall back to independent placement.

        Parameters
        ----------
        topology : Topology
            Net to assemble on.
        bbs : sequence
            One building block per topology slot. ``None`` leaves an edge slot
            empty and creates a direct node-to-node bond.
        permutations : dict, optional
            Preselected connection-point permutations keyed by slot index.
        accuracy : int, optional
            Locator search resolution for slots with more than five ports.
        wrap : bool, optional
            Whether to wrap final atoms into the unit cell.
        tolerance : float, optional
            Maximum X-pair separation in angstrom before fusion.
        scale : float, optional
            Maximum fused-bond length as a multiple of summed covalent radii.
        overlap_scale : float, optional
            Minimum atom separation as a multiple of the smaller covalent
            radius. Set to zero to disable the overlap gate.
        symmetry_tolerance : float, optional
            Cartesian tolerance for site-stabilizer diagnostics.

        Returns
        -------
        Framework
            The assembled framework with placement and scaling diagnostics in
            ``Framework.info``.
        """
        overlap_scale = float(overlap_scale)
        symmetry_tolerance = float(symmetry_tolerance)
        locator = self.locator
        if permutations is None:
            permutations = topology.atoms.info.get("port_mappings")
        assignment = SlotAssignment(
            topology,
            bbs,
            locator,
            PBC(),
            permutations,
            accuracy=accuracy,
            sym_tol=symmetry_tolerance,
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
        }

        return FrameworkAssembler(assignment, placement, overlap_scale).build(
            info, wrap=wrap
        )

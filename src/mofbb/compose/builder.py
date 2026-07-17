"""
Adatpted from https://github.com/Sangwon91/PORMAKE/blob/main/src/pormake/builder.py
"""

from collections import defaultdict

import numpy as np

from .edge import Edge
from .framework import Framework
from .locator import Locator
from .log import logger
from .pbc import PBC
from .scaler import Scaler
from .symmetry_placement import SymmetryPlacement


# bb: building block.
class Builder:
    """Orchestrate assembly of a framework from blocks and a topology.

    Holds the two pluggable strategy objects the assembly depends on --
    a :class:`~pormake.locator.Locator` for orienting building blocks
    onto slots and a :class:`~pormake.scaler.Scaler` for resizing the
    topology cell -- and exposes :meth:`build` (and the per-type
    convenience wrappers) to run the full pipeline. Keeping locator and
    scaler injectable lets callers swap matching/scaling behavior without
    touching the orchestration logic.

    Parameters
    ----------
    locator : Locator, optional
        Strategy for finding the best orientation and permutation of a
        building block on a slot. A default :class:`Locator` is created
        when omitted.
    scaler : Scaler, optional
        Strategy for rescaling the topology unit cell to match real
        building-block sizes. A default :class:`Scaler` is created when
        omitted.

    Attributes
    ----------
    locator : Locator
        The orientation/permutation matcher used during the build.
    scaler : Scaler
        The cell-rescaling strategy used during the build.
    """

    def __init__(self, locator=None, scaler=None):
        """Store the locator and scaler, creating defaults if needed."""
        self.locator = locator if locator is not None else Locator()
        self.scaler = scaler if scaler is not None else Scaler()

    def make_bbs_by_type(self, topology, node_bbs, edge_bbs=None):
        """Build the per-slot building-block list from type maps.

        Expands compact, type-keyed dictionaries into the dense
        slot-indexed list that :meth:`build` expects, copying each
        building block so every slot owns an independent instance.
        Specifying blocks by type (rather than per slot) is the
        ergonomic way to describe a structure, since all nodes of the
        same type and all edges of the same type usually share a block.
        Edge types absent from ``edge_bbs`` are logged and left empty
        (``None``), producing direct node-to-node bonds.

        Parameters
        ----------
        topology : Topology
            Topology whose slots are being populated; supplies the node
            and edge indices and their types.
        node_bbs : dict
            Maps a node type (int) to the building block to place on
            every node of that type.
        edge_bbs : dict, optional
            Maps an edge type (a tuple of two ints) to the linker
            building block for those edges. A value of ``None`` for a
            type leaves those edges empty. Defaults to no edge blocks.

        Returns
        -------
        list
            Length ``topology.n_slots`` list whose entry ``i`` is the
            (copied) building block for slot ``i``, or ``None`` for empty
            edge slots.
        """
        bbs = [None] * topology.n_slots

        for i in topology.node_indices:
            t = topology.node_types[i]
            bbs[i] = node_bbs[t].copy()

        if edge_bbs is None:
            # Empty dictionary.
            edge_bbs = {}

        # Log undefined edge building blocks.
        for t in topology.unique_edge_types:
            t = tuple(t)
            if t not in edge_bbs:
                logger.info(
                    "No edge building block for type %s in edge_bbs.", t
                )

        for i in topology.edge_indices:
            t = tuple(topology.edge_types[i])
            if t in edge_bbs:
                if edge_bbs[t] is None:
                    continue
                bbs[i] = edge_bbs[t].copy()

        return bbs

    def build_by_type(self, topology, node_bbs, edge_bbs=None, **kwargs):
        """Assemble a framework from type-keyed building-block maps.

        Convenience wrapper that turns the type-keyed ``node_bbs`` and
        ``edge_bbs`` dictionaries into a per-slot list via
        :meth:`make_bbs_by_type` and forwards it to :meth:`build`. This
        is the usual public entry point for users who think in terms of
        node and edge *types* rather than individual slots.

        Parameters
        ----------
        topology : Topology
            Net the framework is assembled on.
        node_bbs : dict
            Maps node type (int) to its building block.
        edge_bbs : dict, optional
            Maps edge type (tuple of two ints) to its linker building
            block. Defaults to no edge blocks.
        **kwargs
            Forwarded to :meth:`build` (e.g. ``accuracy``, ``wrap``,
            ``permutations``).

        Returns
        -------
        Framework
            The assembled framework.
        """
        bbs = self.make_bbs_by_type(topology, node_bbs, edge_bbs)
        return self.build(topology, bbs, **kwargs)

    def build(self, topology, bbs, permutations=None, **kwargs):
        """Assemble a framework from a per-slot list of building blocks.

        Runs the full assembly pipeline. It locates one representative
        building block per crystallographic node and edge orbit, derives
        member port permutations from the declared space-group operations,
        and creates compatible remaining placements by transforming the
        representative. Site-stabilizer incompatibilities are reported in
        the framework metadata and fall back to independent placement rather
        than rejecting the structure. The topology cell is scaled to the real
        block sizes before the final placements are generated. Finally all
        intra- and inter-block bonds are collected, the "X" connection-point
        atoms are removed while the bonds they implied are fused, and the
        result is returned as a Framework.

        Parameters
        ----------
        topology : Topology
            Net to assemble on. Node building blocks must correspond to
            the topology's node-type ordering.
        bbs : list
            Length ``topology.n_slots`` sequence where ``bbs[i]`` is the
            building block for node ``i`` (if ``i`` is a node index) or
            edge ``i`` (if an edge index). ``None`` leaves an edge empty.
        permutations : dict, optional
            Maps a slot index to a pre-determined connection-point
            permutation, bypassing the locator's search for that slot.
            Defaults to no pre-set permutations.
        **kwargs
            ``accuracy`` : int
                Euler-angle grid resolution for high-accuracy
                relocation (``max_n_slices``); default 6.
            ``wrap`` : bool
                Whether to wrap the framework into the unit cell;
                default ``True``.
            ``tolerance`` : float
                Maximum minimum-image distance, in Å, allowed between
                paired ``X`` connection atoms before fusion; default
                ``0.3``.
            ``scale`` : float
                Maximum fused-bond distance as a multiplier of the two
                atoms' summed ASE covalent radii; default ``1.2``.
            ``symmetry_tolerance`` : float
                Cartesian tolerance, in Å, used only to diagnose whether a
                representative building block obeys its site stabilizer;
                default ``0.1``. A failed check does not reject the build.
        Returns
        -------
        Framework
            The assembled periodic framework, with bonds, bond types,
            and build metadata in ``info``.

        Raises
        ------
        Exception
            If a node placement yields an RMSD below 99% of the cached
            minimum (with that minimum above ``1e-3``), indicating the
            minimum RMSD was computed incorrectly.
        ValueError
            If an X site lacks exactly one real anchor or a topology
            connection fails either geometric gate. X atoms are not removed
            in these cases.
        """

        # Parse keyword arguments.
        if "accuracy" in kwargs:
            max_n_slices = kwargs["accuracy"]
        else:
            max_n_slices = 6

        tolerance = kwargs.get("tolerance", PBC.tolerance)
        scale = kwargs.get("scale", PBC.scale)
        symmetry_tolerance = float(kwargs.get("symmetry_tolerance", 0.1))
        if symmetry_tolerance <= 0.0:
            raise ValueError("symmetry_tolerance must be positive.")

        # Empty dictionary.
        if permutations is None:
            permutations = {}

        logger.debug("Builder.build starts.")

        # locator for bb locations.
        locator = self.locator
        pbc = PBC()
        symmetry = SymmetryPlacement(topology)

        # Locate nodes and edges.
        located_bbs = [None for _ in range(topology.n_slots)]

        provided_permutations = {
            int(index): np.asarray(permutation, dtype=int)
            for index, permutation in permutations.items()
        }
        _permutations = [None for _ in range(topology.n_slots)]
        for i, perm in provided_permutations.items():
            _permutations[i] = perm.copy()
        permutations = _permutations

        symmetry_diagnostics = {
            "mapping_tolerance": symmetry.mapping_tolerance,
            "stabilizer_tolerance": symmetry_tolerance,
            "nodes": [],
            "edges": [],
        }
        covariant_node_orbits = set()
        covariant_edge_orbits = set()

        slot_min_rmsd = defaultdict(lambda: -1.0)

        def rmsd_ratio(value, minimum):
            if minimum < 1e-12:
                return 1.0 if value < 1e-12 else np.inf
            return value / minimum

        def blocks_share_template(left, right):
            if left is None or right is None:
                return left is right
            if left.name != right.name:
                return False
            if left.atoms.get_chemical_symbols() != right.atoms.get_chemical_symbols():
                return False
            if not np.array_equal(
                left.connection_point_indices,
                right.connection_point_indices,
            ):
                return False
            left_positions = left.atoms.positions - left.centroid
            right_positions = right.atoms.positions - right.centroid
            if not np.allclose(left_positions, right_positions, atol=1e-12):
                return False
            if not np.allclose(
                left.atoms.get_initial_charges(),
                right.atoms.get_initial_charges(),
                atol=1e-12,
            ):
                return False
            if not np.array_equal(left.bonds, right.bonds):
                return False
            return left.bond_types == right.bond_types

        def orbit_has_one_template(orbit):
            representative_block = bbs[orbit.representative]
            return all(
                blocks_share_template(representative_block, bbs[int(member)])
                for member in orbit.members
            )

        def pulled_representative_permutation(orbit):
            supplied = [
                int(member)
                for member in orbit.members
                if int(member) in provided_permutations
            ]
            if not supplied:
                return None

            source = min(supplied)
            representative_permutation = symmetry.to_representative(
                orbit,
                source,
                provided_permutations[source],
            )
            for member in supplied:
                expected = symmetry.to_member(
                    orbit,
                    member,
                    representative_permutation,
                )
                if not np.array_equal(expected, provided_permutations[member]):
                    logger.warning(
                        "Provided permutation for slot %d is inconsistent "
                        "with representative slot %d; using the "
                        "operation-derived permutation.",
                        member,
                        orbit.representative,
                    )
            return representative_permutation

        def locate_node(i, forced_permutation=None):
            node_bb = bbs[i]
            target = topology.local_structure(i)

            if forced_permutation is not None:
                perm = np.asarray(forced_permutation, dtype=int)
                logger.info(
                    "Use given permutation for node slot index %d"
                    ", permutation: %s",
                    i,
                    perm,
                )

                located_node, rmsd = locator.locate_with_permutation(
                    target, node_bb, perm
                )

                logger.info(
                    "Pre-location of node slot %d, RMSD: %.2E",
                    i,
                    rmsd,
                )
                return located_node, perm, rmsd

            t = topology.get_node_type(i)
            key = (t, node_bb.name)
            if slot_min_rmsd[key] < 0.0:
                rmsd = locator.calculate_rmsd(
                    target, node_bb, max_n_slices=max_n_slices
                )

                chiral_node_bb = node_bb.make_chiral_building_block()
                c_rmsd = locator.calculate_rmsd(target, chiral_node_bb)
                slot_min_rmsd[key] = min(rmsd, c_rmsd)
                logger.info(
                    "== Min RMSD of (node type: %s, node bb: %s): %.2E",
                    *key,
                    slot_min_rmsd[key],
                )
            located_node, perm, rmsd = locator.locate(target, node_bb)
            logger.info(
                "Pre-location at node slot %d"
                ", (node type: %s, node bb: %s)"
                ", RMSD: %.2E",
                i,
                *key,
                rmsd,
            )
            # If RMSD is different from min RMSD relocate with high accuracy.
            # 1% error.
            ratio = rmsd_ratio(rmsd, slot_min_rmsd[key])
            if ratio > 1.01:
                located_node, perm, rmsd = locator.locate(
                    target, node_bb, max_n_slices=max_n_slices
                )
                logger.info(
                    "RMSD > MIN_RMSD*1.01, relocate Node %d"
                    " with %d trial orientations, RMSD: %.2E",
                    i,
                    max_n_slices**3,
                    rmsd,
                )

            ratio = rmsd_ratio(rmsd, slot_min_rmsd[key])
            if ratio > 1.01:
                # Make chiral building block.
                node_bb = node_bb.make_chiral_building_block()
                located_node, perm, rmsd = locator.locate(
                    target, node_bb, max_n_slices=max_n_slices
                )
                logger.info(
                    "RMSD > MIN_RMSD*1.01, relocate Node %d"
                    " with %d trial orientations and chiral building block"
                    ", RMSD: %.2E",
                    i,
                    max_n_slices**3,
                    rmsd,
                )

            if (ratio < 0.99) and (slot_min_rmsd[key] > 1e-3):
                message = (
                    "MIN_RMSD is not correct. "
                    "Topology: %s; "
                    "Slot: %s; "
                    "Building block: %s; "
                    "rmsd: %.3E." % (topology, key, node_bb, rmsd)
                )
                logger.error(message)
                raise Exception(message)

            return located_node, np.asarray(perm, dtype=int), rmsd

        # Locate only one node per crystallographic orbit, then transform it
        # to the remaining members. Distinct explicitly assigned templates
        # cannot share one representative and retain the legacy behavior.
        original_cell = np.asarray(topology.atoms.cell, dtype=float)
        for orbit in symmetry.node_orbits:
            representative = orbit.representative
            if not orbit_has_one_template(orbit):
                logger.warning(
                    "Node orbit tag %d contains different building blocks; "
                    "placing its members independently.",
                    orbit.tag,
                )
                symmetry_diagnostics["nodes"].append(
                    {
                        "tag": orbit.tag,
                        "representative": representative,
                        "orbit_size": len(orbit.members),
                        "covariant": False,
                        "reason": "different_building_blocks",
                    }
                )
                for raw_member in orbit.members:
                    member = int(raw_member)
                    located, permutation, _ = locate_node(
                        member,
                        permutations[member],
                    )
                    located_bbs[member] = located
                    permutations[member] = permutation
                continue

            representative_permutation = pulled_representative_permutation(orbit)
            representative_block, representative_permutation, _ = locate_node(
                representative,
                representative_permutation,
            )
            representative_block.set_centroid(
                topology.atoms.positions[representative]
            )
            initial_diagnostic = symmetry.diagnose(
                orbit,
                representative_block,
                representative_permutation,
                original_cell,
                symmetry_tolerance,
            )
            if not initial_diagnostic["compatible"]:
                initial_diagnostic.update(
                    {
                        "kind": "node",
                        "covariant": False,
                        "reason": "stabilizer_incompatible",
                    }
                )
                symmetry_diagnostics["nodes"].append(initial_diagnostic)
                logger.warning(
                    "Node orbit tag %d is not compatible with the site "
                    "stabilizer at %.3f Å; placing its members "
                    "independently without rejection.",
                    orbit.tag,
                    symmetry_tolerance,
                )
                located_bbs[representative] = representative_block
                permutations[representative] = representative_permutation
                for raw_member in orbit.members:
                    member = int(raw_member)
                    if member == representative:
                        continue
                    located, permutation, _ = locate_node(
                        member,
                        permutations[member],
                    )
                    located.set_centroid(topology.atoms.positions[member])
                    located_bbs[member] = located
                    permutations[member] = permutation
                continue

            for raw_member in orbit.members:
                member = int(raw_member)
                permutation = symmetry.to_member(
                    orbit,
                    member,
                    representative_permutation,
                )
                operation_index = symmetry.operation(orbit, member)
                if member == representative:
                    located = representative_block
                else:
                    located = symmetry.place(
                        representative_block,
                        operation_index,
                        topology.atoms.positions[member],
                        original_cell,
                    )
                located_bbs[member] = located
                permutations[member] = permutation
            covariant_node_orbits.add(representative)

        def default_edge_permutation(edge_index):
            neighbor_i, neighbor_j = topology.neighbor_list[edge_index]
            if topology.node_types[neighbor_i.index] <= topology.node_types[
                neighbor_j.index
            ]:
                return np.array([0, 1])
            return np.array([1, 0])

        # Edge orientation is applied after scaling, but its endpoint
        # permutation is already derived orbit-wise for the scaler and port
        # matching stages.
        for orbit in symmetry.edge_orbits:
            representative = orbit.representative
            if all(bbs[int(member)] is None for member in orbit.members):
                continue

            if not orbit_has_one_template(orbit):
                logger.warning(
                    "Edge orbit tag %d has mixed occupancy or different "
                    "building blocks; placing occupied members independently.",
                    orbit.tag,
                )
                symmetry_diagnostics["edges"].append(
                    {
                        "tag": orbit.tag,
                        "representative": representative,
                        "orbit_size": len(orbit.members),
                        "covariant": False,
                        "reason": "different_building_blocks_or_occupancy",
                    }
                )
                for raw_member in orbit.members:
                    member = int(raw_member)
                    edge_block = bbs[member]
                    if edge_block is None:
                        continue
                    if permutations[member] is None:
                        permutations[member] = default_edge_permutation(member)
                    located_bbs[member] = edge_block
                continue

            representative_permutation = pulled_representative_permutation(orbit)
            if representative_permutation is None:
                representative_permutation = default_edge_permutation(
                    representative
                )
            representative_block = bbs[representative]
            for raw_member in orbit.members:
                member = int(raw_member)
                permutations[member] = symmetry.to_member(
                    orbit,
                    member,
                    representative_permutation,
                )
                located_bbs[member] = representative_block.copy()
            covariant_edge_orbits.add(representative)

        for building_block in located_bbs:
            if building_block is not None:
                building_block.connection_point_anchor_indices

        predicted_failures = []
        for edge_index in topology.edge_indices:
            node_x_i, node_x_j = pbc._find_matched_x_sites(
                topology, located_bbs, permutations, edge_index
            )
            neighbor_i, neighbor_j = topology.neighbor_list[edge_index]
            node_i = located_bbs[neighbor_i.index]
            node_j = located_bbs[neighbor_j.index]
            edge = located_bbs[edge_index]
            if edge is None:
                port_pairs = ((node_i, node_x_i, node_j, node_x_j),)
            else:
                edge_x_i, edge_x_j = edge.connection_point_indices[
                    permutations[edge_index]
                ]
                port_pairs = (
                    (node_i, node_x_i, edge, int(edge_x_i)),
                    (edge, int(edge_x_j), node_j, node_x_j),
                )

            for bb_i, x_i, bb_j, x_j in port_pairs:
                symbol_i, _, cap_i = pbc._connection_port(bb_i, x_i)
                symbol_j, _, cap_j = pbc._connection_port(bb_j, x_j)
                predicted = cap_i + cap_j
                limit = pbc.max_bond_len(symbol_i, symbol_j, scale)
                if predicted > limit:
                    predicted_failures.append(
                        (edge_index, symbol_i, symbol_j, predicted, limit)
                    )

        if predicted_failures:
            edge_index, symbol_i, symbol_j, distance, limit = predicted_failures[0]
            raise ValueError(
                "Predicted fused-bond gate rejected assembly: "
                f"{len(predicted_failures)} topology connections exceed "
                f"{scale:.3f} times their summed covalent radii. "
                f"Edge {edge_index} {symbol_i}-{symbol_j}: {distance:.3f} Å "
                f"> {limit:.3f} Å. Resample the building blocks or ports."
            )

        # Scale topology.
        # Change topology to scaled topology.
        original_topology = topology
        topology, scaling_result = self.scaler.scale(
            topology=topology,
            bbs=located_bbs,
            perms=permutations,
            return_result=True,
        )
        edge_targets = Edge(
            original_topology,
            topology,
            permutations,
        ).build()
        edge_frame_diagnostics = []

        rmsd_values = []
        scaled_cell = np.asarray(topology.atoms.cell, dtype=float)
        for orbit in symmetry.node_orbits:
            representative = orbit.representative
            if representative not in covariant_node_orbits:
                for raw_member in orbit.members:
                    member = int(raw_member)
                    permutation = permutations[member]
                    node_block = located_bbs[member]
                    located_node, rmsd = locator.locate_with_permutation(
                        topology.local_structure(member),
                        node_block,
                        permutation,
                    )
                    located_node.set_centroid(topology.atoms.positions[member])
                    located_bbs[member] = located_node
                    rmsd_values.append(rmsd)
                continue

            representative_block = located_bbs[representative]
            representative_permutation = permutations[representative]
            located_representative, rmsd = locator.locate_with_permutation(
                topology.local_structure(representative),
                representative_block,
                representative_permutation,
            )
            located_representative.set_centroid(
                topology.atoms.positions[representative]
            )
            diagnostic = symmetry.diagnose(
                orbit,
                located_representative,
                representative_permutation,
                scaled_cell,
                symmetry_tolerance,
            )
            diagnostic.update({"kind": "node", "covariant": True})
            symmetry_diagnostics["nodes"].append(diagnostic)
            if not diagnostic["compatible"]:
                diagnostic["covariant"] = False
                diagnostic["reason"] = "final_stabilizer_incompatible"
                logger.warning(
                    "Node orbit tag %d is not compatible with the site "
                    "stabilizer at %.3f Å (atom %.3f Å, port %.3f Å); "
                    "placing its members independently without rejection.",
                    orbit.tag,
                    symmetry_tolerance,
                    diagnostic["max_atom_residual"],
                    diagnostic["max_port_residual"],
                )
                for raw_member in orbit.members:
                    member = int(raw_member)
                    located_node, member_rmsd = (
                        locator.locate_with_permutation(
                            topology.local_structure(member),
                            located_bbs[member],
                            permutations[member],
                        )
                    )
                    located_node.set_centroid(topology.atoms.positions[member])
                    located_bbs[member] = located_node
                    rmsd_values.append(member_rmsd)
                continue

            for raw_member in orbit.members:
                member = int(raw_member)
                operation_index = symmetry.operation(orbit, member)
                if member == representative:
                    located_node = located_representative
                else:
                    located_node = symmetry.place(
                        located_representative,
                        operation_index,
                        topology.atoms.positions[member],
                        scaled_cell,
                    )
                located_bbs[member] = located_node
                rmsd_values.append(rmsd)
                logger.info(
                    "Symmetry-covariant node placement at slot %d from "
                    "representative %d with operation %d, RMSD: %.2E",
                    member,
                    representative,
                    operation_index,
                    rmsd,
                )
        rmsd_values = np.array(rmsd_values)

        def calc_image(ni, nj, invc):
            """Compute the periodic image offset between two neighbors.

            Determine the integer unit-cell translation separating the
            two node ends of an edge, so an edge that wraps across a cell
            boundary is placed correctly. Reads the enclosing-scope
            ``topology``.

            Parameters
            ----------
            ni, nj : Neighbor
                The two neighbor records (node ends) of the edge.
            invc : numpy.ndarray
                Inverse of the topology cell matrix, used to convert the
                Cartesian offset into fractional (image) units.

            Returns
            -------
            numpy.ndarray
                The ``(3,)`` periodic image vector (integer-valued) of
                ``nj`` relative to ``ni``.
            """
            # Calculate image.
            # d = d_{ij}
            i = ni.index
            j = nj.index

            d = nj.distance_vector - ni.distance_vector

            ri = topology.atoms.positions[i]
            rj = topology.atoms.positions[j]

            image = (d - (rj - ri)) @ invc

            return image

        # Locate edges.
        logger.info("Start placing edges.")
        c = scaled_cell
        invc = np.linalg.inv(c)

        def edge_geometry(edge_index):
            neighbor_i, neighbor_j = topology.neighbor_list[edge_index]
            block_i = located_bbs[neighbor_i.index]
            block_j = located_bbs[neighbor_j.index]
            a1, a2 = pbc._find_matched_x_sites(
                original_topology,
                located_bbs,
                permutations,
                edge_index,
            )
            r1 = block_i.atoms.positions[a1]
            r2 = block_j.atoms.positions[a2]
            image = calc_image(neighbor_i, neighbor_j, invc)
            d = r2 - r1 + image @ c
            target_points = np.array([r1, r1 + d])
            return target_points, r1 + 0.5 * d

        def place_edge_independently(edge_index):
            edge_block = located_bbs[edge_index]
            target_points, centroid = edge_geometry(edge_index)
            edge_target = edge_targets[edge_index]
            located_edge, rmsd, source_frame = locator.locate_edge_with_frame(
                target_points,
                edge_block,
                permutations[edge_index],
                edge_target.transverse,
            )
            located_edge.set_centroid(centroid)
            edge_frame_diagnostics.append(
                {
                    "edge_index": int(edge_index),
                    "symmetry_edge_type": edge_target.symmetry_edge_type,
                    "representative_edge": edge_target.representative_edge,
                    "operation_index": edge_target.operation_index,
                    "endpoint_reversed": edge_target.endpoint_reversed,
                    "representative_gauge": edge_target.representative_gauge,
                    "pca_gap": source_frame.pca_gap,
                    "torsion_degenerate": source_frame.torsion_degenerate,
                    "sign_degenerate": source_frame.sign_degenerate,
                }
            )
            located_bbs[edge_index] = located_edge
            logger.debug("Edge %d, RMSD: %.2E", edge_index, rmsd)

        for orbit in symmetry.edge_orbits:
            representative = orbit.representative
            occupied_members = [
                int(member)
                for member in orbit.members
                if located_bbs[int(member)] is not None
            ]
            if not occupied_members:
                continue
            if representative not in covariant_edge_orbits:
                for member in occupied_members:
                    place_edge_independently(member)
                continue

            representative_target_points, representative_centroid = (
                edge_geometry(representative)
            )
            representative_target = edge_targets[representative]
            representative_edge, rmsd, source_frame = (
                locator.locate_edge_with_frame(
                    representative_target_points,
                    located_bbs[representative],
                    permutations[representative],
                    representative_target.transverse,
                )
            )
            representative_edge.set_centroid(representative_centroid)
            representative_edge, diagnostic = (
                symmetry.fit_torsion(
                    orbit,
                    representative_edge,
                    permutations[representative],
                    scaled_cell,
                    symmetry_tolerance,
                )
            )
            diagnostic.update({"kind": "edge", "covariant": True})
            symmetry_diagnostics["edges"].append(diagnostic)
            if not diagnostic["compatible"]:
                diagnostic["covariant"] = False
                diagnostic["reason"] = "stabilizer_incompatible"
                logger.warning(
                    "Edge orbit tag %d is not compatible with the site "
                    "stabilizer at %.3f Å (atom %.3f Å, port %.3f Å); "
                    "placing its members independently without rejection.",
                    orbit.tag,
                    symmetry_tolerance,
                    diagnostic["max_atom_residual"],
                    diagnostic["max_port_residual"],
                )
                located_bbs[representative] = representative_edge
                edge_frame_diagnostics.append(
                    {
                        "edge_index": representative,
                        "symmetry_edge_type": orbit.tag,
                        "representative_edge": representative,
                        "operation_index": symmetry.operation(
                            orbit,
                            representative,
                        ),
                        "endpoint_reversed": False,
                        "representative_gauge": (
                            representative_target.representative_gauge
                        ),
                        "pca_gap": source_frame.pca_gap,
                        "torsion_degenerate": source_frame.torsion_degenerate,
                        "sign_degenerate": source_frame.sign_degenerate,
                    }
                )
                for member in occupied_members:
                    if member == representative:
                        continue
                    place_edge_independently(member)
                continue

            fallback_members = []
            for member in occupied_members:
                operation_index = symmetry.operation(orbit, member)
                member_target_points, member_centroid = edge_geometry(member)
                if member == representative:
                    located_edge = representative_edge
                else:
                    located_edge = symmetry.place(
                        representative_edge,
                        operation_index,
                        member_centroid,
                        scaled_cell,
                    )
                port_residual = float(
                    np.max(
                        np.linalg.norm(
                            located_edge.connection_points[
                                permutations[member]
                            ]
                            - member_target_points,
                            axis=1,
                        )
                    )
                )
                if member != representative and port_residual > tolerance:
                    fallback_members.append(member)
                    logger.warning(
                        "Propagated edge slot %d misses its node ports by "
                        "%.3f Å; placing this member independently without "
                        "rejecting the framework.",
                        member,
                        port_residual,
                    )
                    place_edge_independently(member)
                    continue
                located_bbs[member] = located_edge
                endpoint_map = symmetry.offset_map(
                    representative,
                    member,
                    operation_index,
                )
                edge_frame_diagnostics.append(
                    {
                        "edge_index": member,
                        "symmetry_edge_type": orbit.tag,
                        "representative_edge": representative,
                        "operation_index": operation_index,
                        "endpoint_reversed": bool(endpoint_map[0] == 1),
                        "representative_gauge": (
                            representative_target.representative_gauge
                        ),
                        "pca_gap": source_frame.pca_gap,
                        "torsion_degenerate": source_frame.torsion_degenerate,
                        "sign_degenerate": source_frame.sign_degenerate,
                    }
                )
                logger.debug(
                    "Symmetry-covariant edge %d from representative %d with "
                    "operation %d, RMSD: %.2E",
                    member,
                    representative,
                    operation_index,
                    rmsd,
                )
            if fallback_members:
                diagnostic["covariant"] = False
                diagnostic["reason"] = "propagated_port_mismatch"
                diagnostic["fallback_members"] = fallback_members

        logger.info("Start finding bonds in generated framework.")
        logger.info("Start finding bonds in building blocks.")
        # Build bonds of generated framework.
        index_offsets = [None for _ in range(topology.n_slots)]
        index_offsets[0] = 0
        for i, bb in enumerate(located_bbs[:-1]):
            if bb is None:
                index_offsets[i + 1] = index_offsets[i] + 0
            else:
                index_offsets[i + 1] = index_offsets[i] + bb.n_atoms

        bb_bonds = []
        bb_bond_types = []
        for offset, bb in zip(index_offsets, located_bbs):
            if bb is None:
                continue
            bb_bonds.append(bb.bonds + offset)
            bb_bond_types += bb.bond_types
        bb_bonds = np.concatenate(bb_bonds, axis=0)

        logger.info("Start finding bonds between building blocks.")

        # Find bond between building blocks.
        bonds = []
        bond_types = []
        for j in topology.edge_indices:
            a1, a2 = pbc._find_matched_x_sites(
                original_topology, located_bbs, permutations, j
            )

            # i and j: edge index in topology
            n1, n2 = topology.neighbor_list[j]
            i1 = n1.index
            i2 = n2.index
            a1 += index_offsets[i1]
            a2 += index_offsets[i2]

            # Edge exists.
            if located_bbs[j] is not None:
                perm = permutations[j]
                e1, e2 = (
                    located_bbs[j].connection_point_indices[perm]
                    + index_offsets[j]
                )
                bonds.append((e1, a1))
                bonds.append((e2, a2))
                bond_types += ["S", "S"]
                logger.debug(
                    "Bonds on topology edge %s are connected %s, %s.",
                    j,
                    bonds[-2],
                    bonds[-1],
                )
            else:
                bonds.append((a1, a2))
                bond_types += ["S"]
                logger.debug(
                    "Bonds on topology edge %s are connected %s.",
                    j,
                    bonds[-1],
                )

        x_pairs = bonds
        bonds = np.array(bonds)

        # All bonds in generated framework.
        all_bonds = np.concatenate([bb_bonds, bonds], axis=0)
        all_bond_types = bb_bond_types + bond_types

        logger.info("Start making Framework instance.")
        # Make full atoms from located building blocks.
        bb_atoms_list = [v.atoms for v in located_bbs if v is not None]

        logger.debug("Merge list of atoms.")
        framework_atoms = sum(bb_atoms_list[1:], bb_atoms_list[0])
        logger.debug("Set cell and boundary.")
        framework_atoms.set_pbc(True)
        framework_atoms.set_cell(topology.atoms.cell)

        failures = pbc.calc_x_mid(framework_atoms, x_pairs, tolerance)
        if failures:
            x_i, x_j, distance = failures[0]
            raise ValueError(
                "Geometric quality gate rejected assembly: "
                f"{len(failures)} of {len(x_pairs)} topology X-X connections "
                f"exceed the {tolerance:.3f} Å tolerance. "
                f"Atoms {x_i}-{x_j}: {distance:.3f} Å. Resample the "
                "building blocks or orientations before fusion."
            )

        # Remove connection points (X) from the framework.
        count = 0
        new_indices = {}
        for a in framework_atoms:
            if a.symbol == "X":
                continue
            new_indices[a.index] = count
            count += 1

        def is_X(i):
            """Return whether atom ``i`` is an "X" connection point.

            Reads the enclosing-scope ``framework_atoms``; used to
            classify bonds while fusing and removing the dummy
            connection-point atoms.
            """
            return framework_atoms[i].symbol == "X"

        XX_bonds = []
        new_bonds = []
        new_bond_types = []
        X_neighbor_list = {}
        for (i, j), t in zip(all_bonds, all_bond_types):
            if is_X(i) and is_X(j):
                XX_bonds.append((i, j))
            elif is_X(i):
                X_neighbor_list[i] = j
            elif is_X(j):
                X_neighbor_list[j] = i
            else:
                new_bonds.append((i, j))
                new_bond_types.append(t)

        failures = pbc.calc_fail_connection(
            framework_atoms,
            x_pairs,
            X_neighbor_list,
            scale,
        )
        if failures:
            atom_i, atom_j, distance, limit = failures[0]
            symbols = framework_atoms.get_chemical_symbols()
            raise ValueError(
                "Fused-bond quality gate rejected assembly: "
                f"{len(failures)} topology connections exceed "
                f"{scale:.3f} times their summed covalent radii. "
                f"Atoms {atom_i}-{atom_j} ({symbols[atom_i]}-"
                f"{symbols[atom_j]}): {distance:.3f} Å > {limit:.3f} Å. "
                "Resample the building blocks or orientations before fusion."
            )

        for i, j in XX_bonds:
            new_bonds.append((X_neighbor_list[i], X_neighbor_list[j]))
            new_bond_types.append("S")

        all_bonds = [(new_indices[i], new_indices[j]) for i, j in new_bonds]
        all_bonds = np.array(all_bonds)
        all_bond_types = new_bond_types

        del framework_atoms[[a.symbol == "X" for a in framework_atoms]]

        symmetry_diagnostics["nodes"].sort(
            key=lambda item: item["representative"]
        )
        symmetry_diagnostics["edges"].sort(
            key=lambda item: item["representative"]
        )
        info = {
            "topology": topology,
            "located_bbs": located_bbs,
            "permutations": permutations,
            "relax_obj": scaling_result.fun,
            "max_rmsd": np.max(rmsd_values),
            "mean_rmsd": np.mean(rmsd_values),
            "tolerance": tolerance,
            "scale": scale,
            "edge_frame_diagnostics": edge_frame_diagnostics,
            "symmetry_placement_diagnostics": symmetry_diagnostics,
        }

        if "wrap" not in kwargs:
            wrap = True
        else:
            wrap = kwargs["wrap"]

        framework = Framework(
            framework_atoms, all_bonds, all_bond_types, info=info, wrap=wrap
        )
        logger.info("Construction of framework done.")

        return framework

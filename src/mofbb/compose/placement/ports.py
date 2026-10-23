"""Discrete port-assignment constraints for joint framework optimization."""

from __future__ import annotations

import numpy as np
import scipy.optimize
from scipy.sparse import coo_matrix


def connection_residuals(vectors, spans, direct):
    """Return port errors after eliminating independent rigid linker poses."""
    return np.concatenate([
        (np.linalg.norm(vectors[~direct], axis=1) - spans[~direct]) / np.sqrt(2.0),
        vectors[direct].ravel(),
    ])


def _port_mask(pbc, bond_scale, left, right):
    return np.asarray([
        [
            not pbc.is_forbidden_fused_pair(symbol_i, symbol_j)
            and cap_i + cap_j <= pbc.max_bond_len(symbol_i, symbol_j, bond_scale)
            for symbol_j, cap_j in right.port_descriptors
        ]
        for symbol_i, cap_i in left.port_descriptors
    ])


def _variables(shapes, count=0):
    variables = []
    for shape in shapes:
        size = np.prod(shape)
        variables.append(np.arange(count, count + size).reshape(shape))
        count += size
    return variables, count


class Constraints:
    """Build sparse bounded constraints for binary port assignments."""

    def __init__(self):
        self.rows, self.columns, self.values = [], [], []
        self.lower, self.upper = [], []

    def add(self, indices, coefficients, minimum, maximum):
        row = len(self.lower)
        self.rows.extend([row] * len(indices))
        self.columns.extend(indices)
        self.values.extend(coefficients)
        self.lower.append(minimum)
        self.upper.append(maximum)

    def scipy(self, count):
        matrix = coo_matrix(
            (self.values, (self.rows, self.columns)),
            shape=(len(self.lower), count),
        ).tocsr()
        return scipy.optimize.LinearConstraint(matrix, self.lower, self.upper)

    def assignment(self, variables, bounds, fixed_perm=None):
        for row in (*variables, *variables.T):
            self.add(row, np.ones(len(row)), 1.0, 1.0)
        if fixed_perm is not None:
            bounds[variables] = 0.0
            bounds[variables[np.arange(len(fixed_perm)), fixed_perm]] = 1.0

    def link_permutations(self, left, right, offsets, source_map=None):
        if source_map is None:
            source_map = np.arange(len(offsets))
        for target, member_target in enumerate(offsets):
            for source, member_source in enumerate(source_map):
                self.add(
                    (left[target, source], right[member_target, member_source]),
                    (1.0, -1.0), 0.0, 0.0,
                )


def solve_binary(costs, bounds, rules):
    return scipy.optimize.milp(
        costs,
        integrality=np.ones(len(costs)),
        bounds=scipy.optimize.Bounds(0, bounds),
        constraints=rules.scipy(len(costs)),
    )


def _edge_endpoint_offsets(topology, edge: int) -> list[tuple[int, int]]:
    ends = []
    for edge_nb in topology.neighbor_list[edge]:
        node = edge_nb.index
        offset = next(
            offset
            for offset, node_nb in enumerate(
                topology.neighbor_list[node]
            )
            if node_nb.index == edge
            and np.linalg.norm(
                node_nb.distance_vector + edge_nb.distance_vector
            )
            < 0.01
        )
        ends.append((node, offset))
    return ends


class PortAssignmentSolver:
    """Optimize compatible ports using the joint node and linker-span loss."""

    @classmethod
    def initialize(cls, placement):
        """Place initial blocks and enforce connection chemistry by default."""
        assignment = placement.assignment
        assignment.run()
        for slot, perm in assignment.fixed_perms.items():
            if not np.array_equal(placement.perms[slot], perm):
                raise ValueError(f"Provided port assignment at slot {slot} conflicts with declared symmetry.")
        solver = cls(placement)
        if solver.valid(placement.perms):
            return
        blocks = [placement.blocks[slot] for slot in solver.slot_indices]
        candidate = solver.solve(
            np.asarray([block.centroid for block in blocks]),
            [block.port_pos - block.centroid for block in blocks],
        )
        if candidate is None:
            raise ValueError(
                "Predicted fused-bond gate rejected port assignment: "
                "no chemically and symmetry-compatible assignment exists."
            )
        solver._apply(candidate)

    def __init__(self, placement):
        self.placement = placement
        self.slot_indices = list(placement.topology.node_idx) + [
            i for i in placement.topology.edge_idx if placement.blocks[i] is not None
        ]
        self.slot_offsets = {slot: i for i, slot in enumerate(self.slot_indices)}
        self.edges = []
        edge_blocks = [placement.blocks[edge] for edge in placement.topology.edge_idx]
        self.spans = np.array([
            0.0 if block is None else np.linalg.norm(np.diff(block.port_pos, axis=0)[0])
            for block in edge_blocks
        ])
        self.direct = np.array([block is None for block in edge_blocks])
        self.connections = self._connections()
        self.allowed = {}
        # Direct-edge chemistry and geometry share the same joint variables.
        # Linkers also need edge-node variables to constrain endpoint chemistry.
        self.variable_connections = self.edges + [
            link for link in self.connections if link[0] in placement.topology.edge_idx
        ]

    def _apply(self, perms):
        job = self.placement
        cell = np.asarray(job.topology.atoms.cell)
        for orbit in job.symmetry.node_orbits:
            shared = orbit.representative in job.symmetry_compatible_node_orbits
            for node in ([orbit.representative] if shared else orbit.members):
                if np.array_equal(job.perms[node], perms[node]):
                    continue
                block, _ = job.locator.locate_with_permutation(
                    job.topology.local_structure(node), job.blocks[node], perms[node]
                )
                block.set_centroid(job.topology.atoms.positions[node])
                if shared:
                    # A free rotation in the port fit can break the motif's site symmetry.
                    diagnostic = job.symmetry.diagnose(
                        orbit, block, perms[node], cell, job.sym_tol
                    )
                    if not diagnostic["compatible"]:
                        block = job.blocks[node]
                    for member in orbit.members:
                        job.blocks[member] = job.symmetry.place(
                            block, job.symmetry.operation(orbit, member),
                            job.topology.atoms.positions[member], cell,
                        )
                else:
                    job.blocks[node] = block
        job.perms[:] = perms

    def pair_ok(self, left, x_i, right, x_j):
        """Return whether two source ports can form a fused bond."""
        return self._allowed_pairs(left, right)[x_i, x_j]

    def _allowed_pairs(self, left, right):
        if (left, right) not in self.allowed:
            job = self.placement
            self.allowed[left, right] = _port_mask(
                job.pbc, job.bond_scale,
                job.blocks[left], job.blocks[right],
            )
        return self.allowed[left, right]

    def valid(self, perms):
        """Return whether every selected port pair is chemically allowed."""
        return all(
            self.pair_ok(left, perms[left][x_i], right, perms[right][x_j])
            for left, x_i, right, x_j, _ in self.connections
        )

    def _connections(self):
        """Build node-node or edge-node connections with periodic images."""
        job = self.placement
        links = []
        edge_shifts = job.topology.edge_shifts
        cell = np.asarray(job.topology.atoms.cell)
        self.shifts = np.asarray([
            edge_shifts[edge][1] - edge_shifts[edge][0]
            for edge in job.topology.edge_idx
        ], dtype=int).reshape(-1, 3)
        for edge, shift in zip(job.topology.edge_idx, self.shifts, strict=True):
            (node_i, x_i), (node_j, x_j) = _edge_endpoint_offsets(job.source_topology, edge)
            images = edge_shifts[edge]
            image = shift @ cell
            self.edges.append((node_i, x_i, node_j, x_j, image))
            if job.blocks[edge] is None:
                links.append((node_i, x_i, node_j, x_j, image))
                continue

            for end, image, node, target in zip(
                range(2),
                images @ cell,
                (node_i, node_j),
                (x_i, x_j),
                strict=True,
            ):
                links.append((edge, end, node, target, image))
        return links

    def solve(self, centers, points):
        """Find the minimum-loss feasible permutation for every occupied slot."""
        assignments, count = _variables(
            (len(points[i]),) * 2 for i in range(len(self.slot_indices))
        )
        slot_vars = dict(zip(self.slot_indices, assignments, strict=True))
        pair_vars, count = _variables(
            (
                (slot_vars[left].shape[1], slot_vars[right].shape[1])
                for left, _, right, _, _ in self.variable_connections
            ),
            count,
        )
        costs = np.zeros(count)
        bounds = np.ones(count)
        rules = Constraints()

        for slot, variables in slot_vars.items():
            rules.assignment(variables, bounds, self.placement.assignment.fixed_perms.get(slot))
        self._orbit_rules(slot_vars, rules)
        self._pair_rules(slot_vars, pair_vars, bounds, rules)
        for edge, pair, span, direct in zip(
            self.edges, pair_vars[:len(self.edges)], self.spans, self.direct, strict=True
        ):
            left, x_i, right, x_j, image = edge
            left_id, right_id = self.slot_offsets[left], self.slot_offsets[right]
            vectors = (centers[right_id] + image + points[right_id][None, :]
                       - centers[left_id] - points[left_id][:, None])
            costs[pair] = (np.sum(vectors**2, axis=2) if direct else
                           0.5 * (np.linalg.norm(vectors, axis=2) - span)**2)

        result = solve_binary(costs, bounds, rules)
        if not result.success:
            return None
        perms = [
            None if value is None else np.asarray(value).copy()
            for value in self.placement.perms
        ]
        for slot, var in slot_vars.items():
            perms[slot] = np.argmax(result.x[var], axis=1)
        return perms

    def residuals(self, centers, points, perms):
        """Return actual port-gap vectors in their declared periodic images."""
        return np.asarray([
            centers[self.slot_offsets[left]] + points[self.slot_offsets[left]][perms[left][x_i]]
            - centers[self.slot_offsets[right]] - image
            - points[self.slot_offsets[right]][perms[right][x_j]]
            for left, x_i, right, x_j, image in self.connections
        ])

    def loss(self, centers, points, perms):
        """Return the sum of squared port residuals with linker poses eliminated."""
        vectors = np.asarray([
            centers[self.slot_offsets[right]] + image
            + points[self.slot_offsets[right]][perms[right][x_j]]
            - centers[self.slot_offsets[left]]
            - points[self.slot_offsets[left]][perms[left][x_i]]
            for left, x_i, right, x_j, image in self.edges
        ])
        residuals = connection_residuals(vectors, self.spans, self.direct)
        return residuals @ residuals

    def _orbit_rules(self, slot_vars, rules):
        job = self.placement
        groups = (
            (job.symmetry.node_orbits, job.symmetry_compatible_node_orbits),
            (job.symmetry.edge_orbits, job.symmetry_compatible_edge_orbits),
        )
        for orbits, compatible_orbits in groups:
            for orbit in orbits:
                rep = orbit.representative
                if rep not in compatible_orbits or rep not in slot_vars:
                    continue
                current = job.perms[rep]
                for operation in orbit.stabilizers:
                    offsets = job.symmetry.offset_map(rep, rep, operation)
                    source_map = np.empty_like(current)
                    source_map[current] = current[offsets]
                    if not np.array_equal(offsets, np.arange(len(offsets))):
                        rules.link_permutations(slot_vars[rep], slot_vars[rep], offsets, source_map)
                for member in orbit.members:
                    if member == rep or member not in slot_vars:
                        continue
                    offsets = job.symmetry.offset_map(
                        rep,
                        member,
                        job.symmetry.operation(orbit, member),
                    )
                    rules.link_permutations(slot_vars[rep], slot_vars[member], offsets)

    def _pair_rules(self, slot_vars, pair_vars, bounds, rules):
        for index, (link, pair) in enumerate(
            zip(self.variable_connections, pair_vars, strict=True)
        ):
            left, x_i, right, x_j, _ = link
            if index >= len(self.edges) or self.direct[index]:
                bounds[pair[~self._allowed_pairs(left, right)]] = 0.0
            self._link_pair(pair, slot_vars[left][x_i], rules, axis=0)
            self._link_pair(pair, slot_vars[right][x_j], rules, axis=1)

    @staticmethod
    def _link_pair(pair, slot_row, rules, axis):
        width = pair.shape[1 - axis]
        for src in range(pair.shape[axis]):
            row = pair[src] if axis == 0 else pair[:, src]
            rules.add(
                (*row, slot_row[src]),
                (*np.ones(width), -1.0),
                0.0,
                0.0,
            )

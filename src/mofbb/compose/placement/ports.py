"""Port constraints for fixed-cell framework closure."""

from __future__ import annotations

import numpy as np

from mofbb.compose.placement.milp import Constraints, solve_binary


class PortSolver:
    """Build and solve the fixed-cell port-assignment MILP."""

    def __init__(self, cell):
        self.cell = cell

    def pair_ok(self, left, x_i, right, x_j):
        """Return whether two source ports can form a fused bond."""
        job = self.cell
        symbol_i, cap_i = job.templates[left].port_descriptors[x_i]
        symbol_j, cap_j = job.templates[right].port_descriptors[x_j]
        return (
            not job.pbc.is_carbon_metal_pair(symbol_i, symbol_j)
            and cap_i + cap_j
            <= job.pbc.max_bond_len(symbol_i, symbol_j, job.bond_scale)
        )

    def valid(self, perms):
        """Return whether every selected port pair is chemically allowed."""
        return all(
            self.pair_ok(left, perms[left][x_i], right, perms[right][x_j])
            for left, x_i, right, x_j, _ in self.cell.connections
        )

    def connections(self):
        """Build node-node or edge-node connections with periodic images."""
        job = self.cell
        links = []
        for raw in job.topology.edge_indices:
            edge = int(raw)
            (node_i, x_i), (node_j, x_j) = job.edge_endpoint_offsets(edge)
            near, far = job.topology.neighbor_list[edge]
            if edge in job.direct_edge_masks:
                image = (
                    far.distance_vector
                    - near.distance_vector
                    - job.topology.atoms.positions[node_j]
                    + job.topology.atoms.positions[node_i]
                )
                links.append((node_i, x_i, node_j, x_j, image))
                continue

            for end, side, node, target in zip(
                range(2),
                (near, far),
                (node_i, node_j),
                (x_i, x_j),
                strict=True,
            ):
                image = (
                    job.topology.atoms.positions[edge]
                    + side.distance_vector
                    - job.topology.atoms.positions[node]
                )
                links.append((edge, end, node, target, image))
        return links

    def solve(self, centers, points):
        """Find the lowest-gap feasible permutation for every occupied slot."""
        slot_vars, count = self._slot_vars(points)
        pair_vars, count = self._pair_vars(points, count)
        costs = np.zeros(count)
        bounds = np.ones(count)
        rules = Constraints()

        self._slot_rules(slot_vars, bounds, rules)
        self._orbit_rules(slot_vars, rules)
        self._pair_rules(
            centers,
            points,
            slot_vars,
            pair_vars,
            costs,
            bounds,
            rules,
        )

        result = solve_binary(costs, bounds, rules)
        if not result.success:
            return None
        perms = [
            None if value is None else np.asarray(value).copy()
            for value in self.cell.permutations
        ]
        for slot, var in slot_vars.items():
            perms[slot] = np.argmax(result.x[var], axis=1)
        return perms

    def _slot_vars(self, points):
        job = self.cell
        out = {}
        count = 0
        for slot in job.slot_indices:
            degree = len(points[job.slot_offsets[slot]])
            out[slot] = np.arange(count, count + degree**2).reshape(
                degree,
                degree,
            )
            count += degree**2
        return out, count

    def _pair_vars(self, points, count):
        job = self.cell
        out = []
        for left, _, right, _, _ in job.connections:
            shape = (
                len(points[job.slot_offsets[left]]),
                len(points[job.slot_offsets[right]]),
            )
            size = int(np.prod(shape))
            out.append(np.arange(count, count + size).reshape(shape))
            count += size
        return out, count

    def _slot_rules(self, slot_vars, bounds, rules):
        for slot, var in slot_vars.items():
            for assignment in (*var, *var.T):
                rules.add(
                    assignment,
                    np.ones(len(assignment)),
                    1.0,
                    1.0,
                )
            if slot not in self.cell.provided_permutations:
                continue
            fixed = self.cell.provided_permutations[slot]
            bounds[var] = 0.0
            bounds[var[np.arange(len(fixed)), fixed]] = 1.0

    def _orbit_rules(self, slot_vars, rules):
        job = self.cell
        groups = (
            (job.symmetry.node_orbits, job.covariant_node_orbits),
            (job.symmetry.edge_orbits, job.covariant_edge_orbits),
        )
        for orbits, covariant in groups:
            for orbit in orbits:
                rep = orbit.representative
                if rep not in covariant or rep not in slot_vars:
                    continue
                for raw in orbit.members:
                    member = int(raw)
                    if member == rep or member not in slot_vars:
                        continue
                    offsets = job.symmetry.offset_map(
                        rep,
                        member,
                        job.symmetry.operation(orbit, member),
                    )
                    self._link_members(slot_vars, rules, rep, member, offsets)

    @staticmethod
    def _link_members(slot_vars, rules, rep, member, offsets):
        for target, member_target in enumerate(offsets):
            for source in range(len(offsets)):
                rules.add(
                    (
                        slot_vars[rep][target, source],
                        slot_vars[member][member_target, source],
                    ),
                    (1.0, -1.0),
                    0.0,
                    0.0,
                )

    def _pair_rules(
        self,
        centers,
        points,
        slot_vars,
        pair_vars,
        costs,
        bounds,
        rules,
    ):
        for link, pair in zip(
            self.cell.connections,
            pair_vars,
            strict=True,
        ):
            left, x_i, right, x_j, image = link
            left_id = self.cell.slot_offsets[left]
            right_id = self.cell.slot_offsets[right]
            left_pos = centers[left_id] + points[left_id]
            right_pos = centers[right_id] + image + points[right_id]
            costs[pair] = np.sum(
                (left_pos[:, None] - right_pos[None, :]) ** 2,
                axis=2,
            )
            allowed = np.asarray(
                [
                    [
                        self.pair_ok(left, src_i, right, src_j)
                        for src_j in range(pair.shape[1])
                    ]
                    for src_i in range(pair.shape[0])
                ]
            )
            bounds[pair[~allowed]] = 0.0
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

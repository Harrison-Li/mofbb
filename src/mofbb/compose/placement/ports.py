"""Connection-site constraints for fixed-cell framework fitting."""

from __future__ import annotations

from functools import cache

import numpy as np
import scipy.optimize
from scipy.sparse import coo_matrix


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


def solve_binary(costs, bounds, rules):
    return scipy.optimize.milp(
        costs,
        integrality=np.ones(len(costs)),
        bounds=scipy.optimize.Bounds(0, bounds),
        constraints=rules.scipy(len(costs)),
    )


class PortAssignmentSolver:
    """Build and solve the fixed-cell port-assignment MILP."""

    def __init__(self, fit_stage):
        self.fit_stage = fit_stage

    def pair_ok(self, left, x_i, right, x_j):
        """Return whether two source ports can form a fused bond."""
        job = self.fit_stage
        symbol_i, cap_i = job.templates[left].port_descriptors[x_i]
        symbol_j, cap_j = job.templates[right].port_descriptors[x_j]
        return (
            not job.pbc.is_forbidden_fused_pair(symbol_i, symbol_j)
            and cap_i + cap_j
            <= job.pbc.max_bond_len(symbol_i, symbol_j, job.bond_scale)
        )

    def valid(self, perms):
        """Return whether every selected port pair is chemically allowed."""
        return all(
            self.pair_ok(left, perms[left][x_i], right, perms[right][x_j])
            for left, x_i, right, x_j, _ in self.fit_stage.connections
        )

    def connections(self):
        """Build node-node or edge-node connections with periodic images."""
        job = self.fit_stage
        links = []
        edge_shifts = job.topology.edge_shifts
        cell = np.asarray(job.topology.atoms.cell)
        for raw in job.topology.edge_indices:
            edge = int(raw)
            (node_i, x_i), (node_j, x_j) = job.edge_endpoint_offsets(edge)
            images = edge_shifts[edge]
            if edge in job.direct_edge_masks:
                image = (images[1] - images[0]) @ cell
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
            for value in self.fit_stage.permutations
        ]
        for slot, var in slot_vars.items():
            perms[slot] = np.argmax(result.x[var], axis=1)
        return perms

    def _slot_vars(self, points):
        job = self.fit_stage
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
        job = self.fit_stage
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
            if slot not in self.fit_stage.provided_permutations:
                continue
            fixed = self.fit_stage.provided_permutations[slot]
            bounds[var] = 0.0
            bounds[var[np.arange(len(fixed)), fixed]] = 1.0

    def _orbit_rules(self, slot_vars, rules):
        job = self.fit_stage
        groups = (
            (job.symmetry.node_orbits, job.symmetry_compatible_node_orbits),
            (job.symmetry.edge_orbits, job.symmetry_compatible_edge_orbits),
        )
        for orbits, compatible_orbits in groups:
            for orbit in orbits:
                rep = orbit.representative
                if rep not in compatible_orbits or rep not in slot_vars:
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
            self.fit_stage.connections,
            pair_vars,
            strict=True,
        ):
            left, x_i, right, x_j, image = link
            left_id = self.fit_stage.slot_offsets[left]
            right_id = self.fit_stage.slot_offsets[right]
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

PortSolver = PortAssignmentSolver


class DirectEdgeRepair:
    """Repair node permutations that make forbidden direct connections."""

    def __init__(self, state, scale: float) -> None:
        self.topology = state.topology
        self.templates = state.blocks
        self.blocks = state.placed
        self.perms = state.perms
        self.fixed = state.fixed
        self.symmetry = state.symmetry
        self.locator = state.locator
        self.pbc = state.pbc
        self.symmetry_compatible_orbits = (
            state.symmetry_compatible_node_orbits
        )
        self.port_mask = state.node_port_compatibility
        self.scale = scale
        self.masks = self._make_masks()

    def repair_if_needed(self) -> None:
        if not self._has_invalid_edge():
            return

        variables, count = self._variables()
        costs, bounds = self._costs(variables, count)
        rules = Constraints()
        self._add_bijections(variables, rules)
        self._add_orbits(variables, rules)
        self._block_pairs(variables, rules)
        result = solve_binary(costs, bounds, rules)
        if not result.success:
            raise ValueError(
                "Predicted fused-bond gate rejected joint direct-edge "
                "permutation search: no compatible assignment exists."
            )

        solution = {
            node: np.argmax(result.x[node_vars], axis=1)
            for node, node_vars in variables.items()
        }
        self._apply(solution)

    @cache
    def ends(self, edge: int) -> list[tuple[int, int]]:
        ends = []
        for edge_nb in self.topology.neighbor_list[edge]:
            node = int(edge_nb.index)
            offset = next(
                offset
                for offset, node_nb in enumerate(
                    self.topology.neighbor_list[node]
                )
                if node_nb.index == edge
                and np.linalg.norm(
                    node_nb.distance_vector + edge_nb.distance_vector
                )
                < 0.01
            )
            ends.append((node, offset))
        return ends

    def _make_masks(self):
        masks = {}
        for raw_edge in self.topology.edge_indices:
            edge = int(raw_edge)
            if self.templates[edge] is not None:
                continue
            end_i, end_j = self.ends(edge)
            node_i, _ = end_i
            node_j, _ = end_j
            allowed = np.asarray(
                [
                    [
                        self._ports_ok(
                            symbol_i,
                            cap_i,
                            symbol_j,
                            cap_j,
                        )
                        for symbol_j, cap_j in self.templates[
                            node_j
                        ].port_descriptors
                    ]
                    for symbol_i, cap_i in self.templates[node_i].port_descriptors
                ]
            )
            masks[edge] = end_i, end_j, allowed
        return masks

    def _ports_ok(
        self,
        symbol_i: str,
        cap_i: float,
        symbol_j: str,
        cap_j: float,
    ) -> bool:
        return (
            not self.pbc.is_forbidden_fused_pair(symbol_i, symbol_j)
            and cap_i + cap_j
            <= self.pbc.max_bond_len(symbol_i, symbol_j, self.scale)
        )

    def _has_invalid_edge(self) -> bool:
        return any(
            not allowed[
                self.perms[node_i][target_i],
                self.perms[node_j][target_j],
            ]
            for (node_i, target_i), (node_j, target_j), allowed in self.masks.values()
        )

    def _variables(self) -> tuple[dict[int, np.ndarray], int]:
        variables = {}
        count = 0
        for raw_node in self.topology.node_indices:
            node = int(raw_node)
            degree = len(self.topology.neighbor_list[node])
            variables[node] = np.arange(
                count,
                count + degree**2,
            ).reshape(degree, degree)
            count += degree**2
        return variables, count

    def _costs(
        self,
        variables: dict[int, np.ndarray],
        count: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        costs = np.zeros(count)
        bounds = np.ones(count)
        for node, node_vars in variables.items():
            target = self.topology.local_structure(node).atoms.positions
            source = self.blocks[node].local_structure().atoms.positions
            residuals = target[:, None] - source[None, :]
            costs[node_vars] = np.sum(residuals**2, axis=2)

            allowed, _ = self.port_mask(
                node,
                self.perms,
            )
            bounds[node_vars[~allowed]] = 0.0
            if node in self.fixed:
                forced = self.perms[node]
                bounds[node_vars] = 0.0
                bounds[
                    node_vars[np.arange(len(forced)), forced]
                ] = 1.0
        return costs, bounds

    @staticmethod
    def _add_bijections(
        variables: dict[int, np.ndarray],
        rules: Constraints,
    ) -> None:
        for node_vars in variables.values():
            for assignment in (*node_vars, *node_vars.T):
                rules.add(
                    assignment,
                    np.ones(len(assignment)),
                    1.0,
                    1.0,
                )

    def _add_orbits(
        self,
        variables: dict[int, np.ndarray],
        rules: Constraints,
    ) -> None:
        for orbit in self.symmetry.node_orbits:
            rep = orbit.representative
            if rep not in self.symmetry_compatible_orbits:
                continue
            for raw_member in orbit.members:
                member = int(raw_member)
                if member == rep:
                    continue
                op = self.symmetry.operation(orbit, member)
                offsets = self.symmetry.offset_map(
                    rep,
                    member,
                    op,
                )
                self._link_orbit(
                    variables,
                    rules,
                    rep,
                    member,
                    offsets,
                )

    @staticmethod
    def _link_orbit(
        variables: dict[int, np.ndarray],
        rules: Constraints,
        rep: int,
        member: int,
        offsets: np.ndarray,
    ) -> None:
        for source_index in range(len(offsets)):
            for target_index, member_target in enumerate(offsets):
                rules.add(
                    (
                        variables[rep][target_index, source_index],
                        variables[member][member_target, source_index],
                    ),
                    (1.0, -1.0),
                    0.0,
                    0.0,
                )

    def _block_pairs(
        self,
        variables: dict[int, np.ndarray],
        rules: Constraints,
    ) -> None:
        for endpoint_i, endpoint_j, allowed in self.masks.values():
            node_i, target_i = endpoint_i
            node_j, target_j = endpoint_j
            for source_i, source_j in zip(*np.nonzero(~allowed), strict=True):
                rules.add(
                    (
                        variables[node_i][target_i, source_i],
                        variables[node_j][target_j, source_j],
                    ),
                    (1.0, 1.0),
                    -np.inf,
                    1.0,
                )

    def _apply(self, solution: dict[int, np.ndarray]) -> None:
        for node, permutation in solution.items():
            self.blocks[node], _ = self.locator.locate_with_permutation(
                self.topology.local_structure(node),
                self.blocks[node],
                permutation,
            )
            self.perms[node] = permutation

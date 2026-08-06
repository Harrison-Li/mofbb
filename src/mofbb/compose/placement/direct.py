"""Chemically constrained port repair for direct topology edges."""

from __future__ import annotations

from functools import cache

import numpy as np

from mofbb.compose.placement.milp import Constraints, solve_binary


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
        self.covariant = state.node_cov
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
            not self.pbc.is_carbon_metal_pair(symbol_i, symbol_j)
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
            if rep not in self.covariant:
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

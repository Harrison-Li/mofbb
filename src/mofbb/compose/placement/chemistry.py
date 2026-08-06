"""Choose and validate chemically compatible port permutations."""

from __future__ import annotations

import itertools

import numpy as np

from mofbb.compose.locator import PortAssignmentError


class PortRules:
    """Centralize chemistry checks shared by placement and cell closure."""

    def __init__(
        self,
        topology,
        blocks,
        locator,
        pbc,
        symmetry,
        fixed,
        groups,
    ):
        self.topology = topology
        self.blocks = blocks
        self.locator = locator
        self.pbc = pbc
        self.symmetry = symmetry
        self.fixed = fixed
        self.groups = groups

    def pref(self, edge):
        """Return a deterministic preferred orientation for an edge slot."""
        near, far = self.topology.neighbor_list[edge]
        forward = (
            self.topology.node_types[near.index]
            <= self.topology.node_types[far.index]
        )
        return np.array([0, 1] if forward else [1, 0])

    def node_mask(self, node, edge_perms):
        """Return a target/source feasibility mask for one node slot."""
        src_symbols = self.blocks[node].connection_point_anchor_symbols
        neighbors = self.topology.neighbor_list[node]
        allowed = np.ones((len(neighbors), len(src_symbols)), dtype=bool)
        bad_pair = None
        for row, node_near in enumerate(neighbors):
            edge = int(node_near.index)
            other = self.blocks[edge]
            if other is None:
                ends = [
                    int(item.index)
                    for item in self.topology.neighbor_list[edge]
                ]
                peer = ends[1] if ends[0] == node else ends[0]
                dst_symbols = self.blocks[peer].connection_point_anchor_symbols
            else:
                end = next(
                    i
                    for i, edge_near in enumerate(
                        self.topology.neighbor_list[edge]
                    )
                    if edge_near.index == node
                    and np.linalg.norm(
                        node_near.distance_vector + edge_near.distance_vector
                    )
                    < 0.01
                )
                dst_symbols = (
                    other.connection_point_anchor_symbols[edge_perms[edge][end]],
                )

            for col, src_symbol in enumerate(src_symbols):
                valid = [
                    not self.pbc.is_carbon_metal_pair(src_symbol, dst_symbol)
                    for dst_symbol in dst_symbols
                ]
                if bad_pair is None:
                    bad_pair = next(
                        (
                            (src_symbol, symbol)
                            for symbol, ok in zip(dst_symbols, valid)
                            if not ok
                        ),
                        None,
                    )
                allowed[row, col] = any(valid)
        return allowed, bad_pair

    def choose(self):
        """Choose edge orientations jointly with all node port assignments."""
        fixed, choices = self._options()
        if not choices:
            return fixed

        best = None
        scores = {}
        for selection in itertools.product(*choices):
            perms = fixed.copy()
            for item in selection:
                perms.update(item)
            score = self._score(perms, scores)
            if score is not None and (best is None or score < best[0]):
                best = score, perms
        if best is None:
            raise ValueError(
                "Forbidden fused-bond chemistry rejected edge slot orientation "
                "search: no joint node-port assignment exists."
            )
        return best[1]

    def edge_ok(self, edge, perm, placed, perms):
        """Return whether an edge orientation avoids carbon-metal pairs."""
        block = self.blocks[edge]
        if block is None:
            return True
        node_x = self.pbc._find_matched_x_sites(
            self.topology,
            placed,
            perms,
            edge,
        )
        node_symbols = [
            self.pbc._connection_port(placed[near.index], x)[0]
            for near, x in zip(
                self.topology.neighbor_list[edge],
                node_x,
                strict=True,
            )
        ]
        edge_symbols = block.connection_point_anchor_symbols
        return all(
            not self.pbc.is_carbon_metal_pair(
                node_symbol,
                edge_symbols[int(src)],
            )
            for node_symbol, src in zip(node_symbols, perm, strict=True)
        )

    def edge_perm(self, edge, chosen, placed, perms):
        """Return the first compatible preferred edge orientation."""
        pref = chosen.get(edge, self.pref(edge))
        for perm in (pref, pref[::-1]):
            if self.edge_ok(edge, perm, placed, perms):
                return perm.copy()
        raise ValueError(
            "Forbidden fused-bond chemistry rejected permutation optimization "
            f"for edge slot {edge}: neither endpoint permutation avoids "
            "a direct carbon-metal connection."
        )

    def validate(self, placed, perms, scale):
        """Reject assigned ports that predict forbidden or overlong bonds."""
        for raw_edge in self.topology.edge_indices:
            edge = int(raw_edge)
            node_x_i, node_x_j = self.pbc._find_matched_x_sites(
                self.topology,
                placed,
                perms,
                edge,
            )
            near, far = self.topology.neighbor_list[edge]
            node_i = placed[near.index]
            node_j = placed[far.index]
            block = placed[edge]
            if block is None:
                pairs = ((node_i, node_x_i, node_j, node_x_j),)
            else:
                edge_x_i, edge_x_j = block.connection_point_indices[perms[edge]]
                pairs = (
                    (node_i, node_x_i, block, int(edge_x_i)),
                    (block, int(edge_x_j), node_j, node_x_j),
                )
            for left, x_i, right, x_j in pairs:
                symbol_i, _, cap_i = self.pbc._connection_port(left, x_i)
                symbol_j, _, cap_j = self.pbc._connection_port(right, x_j)
                if self.pbc.is_carbon_metal_pair(symbol_i, symbol_j):
                    raise ValueError(
                        "Forbidden fused-bond chemistry rejected assembly: "
                        f"edge {edge} joins {symbol_i}-{symbol_j}."
                    )
                predicted = cap_i + cap_j
                limit = self.pbc.max_bond_len(symbol_i, symbol_j, scale)
                if predicted > limit:
                    raise ValueError(
                        "Predicted fused-bond gate rejected assembly: "
                        f"edge {edge} {symbol_i}-{symbol_j} is "
                        f"{predicted:.3f} Å; limit {limit:.3f} Å."
                    )

    def _flip_matters(self, edge):
        symbols = self.blocks[edge].connection_point_anchor_symbols
        return any(
            self.pbc.is_carbon_metal_pair(node_symbol, symbols[0])
            != self.pbc.is_carbon_metal_pair(node_symbol, symbols[1])
            for near in self.topology.neighbor_list[edge]
            for node_symbol in self.blocks[
                int(near.index)
            ].connection_point_anchor_symbols
        )

    @staticmethod
    def _candidates(pref, fixed=None, sensitive=False):
        if fixed is not None:
            return (fixed,)
        if sensitive:
            return pref, pref[::-1]
        return (pref,)

    def _options(self):
        fixed = {}
        choices = []
        for orbit in self.symmetry.edge_orbits:
            members = [
                int(i) for i in orbit.members if self.blocks[int(i)] is not None
            ]
            if not members:
                continue
            if self.groups.one(orbit):
                forced = self.groups.perm(orbit)
                pref = self.pref(orbit.representative)
                candidates = self._candidates(
                    pref,
                    forced,
                    any(self._flip_matters(i) for i in members),
                )
                options = [
                    {
                        i: self.symmetry.to_member(orbit, i, candidate)
                        for i in members
                    }
                    for candidate in candidates
                ]
                if len(options) == 1:
                    fixed.update(options[0])
                else:
                    choices.append(options)
                continue

            for edge in members:
                pref = self.pref(edge)
                candidates = self._candidates(
                    pref,
                    self.fixed.get(edge),
                    self._flip_matters(edge),
                )
                options = [{edge: candidate} for candidate in candidates]
                if len(options) == 1:
                    fixed.update(options[0])
                else:
                    choices.append(options)
        return fixed, choices

    def _score(self, perms, cache):
        score = 0.0
        for raw_node in self.topology.node_indices:
            node = int(raw_node)
            allowed, _ = self.node_mask(node, perms)
            forced = self.fixed.get(node)
            if forced is not None:
                if not np.all(allowed[np.arange(len(forced)), forced]):
                    return None
                continue

            key = (node, allowed.tobytes())
            if key not in cache:
                try:
                    cache[key] = self.locator.locate(
                        self.topology.local_structure(node),
                        self.blocks[node],
                        allowed=allowed,
                    )[2]
                except PortAssignmentError:
                    cache[key] = None
            if cache[key] is None:
                return None
            score += cache[key]
        return score

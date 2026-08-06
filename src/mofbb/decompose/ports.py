"""Logical connection-port assignment for decomposed fragments."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from math import gcd

from mofbb.decompose.object import Fragment, LogicalPort, TopologyEdge, XSite
from mofbb.decompose.pbc import PeriodicBond


class LogicalPortAssigner:
    """Group physical X sites into stable logical fragment ports."""

    def __init__(
        self,
        fragments: list[Fragment],
        connection_records: tuple[PeriodicBond, ...],
        topology_edges: list[TopologyEdge],
    ) -> None:
        self.fragments = fragments
        self.atom_to_fragment = {
            atom_index: (fragment.kind, fragment.index)
            for fragment in fragments
            for atom_index in fragment.source_indices
        }
        self.record_by_pair = {record.pair: record for record in connection_records}

        self.edge_by_pair = {
            edge.source_cut_bond: edge
            for edge in topology_edges
            if edge.source_cut_bond is not None
        }
        self_translations: dict[tuple[str, int], list[tuple[int, int, int]]] = (
            defaultdict(list)
        )
        for edge in topology_edges:
            begin_key = (edge.begin_kind, edge.begin_index)
            end_key = (edge.end_kind, edge.end_index)
            if begin_key == end_key:
                self_translations[begin_key].append(edge.slot_translation)

        self.self_lattices = {
            fragment_key: self.self_edge_lattice(fragment_key, translations)
            for fragment_key, translations in self_translations.items()
        }

    def assign(self) -> None:
        for fragment in self.fragments:
            fragment.logical_ports = self.ports_for_fragment(fragment)
            fragment.logical_kind = fragment.kind
            if fragment.kind not in {"organic_node", "edge"}:
                continue
            if len(fragment.logical_ports) == 2:
                fragment.logical_kind = "edge"
            elif len(fragment.logical_ports) > 2:
                fragment.logical_kind = "organic_node"

    def ports_for_fragment(
        self,
        fragment: Fragment,
    ) -> tuple[LogicalPort, ...]:
        grouped_x_indices: dict[tuple[str, int, tuple[int, int, int]], list[int]] = (
            defaultdict(list)
        )

        for site in fragment.x_sites:
            group_key = self.group_key_for_site(
                fragment,
                site,
                grouped_x_indices,
            )
            grouped_x_indices[group_key].append(site.x_index)

        return tuple(
            LogicalPort(
                logical_port_id=port_id,
                x_indices=tuple(sorted(grouped_x_indices[group_key])),
                neighbor_kind=group_key[0],
                neighbor_index=group_key[1],
                translation=group_key[2],
            )
            for port_id, group_key in enumerate(sorted(grouped_x_indices))
        )

    def group_key_for_site(
        self,
        fragment: Fragment,
        site: XSite,
        grouped_x_indices: dict[tuple[str, int, tuple[int, int, int]], list[int]],
    ) -> tuple[str, int, tuple[int, int, int]]:
        fragment_key = (fragment.kind, fragment.index)
        record = self.record_by_pair[site.source_cut_bond]

        internal = site.internal_source_index
        external = site.external_source_index
        neighbor_key = self.atom_to_fragment[external]

        edge = self.edge_by_pair[record.pair]
        begin_key = (edge.begin_kind, edge.begin_index)
        end_key = (edge.end_kind, edge.end_index)
        if begin_key == end_key:
            translation = (
                edge.slot_translation
                if internal == record.i
                else tuple(-value for value in edge.slot_translation)
            )
        elif fragment_key == begin_key and neighbor_key == end_key:
            translation = edge.slot_translation
        elif fragment_key == end_key and neighbor_key == begin_key:
            translation = tuple(-value for value in edge.slot_translation)
        else:
            translation = edge.slot_translation

        neighbor_lattice = self.self_lattices.get(neighbor_key)
        if neighbor_key != fragment_key and isinstance(neighbor_lattice, tuple):
            translation = self.effective_slot_translation(
                translation,
                neighbor_lattice,
            )

        group_key = (neighbor_key[0], neighbor_key[1], translation)
        if neighbor_key != fragment_key and isinstance(
            neighbor_lattice, _MultiRankSelfEdgeLattice
        ):
            equivalent_keys = [
                existing_key
                for existing_key in grouped_x_indices
                if existing_key[:2] == neighbor_key
                and neighbor_lattice.contains(
                    tuple(
                        existing - current
                        for existing, current in zip(existing_key[2], translation)
                    )
                )
            ]
            canonical_translation = min(
                (translation, *(key[2] for key in equivalent_keys))
            )
            group_key = (
                neighbor_key[0],
                neighbor_key[1],
                canonical_translation,
            )
            for equivalent_key in equivalent_keys:
                if equivalent_key != group_key:
                    grouped_x_indices[group_key].extend(
                        grouped_x_indices.pop(equivalent_key)
                    )

        return group_key

    @classmethod
    def self_edge_lattice(
        cls,
        fragment_key: tuple[str, int],
        translations: list[tuple[int, int, int]],
    ) -> tuple[int, int, int] | _MultiRankSelfEdgeLattice:
        rank, _ = cls.lattice_rank_and_divisor(tuple(translations))
        if rank > 1:
            generators = tuple(sorted(set(translations)))
            rank, divisor = cls.lattice_rank_and_divisor(generators)
            return _MultiRankSelfEdgeLattice(generators, rank, divisor)

        first = translations[0]
        divisor = gcd(*(abs(value) for value in first))

        primitive = tuple(value // divisor for value in first)
        axis = next(index for index, value in enumerate(primitive) if value)
        if primitive[axis] < 0:
            primitive = tuple(-value for value in primitive)

        period = gcd(
            *(abs(translation[axis] // primitive[axis]) for translation in translations)
        )
        return tuple(period * value for value in primitive)

    @staticmethod
    def lattice_rank_and_divisor(
        vectors: tuple[tuple[int, int, int], ...],
    ) -> tuple[int, int]:
        nonzero = tuple(vector for vector in vectors if any(vector))
        if not nonzero:
            return 0, 0

        cross_products = [
            (
                left[1] * right[2] - left[2] * right[1],
                left[2] * right[0] - left[0] * right[2],
                left[0] * right[1] - left[1] * right[0],
            )
            for left, right in combinations(nonzero, 2)
        ]
        nonzero_cross_products = [cross for cross in cross_products if any(cross)]
        if not nonzero_cross_products:
            divisor = gcd(*(abs(value) for vector in nonzero for value in vector))
            return 1, divisor

        determinants = [
            sum(left[index] * cross[index] for index in range(3))
            for left, middle, right in combinations(nonzero, 3)
            for cross in (
                (
                    middle[1] * right[2] - middle[2] * right[1],
                    middle[2] * right[0] - middle[0] * right[2],
                    middle[0] * right[1] - middle[1] * right[0],
                ),
            )
        ]
        if not any(determinants):
            divisor = gcd(
                *(abs(value) for cross in nonzero_cross_products for value in cross)
            )
            return 2, divisor

        divisor = gcd(*(abs(determinant) for determinant in determinants))
        return 3, divisor

    @staticmethod
    def effective_slot_translation(
        translation: tuple[int, int, int],
        neighbor_lattice: tuple[int, int, int],
    ) -> tuple[int, int, int]:
        axis = next(index for index, value in enumerate(neighbor_lattice) if value)
        quotient = translation[axis] // neighbor_lattice[axis]
        return tuple(
            value - quotient * lattice_value
            for value, lattice_value in zip(translation, neighbor_lattice)
        )


@dataclass(frozen=True)
class _MultiRankSelfEdgeLattice:
    generators: tuple[tuple[int, int, int], ...]
    rank: int
    determinant_divisor: int

    def contains(self, translation: tuple[int, int, int]) -> bool:
        if not any(translation):
            return True
        rank, divisor = LogicalPortAssigner.lattice_rank_and_divisor(
            (*self.generators, translation)
        )
        return rank == self.rank and divisor == self.determinant_divisor

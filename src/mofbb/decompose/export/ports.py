"""Incidence-to-building-block port mappings for contracted topologies."""

from __future__ import annotations

from collections import defaultdict
from functools import cached_property

import numpy as np

from mofbb.decompose.object import Fragment, TopologyEdge, TopologyVertex, XSite

Key = tuple[str, int]
SlotMap = dict[Key, tuple[str, int]]
End = tuple[Key, TopologyEdge]
Incident = tuple[Key, np.ndarray, TopologyEdge]


class PortMapper:
    """Map topology incidences to each emitted BB's connection ports."""

    def __init__(
        self,
        src_nodes: list[TopologyVertex],
        src_edges: list[TopologyEdge],
        nodes: list[TopologyVertex],
        edges: list[TopologyEdge],
        slot_map: SlotMap,
        frags: list[Fragment],
    ) -> None:
        self.src_nodes = src_nodes
        self.src_edges = src_edges
        self.nodes = nodes
        self.edges = edges
        self.slot_map = slot_map
        self.frags = {self._key(frag): frag for frag in frags}

    def build(self) -> dict[int, list[int]]:
        maps = {index: [] for index in range(len(self.nodes))}
        for edge_id, edge in enumerate(self.edges):
            edge_key = self.edge_frags.get(edge_id)
            ends = self.edge_ends.get(
                edge_id,
                self._direct_ends(edge),
            )
            edge_ports = self._map_edge(
                maps,
                ends,
                edge_key,
            )
            if edge_key is not None:
                maps[len(self.nodes) + edge_id] = edge_ports

        self._validate(maps)
        return maps

    @cached_property
    def edge_keys(self) -> set[Key]:
        return {
            key
            for key, (kind, _) in self.slot_map.items()
            if kind == "edge"
        }

    @cached_property
    def node_ids(self) -> dict[Key, int]:
        return {
            self._key(node): index
            for index, node in enumerate(self.nodes)
        }

    @cached_property
    def src_pos(self) -> dict[Key, np.ndarray]:
        return {
            self._key(node): np.asarray(node.frac_position)
            for node in self.src_nodes
        }

    @cached_property
    def port_ids(self) -> dict[Key, dict[int, int]]:
        return {
            key: {
                x_id: port_id
                for port_id, x_id in enumerate(
                    sorted(site.x_index for site in frag.x_sites)
                )
            }
            for key, frag in self.frags.items()
        }

    @cached_property
    def edge_frags(self) -> dict[int, Key]:
        return {
            slot_id: key
            for key, (kind, slot_id) in self.slot_map.items()
            if kind == "edge"
        }

    @cached_property
    def edge_ends(self) -> dict[int, tuple[End, End]]:
        incidents = self._incidents()
        return {
            self.slot_map[key][1]: self._order(incidents[key])
            for key in self.edge_keys
        }

    def _incidents(self) -> dict[Key, list[Incident]]:
        incidents: dict[Key, list[Incident]] = defaultdict(list)
        for edge in self.src_edges:
            begin = edge.begin_kind, edge.begin_index
            end = edge.end_kind, edge.end_index
            begin_edge = begin in self.edge_keys
            end_edge = end in self.edge_keys
            if not (begin_edge or end_edge):
                continue

            begin_img = self._image(begin, edge.begin_frac)
            end_img = self._image(end, edge.end_frac)
            shift = end_img - begin_img
            if begin_edge:
                incidents[begin].append((end, shift, edge))
            else:
                incidents[end].append((begin, -shift, edge))
        return incidents

    def _order(
        self,
        incidents: list[Incident],
    ) -> tuple[End, End]:
        (begin, begin_shift, begin_edge), (end, end_shift, end_edge) = incidents
        shift = end_shift - begin_shift
        reverse = self.node_ids[begin] > self.node_ids[end]
        reverse |= begin == end and tuple(shift) > tuple(-shift)
        if reverse:
            begin, end = end, begin
            begin_edge, end_edge = end_edge, begin_edge
        return (begin, begin_edge), (end, end_edge)

    def _map_edge(
        self,
        maps: dict[int, list[int]],
        ends: tuple[End, End],
        edge_key: Key | None,
    ) -> list[int]:
        edge_ports = []
        for endpoint, (key, src_edge) in enumerate(ends):
            site = self._end_site(key, src_edge, endpoint)
            maps[self.node_ids[key]].append(
                self.port_ids[key][site.x_index]
            )
            if edge_key is not None:
                edge_site = self._edge_site(edge_key, src_edge)
                edge_ports.append(self.port_ids[edge_key][edge_site.x_index])
        return edge_ports

    def _end_site(
        self,
        key: Key,
        src_edge: TopologyEdge,
        endpoint: int,
    ) -> XSite:
        candidates = self._cut_sites(key, src_edge)
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) == 2:
            source_atom = src_edge.source_cut_bond[endpoint]
            matches = [
                site
                for site in candidates
                if site.internal_source_index == source_atom
            ]
            if len(matches) == 1:
                return matches[0]
        raise ValueError(
            f"Cannot map topology incidence for {key!r} "
            f"and cut {src_edge.source_cut_bond}."
        )

    def _edge_site(
        self,
        key: Key,
        src_edge: TopologyEdge,
    ) -> XSite:
        return next(iter(self._cut_sites(key, src_edge)))

    def _cut_sites(
        self,
        key: Key,
        src_edge: TopologyEdge,
    ) -> list[XSite]:
        return [
            site
            for site in self.frags[key].x_sites
            if site.source_cut_bond == src_edge.source_cut_bond
            and site.source_cut_shift == src_edge.source_cut_shift
        ]

    def _image(
        self,
        key: Key,
        frac: tuple[float, float, float],
    ) -> np.ndarray:
        return np.rint(np.asarray(frac) - self.src_pos[key]).astype(int)

    @staticmethod
    def _direct_ends(
        edge: TopologyEdge,
    ) -> tuple[End, End]:
        return (
            ((edge.begin_kind, edge.begin_index), edge),
            ((edge.end_kind, edge.end_index), edge),
        )

    @staticmethod
    def _validate(maps: dict[int, list[int]]) -> None:
        for slot, mapping in maps.items():
            if sorted(mapping) != list(range(len(mapping))):
                raise ValueError(
                    f"Topology port map for slot {slot} is not a permutation: "
                    f"{mapping}."
                )

    @staticmethod
    def _key(item: Fragment | TopologyVertex) -> Key:
        return item.kind, item.index

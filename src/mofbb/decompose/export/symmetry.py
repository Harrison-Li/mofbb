"""Source-symmetry matching for decorated abstract topologies."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import spglib
from ase.geometry import cell_to_cellpar, cellpar_to_cell

from mofbb.decompose.object import Fragment, TopologyEdge, TopologyVertex
from mofbb.decompose.pbc import PeriodicBond
from mofbb.decompose.symmetry import (
    AtomicSymmetry,
    SpaceGroupOperations,
    TopologySymmetry,
    map_periodic_bond,
)

Cell = tuple[float, float, float, float, float, float]
Key = tuple[str, int]
SlotMap = dict[Key, tuple[str, int]]
GraphEdge = tuple[int, int, tuple[int, int, int]]


def canonical_edge(
    begin: int,
    end: int,
    shift: np.ndarray | tuple[int, int, int],
) -> GraphEdge:
    """Return an orientation-independent periodic graph edge."""

    values = tuple(int(value) for value in shift)
    direct = (begin, end, values)
    reverse = (end, begin, tuple(-value for value in values))
    return min(direct, reverse)


def group_symbol(
    rots: np.ndarray,
    shifts: np.ndarray,
    cell: Cell,
) -> str:
    """Identify a subgroup from its fractional-coordinate operations."""

    group = spglib.get_spacegroup_type_from_symmetry(
        rots,
        shifts,
        np.asarray(cellpar_to_cell(cell), dtype=float),
        symprec=0.1,
    )
    return str(group.international_short) if group is not None else "P1"


def fit_cell(
    cell: Cell,
    rots: np.ndarray,
) -> Cell:
    """Project a rounded experimental cell metric onto its point group."""

    matrix = np.asarray(cellpar_to_cell(cell), dtype=float)
    metric = matrix @ matrix.T
    projected = np.mean(
        [rotation.T @ metric @ rotation for rotation in rots],
        axis=0,
    )
    projected = 0.5 * (projected + projected.T)
    compatible_cell = np.linalg.cholesky(projected)
    return tuple(float(value) for value in cell_to_cellpar(compatible_cell))


@dataclass(frozen=True)
class SymChoice:
    """The operations written to CGD and their source-group relationship."""

    sym: TopologySymmetry
    source: str | None
    compatible: bool | None


def choose_symmetry(
    net: TopologySymmetry,
    src: SpaceGroupOperations | None,
    decorated: TopologySymmetry | None,
    colors: list[int],
) -> SymChoice:
    """Prefer decorated source symmetry, then a source subgroup of the net."""

    if src is None:
        return SymChoice(net, None, None)
    if decorated is not None:
        return SymChoice(decorated, src.symbol, True)

    indices = _source_ops(net, src)
    if not indices:
        return SymChoice(net, src.symbol, False)

    sym = TopologySymmetry(
        src.symbol,
        net.rotations[indices],
        net.translations[indices],
        np.asarray(colors, dtype=int),
        net.node_maps[indices],
        net.node_images[indices],
        net.edge_maps[indices],
    )
    return SymChoice(sym, src.symbol, True)


def _source_ops(
    net: TopologySymmetry,
    src: SpaceGroupOperations,
) -> list[int]:
    indices = []
    for rotation, translation in zip(
        src.rotations,
        src.translations,
        strict=True,
    ):
        matches = [
            index
            for index, (net_rotation, net_translation) in enumerate(
                zip(
                    net.rotations,
                    net.translations,
                    strict=True,
                )
            )
            if np.array_equal(rotation, net_rotation)
            and _same_shift(translation, net_translation)
        ]
        if len(matches) != 1:
            return []
        indices.append(matches[0])
    return indices


def _same_shift(
    left: np.ndarray,
    right: np.ndarray,
    tolerance: float = 1.0e-6,
) -> bool:
    difference = left - right
    residual = difference - np.rint(difference)
    return bool(np.max(np.abs(residual)) <= tolerance)


class SymmetryBuilder:
    """Find source operations preserving BB slots and selected cuts."""

    def __init__(
        self,
        cell: Cell,
        nodes: list[TopologyVertex],
        edges: list[TopologyEdge],
        slot_map: SlotMap,
        frags: list[Fragment],
        bonds: tuple[PeriodicBond, ...],
        bb_ids: dict[Key, str],
        colors: list[int],
        src: AtomicSymmetry,
    ) -> None:
        self.cell = cell
        self.nodes = nodes
        self.edges = edges
        self.frags = frags
        self.bonds = bonds
        self.bb_ids = bb_ids
        self.colors = colors
        self.src = src

        self.frag_by_atoms = {
            frozenset(frag.source_indices): self._key(frag)
            for frag in frags
        }
        self.node_ids = {
            self._key(node): index
            for index, node in enumerate(nodes)
        }
        self.pos = np.asarray(
            [node.frac_position for node in nodes],
            dtype=float,
        )
        self.edge_frags = {
            index: key
            for key, (kind, index) in slot_map.items()
            if kind == "edge"
        }
        self.edge_ids = {
            key: index for index, key in self.edge_frags.items()
        }
        self.direct = self._index_direct()
        self.graph = self._graph()

    def build(self) -> TopologySymmetry | None:
        """Return the source subgroup preserving the decorated topology."""

        rots = []
        shifts = []
        node_maps = []
        images = []
        edge_maps = []
        for op in zip(
            self.src.rotations,
            self.src.translations,
            self.src.atom_maps,
            self.src.atom_images,
            strict=True,
        ):
            mapping = self._map_op(*op)
            if mapping is None:
                continue
            node_map, node_images, edge_map = mapping
            rots.append(op[0])
            shifts.append(op[1])
            node_maps.append(node_map)
            images.append(node_images)
            edge_maps.append(edge_map)

        if not rots:
            return None
        rots = np.asarray(rots, dtype=int)
        shifts = np.asarray(shifts, dtype=float)
        symbol = self._symbol(rots, shifts)
        return TopologySymmetry(
            symbol,
            rots,
            shifts,
            np.asarray(self.colors, dtype=int),
            np.asarray(node_maps, dtype=int),
            np.asarray(images, dtype=int),
            np.asarray(edge_maps, dtype=int),
        )

    def _map_op(
        self,
        rotation: np.ndarray,
        shift: np.ndarray,
        atom_map: np.ndarray,
        images: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        def map_bond(bond: PeriodicBond) -> PeriodicBond:
            return map_periodic_bond(bond, rotation, atom_map, images)

        if not self._maps_bonds(map_bond):
            return None

        frag_map = self._map_frags(atom_map)
        if frag_map is None:
            return None
        node_mapping = self._map_nodes(frag_map, rotation, shift)
        if node_mapping is None:
            return None
        node_map, node_images = node_mapping
        edge_map = self._map_edges(frag_map, map_bond)
        if edge_map is None:
            return None
        if not self._maps_graph(rotation, node_map, node_images, edge_map):
            return None
        return node_map, node_images, edge_map

    def _maps_bonds(self, map_bond) -> bool:
        mapped = {map_bond(bond) for bond in self.bonds}
        return mapped == set(self.bonds)

    def _map_frags(
        self,
        atom_map: np.ndarray,
    ) -> dict[Key, Key] | None:
        mapping = {}
        for frag in self.frags:
            source = self._key(frag)
            mapped_atoms = frozenset(
                int(atom_map[index]) for index in frag.source_indices
            )
            target = self.frag_by_atoms.get(mapped_atoms)
            if target is None:
                return None
            if self.bb_ids[source] != self.bb_ids[target]:
                return None
            mapping[source] = target
        return mapping

    def _map_nodes(
        self,
        frag_map: dict[Key, Key],
        rotation: np.ndarray,
        shift: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        try:
            node_map = np.asarray(
                [
                    self.node_ids[frag_map[self._key(node)]]
                    for node in self.nodes
                ],
                dtype=int,
            )
        except KeyError:
            return None
        if len(np.unique(node_map)) != len(self.nodes):
            return None
        differences = (
            self.pos @ np.asarray(rotation, dtype=int).T
            + shift
            - self.pos[node_map]
        )
        return node_map, np.rint(differences).astype(int)

    def _map_edges(self, frag_map, map_bond) -> np.ndarray | None:
        edge_map = np.empty(len(self.edges), dtype=int)
        for edge_id, edge in enumerate(self.edges):
            target = self._edge_target(
                edge_id,
                edge,
                frag_map,
                map_bond,
            )
            if target is None:
                return None
            if self.colors[edge_id] != self.colors[target]:
                return None
            edge_map[edge_id] = target
        if len(np.unique(edge_map)) != len(self.edges):
            return None
        return edge_map

    def _edge_target(
        self,
        edge_id: int,
        edge: TopologyEdge,
        frag_map: dict[Key, Key],
        map_bond,
    ) -> int | None:
        frag_key = self.edge_frags.get(edge_id)
        if frag_key is not None:
            return self.edge_ids.get(frag_map[frag_key])

        record = PeriodicBond(
            edge.source_cut_bond[0],
            edge.source_cut_bond[1],
            edge.source_cut_shift,
        )
        source_group = self.direct[record]
        target_group = self.direct.get(map_bond(record), ())
        rank = source_group.index(edge_id)
        return target_group[rank] if rank < len(target_group) else None

    def _maps_graph(
        self,
        rotation: np.ndarray,
        node_map: np.ndarray,
        images: np.ndarray,
        edge_map: np.ndarray,
    ) -> bool:
        for edge_id, (begin, end, shift) in enumerate(self.graph):
            mapped = canonical_edge(
                int(node_map[begin]),
                int(node_map[end]),
                images[end]
                + rotation @ np.asarray(shift, dtype=int)
                - images[begin],
            )
            if mapped != self.graph[int(edge_map[edge_id])]:
                return False
        return True

    def _index_direct(self) -> dict[PeriodicBond, list[int]]:
        direct: dict[PeriodicBond, list[int]] = defaultdict(list)
        for index, edge in enumerate(self.edges):
            if edge.source_cut_bond is None:
                continue
            direct[
                PeriodicBond(
                    edge.source_cut_bond[0],
                    edge.source_cut_bond[1],
                    edge.source_cut_shift,
                )
            ].append(index)
        return direct

    def _graph(self) -> list[GraphEdge]:
        graph = []
        for edge in self.edges:
            begin = self.node_ids[(edge.begin_kind, edge.begin_index)]
            end = self.node_ids[(edge.end_kind, edge.end_index)]
            begin_image = np.rint(
                np.asarray(edge.begin_frac) - self.pos[begin]
            )
            end_image = np.rint(np.asarray(edge.end_frac) - self.pos[end])
            graph.append(
                canonical_edge(begin, end, end_image - begin_image)
            )
        return graph

    def _symbol(
        self,
        rots: np.ndarray,
        shifts: np.ndarray,
    ) -> str:
        if len(rots) == len(self.src.rotations):
            return self.src.symbol
        return group_symbol(rots, shifts, self.cell)

    @staticmethod
    def _key(item: Fragment | TopologyVertex) -> Key:
        return item.kind, item.index

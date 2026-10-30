'''
Atom-order-invariant matching for MOF building blocks.
'''

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import networkx as nx

if TYPE_CHECKING:
    from mofbb.compose.building_block import BuildingBlock
    from mofbb.decompose.object import Fragment


BuildingBlockKind = Literal["metal_node", "organic_node", "edge", "linker"]
Bond = tuple[int, int, str]


@dataclass(frozen=True)
class BuildingBlockGraph:
    '''
    Store an element-labeled building-block graph for comparison.

    Args:
        name (str):
            Name used in comparison results.
        kind (str):
            Building-block category, such as metal_node or edge.
        symbols (tuple[str, ...]):
            Element symbols in atom-index order; X marks connection sites.
        bonds (tuple):
            Zero-based (i, j, bond_type) records. Duplicate and self-bonds are rejected.
    '''

    name: str
    kind: str
    symbols: tuple[str, ...]
    bonds: tuple[Bond, ...]

    def __post_init__(self) -> None:
        atom_count = len(self.symbols)
        normalized: list[Bond] = []
        seen_pairs: set[tuple[int, int]] = set()
        for raw_i, raw_j, raw_type in self.bonds:
            i, j = sorted((int(raw_i), int(raw_j)))
            if i == j:
                raise ValueError(f"Building block {self.name!r} contains a self bond.")
            if i < 0 or j >= atom_count:
                raise ValueError(
                    f"Building block {self.name!r} contains an out-of-range bond "
                    f"{(i, j)}."
                )
            if (i, j) in seen_pairs:
                raise ValueError(
                    f"Building block {self.name!r} repeats bond {(i, j)}."
                )
            seen_pairs.add((i, j))
            normalized.append((i, j, str(raw_type)))
        object.__setattr__(self, "bonds", tuple(sorted(normalized)))

    @property
    def num_x(self) -> int:
        """Return the number of dummy connection sites."""

        return self.symbols.count("X")

    @property
    def composition(self) -> tuple[tuple[str, int], ...]:
        """Return a stable element-count signature."""

        return tuple(sorted(Counter(self.symbols).items()))

    @classmethod
    def from_xyz(
        cls,
        path: str | Path,
        *,
        kind: str | None = None,
        name: str | None = None,
    ) -> BuildingBlockGraph:
        '''
        Read a building-block XYZ file and keep its labeled molecular graph.

        Args:
            path (str or Path):
                Building-block XYZ file with appended bond records.
            kind (str or None):
                Override the building-block kind; otherwise keep the source kind.
            name (str or None):
                Override the name; otherwise use the source name.

        Returns:
            A BuildingBlockGraph with normalized bond records.
        '''

        from mofbb.compose.building_block import BuildingBlock

        block = BuildingBlock(path)
        return cls.from_building_block(block, kind=kind, name=name)

    @classmethod
    def from_building_block(
        cls,
        block: BuildingBlock,
        *,
        kind: str | None = None,
        name: str | None = None,
    ) -> BuildingBlockGraph:
        '''
        Extract the labeled graph from a composition building block.

        Args:
            block (BuildingBlock):
                Building block containing atoms, bonds, and bond labels.
            kind (str or None):
                Override the building-block kind; otherwise keep the source kind.
            name (str or None):
                Override the name; otherwise use the source name.

        Returns:
            A BuildingBlockGraph preserving the source atom order.
        '''

        symbols = tuple(block.atoms.get_chemical_symbols())
        if kind is None:
            kind = block.kind
        bonds = tuple(
            (int(pair[0]), int(pair[1]), str(bond_type))
            for pair, bond_type in zip(
                block.bonds,
                block.bond_types,
                strict=True,
            )
        )
        return cls(
            name=name or str(block.name),
            kind=kind,
            symbols=symbols,
            bonds=bonds,
        )

    @classmethod
    def from_fragment(
        cls,
        fragment: Fragment,
        *,
        name: str | None = None,
    ) -> BuildingBlockGraph:
        '''
        Extract the labeled graph from a decomposition fragment.

        Args:
            fragment (Fragment):
                Fragment containing atoms, bonds, kind, and index.
            name (str or None):
                Override the name; otherwise combine the fragment kind and index.

        Returns:
            A BuildingBlockGraph preserving the fragment atom order.
        '''

        return cls(
            name=name or f"{fragment.kind}_{fragment.index:03d}",
            kind=fragment.kind,
            symbols=tuple(fragment.atoms.get_chemical_symbols()),
            bonds=tuple(fragment.bonds),
        )

    def to_networkx(self) -> nx.Graph:
        """Return a new NetworkX graph with element and bond labels."""

        graph = nx.Graph()
        graph.add_nodes_from(
            (index, {"symbol": symbol})
            for index, symbol in enumerate(self.symbols)
        )
        graph.add_edges_from(
            (i, j, {"bond_type": bond_type})
            for i, j, bond_type in self.bonds
        )
        return graph


@dataclass(frozen=True)
class BuildingBlockMatch:
    '''
    Store the outcome of comparing a reference block with a candidate.

    The individual flags record kind, composition, connectivity, and bond-label
    agreement. matched follows the comparison options. mapping contains pairs
    of reference and candidate atom indices when an isomorphism was found.
    '''

    reference_name: str
    candidate_name: str
    kind_match: bool
    composition_match: bool
    connectivity_match: bool
    bond_type_match: bool
    matched: bool
    mapping: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class BuildingBlockMultisetMatch:
    '''
    Store a one-to-one assignment between reference and candidate blocks.

    Args:
        reference_count (int):
            Total number of expected blocks, including repeated copies.
        candidate_count (int):
            Total number of predicted blocks, including repeated copies.
        matched_pairs (tuple[tuple[int, int], ...]):
            Matched zero-based reference and candidate list indices.
    '''

    reference_count: int
    candidate_count: int
    matched_pairs: tuple[tuple[int, int], ...]

    @property
    def matched_count(self) -> int:
        return len(self.matched_pairs)

    @property
    def precision(self) -> float:
        if self.candidate_count == 0:
            return float(self.reference_count == 0)
        return self.matched_count / self.candidate_count

    @property
    def recall(self) -> float:
        if self.reference_count == 0:
            return float(self.candidate_count == 0)
        return self.matched_count / self.reference_count

    @property
    def f1(self) -> float:
        denominator = self.precision + self.recall
        if denominator == 0.0:
            return 0.0
        return 2.0 * self.precision * self.recall / denominator

    @property
    def exact(self) -> bool:
        return (
            self.matched_count
            == self.reference_count
            == self.candidate_count
        )


def compare_building_blocks(
    reference: BuildingBlockGraph,
    candidate: BuildingBlockGraph,
    *,
    require_kind: bool = True,
    require_bond_types: bool = False,
) -> BuildingBlockMatch:
    '''
    Compare molecular graphs without depending on atom order or geometry.

    Args:
        reference (BuildingBlockGraph):
            Expected building-block graph.
        candidate (BuildingBlockGraph):
            Graph to compare against the reference.
        require_kind (bool):
            Require matching building-block categories. Defaults to True.
        require_bond_types (bool):
            Also require identical bond labels. Defaults to False.

    Returns:
        A BuildingBlockMatch with comparison flags and, when found, a mapping
        from reference atom indices to candidate atom indices.
    '''

    kind_match = reference.kind == candidate.kind
    composition_match = reference.composition == candidate.composition
    connectivity_match = False
    bond_type_match = False
    mapping: tuple[tuple[int, int], ...] = ()

    if (
        composition_match
        and len(reference.bonds) == len(candidate.bonds)
        and (kind_match or not require_kind)
    ):
        reference_graph = reference.to_networkx()
        candidate_graph = candidate.to_networkx()
        node_match = nx.algorithms.isomorphism.categorical_node_match(
            "symbol", None
        )
        edge_match = nx.algorithms.isomorphism.categorical_edge_match(
            "bond_type", None
        )

        strict_matcher = nx.algorithms.isomorphism.GraphMatcher(
            reference_graph,
            candidate_graph,
            node_match=node_match,
            edge_match=edge_match,
        )
        if strict_matcher.is_isomorphic():
            connectivity_match = True
            bond_type_match = True
            mapping = tuple(sorted(strict_matcher.mapping.items()))
        else:
            connectivity_matcher = nx.algorithms.isomorphism.GraphMatcher(
                reference_graph,
                candidate_graph,
                node_match=node_match,
            )
            if connectivity_matcher.is_isomorphic():
                connectivity_match = True
                mapping = tuple(sorted(connectivity_matcher.mapping.items()))

    matched = connectivity_match and (
        bond_type_match or not require_bond_types
    )
    return BuildingBlockMatch(
        reference_name=reference.name,
        candidate_name=candidate.name,
        kind_match=kind_match,
        composition_match=composition_match,
        connectivity_match=connectivity_match,
        bond_type_match=bond_type_match,
        matched=matched,
        mapping=mapping,
    )


def match_building_block_multisets(
    references: list[BuildingBlockGraph],
    candidates: list[BuildingBlockGraph],
    *,
    require_kind: bool = True,
    require_bond_types: bool = False,
) -> BuildingBlockMultisetMatch:
    '''
    Find the largest one-to-one assignment of matching building blocks.

    Args:
        references (list[BuildingBlockGraph]):
            Expected blocks, including repeated copies.
        candidates (list[BuildingBlockGraph]):
            Predicted blocks, including repeated copies.
        require_kind (bool):
            Require matching building-block categories. Defaults to True.
        require_bond_types (bool):
            Also require identical bond labels. Defaults to False.

    Returns:
        A BuildingBlockMultisetMatch with matched (reference, candidate) list
        indices and precision, recall, F1, and exact-match properties.
    '''

    graph = nx.Graph()
    reference_nodes = [("reference", index) for index in range(len(references))]
    candidate_nodes = [("candidate", index) for index in range(len(candidates))]
    graph.add_nodes_from(reference_nodes, bipartite=0)
    graph.add_nodes_from(candidate_nodes, bipartite=1)

    for reference_index, reference in enumerate(references):
        for candidate_index, candidate in enumerate(candidates):
            comparison = compare_building_blocks(
                reference,
                candidate,
                require_kind=require_kind,
                require_bond_types=require_bond_types,
            )
            if comparison.matched:
                graph.add_edge(
                    ("reference", reference_index),
                    ("candidate", candidate_index),
                )

    matching = nx.algorithms.bipartite.maximum_matching(
        graph,
        top_nodes=set(reference_nodes),
    )
    pairs = tuple(
        sorted(
            (
                reference_index,
                matching[("reference", reference_index)][1],
            )
            for reference_index in range(len(references))
            if ("reference", reference_index) in matching
        )
    )
    return BuildingBlockMultisetMatch(
        reference_count=len(references),
        candidate_count=len(candidates),
        matched_pairs=pairs,
    )


__all__ = [
    "BuildingBlockGraph",
    "BuildingBlockMatch",
    "BuildingBlockMultisetMatch",
    "compare_building_blocks",
    "match_building_block_multisets",
]

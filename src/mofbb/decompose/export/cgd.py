"""Prepare decorated topologies and delegate CGD serialization."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from mofbb.io.cgd.writer import CgdWriter as TopologyCgdWriter

from mofbb.decompose.export.symmetry import (
    Cell,
    Key,
    SymChoice,
    choose_symmetry,
    fit_cell,
)
from mofbb.decompose.object import TopologyEdge, TopologyVertex
from mofbb.decompose.symmetry import (
    SpaceGroupOperations,
    TopologySymmetry,
    detect_topology_symmetry,
)


def edge_types(
    count: int,
    edge_bbs: dict[int, str] | None,
) -> tuple[list[int], dict[str | None, int]]:
    """Assign stable positive colors to edge building-block identities."""

    by_bb: dict[str | None, int] = {}
    colors = [
        by_bb.setdefault(
            None if edge_bbs is None else edge_bbs.get(index),
            len(by_bb) + 1,
        )
        for index in range(count)
    ]
    return colors, by_bb


@dataclass
class CgdWriter:
    """Write graph geometry and discrete assembly metadata to one CGD file."""

    path: Path
    name: str
    cell: Cell
    nodes: list[TopologyVertex]
    edges: list[TopologyEdge]
    bb_ids: dict[Key, str]
    edge_bbs: dict[int, str] | None = None
    src_sym: SpaceGroupOperations | None = None
    slot_sym: TopologySymmetry | None = None
    ports: dict[int, list[int]] | None = None

    def write(self) -> dict[str, object]:
        """Write the explicit topology with its graph-derived symmetry."""

        positions, endpoints = self.sym.project(self.nodes, self.edges)
        TopologyCgdWriter(
            name=self.name,
            cell=self.output_cell,
            spacegroup=self.sym.symbol,
            net_spacegroup=self.net_sym.symbol,
            source_group=self.choice.source,
            source_compatible=self.choice.compatible,
            rotations=self.sym.rotations,
            translations=self.sym.translations,
            node_maps=self.sym.node_maps,
            edge_maps=self.sym.edge_maps,
            node_types=self.node_types,
            node_tags=self.node_tags,
            coordinations=[node.coordination for node in self.nodes],
            edge_tags=self.edge_tags,
            colors=self.colors,
            ports=self.ports,
        ).write(self.path, positions, endpoints)
        return self.summary()

    @cached_property
    def node_types(self) -> dict[str, int]:
        types: dict[str, int] = {}
        for node in self.nodes:
            bb_id = self.bb_ids[(node.kind, node.index)]
            types.setdefault(bb_id, len(types) + 1)
        return types

    @cached_property
    def node_tags(self) -> list[int]:
        return [
            self.node_types[self.bb_ids[(node.kind, node.index)]]
            for node in self.nodes
        ]

    @cached_property
    def edge_data(self) -> tuple[list[int], dict[str | None, int]]:
        return edge_types(len(self.edges), self.edge_bbs)

    @property
    def colors(self) -> list[int]:
        return self.edge_data[0]

    @property
    def edge_tags(self) -> dict[str | None, int]:
        return self.edge_data[1]

    @cached_property
    def net_sym(self) -> TopologySymmetry:
        return detect_topology_symmetry(
            self.cell,
            self.nodes,
            self.edges,
            self.node_tags,
            self.colors,
        )

    @cached_property
    def choice(self) -> SymChoice:
        return choose_symmetry(
            self.net_sym,
            self.src_sym,
            self.slot_sym,
            self.colors,
        )

    @property
    def sym(self) -> TopologySymmetry:
        return self.choice.sym

    @cached_property
    def output_cell(self) -> Cell:
        return fit_cell(
            self.cell,
            self.sym.rotations,
        )

    def summary(self) -> dict[str, object]:
        return {
            "num_vertices": len(self.nodes),
            "num_edges": len(self.edges),
            "num_node_types": len(self.node_types),
            "spacegroup": self.sym.symbol,
            "net_spacegroup": self.net_sym.symbol,
            "num_symmetry_operations": len(self.sym.rotations),
            "node_bb_types": {
                str(node_type - 1): bb_id
                for bb_id, node_type in self.node_types.items()
            },
            "edge_bb_types": {
                str(edge_type): bb_id
                for bb_id, edge_type in self.edge_tags.items()
                if bb_id is not None
            },
        }

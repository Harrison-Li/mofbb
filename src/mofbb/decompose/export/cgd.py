"""Serialize decorated abstract topologies in CGD format."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TextIO

import numpy as np

from mofbb.decompose.geometry import format_float
from mofbb.decompose.object import TopologyEdge, TopologyVertex
from mofbb.decompose.symmetry import (
    SpaceGroupOperations,
    TopologySymmetry,
    detect_topology_symmetry,
)
from mofbb.decompose.export.symmetry import (
    Cell,
    Key,
    choose_symmetry,
    fit_cell,
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
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as out:
            self._write_header(out)
            self._write_symmetry(out)
            self._write_bbs(out)
            self._write_ports(out)
            self._write_nodes(out, positions)
            self._write_edges(out, endpoints)
            out.write("END\n")
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
    def choice(self):
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

    def _write_header(self, out: TextIO) -> None:
        out.write("CRYSTAL\n")
        out.write(f"  NAME {self.name}\n")
        out.write(f"  GROUP {self.sym.symbol}\n")
        out.write(f"  # NET_GROUP {self.net_sym.symbol}\n")
        if self.choice.source is not None:
            out.write(f"  # SOURCE_GROUP {self.choice.source}\n")
            compatible = int(self.choice.compatible)
            out.write(f"  # SOURCE_GROUP_COMPATIBLE {compatible}\n")
        cell = " ".join(f"{value:.16g}" for value in self.output_cell)
        out.write(f"  CELL {cell}\n")
        out.write("  # EXPANDED\n")

    def _write_symmetry(self, out: TextIO) -> None:
        for rotation, translation in zip(
            self.sym.rotations,
            self.sym.translations,
            strict=True,
        ):
            values = [
                *(str(int(value)) for value in rotation.ravel()),
                *(format_float(value) for value in translation),
            ]
            out.write(f"  # SYMOP {' '.join(values)}\n")

        for op_id, (node_map, edge_map) in enumerate(
            zip(
                self.sym.node_maps,
                self.sym.edge_maps,
                strict=True,
            ),
            start=1,
        ):
            mapping = np.concatenate((node_map, len(node_map) + edge_map)) + 1
            targets = " ".join(str(int(value)) for value in mapping)
            out.write(f"  # SLOT_MAP {op_id} {targets}\n")

    def _write_bbs(self, out: TextIO) -> None:
        for bb_id, node_type in sorted(
            self.node_types.items(),
            key=lambda item: item[1],
        ):
            out.write(f"  # NODE_BB {node_type} {bb_id}\n")

        for bb_id, edge_type in sorted(
            self.edge_tags.items(),
            key=lambda item: item[1],
        ):
            value = bb_id if bb_id is not None else "NONE"
            out.write(f"  # EDGE_BB {edge_type} {value}\n")

    def _write_ports(self, out: TextIO) -> None:
        for slot, mapping in sorted((self.ports or {}).items()):
            ports = " ".join(str(value + 1) for value in mapping)
            out.write(f"  # PORT_MAP {slot + 1} {ports}\n")

    def _write_nodes(self, out: TextIO, positions: np.ndarray) -> None:
        for node_type, node, position in zip(
            self.node_tags,
            self.nodes,
            positions,
            strict=True,
        ):
            coordinates = " ".join(map(format_float, position))
            out.write(
                f"  NODE {node_type} {node.coordination} {coordinates}\n"
            )

    def _write_edges(
        self,
        out: TextIO,
        endpoints: list[tuple[np.ndarray, np.ndarray]],
    ) -> None:
        for index, (edge_type, (begin, end)) in enumerate(
            zip(self.colors, endpoints, strict=True),
            start=1,
        ):
            out.write(f"  # EDGE_TYPE {index} {edge_type}\n")
            begin_text = " ".join(map(format_float, begin))
            end_text = " ".join(map(format_float, end))
            out.write(f"  EDGE  {begin_text}   {end_text}\n")
        for begin, end in endpoints:
            center = " ".join(map(format_float, (begin + end) / 2))
            out.write(f"  # EDGE_CENTER  {center}\n")

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

'''
Serialize prepared topology geometry and metadata as CGD records.
'''

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import numpy as np


def format_float(value: float) -> str:
    '''
    Format a number with up to ten significant digits.

    Args:
        value (float):
            Number to place in a CGD record.

    Returns:
        The formatted string.
    '''

    return f"{value:.10g}"


@dataclass
class CgdWriter:
    '''
    Write prepared topology geometry with MOFBB symmetry and assembly metadata.

    Cell lengths are in angstroms and angles in degrees. Symmetry operations,
    node maps, and edge maps share the same operation order. Node tags and
    coordinations follow node order; colors follow edge order. Slot and port
    maps use zero-based indices internally and are written as one-based IDs.
    '''

    name: str
    cell: tuple[float, float, float, float, float, float]
    spacegroup: str
    net_spacegroup: str
    source_group: str | None
    source_compatible: bool | None
    rotations: np.ndarray
    translations: np.ndarray
    node_maps: np.ndarray
    edge_maps: np.ndarray
    node_types: dict[str, int]
    node_tags: list[int]
    coordinations: list[int]
    edge_tags: dict[str | None, int]
    colors: list[int]
    ports: dict[int, list[int]] | None

    def write(
        self,
        path: Path,
        positions: np.ndarray,
        endpoints: list[tuple[np.ndarray, np.ndarray]],
    ) -> None:
        '''
        Write prepared topology sites and metadata to a decorated CGD file.

        Args:
            path (Path):
                Output file. Parent directories are created if needed.
            positions (numpy.ndarray):
                (n, 3) fractional node positions in the same order as node_tags.
            endpoints (list of tuple):
                Pairs of (3,) fractional edge endpoints, including periodic images,
                in the same order as colors.

        Returns:
            None. The file is created or overwritten.
        '''
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as out:
            self._write_header(out)
            self._write_symmetry(out)
            self._write_bbs(out)
            self._write_ports(out)
            self._write_nodes(out, positions)
            self._write_edges(out, endpoints)
            out.write("END\n")

    def _write_header(self, out: TextIO) -> None:
        out.write("CRYSTAL\n")
        out.write(f"  NAME {self.name}\n")
        out.write(f"  GROUP {self.spacegroup}\n")
        out.write(f"  # NET_GROUP {self.net_spacegroup}\n")
        if self.source_group is not None:
            out.write(f"  # SOURCE_GROUP {self.source_group}\n")
            compatible = int(self.source_compatible)
            out.write(f"  # SOURCE_GROUP_COMPATIBLE {compatible}\n")
        cell = " ".join(f"{value:.16g}" for value in self.cell)
        out.write(f"  CELL {cell}\n")
        out.write("  # EXPANDED\n")

    def _write_symmetry(self, out: TextIO) -> None:
        for rotation, translation in zip(self.rotations, self.translations, strict=True):
            values = [
                *(str(int(value)) for value in rotation.ravel()),
                *(format_float(value) for value in translation),
            ]
            out.write(f"  # SYMOP {' '.join(values)}\n")

        for op_id, (node_map, edge_map) in enumerate(
            zip(self.node_maps, self.edge_maps, strict=True),
            start=1,
        ):
            mapping = np.concatenate((node_map, len(node_map) + edge_map)) + 1
            targets = " ".join(str(value) for value in mapping)
            out.write(f"  # SLOT_MAP {op_id} {targets}\n")

    def _write_bbs(self, out: TextIO) -> None:
        for bb_id, node_type in sorted(self.node_types.items(), key=lambda item: item[1]):
            out.write(f"  # NODE_BB {node_type} {bb_id}\n")

        for bb_id, edge_type in sorted(self.edge_tags.items(), key=lambda item: item[1]):
            value = bb_id if bb_id is not None else "NONE"
            out.write(f"  # EDGE_BB {edge_type} {value}\n")

    def _write_ports(self, out: TextIO) -> None:
        for slot, mapping in sorted((self.ports or {}).items()):
            ports = " ".join(str(value + 1) for value in mapping)
            out.write(f"  # PORT_MAP {slot + 1} {ports}\n")

    def _write_nodes(self, out: TextIO, positions: np.ndarray) -> None:
        for node_type, coordination, position in zip(
            self.node_tags, self.coordinations, positions, strict=True,
        ):
            coordinates = " ".join(map(format_float, position))
            out.write(f"  NODE {node_type} {coordination} {coordinates}\n")

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

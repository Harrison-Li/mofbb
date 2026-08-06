"""Decorated declarations embedded in CGD comments."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CgdMeta:
    """Optional declarations embedded in CGD comment records."""

    expanded: bool = False
    symops: list[dict[str, object]] = field(default_factory=list)
    edge_types: dict[int, int] = field(default_factory=dict)
    node_bbs: dict[int, str] = field(default_factory=dict)
    edge_bbs: dict[int, str | None] = field(default_factory=dict)
    slot_maps: dict[int, list[int]] = field(default_factory=dict)
    port_maps: dict[int, list[int]] = field(default_factory=dict)
    source_ok: bool | None = None

    @classmethod
    def from_lines(cls, lines: list[str]) -> CgdMeta:
        meta = cls()
        readers = {
            "SYMOP": meta._read_symop,
            "EDGE_TYPE": meta._read_edge_type,
            "NODE_BB": meta._read_node_bb,
            "EDGE_BB": meta._read_edge_bb,
            "SLOT_MAP": meta._read_slot_map,
            "PORT_MAP": meta._read_port_map,
            "SOURCE_GROUP_COMPATIBLE": meta._read_source_ok,
        }
        for line in lines:
            clean = line.strip()
            if not clean.startswith("#"):
                continue
            parts = clean[1:].split()
            if parts == ["EXPANDED"]:
                meta.expanded = True
            elif parts and parts[0] in readers:
                readers[parts[0]](parts)
        return meta

    def slots(self, count: int) -> list[list[int]] | None:
        if not self.slot_maps:
            return None
        expected = set(range(len(self.symops)))
        if set(self.slot_maps) != expected:
            raise ValueError(
                "CGD SLOT_MAP operation IDs must match the declared SYMOP records."
            )

        maps = []
        for op in range(len(self.symops)):
            mapping = self.slot_maps[op]
            if len(mapping) != count:
                raise ValueError(
                    f"CGD SLOT_MAP {op + 1} has {len(mapping)} "
                    f"targets; expected {count}."
                )
            if sorted(mapping) != list(range(count)):
                raise ValueError(
                    f"CGD SLOT_MAP {op + 1} is not a slot permutation."
                )
            maps.append(mapping)
        return maps

    def check_ports(self, cn: list[int]) -> None:
        for slot, mapping in self.port_maps.items():
            if slot >= len(cn):
                raise ValueError(f"CGD PORT_MAP references missing slot {slot + 1}.")
            count = cn[slot]
            if sorted(mapping) != list(range(count)):
                raise ValueError(
                    f"CGD PORT_MAP {slot + 1} is not a permutation of "
                    f"{count} ports."
                )

    def add_info(
        self,
        info: dict[str, object],
        maps: list[list[int]] | None,
    ) -> None:
        if self.symops:
            info["symmetry_operations"] = self.symops
        if maps is not None:
            info["slot_mappings"] = maps
        if self.node_bbs:
            info["node_bb_ids"] = self.node_bbs
        if self.edge_bbs:
            info["edge_bb_ids"] = self.edge_bbs
        if self.port_maps:
            info["port_mappings"] = self.port_maps
        if self.source_ok is not None:
            info["source_group_compatible"] = self.source_ok

    def _read_symop(self, parts: list[str]) -> None:
        if len(parts) != 13:
            raise ValueError("A CGD SYMOP comment must contain 12 numbers.")
        values = np.asarray(parts[1:], dtype=float)
        self.symops.append(
            {
                "rotation": values[:9].reshape(3, 3).tolist(),
                "translation": values[9:].tolist(),
            }
        )

    def _read_edge_type(self, parts: list[str]) -> None:
        if len(parts) != 3:
            raise ValueError("A CGD EDGE_TYPE comment requires edge and type IDs.")
        edge, kind = (int(value) for value in parts[1:])
        if edge < 1 or kind < 1:
            raise ValueError("CGD EDGE_TYPE IDs must be positive.")
        self.edge_types[edge - 1] = kind

    def _read_node_bb(self, parts: list[str]) -> None:
        if len(parts) != 3:
            raise ValueError("A CGD NODE_BB comment requires a type and BB ID.")
        kind = int(parts[1])
        if kind < 1:
            raise ValueError("CGD NODE_BB type IDs must be positive.")
        self.node_bbs[kind - 1] = parts[2]

    def _read_edge_bb(self, parts: list[str]) -> None:
        if len(parts) != 3:
            raise ValueError("A CGD EDGE_BB comment requires a type and BB ID.")
        kind = int(parts[1])
        if kind < 1:
            raise ValueError("CGD EDGE_BB type IDs must be positive.")
        self.edge_bbs[-kind] = None if parts[2].upper() == "NONE" else parts[2]

    def _read_slot_map(self, parts: list[str]) -> None:
        if len(parts) < 3:
            raise ValueError(
                "A CGD SLOT_MAP comment requires an operation and targets."
            )
        op = int(parts[1])
        targets = [int(value) for value in parts[2:]]
        if op < 1 or any(target < 1 for target in targets):
            raise ValueError("CGD SLOT_MAP IDs must be positive.")
        self.slot_maps[op - 1] = [target - 1 for target in targets]

    def _read_port_map(self, parts: list[str]) -> None:
        if len(parts) < 2:
            raise ValueError("A CGD PORT_MAP comment requires a slot ID.")
        slot = int(parts[1])
        targets = [int(value) for value in parts[2:]]
        if slot < 1 or any(target < 1 for target in targets):
            raise ValueError("CGD PORT_MAP IDs must be positive.")
        self.port_maps[slot - 1] = [target - 1 for target in targets]

    def _read_source_ok(self, parts: list[str]) -> None:
        if len(parts) != 2:
            raise ValueError(
                "A CGD SOURCE_GROUP_COMPATIBLE comment requires one flag."
            )
        self.source_ok = bool(int(parts[1]))

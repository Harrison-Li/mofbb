"""Required geometry records from the non-comment CGD body."""

from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np

from mofbb.compose.formats.meta import CgdMeta


@dataclass
class CgdData:
    """Geometry and required records from the non-comment CGD body."""

    name: str
    spacegroup: str
    cell: np.ndarray
    node_pos: np.ndarray
    node_labels: list[str]
    node_cn: list[int]
    edge_pos: np.ndarray
    edge_ends: list[list[list[float]]]

    @classmethod
    def from_lines(cls, lines: list[str]) -> CgdData:
        # Get topology name.
        name = lines[0].split()[1]
        # Get spacegroup.
        group = lines[1].split()[1]

        # Get cell paremeters and expand cell lengths by 10.
        cell = np.array(lines[2].split()[1:], dtype=float)

        # Parse node information.
        node_pos = []
        labels = []
        node_cn = []
        for line in lines[3:]:
            parts = line.split()
            if parts[0] != "NODE":
                continue
            cn = int(parts[2])
            pos = [float(value) for value in parts[3:]]
            labels.append(parts[1])
            node_pos.append(pos)
            node_cn.append(cn)
        node_pos = np.array(node_pos)
        # node_cn = np.array(node_cn)

        # Parse edge information.
        edge_pos = []
        edge_ends = []
        for line in lines[3:]:
            parts = line.split()
            if parts[0] != "EDGE":
                continue
            begin = np.array([float(value) for value in parts[1:4]])
            end = np.array([float(value) for value in parts[4:]])
            edge_pos.append(0.5 * (begin + end))
            edge_ends.append([begin.tolist(), end.tolist()])

        # New feature. Read EDGE_CENTER.
        for line in lines[3:]:
            parts = line.split()
            if parts[0] != "EDGE_CENTER":
                continue
            edge_pos.append(np.array([float(value) for value in parts[1:]]))

        return cls(
            name=name,
            spacegroup=group,
            cell=cell,
            node_pos=node_pos,
            node_labels=labels,
            node_cn=node_cn,
            edge_pos=np.array(edge_pos),
            edge_ends=edge_ends,
        )

    @property
    def num_nodes(self) -> int:
        return self.node_pos.shape[0]

    @property
    def num_edges(self) -> int:
        return self.edge_pos.shape[0]

    def species(self, node: str, edge: str) -> np.ndarray:
        # Carbon for nodes, oxygen for edges.
        return np.concatenate(
            [
                np.full(shape=self.num_nodes, fill_value=node),
                np.full(shape=self.num_edges, fill_value=edge),
            ]
        )

    def coords(self) -> np.ndarray:
        return np.concatenate([self.node_pos, self.edge_pos], axis=0)

    def node_types(self) -> list[int]:
        # Pymatget can handle : indicator in spacegroup.
        # Mark symmetrically equivalent sites.
        numeric = {}
        used = set()
        for label in self.node_labels:
            match = re.fullmatch(r"[Vv]?([0-9]+)", label)
            if match:
                node_id = int(match.group(1))
                if node_id < 1:
                    raise ValueError("CGD NODE IDs must be positive.")
                numeric[label] = node_id - 1
                used.add(node_id - 1)

        named = {}
        types = []
        for label in self.node_labels:
            if label in numeric:
                types.append(numeric[label])
                continue
            if label not in named:
                kind = 0
                while kind in used:
                    kind += 1
                named[label] = kind
                used.add(kind)
            types.append(named[label])
        return types

    def props(self, meta: CgdMeta) -> dict[str, list[int]]:
        node_types = self.node_types()
        self._check_cn(node_types)
        edge_types = [
            -meta.edge_types.get(i, i + 1) for i in range(self.num_edges)
        ]
        return {
            "type": node_types + edge_types,
            "cn": self.node_cn + [2 for _ in range(self.num_edges)],
        }

    def _check_cn(self, node_types: list[int]) -> None:
        by_type = {}
        for kind, cn in zip(node_types, self.node_cn, strict=True):
            old = by_type.setdefault(kind, cn)
            if old != cn:
                raise ValueError(
                    f"CGD NODE type {kind} declares both CN {old} and CN {cn}."
                )

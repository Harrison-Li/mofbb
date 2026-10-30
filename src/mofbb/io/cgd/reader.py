'''
CGD topology reader.
'''

from __future__ import annotations

import re
from pathlib import Path

import ase
import ase.neighborlist
from ase.geometry import cellpar_to_cell
import numpy as np
import pymatgen.core as mg


class CgdReader:
    '''
    Read a CGD topology, including MOFBB building-block and port metadata.

    Args:
        path (str or Path):
            CGD file to read when read() is called.
    '''

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.expanded = False
        self.symops = []
        self.edge_types = {}
        self.node_bbs = {}
        self.edge_bbs = {}
        self.slot_maps = {}
        self.port_maps = {}
        self.source_group_compatible = None

    def read(self) -> ase.Atoms:
        '''
        Load topology sites and expand space-group symmetry when needed.

        Returns:
            An ase.Atoms object with C placeholders for nodes and O placeholders
            for edge centers. Tags identify site types; info holds coordination
            numbers and available MOFBB metadata.
        '''
        lines = self.path.read_text(encoding="utf-8").splitlines()[1:-1]
        self._parse_comments(lines)
        self._parse_body([line for line in lines if not line.lstrip().startswith("#")])
        types = self._node_types()
        self._check_coordination(types)
        properties = {
            "type": types
            + [-self.edge_types.get(i, i + 1) for i in range(len(self.edge_pos))],
            "cn": self.node_cn + [2] * len(self.edge_pos),
        }
        self._check_ports(properties["cn"])
        mappings = self._slot_mappings(len(properties["cn"]))
        group = self.spacegroup.replace(" ", "")
        if group == "P1" or self.expanded:
            return self._explicit_atoms(properties, mappings)
        return self._expanded_atoms(properties)

    def _parse_body(self, lines: list[str]) -> None:
        self.name = lines[0].split()[1]
        self.spacegroup = lines[1].split()[1]
        self.cell = np.asarray(lines[2].split()[1:], dtype=float)
        self.node_pos, self.node_labels, self.node_cn = [], [], []
        self.edge_pos, self.edge_ends = [], []
        for line in lines[3:]:
            parts = line.split()
            if parts[0] == "NODE":
                self.node_labels.append(parts[1])
                self.node_cn.append(int(parts[2]))
                self.node_pos.append([float(value) for value in parts[3:]])
            elif parts[0] == "EDGE":
                begin = np.asarray(parts[1:4], dtype=float)
                end = np.asarray(parts[4:], dtype=float)
                self.edge_pos.append((begin + end) / 2)
                self.edge_ends.append([begin.tolist(), end.tolist()])
            elif parts[0] == "EDGE_CENTER":
                self.edge_pos.append(np.asarray(parts[1:], dtype=float))
        self.node_pos = np.asarray(self.node_pos, dtype=float).reshape(-1, 3)
        self.edge_pos = np.asarray(self.edge_pos, dtype=float).reshape(-1, 3)

    def _parse_comments(self, lines: list[str]) -> None:
        """Read MOFBB metadata; CGD identifiers are one-based on disk."""
        for line in lines:
            if not line.lstrip().startswith("#"):
                continue
            parts = line.lstrip()[1:].split()
            if not parts:
                continue
            record, *values = parts
            match record:
                case "EXPANDED":
                    if not values:
                        self.expanded = True
                case "SYMOP":
                    if len(values) != 12:
                        raise ValueError("A CGD SYMOP comment must contain 12 numbers.")
                    operation = np.asarray(values, dtype=float)
                    self.symops.append({
                        "rotation": operation[:9].reshape(3, 3).tolist(),
                        "translation": operation[9:].tolist(),
                    })
                case "EDGE_TYPE":
                    edge, kind = self._positive_ids(values, record)
                    self.edge_types[edge - 1] = kind
                case "NODE_BB" | "EDGE_BB":
                    if len(values) != 2 or int(values[0]) < 1:
                        raise ValueError(
                            f"A CGD {record} comment requires a positive type and BB ID."
                        )
                    kind, block = int(values[0]), values[1]
                    if record == "NODE_BB":
                        self.node_bbs[kind - 1] = block
                    else:
                        self.edge_bbs[-kind] = None if block.upper() == "NONE" else block
                case "SLOT_MAP" | "PORT_MAP":
                    minimum = 2 if record == "SLOT_MAP" else 1
                    slot, *mapping = self._positive_ids(values, record, minimum)
                    target = self.slot_maps if record == "SLOT_MAP" else self.port_maps
                    target[slot - 1] = [value - 1 for value in mapping]
                case "SOURCE_GROUP_COMPATIBLE":
                    if len(values) != 1:
                        raise ValueError(
                            "A CGD SOURCE_GROUP_COMPATIBLE comment requires one flag."
                        )
                    self.source_group_compatible = bool(int(values[0]))

    @staticmethod
    def _positive_ids(
        values: list[str], record: str, minimum: int = 2
    ) -> list[int]:
        if len(values) < minimum:
            raise ValueError(f"A CGD {record} comment requires {minimum} IDs.")
        ids = [int(value) for value in values]
        if any(value < 1 for value in ids):
            raise ValueError(f"CGD {record} IDs must be positive.")
        return ids

    def _node_types(self) -> list[int]:
        by_label, used = {}, set()
        for label in self.node_labels:
            match = re.fullmatch(r"[Vv]?(\d+)", label)
            if match:
                kind = int(match.group(1)) - 1
                if kind < 0:
                    raise ValueError("CGD NODE IDs must be positive.")
                by_label[label] = kind
                used.add(kind)
        result = []
        for label in self.node_labels:
            if label not in by_label:
                kind = next(
                    value
                    for value in range(len(used) + 1)
                    if value not in used
                )
                by_label[label] = kind
                used.add(kind)
            result.append(by_label[label])
        return result

    def _check_coordination(self, types: list[int]) -> None:
        by_type = {}
        for kind, cn in zip(types, self.node_cn):
            previous = by_type.setdefault(kind, cn)
            if previous != cn:
                raise ValueError(
                    f"CGD NODE type {kind} declares both CN {previous} and CN {cn}."
                )

    def _check_ports(self, coordination: list[int]) -> None:
        for slot, mapping in self.port_maps.items():
            if slot >= len(coordination):
                raise ValueError(f"CGD PORT_MAP references missing slot {slot + 1}.")
            if sorted(mapping) != list(range(coordination[slot])):
                raise ValueError(
                    f"CGD PORT_MAP {slot + 1} is not a permutation of "
                    f"{coordination[slot]} ports."
                )

    def _slot_mappings(self, count: int) -> list[list[int]] | None:
        if not self.slot_maps:
            return None
        if set(self.slot_maps) != set(range(len(self.symops))):
            raise ValueError(
                "CGD SLOT_MAP operation IDs must match the declared SYMOP records."
            )
        mappings = [self.slot_maps[op] for op in range(len(self.symops))]
        for op, mapping in enumerate(mappings, start=1):
            if sorted(mapping) != list(range(count)):
                raise ValueError(f"CGD SLOT_MAP {op} is not a {count}-slot permutation.")
        return mappings

    def _info(
        self, properties: dict[str, list[int]], mappings: list[list[int]] | None
    ) -> dict[str, object]:
        info = {
            "spacegroup": self.spacegroup,
            "name": self.name,
            "cn": properties["cn"],
        }
        optional = {
            "symmetry_operations": self.symops,
            "slot_mappings": mappings,
            "node_bb_ids": self.node_bbs,
            "edge_bb_ids": self.edge_bbs,
            "port_mappings": self.port_maps,
        }
        info.update({key: value for key, value in optional.items() if value})
        if self.source_group_compatible is not None:
            info["source_group_compatible"] = self.source_group_compatible
        if len(self.edge_ends) == len(self.edge_pos):
            info["edge_endpoints"] = self.edge_ends
        return info

    def _explicit_atoms(
        self, properties: dict[str, list[int]], mappings: list[list[int]] | None
    ) -> ase.Atoms:
        coordinates = np.concatenate((self.node_pos, self.edge_pos))
        return ase.Atoms(
            symbols=["C"] * len(self.node_pos) + ["O"] * len(self.edge_pos),
            scaled_positions=np.mod(coordinates, 1.0),
            cell=cellpar_to_cell(self.cell),
            tags=properties["type"],
            pbc=True,
            info=self._info(properties, mappings),
        )

    def _expanded_atoms(self, properties: dict[str, list[int]]) -> ase.Atoms:
        group = "Cmce" if self.spacegroup == "Cmca" else self.spacegroup
        structure = mg.Structure.from_spacegroup(
            sg=group,
            lattice=mg.Lattice.from_parameters(*self.cell),
            species=["C"] * len(self.node_pos) + ["O"] * len(self.edge_pos),
            coords=np.concatenate((self.node_pos, self.edge_pos)),
            site_properties=properties,
        )
        atoms = ase.Atoms(
            symbols=[species.name for species in structure.species],
            positions=structure.cart_coords,
            cell=structure.lattice.matrix,
            tags=structure.site_properties["type"],
            pbc=True,
            info={
                "spacegroup": group,
                "name": self.name,
                "cn": structure.site_properties["cn"],
            },
        )
        self._remove_overlaps(atoms)
        return atoms

    @staticmethod
    def _remove_overlaps(atoms: ase.Atoms) -> None:
        first, second = ase.neighborlist.neighbor_list("ij", atoms, cutoff=0.1)
        tags = atoms.get_tags()
        removed = {j for i, j in zip(first, second) if j > i and tags[i] == tags[j]}
        if not removed:
            return
        del atoms[sorted(removed)]
        atoms.info["cn"] = [
            cn for index, cn in enumerate(atoms.info["cn"]) if index not in removed
        ]


def read_topology_geometry(path: Path) -> tuple[np.ndarray, list[np.ndarray]]:
    '''
    Read the unit cell and listed node positions without symmetry expansion.

    Args:
        path (Path):
            CGD file containing CELL and NODE records.

    Returns:
        A (3, 3) cell matrix in angstroms and a list of (3,) fractional node
        coordinates. Missing cell or node records raise ValueError.
    '''
    cell_parameters = None
    vertices = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        tokens = raw_line.split()
        if not tokens:
            continue
        if tokens[0] == "CELL":
            cell_parameters = [float(value) for value in tokens[1:]]
        elif tokens[0] == "NODE":
            vertices.append(np.asarray(tokens[3:], dtype=float))

    if cell_parameters is None:
        raise ValueError(f"Topology {path} does not contain a CELL record.")
    if not vertices:
        raise ValueError(f"Topology {path} does not contain any NODE records.")
    return cellpar_to_cell(cell_parameters), vertices

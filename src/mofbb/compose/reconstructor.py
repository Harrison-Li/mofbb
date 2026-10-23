"""Exact reconstruction from a decomposition output directory."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import ase
import numpy as np
from ase.geometry import cellpar_to_cell

from mofbb.compose.building_block import BuildingBlock
from mofbb.compose.framework import Framework
from mofbb.compose.topology import Topology
from mofbb.compose.utils import wrap_frac


def _read_topology_geometry(path: Path) -> tuple[np.ndarray, list[np.ndarray]]:
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


class Reconstructor:
    """Rebuild exact source geometry from decomposition artifacts."""

    def __init__(self, decomp_dir: str | Path) -> None:
        self.decomp_dir = Path(decomp_dir)
        self.map_path = self.decomp_dir / "reconstruction_map.json"

    def reconstruct(self) -> Framework:
        """Return the reconstructed periodic framework."""

        data = json.loads(self.map_path.read_text(encoding="utf-8"))
        topology = data["topology"]
        topology_path = topology.get("path")
        slots = sorted(data["slots"], key=lambda slot: slot["slot_id"])
        exact_vertices = [slot.get("exact_frac_position") for slot in slots]
        if all(vertex is not None for vertex in exact_vertices):
            cell = cellpar_to_cell(topology["cell"])
            vertices = [np.asarray(vertex, dtype=float) for vertex in exact_vertices]
        elif any(vertex is not None for vertex in exact_vertices):
            raise ValueError(
                "Reconstruction slots must either all define "
                "exact_frac_position or all use topology NODE positions."
            )
        elif topology_path:
            cell, vertices = _read_topology_geometry(
                self.decomp_dir / topology_path
            )
        else:
            raise ValueError(
                "Reconstruction requires exact slot positions when topology "
                "output is disabled."
            )
        if len(slots) != len(vertices):
            raise ValueError(
                f"Reconstruction map has {len(slots)} slots, but topology has "
                f"{len(vertices)} vertices."
            )

        symbols_by_source: dict[int, str] = {}
        positions_by_source: dict[int, np.ndarray] = {}
        bond_types_by_source_pair: dict[tuple[int, int], str] = {}
        cap_bond_types: dict[tuple[int, int], str] = {}

        for slot, vertex in zip(slots, vertices):
            slot_id = int(slot["slot_id"])
            block = BuildingBlock(self.decomp_dir / slot["exact_instance_path"])
            source_indices = [int(value) for value in slot["source_indices"]]
            source_count = len(source_indices)
            block_symbols = block.atoms.get_chemical_symbols()
            if len(block_symbols) - block_symbols.count("X") != source_count:
                raise ValueError(
                    f"Slot {slot_id} source-index count does not match its "
                    "non-X atom count."
                )
            if any(symbol == "X" for symbol in block_symbols[:source_count]):
                raise ValueError(
                    f"Slot {slot_id} does not store source atoms before X caps."
                )

            source_positions = block.atoms.positions[:source_count]
            source_centroid = np.mean(source_positions, axis=0)
            target_centroid = np.mod(vertex, 1.0) @ cell
            placed_positions = source_positions - source_centroid + target_centroid
            for source_index, symbol, position in zip(
                source_indices,
                block_symbols[:source_count],
                placed_positions,
            ):
                if source_index in positions_by_source:
                    raise ValueError(
                        f"Source atom {source_index} occurs in multiple slots."
                    )
                symbols_by_source[source_index] = symbol
                positions_by_source[source_index] = position

            for (left, right), bond_type in zip(block.bonds, block.bond_types):
                if left < source_count and right < source_count:
                    source_pair = tuple(
                        sorted((source_indices[left], source_indices[right]))
                    )
                    bond_types_by_source_pair[source_pair] = bond_type
                elif (left < source_count) != (right < source_count):
                    x_index = right if left < source_count else left
                    cap_bond_types[(slot_id, x_index)] = bond_type

        for edge in data["edges"]:
            source_pair = tuple(sorted(int(value) for value in edge["source_cut_bond"]))
            endpoint_types = [
                cap_bond_types.get((int(endpoint["slot_id"]), int(endpoint["x_index"])))
                for endpoint in edge["endpoint_x_sites"]
            ]
            known_types = {value for value in endpoint_types if value is not None}
            if len(known_types) > 1:
                raise ValueError(
                    f"Edge {edge['edge_id']} has conflicting endpoint bond types."
                )
            bond_types_by_source_pair[source_pair] = (
                next(iter(known_types)) if known_types else "S"
            )

        source_order = sorted(positions_by_source)
        source_to_output = {
            source_index: output_index
            for output_index, source_index in enumerate(source_order)
        }
        atoms = ase.Atoms(
            symbols=[symbols_by_source[index] for index in source_order],
            positions=[positions_by_source[index] for index in source_order],
            cell=cell,
            pbc=True,
        )
        atoms.set_scaled_positions(
            wrap_frac(atoms.get_scaled_positions(wrap=False))
        )

        sorted_bonds = sorted(bond_types_by_source_pair.items())
        bonds = np.asarray(
            [
                (source_to_output[left], source_to_output[right])
                for (left, right), _ in sorted_bonds
            ],
            dtype=int,
        ).reshape(-1, 2)
        bond_types = [bond_type for _, bond_type in sorted_bonds]
        return Framework(
            atoms,
            bonds,
            bond_types,
            info={
                "composition_mode": "exact-decomposition",
                "decomposition_dir": str(self.decomp_dir),
            },
            wrap=False,
        )

@dataclass
class TopoInput:
    """Topology and grouped BB templates loaded without reconstruction data."""

    topology: Topology
    templates: dict[str, BuildingBlock]
    slot_bb_ids: tuple[str | None, ...]
    template_paths: dict[str, str]

    @classmethod
    def from_dir(cls, decomp_dir: str | Path):
        """Read ``topology.cgd`` and grouped BBs from a decomposition directory."""

        root = Path(decomp_dir)
        summary = json.loads((root / "decomposition_summary.json").read_text(encoding="utf-8"))
        topology_record = summary.get("topology")
        if not topology_record:
            raise ValueError(
                f"Decomposition directory {root} has no abstract topology."
            )
        topology_path = root / topology_record.get("path", "topology.cgd")
        topology = Topology(topology_path)

        group_paths = {}
        for group in summary.get("groups", ()):
            bb_id = str(group["bb_id"])
            if bb_id in group_paths:
                raise ValueError(f"Duplicate grouped BB identifier {bb_id!r}.")
            group_paths[bb_id] = str(group["path"])

        node_bb_ids = topology.atoms.info.get("node_bb_ids", {})
        edge_bb_ids = topology.atoms.info.get("edge_bb_ids", {})
        slot_bb_ids = []
        for slot, bb_type in enumerate(topology.atoms.get_tags()):
            bb_ids = node_bb_ids if bb_type >= 0 else edge_bb_ids
            if bb_type not in bb_ids:
                raise ValueError(
                    f"Abstract topology slot {slot} has no declared BB type."
                )
            bb_id = bb_ids[bb_type]
            if bb_id is not None and bb_id not in group_paths:
                raise ValueError(
                    f"Abstract topology slot {slot} references missing BB "
                    f"group {bb_id!r}."
                )
            slot_bb_ids.append(bb_id)

        referenced = sorted({bb_id for bb_id in slot_bb_ids if bb_id is not None})
        templates = {
            bb_id: BuildingBlock(root / group_paths[bb_id])
            for bb_id in referenced
        }
        for slot, bb_id in enumerate(slot_bb_ids):
            if bb_id is None:
                continue
            expected = topology.cn[slot]
            actual = templates[bb_id].n_ports
            if actual != expected:
                raise ValueError(
                    f"Topology slot {slot} requires {expected} connection points, "
                    f"but BB group {bb_id!r} has {actual}."
                )
        return cls(
            topology=topology,
            templates=templates,
            slot_bb_ids=tuple(slot_bb_ids),
            template_paths={bb_id: group_paths[bb_id] for bb_id in referenced},
        )

    def copy_bbs(self) -> list[BuildingBlock | None]:
        """Return independent per-slot copies of the grouped templates."""

        return [
            None if bb_id is None else self.templates[bb_id].copy()
            for bb_id in self.slot_bb_ids
        ]

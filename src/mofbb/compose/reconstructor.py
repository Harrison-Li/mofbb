"""Exact reconstruction from a decomposition output directory."""

from __future__ import annotations

import json
from pathlib import Path

import ase
from ase.geometry import cellpar_to_cell
import numpy as np

from .building_block import BuildingBlock
from .framework import Framework


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

    def __init__(self, decomposition_dir: str | Path) -> None:
        self.decomposition_dir = Path(decomposition_dir)
        self.map_path = self.decomposition_dir / "reconstruction_map.json"

    def reconstruct(self) -> Framework:
        """Return the reconstructed periodic framework."""

        if not self.map_path.is_file():
            raise ValueError(
                f"Decomposition directory {self.decomposition_dir} does not "
                "contain reconstruction_map.json."
            )
        data = json.loads(self.map_path.read_text(encoding="utf-8"))
        topology_path = self.decomposition_dir / data["topology"]["path"]
        cell, vertices = _read_topology_geometry(topology_path)
        slots = sorted(data["slots"], key=lambda slot: slot["slot_id"])
        exact_vertices = [slot.get("exact_frac_position") for slot in slots]
        if all(vertex is not None for vertex in exact_vertices):
            vertices = [np.asarray(vertex, dtype=float) for vertex in exact_vertices]
        elif any(vertex is not None for vertex in exact_vertices):
            raise ValueError(
                "Reconstruction slots must either all define "
                "exact_frac_position or all use topology NODE positions."
            )
        elif len(slots) != len(vertices):
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
            relative_path = slot.get("exact_instance_path") or slot.get(
                "debug_instance"
            )
            if relative_path is None:
                raise ValueError(f"Slot {slot_id} has no exact instance path.")
            block = BuildingBlock(self.decomposition_dir / relative_path)
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
            placed_positions = (
                source_positions - source_centroid + target_centroid
            )
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
                left = int(left)
                right = int(right)
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
                cap_bond_types.get(
                    (int(endpoint["slot_id"]), int(endpoint["x_index"]))
                )
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
            np.mod(atoms.get_scaled_positions(wrap=False), 1.0)
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
                "decomposition_dir": str(self.decomposition_dir),
            },
            wrap=False,
        )

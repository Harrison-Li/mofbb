"""Distance-based bond graph construction for MOF decomposition."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
import re
import shlex
import warnings

import ase
import ase.io
from ase.geometry import find_mic
import networkx as nx
import numpy as np

try:  # ASE has moved this helper across versions.
    from ase.neighborlist import natural_cutoffs, neighbor_list
except ImportError:  # pragma: no cover - compatibility for unusual ASE layouts
    from ase.utils import natural_cutoffs  # type: ignore[attr-defined]
    from ase.neighborlist import neighbor_list

from .constants import is_mofid_metal


@dataclass(frozen=True)
class BondNetwork:
    """Undirected minimum-image bond graph for an ASE structure.

    Attributes
    ----------
    adjacency
        Atom-neighbor sets indexed by source atom index.
    vectors
        Minimum-image vectors keyed by sorted atom-index pairs.
    """

    adjacency: tuple[frozenset[int], ...]
    vectors: dict[tuple[int, int], np.ndarray]
    bond_types: dict[tuple[int, int], str] = field(default_factory=dict)

    @property
    def pairs(self) -> list[tuple[int, int]]:
        """Return sorted bonded atom-index pairs."""

        return sorted(self.vectors)

    def vector(self, i: int, j: int) -> np.ndarray:
        """Return the minimum-image vector from atom ``i`` to atom ``j``."""

        if i < j:
            return self.vectors[(i, j)]
        return -self.vectors[(j, i)]

    def bond_type(self, i: int, j: int) -> str:
        """Return the source bond label for a bonded atom pair."""

        return self.bond_types.get(tuple(sorted((i, j))), "S")


def build_bond_network(atoms: ase.Atoms, scale: float = 1.2) -> BondNetwork:
    """Build an undirected minimum-image bond graph from covalent cutoffs.

    Parameters
    ----------
    atoms
        Structure to analyze.
    scale
        Multiplier applied to ASE natural covalent cutoffs.

    Returns
    -------
    BondNetwork
        Bond adjacency and minimum-image vectors.
    """

    cutoffs = [scale * cutoff for cutoff in natural_cutoffs(atoms)]
    i_list, j_list, d_list = neighbor_list(
        "ijD", atoms, cutoff=cutoffs, self_interaction=False
    )

    vector_by_pair: dict[tuple[int, int], np.ndarray] = {}
    for i, j, vector in zip(i_list, j_list, d_list):
        atom_i = int(i)
        atom_j = int(j)
        if atom_i == atom_j:
            continue
        a, b = sorted((atom_i, atom_j))
        pair_vector = np.array(vector if atom_i == a else -vector, dtype=float)
        previous = vector_by_pair.get((a, b))
        if previous is None or np.linalg.norm(pair_vector) < np.linalg.norm(previous):
            vector_by_pair[(a, b)] = pair_vector

    vector_by_pair = _repair_inferred_bonds(atoms, vector_by_pair)

    adjacency = [set() for _ in atoms]
    for i, j in vector_by_pair:
        adjacency[i].add(j)
        adjacency[j].add(i)

    return BondNetwork(
        adjacency=tuple(frozenset(neighbors) for neighbors in adjacency),
        vectors=vector_by_pair,
    )


def build_bond_network_from_cif(cif: str | Path, atoms: ase.Atoms) -> BondNetwork | None:
    """Build a bond graph from a CIF ``_geom_bond`` loop when present."""

    loops = list(_cif_loops(Path(cif)))
    label_to_index = _atom_label_to_index(loops)
    if not label_to_index:
        return None
    if len(label_to_index) != len(atoms):
        # ASE may expand symmetry-equivalent sites or remove duplicate sites
        # while reading a CIF.  In either case, atom-row indices no longer map
        # one-to-one onto the parsed Atoms object, so the raw bond loop is
        # unsafe to use.  Let the caller fall back to distance-based bonding.
        return None

    bond_loop = _geom_bond_loop(loops)
    if bond_loop is None:
        return None

    headers, rows = bond_loop
    label_1_index = headers.index("_geom_bond_atom_site_label_1")
    label_2_index = headers.index("_geom_bond_atom_site_label_2")
    symmetry_index = (
        headers.index("_geom_bond_site_symmetry_2")
        if "_geom_bond_site_symmetry_2" in headers
        else None
    )
    bond_type_index = (
        headers.index("_ccdc_geom_bond_type")
        if "_ccdc_geom_bond_type" in headers
        else None
    )

    vector_by_pair: dict[tuple[int, int], np.ndarray] = {}
    bond_types: dict[tuple[int, int], str] = {}
    cell = np.array(atoms.cell.array, dtype=float)
    for row in rows:
        if len(row) <= max(label_1_index, label_2_index):
            continue
        i = _atom_index_from_label(row[label_1_index], label_to_index)
        j = _atom_index_from_label(row[label_2_index], label_to_index)
        if i is None or j is None or i == j:
            continue

        image = (
            _site_symmetry_offset(row[symmetry_index])
            if symmetry_index is not None and len(row) > symmetry_index
            else np.zeros(3, dtype=int)
        )
        vector = atoms.positions[j] + image @ cell - atoms.positions[i]
        a, b = sorted((i, j))
        pair_vector = np.array(vector if i == a else -vector, dtype=float)
        pair = (a, b)
        previous = vector_by_pair.get(pair)
        if previous is None or np.linalg.norm(pair_vector) < np.linalg.norm(previous):
            vector_by_pair[pair] = pair_vector
            if bond_type_index is not None and len(row) > bond_type_index:
                bond_types[pair] = row[bond_type_index]

    if not vector_by_pair:
        return None

    adjacency = [set() for _ in atoms]
    for i, j in vector_by_pair:
        adjacency[i].add(j)
        adjacency[j].add(i)

    return BondNetwork(
        adjacency=tuple(frozenset(neighbors) for neighbors in adjacency),
        vectors=vector_by_pair,
        bond_types=bond_types,
    )


def read_bonded_structure(
    cif: str | Path,
    scale: float = 1.2,
    cleanup: bool = True,
) -> tuple[ase.Atoms, BondNetwork]:
    """Read a CIF and return atoms with the best available bond graph."""

    atoms = ase.io.read(str(cif))
    bonds = build_bond_network_from_cif(cif, atoms)
    if bonds is None:
        bonds = build_bond_network(atoms, scale=scale)
    if not cleanup:
        return atoms, bonds
    return largest_connected_component_with_bonds(atoms, bonds)


def largest_connected_component(atoms: ase.Atoms, scale: float = 1.2) -> ase.Atoms:
    """Return a copy containing only the largest bonded component.

    Structures from databases sometimes include solvents, counterions, or other
    disconnected species.  Keeping this helper in the bonding module makes that
    cleanup behavior explicit and reusable.
    """

    bonds = build_bond_network(atoms, scale=scale)
    graph = nx.Graph()
    graph.add_nodes_from(range(len(atoms)))
    graph.add_edges_from(bonds.pairs)

    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    if len(components) <= 1:
        return atoms
    return atoms[sorted(components[0])]


def subset_bonded_structure(
    atoms: ase.Atoms,
    bonds: BondNetwork,
    indices: Iterable[int],
) -> tuple[ase.Atoms, BondNetwork, tuple[int, ...]]:
    """Return an ordered atom subset with its reindexed bond graph."""

    selected_to_original = tuple(int(index) for index in indices)
    if len(set(selected_to_original)) != len(selected_to_original):
        raise ValueError("Bonded-structure subset indices must be unique.")
    if any(index < 0 or index >= len(atoms) for index in selected_to_original):
        raise ValueError("Bonded-structure subset index is out of range.")

    reindex = {old: new for new, old in enumerate(selected_to_original)}
    selected = set(selected_to_original)
    vectors: dict[tuple[int, int], np.ndarray] = {}
    bond_types: dict[tuple[int, int], str] = {}
    for i, j in bonds.pairs:
        if i not in selected or j not in selected:
            continue
        new_i = reindex[i]
        new_j = reindex[j]
        pair = tuple(sorted((new_i, new_j)))
        vector = bonds.vector(i, j)
        vectors[pair] = np.array(vector if new_i < new_j else -vector, dtype=float)
        bond_types[pair] = bonds.bond_type(i, j)

    adjacency = [set() for _ in selected_to_original]
    for i, j in vectors:
        adjacency[i].add(j)
        adjacency[j].add(i)
    subset_bonds = BondNetwork(
        adjacency=tuple(frozenset(neighbors) for neighbors in adjacency),
        vectors=vectors,
        bond_types=bond_types,
    )
    return atoms[list(selected_to_original)], subset_bonds, selected_to_original


def largest_connected_component_with_bonds(
    atoms: ase.Atoms,
    bonds: BondNetwork,
) -> tuple[ase.Atoms, BondNetwork]:
    """Return the largest bonded component and a reindexed bond graph."""

    graph = nx.Graph()
    graph.add_nodes_from(range(len(atoms)))
    graph.add_edges_from(bonds.pairs)

    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    if len(components) <= 1:
        return atoms, bonds

    _warn_about_dropped_components(atoms, components)
    kept = sorted(components[0])
    kept_atoms, kept_bonds, _ = subset_bonded_structure(atoms, bonds, kept)
    return kept_atoms, kept_bonds


def _cif_loops(path: Path) -> Iterator[tuple[list[str], list[list[str]]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        if lines[i].strip() != "loop_":
            i += 1
            continue

        i += 1
        headers: list[str] = []
        while i < len(lines) and lines[i].strip().startswith("_"):
            headers.append(lines[i].strip())
            i += 1

        rows: list[list[str]] = []
        while i < len(lines):
            stripped = lines[i].strip()
            if (
                not stripped
                or stripped == "loop_"
                or stripped.startswith("_")
                or stripped.startswith("data_")
            ):
                break
            if not stripped.startswith("#"):
                rows.append(shlex.split(stripped))
            i += 1

        yield headers, rows


def _atom_label_to_index(loops: list[tuple[list[str], list[list[str]]]]) -> dict[str, int]:
    for headers, rows in loops:
        if "_atom_site_label" not in headers:
            continue
        label_index = headers.index("_atom_site_label")
        return {
            row[label_index]: atom_index
            for atom_index, row in enumerate(rows)
            if len(row) > label_index
        }
    return {}


def _geom_bond_loop(
    loops: list[tuple[list[str], list[list[str]]]],
) -> tuple[list[str], list[list[str]]] | None:
    for headers, rows in loops:
        if (
            "_geom_bond_atom_site_label_1" in headers
            and "_geom_bond_atom_site_label_2" in headers
        ):
            return headers, rows
    return None


def _atom_index_from_label(label: str, label_to_index: dict[str, int]) -> int | None:
    if label in label_to_index:
        return label_to_index[label]
    match = re.search(r"(\d+)$", label)
    if match is None:
        return None
    return int(match.group(1))


def _site_symmetry_offset(symmetry: str) -> np.ndarray:
    if symmetry in {".", "?", ""}:
        return np.zeros(3, dtype=int)
    match = re.fullmatch(r"\d_([0-9])([0-9])([0-9])", symmetry)
    if match is None:
        return np.zeros(3, dtype=int)
    return np.array([int(value) - 5 for value in match.groups()], dtype=int)


def _warn_about_dropped_components(
    atoms: ase.Atoms,
    components: list[set[int]],
) -> None:
    dropped = components[1:]
    if not dropped:
        return
    summaries = []
    for component in dropped[:5]:
        fragment = atoms[sorted(component)]
        summaries.append(f"{len(component)} atoms {fragment.get_chemical_formula()}")
    more = "" if len(dropped) <= 5 else f"; {len(dropped) - 5} more"
    warnings.warn(
        "Cleanup dropped disconnected bonded components: "
        + "; ".join(summaries)
        + more,
        RuntimeWarning,
        stacklevel=2,
    )


def _repair_inferred_bonds(
    atoms: ase.Atoms,
    vectors: dict[tuple[int, int], np.ndarray],
) -> dict[tuple[int, int], np.ndarray]:
    symbols = atoms.get_chemical_symbols()
    repaired = dict(vectors)
    _remove_suspicious_metal_contacts(symbols, repaired)
    _prune_carboxylate_oxygen_extra_neighbors(symbols, repaired)
    _prune_coordinating_oxygen_extra_neighbors(symbols, repaired)
    _prune_carboxylate_extra_neighbors(symbols, repaired)
    _reconnect_nearby_terminal_components(atoms, symbols, repaired)
    return repaired


def _remove_suspicious_metal_contacts(
    symbols: list[str],
    vectors: dict[tuple[int, int], np.ndarray],
) -> None:
    adjacency = _adjacency_from_vectors(len(symbols), vectors)
    for pair in list(vectors):
        i, j = pair
        metal, other = _metal_nonmetal_pair(symbols, i, j)
        if metal is None or other is None:
            continue
        other_symbol = symbols[other]
        if other_symbol == "C":
            del vectors[pair]
        elif other_symbol == "S" and any(
            symbols[neighbor] == "C" for neighbor in adjacency[other] if neighbor != metal
        ):
            del vectors[pair]


def _prune_carboxylate_extra_neighbors(
    symbols: list[str],
    vectors: dict[tuple[int, int], np.ndarray],
) -> None:
    adjacency = _adjacency_from_vectors(len(symbols), vectors)
    for atom_index, symbol in enumerate(symbols):
        if symbol != "C":
            continue
        oxygen_neighbors = [
            neighbor for neighbor in adjacency[atom_index] if symbols[neighbor] == "O"
        ]
        if len(oxygen_neighbors) < 2:
            continue
        extension_neighbors = [
            neighbor
            for neighbor in adjacency[atom_index]
            if symbols[neighbor] not in {"H", "O"} and not is_mofid_metal(symbols[neighbor])
        ]
        if len(extension_neighbors) <= 1:
            continue
        keep = min(
            extension_neighbors,
            key=lambda neighbor: _pair_length(atom_index, neighbor, vectors),
        )
        for neighbor in extension_neighbors:
            if neighbor != keep:
                pair = tuple(sorted((atom_index, neighbor)))
                del vectors[pair]
                adjacency[atom_index].remove(neighbor)
                adjacency[neighbor].remove(atom_index)


def _prune_carboxylate_oxygen_extra_neighbors(
    symbols: list[str],
    vectors: dict[tuple[int, int], np.ndarray],
) -> None:
    adjacency = _adjacency_from_vectors(len(symbols), vectors)
    for atom_index, symbol in enumerate(symbols):
        if symbol != "O":
            continue
        carbon_neighbors = [
            neighbor for neighbor in adjacency[atom_index] if symbols[neighbor] == "C"
        ]
        if len(carbon_neighbors) <= 1:
            continue

        ranked_neighbors = sorted(
            carbon_neighbors,
            key=lambda neighbor: _pair_length(atom_index, neighbor, vectors),
        )
        keep = ranked_neighbors[0]
        keep_length = _pair_length(atom_index, keep, vectors)
        if keep_length > 1.35:
            continue

        for neighbor in ranked_neighbors[1:]:
            neighbor_length = _pair_length(atom_index, neighbor, vectors)
            if neighbor_length >= 1.45 and neighbor_length - keep_length >= 0.15:
                vectors.pop(tuple(sorted((atom_index, neighbor))), None)


def _prune_coordinating_oxygen_extra_neighbors(
    symbols: list[str],
    vectors: dict[tuple[int, int], np.ndarray],
) -> None:
    adjacency = _adjacency_from_vectors(len(symbols), vectors)
    for atom_index, symbol in enumerate(symbols):
        if symbol != "O":
            continue
        if not any(is_mofid_metal(symbols[neighbor]) for neighbor in adjacency[atom_index]):
            continue
        carbonyl_neighbors = [
            neighbor
            for neighbor in adjacency[atom_index]
            if symbols[neighbor] == "C"
            and _pair_length(atom_index, neighbor, vectors) <= 1.30
        ]
        if carbonyl_neighbors:
            for neighbor in list(adjacency[atom_index]):
                if symbols[neighbor] == "H":
                    vectors.pop(tuple(sorted((atom_index, neighbor))), None)
        nonmetal_neighbors = [
            neighbor
            for neighbor in adjacency[atom_index]
            if symbols[neighbor] != "H" and not is_mofid_metal(symbols[neighbor])
        ]
        if len(nonmetal_neighbors) <= 1:
            continue
        carbon_neighbors = [
            neighbor for neighbor in nonmetal_neighbors if symbols[neighbor] == "C"
        ]
        keep_candidates = carbon_neighbors or nonmetal_neighbors
        keep = min(
            keep_candidates,
            key=lambda neighbor: _pair_length(atom_index, neighbor, vectors),
        )
        for neighbor in nonmetal_neighbors:
            if neighbor != keep:
                vectors.pop(tuple(sorted((atom_index, neighbor))), None)


def _reconnect_nearby_terminal_components(
    atoms: ase.Atoms,
    symbols: list[str],
    vectors: dict[tuple[int, int], np.ndarray],
) -> None:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(symbols)))
    graph.add_edges_from(vectors)
    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    if len(components) <= 1:
        return

    adjacency = _adjacency_from_vectors(len(symbols), vectors)
    component_index = {
        atom_index: index
        for index, component in enumerate(components)
        for atom_index in component
    }
    candidates: list[tuple[float, int, int]] = []
    for i, symbol_i in enumerate(symbols):
        if not _is_terminal_carbon(i, symbols, adjacency):
            continue
        for j in range(i + 1, len(symbols)):
            if component_index[i] == component_index[j]:
                continue
            if not _is_terminal_carbon(j, symbols, adjacency):
                continue
            vector, distance = _minimum_image_vector(atoms, i, j)
            if 1.85 <= distance <= 2.65:
                candidates.append((distance, i, j))

    added_by_component = {index: 0 for index in range(len(components))}
    used_atoms: set[int] = set()
    for _, i, j in sorted(candidates):
        i_component = component_index[i]
        j_component = component_index[j]
        non_main_components = [
            component for component in (i_component, j_component) if component != 0
        ]
        if not non_main_components:
            continue
        if i in used_atoms or j in used_atoms:
            continue
        if any(added_by_component[component] >= 2 for component in non_main_components):
            continue

        vector, _ = _minimum_image_vector(atoms, i, j)
        a, b = sorted((i, j))
        vectors[(a, b)] = np.array(vector if i == a else -vector, dtype=float)
        used_atoms.update({i, j})
        for component in non_main_components:
            added_by_component[component] += 1


def _adjacency_from_vectors(
    num_atoms: int,
    vectors: dict[tuple[int, int], np.ndarray],
) -> list[set[int]]:
    adjacency = [set() for _ in range(num_atoms)]
    for i, j in vectors:
        adjacency[i].add(j)
        adjacency[j].add(i)
    return adjacency


def _metal_nonmetal_pair(
    symbols: list[str],
    i: int,
    j: int,
) -> tuple[int | None, int | None]:
    if is_mofid_metal(symbols[i]) and not is_mofid_metal(symbols[j]):
        return i, j
    if is_mofid_metal(symbols[j]) and not is_mofid_metal(symbols[i]):
        return j, i
    return None, None


def _pair_length(
    i: int,
    j: int,
    vectors: dict[tuple[int, int], np.ndarray],
) -> float:
    return float(np.linalg.norm(vectors[tuple(sorted((i, j)))]))


def _is_terminal_carbon(
    atom_index: int,
    symbols: list[str],
    adjacency: list[set[int]],
) -> bool:
    if symbols[atom_index] != "C":
        return False
    heavy_degree = sum(1 for neighbor in adjacency[atom_index] if symbols[neighbor] != "H")
    return heavy_degree <= 2


def _minimum_image_vector(
    atoms: ase.Atoms,
    i: int,
    j: int,
) -> tuple[np.ndarray, float]:
    raw_vector = atoms.positions[j] - atoms.positions[i]
    vector, distance = find_mic(raw_vector, atoms.cell, atoms.pbc)
    return np.array(vector, dtype=float), float(distance)

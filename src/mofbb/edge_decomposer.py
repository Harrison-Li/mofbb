"""Alternative decomposition into metal nodes, organic nodes, and edges."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import product
import json
from pathlib import Path
from typing import Literal

import ase
from ase.geometry import cell_to_cellpar
import networkx as nx
import numpy as np

from .assignment import SingleNodeAtomAssigner
from .bonding import BondNetwork, read_bonded_structure
from .decomposer import find_connecting_bonds
from .fragments import FragmentBuilder
from .geometry import format_float, unwrap_fractional_near
from .models import Fragment, TopologyEdge, TopologyVertex
from .results import fragment_instance_path
from .solvent import MofidSolventRemover, SolventMode, SolventRemovalResult

OrganicFragmentKind = Literal["organic_node", "edge"]


@dataclass
class OrganicLinkerSplit:
    """Fragments and internal cut bonds from splitting organic linkers."""

    organic_node_fragments: list[Fragment] = field(default_factory=list)
    edge_fragments: list[Fragment] = field(default_factory=list)
    organic_connecting_bonds: tuple[tuple[int, int], ...] = ()


def classify_organic_fragment_kind(fragment: Fragment) -> OrganicFragmentKind:
    """Classify an organic fragment from its dummy connection-site count."""

    num_connection_sites = len(fragment.x_sites)
    if num_connection_sites > 2:
        return "organic_node"
    if num_connection_sites == 2:
        return "edge"
    raise ValueError(
        f"Organic fragment {fragment.index} has fewer than 2 connection sites "
        f"({num_connection_sites}); it cannot be classified as an edge or node."
    )


def _partition_bound_linkers(
    organic_fragments: list[Fragment],
    selected_to_original: tuple[int, ...],
    bound_solvent_indices: tuple[int, ...],
) -> tuple[list[Fragment], list[Fragment]]:
    """Partition FSR organics using whole-fragment original-index containment.

    MOFid bound-solvent degree counts unique collapsed site connections after
    endpoint/translation deduplication, not atom-level boundary bonds. An
    accessory linker may therefore retain zero, one, or many X sites; its
    preserved ``linker`` role comes from containment, never from X-site count.
    """

    bound_original = set(bound_solvent_indices)
    accessories: list[Fragment] = []
    splittable: list[Fragment] = []
    for fragment in organic_fragments:
        original_indices = {
            selected_to_original[index] for index in fragment.source_indices
        }
        if bound_original and original_indices <= bound_original:
            accessories.append(fragment)
        else:
            splittable.append(fragment)
    return accessories, splittable


def decompose_mof_with_edges(
    cif: str | Path,
    x_symbol: str = "X",
    bond_scale: float = 1.2,
    cleanup: bool = True,
    solvent_mode: SolventMode = "none",
) -> NodeEdgeDecompositionResult:
    """Convenience function for the node/edge decomposition workflow."""

    return NodeEdgePorDecomposer(
        cif,
        x_symbol=x_symbol,
        bond_scale=bond_scale,
        cleanup=cleanup,
        solvent_mode=solvent_mode,
    ).decompose()


class NodeEdgePorDecomposer:
    """Decompose a MOF into metal nodes, organic nodes, and edge fragments."""

    def __init__(
        self,
        cif: str | Path,
        x_symbol: str = "X",
        bond_scale: float = 1.2,
        cleanup: bool = True,
        solvent_mode: SolventMode = "none",
    ) -> None:
        if solvent_mode not in {"none", "free", "all"}:
            raise ValueError(f"Unsupported solvent mode: {solvent_mode!r}.")
        self.cif = Path(cif)
        self.x_symbol = x_symbol
        self.bond_scale = bond_scale
        self.cleanup = cleanup
        self.solvent_mode = solvent_mode
        self.solvent: SolventRemovalResult | None = None
        self.selected_to_original: tuple[int, ...] | None = None
        self.atoms, self.bonds = read_bonded_structure(
            self.cif,
            scale=self.bond_scale,
            cleanup=self.cleanup if solvent_mode == "none" else False,
        )
        if solvent_mode != "none":
            self.solvent = MofidSolventRemover(self.atoms, self.bonds).classify()
            self.atoms, self.bonds, self.selected_to_original = self.solvent.subset(
                solvent_mode
            )
        self.symbols = self.atoms.get_chemical_symbols()

    def decompose(self) -> NodeEdgeDecompositionResult:
        """Run the node/edge decomposition workflow."""

        assignment = SingleNodeAtomAssigner(self.symbols, self.bonds).assign()
        metal_node_indices = set(assignment.node_indices)
        organic_indices = set(range(len(self.atoms))) - metal_node_indices
        connecting_bonds = find_connecting_bonds(self.bonds.pairs, metal_node_indices)

        fragment_builder = FragmentBuilder(
            self.atoms,
            self.bonds,
            self.symbols,
            x_symbol=self.x_symbol,
        )
        raw_metal_nodes = fragment_builder.build_role_fragments(
            "node", metal_node_indices, connecting_bonds
        )
        raw_organics = fragment_builder.build_role_fragments(
            "linker", organic_indices, connecting_bonds
        )

        if (
            self.solvent_mode == "free"
            and self.solvent is not None
            and self.selected_to_original is not None
        ):
            bound_linker_fragments, splittable_organics = _partition_bound_linkers(
                raw_organics,
                self.selected_to_original,
                self.solvent.bound_solvent_indices,
            )
        else:
            bound_linker_fragments = []
            splittable_organics = raw_organics

        metal_node_fragments = relabel_fragments(raw_metal_nodes, "metal_node")
        organic_split = OrganicLinkerSplitter(
            self.atoms,
            self.bonds,
            self.symbols,
            x_symbol=self.x_symbol,
        ).split(splittable_organics, connecting_bonds)
        all_connecting_bonds = tuple(
            sorted((*connecting_bonds, *organic_split.organic_connecting_bonds))
        )

        topology_vertices, topology_edges = NodeEdgeTopologyBuilder(
            self.atoms,
            self.bonds,
        ).build(
            [
                *metal_node_fragments,
                *organic_split.organic_node_fragments,
                *organic_split.edge_fragments,
                *bound_linker_fragments,
            ],
            all_connecting_bonds,
        )

        return NodeEdgeDecompositionResult(
            name=self.cif.stem,
            metal_node_indices=tuple(sorted(metal_node_indices)),
            organic_indices=tuple(sorted(organic_indices)),
            points_of_extension=tuple(sorted(assignment.points_of_extension)),
            connecting_bonds=all_connecting_bonds,
            cell_parameters=tuple(float(value) for value in cell_to_cellpar(self.atoms.cell)),
            topology_vertices=topology_vertices,
            topology_edges=topology_edges,
            metal_node_fragments=metal_node_fragments,
            organic_node_fragments=organic_split.organic_node_fragments,
            edge_fragments=organic_split.edge_fragments,
            bound_linker_fragments=bound_linker_fragments,
            metal_connecting_bonds=connecting_bonds,
            organic_connecting_bonds=organic_split.organic_connecting_bonds,
            solvent_mode=self.solvent_mode,
            solvent=self.solvent,
            selected_to_original=self.selected_to_original,
        )

    def _read_atoms(self) -> ase.Atoms:
        return self.atoms


@dataclass(frozen=True)
class _RoleComponent:
    kind: OrganicFragmentKind
    atom_indices: frozenset[int]


class OrganicLinkerSplitter:
    """Split original organic linkers into PORMAKE-style organic nodes and edges."""

    def __init__(
        self,
        atoms: ase.Atoms,
        bonds: BondNetwork,
        symbols: list[str],
        x_symbol: str = "X",
    ) -> None:
        self.atoms = atoms
        self.bonds = bonds
        self.symbols = symbols
        self.x_symbol = x_symbol

    def split(
        self,
        organic_fragments: list[Fragment],
        metal_connecting_bonds: tuple[tuple[int, int], ...],
    ) -> OrganicLinkerSplit:
        """Split each original organic linker into smaller node/edge pieces."""

        role_components: list[_RoleComponent] = []
        organic_cut_bonds: set[tuple[int, int]] = set()

        for fragment in organic_fragments:
            components = self._split_fragment_components(
                fragment,
                metal_connecting_bonds,
            )
            role_components.extend(components)
            organic_cut_bonds.update(
                self._cut_bonds_between_components(
                    [set(component.atom_indices) for component in components]
                )
            )

        all_cut_bonds = tuple(sorted((*metal_connecting_bonds, *organic_cut_bonds)))
        fragment_builder = FragmentBuilder(
            self.atoms,
            self.bonds,
            self.symbols,
            x_symbol=self.x_symbol,
        )

        organic_nodes: list[Fragment] = []
        edges: list[Fragment] = []
        for component in role_components:
            if component.kind == "organic_node":
                organic_nodes.append(
                    fragment_builder.materialize_fragment(
                        "organic_node",
                        len(organic_nodes) + 1,
                        set(component.atom_indices),
                        all_cut_bonds,
                    )
                )
            else:
                edges.append(
                    fragment_builder.materialize_fragment(
                        "edge",
                        len(edges) + 1,
                        set(component.atom_indices),
                        all_cut_bonds,
                    )
                )

        self._validate_split_fragments(organic_nodes, edges)
        return OrganicLinkerSplit(
            organic_node_fragments=organic_nodes,
            edge_fragments=edges,
            organic_connecting_bonds=tuple(sorted(organic_cut_bonds)),
        )

    def _split_fragment_components(
        self,
        fragment: Fragment,
        metal_connecting_bonds: tuple[tuple[int, int], ...],
    ) -> list[_RoleComponent]:
        source_atoms = set(fragment.source_indices)
        external_cut_bonds = tuple(
            cut_bond
            for cut_bond in metal_connecting_bonds
            if (cut_bond[0] in source_atoms) != (cut_bond[1] in source_atoms)
        )
        heavy_atoms = {
            atom_index
            for atom_index in source_atoms
            if self.symbols[atom_index] != "H"
        }
        heavy_graph = self._heavy_graph(heavy_atoms)
        node_heavy_atoms = self._candidate_node_heavy_atoms(
            heavy_graph,
            external_cut_bonds,
            source_atoms,
        )

        if not node_heavy_atoms:
            return [
                _RoleComponent(
                    classify_organic_fragment_kind(fragment),
                    frozenset(source_atoms),
                )
            ]

        heavy_components = self._classify_heavy_components(
            heavy_graph,
            heavy_atoms,
            node_heavy_atoms,
            external_cut_bonds,
            source_atoms,
        )
        return self._attach_hydrogens(heavy_components, source_atoms)

    def _heavy_graph(self, heavy_atoms: set[int]) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(heavy_atoms)
        for i, j in self.bonds.pairs:
            if i in heavy_atoms and j in heavy_atoms:
                graph.add_edge(i, j)
        return graph

    def _candidate_node_heavy_atoms(
        self,
        heavy_graph: nx.Graph,
        external_cut_bonds: tuple[tuple[int, int], ...],
        source_atoms: set[int],
    ) -> set[int]:
        node_atoms: set[int] = set()
        for block in nx.biconnected_components(heavy_graph):
            if len(block) < 3:
                continue

            port_count = self._external_cut_count(block, external_cut_bonds, source_atoms)
            for atom_index in block:
                for neighbor in heavy_graph.neighbors(atom_index):
                    if neighbor not in block:
                        port_count += 1

            if port_count > 2:
                node_atoms.update(block)
        return node_atoms

    @staticmethod
    def _external_cut_count(
        atom_indices: set[int] | frozenset[int],
        external_cut_bonds: tuple[tuple[int, int], ...],
        source_atoms: set[int],
    ) -> int:
        return sum(
            1
            for i, j in external_cut_bonds
            if (i in atom_indices and i in source_atoms)
            or (j in atom_indices and j in source_atoms)
        )

    def _classify_heavy_components(
        self,
        heavy_graph: nx.Graph,
        heavy_atoms: set[int],
        node_heavy_atoms: set[int],
        external_cut_bonds: tuple[tuple[int, int], ...],
        source_atoms: set[int],
    ) -> list[_RoleComponent]:
        node_components = [
            set(component)
            for component in nx.connected_components(
                heavy_graph.subgraph(node_heavy_atoms).copy()
            )
        ]
        remaining_heavy_atoms = heavy_atoms - node_heavy_atoms
        edge_components = [
            set(component)
            for component in nx.connected_components(
                heavy_graph.subgraph(remaining_heavy_atoms).copy()
            )
        ]

        absorbed: set[int] = set()
        role_components: list[_RoleComponent] = []
        node_component_by_atom = {
            atom_index: component_index
            for component_index, component in enumerate(node_components)
            for atom_index in component
        }

        for component in edge_components:
            boundary_node_indices = self._boundary_node_indices(
                component,
                node_component_by_atom,
            )
            port_count = self._component_port_count(
                component,
                external_cut_bonds,
                source_atoms,
                node_component_by_atom,
            )
            if port_count < 2 and len(boundary_node_indices) == 1:
                node_components[next(iter(boundary_node_indices))].update(component)
                absorbed.update(component)

        node_component_by_atom = {
            atom_index: component_index
            for component_index, component in enumerate(node_components)
            for atom_index in component
        }
        for component in node_components:
            role_components.append(
                _RoleComponent("organic_node", frozenset(component))
            )

        for component in edge_components:
            if component & absorbed:
                continue
            port_count = self._component_port_count(
                component,
                external_cut_bonds,
                source_atoms,
                node_component_by_atom,
            )
            kind: OrganicFragmentKind = "edge" if port_count == 2 else "organic_node"
            role_components.append(_RoleComponent(kind, frozenset(component)))

        return role_components

    def _boundary_node_indices(
        self,
        component: set[int],
        node_component_by_atom: dict[int, int],
    ) -> set[int]:
        boundary_node_indices: set[int] = set()
        for atom_index in component:
            for neighbor in self.bonds.adjacency[atom_index]:
                node_component_index = node_component_by_atom.get(neighbor)
                if node_component_index is not None:
                    boundary_node_indices.add(node_component_index)
        return boundary_node_indices

    def _component_port_count(
        self,
        component: set[int],
        external_cut_bonds: tuple[tuple[int, int], ...],
        source_atoms: set[int],
        node_component_by_atom: dict[int, int],
    ) -> int:
        port_count = self._external_cut_count(component, external_cut_bonds, source_atoms)
        for atom_index in component:
            for neighbor in self.bonds.adjacency[atom_index]:
                if neighbor in node_component_by_atom:
                    port_count += 1
        return port_count

    def _attach_hydrogens(
        self,
        heavy_components: list[_RoleComponent],
        source_atoms: set[int],
    ) -> list[_RoleComponent]:
        component_by_heavy_atom = {
            atom_index: component_index
            for component_index, component in enumerate(heavy_components)
            for atom_index in component.atom_indices
        }
        expanded = [set(component.atom_indices) for component in heavy_components]

        for atom_index in sorted(source_atoms):
            if self.symbols[atom_index] != "H":
                continue
            neighboring_component = next(
                (
                    component_by_heavy_atom[neighbor]
                    for neighbor in self.bonds.adjacency[atom_index]
                    if neighbor in component_by_heavy_atom
                ),
                None,
            )
            if neighboring_component is not None:
                expanded[neighboring_component].add(atom_index)

        return [
            _RoleComponent(component.kind, frozenset(atom_indices))
            for component, atom_indices in zip(heavy_components, expanded)
        ]

    def _cut_bonds_between_components(
        self,
        components: list[set[int]],
    ) -> set[tuple[int, int]]:
        component_by_atom = {
            atom_index: component_index
            for component_index, component in enumerate(components)
            for atom_index in component
        }
        cut_bonds: set[tuple[int, int]] = set()
        for i, j in self.bonds.pairs:
            i_component = component_by_atom.get(i)
            j_component = component_by_atom.get(j)
            if i_component is None or j_component is None or i_component == j_component:
                continue
            cut_bonds.add(tuple(sorted((i, j))))
        return cut_bonds

    @staticmethod
    def _validate_split_fragments(
        organic_nodes: list[Fragment],
        edges: list[Fragment],
    ) -> None:
        invalid_nodes = [
            fragment.index for fragment in organic_nodes if len(fragment.x_sites) <= 2
        ]
        invalid_edges = [
            fragment.index for fragment in edges if len(fragment.x_sites) != 2
        ]
        if invalid_nodes:
            raise ValueError(
                "Generated organic node fragments with fewer than 3 connection "
                f"sites: {invalid_nodes}"
            )
        if invalid_edges:
            raise ValueError(
                "Generated edge fragments without exactly 2 connection sites: "
                f"{invalid_edges}"
            )


def relabel_fragments(fragments: list[Fragment], kind: str) -> list[Fragment]:
    """Return fragments copied with a new kind and stable one-based indices."""

    return [
        relabel_fragment(fragment, kind, index)
        for index, fragment in enumerate(fragments, start=1)
    ]


def relabel_fragment(fragment: Fragment, kind: str, index: int) -> Fragment:
    """Return one fragment copied with a new output role label."""

    return Fragment(
        kind=kind,
        index=index,
        source_indices=fragment.source_indices,
        atoms=fragment.atoms.copy(),
        bonds=fragment.bonds,
        fingerprint=fragment.fingerprint,
        x_sites=fragment.x_sites,
    )


class NodeEdgeTopologyBuilder:
    """Build collapsed topology records for the node/edge decomposition."""

    def __init__(self, atoms: ase.Atoms, bonds: BondNetwork) -> None:
        self.atoms = atoms
        self.bonds = bonds

    def build(
        self,
        fragments: list[Fragment],
        connecting_bonds: tuple[tuple[int, int], ...],
    ) -> tuple[list[TopologyVertex], list[TopologyEdge]]:
        """Return topology vertices and source cut-bond edges."""

        atom_to_vertex = self._atom_to_vertex(fragments)
        centroids = {
            (fragment.kind, fragment.index): self._fragment_centroid_fractional(fragment)
            for fragment in fragments
        }
        x_site_directions = self._x_site_directions(fragments)
        coordination: dict[tuple[str, int], int] = defaultdict(int)
        edges = self._build_edges(
            connecting_bonds,
            atom_to_vertex,
            centroids,
            x_site_directions,
            coordination,
        )
        vertices = self._build_vertices(fragments, centroids, coordination)
        return vertices, edges

    @staticmethod
    def _atom_to_vertex(fragments: list[Fragment]) -> dict[int, tuple[str, int]]:
        atom_to_vertex: dict[int, tuple[str, int]] = {}
        for fragment in fragments:
            for atom_index in fragment.source_indices:
                atom_to_vertex[atom_index] = (fragment.kind, fragment.index)
        return atom_to_vertex

    def _build_edges(
        self,
        connecting_bonds: tuple[tuple[int, int], ...],
        atom_to_vertex: dict[int, tuple[str, int]],
        centroids: dict[tuple[str, int], tuple[float, float, float]],
        x_site_directions: dict[tuple[tuple[str, int], tuple[int, int]], np.ndarray],
        coordination: dict[tuple[str, int], int],
    ) -> list[TopologyEdge]:
        edges: list[TopologyEdge] = []

        for i, j in connecting_bonds:
            begin_key = atom_to_vertex[i]
            end_key = atom_to_vertex[j]
            if begin_key == end_key:
                continue
            if begin_key[0] != "metal_node" and end_key[0] == "metal_node":
                begin_key, end_key = end_key, begin_key

            begin_frac = centroids[begin_key]
            source_cut_bond = tuple(sorted((i, j)))
            end_frac = self._unwrap_endpoint_for_connection(
                begin_key,
                end_key,
                source_cut_bond,
                centroids,
                x_site_directions,
            )
            center_frac = tuple(0.5 * (np.array(begin_frac) + np.array(end_frac)))
            edges.append(
                TopologyEdge(
                    begin_kind=begin_key[0],
                    begin_index=begin_key[1],
                    end_kind=end_key[0],
                    end_index=end_key[1],
                    begin_frac=tuple(float(value) for value in begin_frac),
                    end_frac=tuple(float(value) for value in end_frac),
                    center_frac=tuple(float(value) for value in center_frac),
                )
            )
            coordination[begin_key] += 1
            coordination[end_key] += 1

        return edges

    @staticmethod
    def _x_site_directions(
        fragments: list[Fragment],
    ) -> dict[tuple[tuple[str, int], tuple[int, int]], np.ndarray]:
        directions: dict[tuple[tuple[str, int], tuple[int, int]], np.ndarray] = {}
        for fragment in fragments:
            if not fragment.x_sites:
                continue
            x_positions = np.array(
                [fragment.atoms.positions[site.x_index] for site in fragment.x_sites],
                dtype=float,
            )
            x_centroid = x_positions.mean(axis=0)
            fragment_key = (fragment.kind, fragment.index)
            for site in fragment.x_sites:
                direction = np.array(fragment.atoms.positions[site.x_index], dtype=float)
                direction -= x_centroid
                if np.linalg.norm(direction) > 1.0e-10:
                    directions[(fragment_key, site.source_cut_bond)] = direction
        return directions

    def _unwrap_endpoint_for_connection(
        self,
        begin_key: tuple[str, int],
        end_key: tuple[str, int],
        source_cut_bond: tuple[int, int],
        centroids: dict[tuple[str, int], tuple[float, float, float]],
        x_site_directions: dict[tuple[tuple[str, int], tuple[int, int]], np.ndarray],
    ) -> tuple[float, float, float]:
        begin = np.array(centroids[begin_key], dtype=float)
        end = np.array(centroids[end_key], dtype=float)
        if begin_key[0] != "edge" and end_key[0] != "edge":
            return unwrap_fractional_near(end, begin)

        begin_direction = self._unit_vector(
            x_site_directions.get((begin_key, source_cut_bond))
        )
        end_direction = self._unit_vector(
            x_site_directions.get((end_key, source_cut_bond))
        )

        if begin_direction is None and end_direction is None:
            return unwrap_fractional_near(end, begin)

        best_score = -np.inf
        best_endpoint = np.array(unwrap_fractional_near(end, begin), dtype=float)
        cell = np.array(self.atoms.cell.array, dtype=float)

        for shift in product((-1, 0, 1), repeat=3):
            candidate = end + np.array(shift, dtype=float)
            delta_cart = (candidate - begin) @ cell
            delta_length = float(np.linalg.norm(delta_cart))
            if delta_length <= 1.0e-10:
                continue

            delta_direction = delta_cart / delta_length
            score = 0.0
            score_terms = 0
            if begin_direction is not None:
                score += float(np.dot(delta_direction, begin_direction))
                score_terms += 1
            if end_direction is not None:
                score += float(np.dot(-delta_direction, end_direction))
                score_terms += 1
            score = score / score_terms - 1.0e-6 * delta_length

            if score > best_score:
                best_score = score
                best_endpoint = candidate

        return tuple(float(value) for value in best_endpoint)

    @staticmethod
    def _unit_vector(vector: np.ndarray | None) -> np.ndarray | None:
        if vector is None:
            return None
        norm = float(np.linalg.norm(vector))
        if norm <= 1.0e-10:
            return None
        return np.array(vector, dtype=float) / norm

    @staticmethod
    def _build_vertices(
        fragments: list[Fragment],
        centroids: dict[tuple[str, int], tuple[float, float, float]],
        coordination: dict[tuple[str, int], int],
    ) -> list[TopologyVertex]:
        return [
            TopologyVertex(
                kind=fragment.kind,
                index=fragment.index,
                coordination=coordination[(fragment.kind, fragment.index)],
                frac_position=tuple(
                    float(value) for value in centroids[(fragment.kind, fragment.index)]
                ),
            )
            for fragment in fragments
        ]

    def _fragment_centroid_fractional(
        self,
        fragment: Fragment,
    ) -> tuple[float, float, float]:
        from .topology import TopologyBuilder

        return TopologyBuilder(self.atoms, self.bonds).fragment_centroid_fractional(
            fragment.source_indices
        )


@dataclass
class NodeEdgeDecompositionResult:
    """Complete decomposition into metal nodes, organic nodes, and edges."""

    name: str
    metal_node_indices: tuple[int, ...]
    organic_indices: tuple[int, ...]
    points_of_extension: tuple[int, ...]
    connecting_bonds: tuple[tuple[int, int], ...]
    cell_parameters: tuple[float, float, float, float, float, float]
    topology_vertices: list[TopologyVertex]
    topology_edges: list[TopologyEdge]
    metal_node_fragments: list[Fragment] = field(default_factory=list)
    organic_node_fragments: list[Fragment] = field(default_factory=list)
    edge_fragments: list[Fragment] = field(default_factory=list)
    metal_connecting_bonds: tuple[tuple[int, int], ...] = ()
    organic_connecting_bonds: tuple[tuple[int, int], ...] = ()
    bound_linker_fragments: list[Fragment] = field(default_factory=list)
    solvent_mode: SolventMode = "none"
    solvent: SolventRemovalResult | None = None
    selected_to_original: tuple[int, ...] | None = None

    @property
    def fragments(self) -> list[Fragment]:
        """Return all fragments in output slot order."""

        return [
            *self.metal_node_fragments,
            *self.organic_node_fragments,
            *self.edge_fragments,
            *self.bound_linker_fragments,
        ]

    def write(self, output_dir: Path) -> dict[str, object]:
        """Write all node/edge decomposition outputs and return a summary."""

        if self.solvent_mode == "none":
            if self.solvent is not None or self.selected_to_original is not None:
                raise ValueError(
                    "Inconsistent solvent metadata: mode 'none' requires both "
                    "the solvent result and selected-to-original mapping to be absent."
                )
        elif self.solvent_mode in {"free", "all"}:
            if self.solvent is None or self.selected_to_original is None:
                raise ValueError(
                    f"Inconsistent solvent metadata: active mode {self.solvent_mode!r} "
                    "requires both a solvent result and selected-to-original mapping."
                )
            self.solvent.validate_write_request(
                self.solvent_mode,
                self.selected_to_original,
            )
        else:
            raise ValueError(f"Unsupported solvent mode: {self.solvent_mode!r}.")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for fragment in self.fragments:
            fragment.write_xyz(output_dir / fragment_instance_path(fragment))
            fragment.write_xyz(
                output_dir
                / "all_fragments"
                / f"{fragment.kind}_{fragment.index:03d}.xyz"
            )

        group_records, fragment_to_bb_id = self._write_group_representatives(output_dir)

        topology_path = output_dir / "topology.cgd"
        topology_summary = self.write_topology_cgd(topology_path)
        topology_summary["path"] = str(topology_path.relative_to(output_dir))

        reconstruction_path = output_dir / "reconstruction_map.json"
        self.write_reconstruction_map(
            reconstruction_path,
            group_records,
            fragment_to_bb_id,
        )

        summary: dict[str, object] = {
            "name": self.name,
            "decomposition": "node_edge",
            "num_metal_nodes": len(self.metal_node_fragments),
            "num_organic_nodes": len(self.organic_node_fragments),
            "num_edges": len(self.edge_fragments),
            "num_connecting_bonds": len(self.connecting_bonds),
            "num_metal_connecting_bonds": len(self.metal_connecting_bonds),
            "num_organic_connecting_bonds": len(self.organic_connecting_bonds),
            "metal_node_indices": list(self.metal_node_indices),
            "organic_indices": list(self.organic_indices),
            "points_of_extension": list(self.points_of_extension),
            "connecting_bonds": [list(pair) for pair in self.connecting_bonds],
            "metal_connecting_bonds": [
                list(pair) for pair in self.metal_connecting_bonds
            ],
            "organic_connecting_bonds": [
                list(pair) for pair in self.organic_connecting_bonds
            ],
            "topology": topology_summary,
            "reconstruction_map": str(reconstruction_path.relative_to(output_dir)),
            "groups": group_records,
            "metal_nodes": [
                fragment.as_dict(fragment_instance_path(fragment))
                for fragment in self.metal_node_fragments
            ],
            "organic_nodes": [
                fragment.as_dict(fragment_instance_path(fragment))
                for fragment in self.organic_node_fragments
            ],
            "edges": [
                fragment.as_dict(fragment_instance_path(fragment))
                for fragment in self.edge_fragments
            ],
        }
        if self.bound_linker_fragments:
            summary["num_bound_linkers"] = len(self.bound_linker_fragments)
            summary["bound_linkers"] = [
                fragment.as_dict(fragment_instance_path(fragment))
                for fragment in self.bound_linker_fragments
            ]
        if self.solvent is not None and self.selected_to_original is not None:
            summary["solvent"] = self.solvent.write(
                output_dir,
                self.solvent_mode,
                self.selected_to_original,
            )
        with (output_dir / "decomposition_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return summary

    def _write_group_representatives(
        self,
        output_dir: Path,
    ) -> tuple[list[dict[str, object]], dict[tuple[str, int], str]]:
        group_records: list[dict[str, object]] = []
        fragment_to_bb_id: dict[tuple[str, int], str] = {}
        for kind, fragments in (
            ("metal_node", self.metal_node_fragments),
            ("organic_node", self.organic_node_fragments),
            ("edge", self.edge_fragments),
            ("linker", self.bound_linker_fragments),
        ):
            grouped: dict[str, list[Fragment]] = defaultdict(list)
            for fragment in fragments:
                grouped[fragment.fingerprint].append(fragment)

            for group_index, (_, group) in enumerate(sorted(grouped.items()), start=1):
                representative = group[0]
                bb_id = f"{kind}_group_{group_index:02d}"
                group_path = output_dir / "bbs" / f"{kind}_group_{group_index:02d}.xyz"
                representative.write_xyz(group_path)
                for fragment in group:
                    fragment_to_bb_id[(fragment.kind, fragment.index)] = bb_id
                group_records.append(
                    {
                        "bb_id": bb_id,
                        "kind": kind,
                        "group": group_index,
                        "count": len(group),
                        "cn": len(representative.x_sites),
                        "formula": representative.formula,
                        "num_atoms": representative.num_atoms,
                        "num_x": representative.num_x,
                        "x_sites": [
                            {"site_id": site_id, "x_index": site.x_index}
                            for site_id, site in enumerate(representative.x_sites)
                        ],
                        "fingerprint": representative.fingerprint,
                        "fragment_indices": [fragment.index for fragment in group],
                        "path": str(group_path.relative_to(output_dir)),
                    }
                )
        return group_records, fragment_to_bb_id

    def write_reconstruction_map(
        self,
        path: Path,
        group_records: list[dict[str, object]],
        fragment_to_bb_id: dict[tuple[str, int], str],
    ) -> dict[str, object]:
        """Write group-first metadata for exact reconstruction experiments."""

        path.parent.mkdir(parents=True, exist_ok=True)
        slot_by_key = {
            (fragment.kind, fragment.index): slot_id
            for slot_id, fragment in enumerate(self.fragments)
        }
        slot_by_source_atom: dict[int, int] = {}
        for fragment in self.fragments:
            slot_id = slot_by_key[(fragment.kind, fragment.index)]
            for atom_index in fragment.source_indices:
                slot_by_source_atom[atom_index] = slot_id

        slots = []
        for slot_id, fragment in enumerate(self.fragments):
            key = (fragment.kind, fragment.index)
            slots.append(
                {
                    "slot_id": slot_id,
                    "kind": fragment.kind,
                    "cn": len(fragment.x_sites),
                    "bb_id": fragment_to_bb_id[key],
                    "debug_instance": fragment_instance_path(fragment),
                    "source_indices": list(fragment.source_indices),
                    "formula": fragment.formula,
                }
            )

        x_site_by_slot_and_cut: dict[tuple[int, tuple[int, int]], int] = {}
        all_x_site_keys: set[tuple[int, int]] = set()
        for slot_id, fragment in enumerate(self.fragments):
            for site in fragment.x_sites:
                key = (slot_id, site.x_index)
                all_x_site_keys.add(key)
                x_site_by_slot_and_cut[(slot_id, site.source_cut_bond)] = site.x_index

        edges = []
        used_x_site_keys: set[tuple[int, int]] = set()
        all_edges_mapped = True
        slot_pair_counts: dict[tuple[int, int], int] = defaultdict(int)
        for edge_id, cut_bond in enumerate(self.connecting_bonds):
            source_cut_bond = tuple(sorted(cut_bond))
            endpoint_slots = [
                slot_by_source_atom[source_cut_bond[0]],
                slot_by_source_atom[source_cut_bond[1]],
            ]
            endpoint_x_sites = []
            for slot_id in endpoint_slots:
                x_index = x_site_by_slot_and_cut.get((slot_id, source_cut_bond))
                if x_index is None:
                    all_edges_mapped = False
                    continue
                used_x_site_keys.add((slot_id, x_index))
                endpoint_x_sites.append({"slot_id": slot_id, "x_index": x_index})

            if len(endpoint_x_sites) != 2:
                all_edges_mapped = False

            slot_pair_counts[tuple(sorted(endpoint_slots))] += 1
            edges.append(
                {
                    "edge_id": edge_id,
                    "endpoint_slots": endpoint_slots,
                    "endpoint_x_sites": endpoint_x_sites,
                    "source_cut_bond": list(source_cut_bond),
                }
            )

        duplicate_edge_warnings = [
            {
                "endpoint_slots": list(slot_pair),
                "edge_count": count,
            }
            for slot_pair, count in sorted(slot_pair_counts.items())
            if count > 1
        ]

        data: dict[str, object] = {
            "schema_version": 1,
            "topology": {
                "kind": "node_edge_slots",
                "path": "topology.cgd",
                "stock_pormake_reader_safe": False,
                "cell": list(self.cell_parameters),
            },
            "bb_groups": [
                {
                    "bb_id": group["bb_id"],
                    "kind": group["kind"],
                    "path": group["path"],
                    "cn": group["cn"],
                    "formula": group["formula"],
                    "x_sites": group["x_sites"],
                }
                for group in group_records
            ],
            "slots": slots,
            "edges": edges,
            "validation": {
                "all_edges_mapped": all_edges_mapped,
                "all_x_sites_used_once": all_x_site_keys == used_x_site_keys,
                "duplicate_edge_warnings": duplicate_edge_warnings,
            },
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return data

    def write_topology_cgd(self, path: Path) -> dict[str, int]:
        """Write a P1 CGD topology file for the node/edge result."""

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            handle.write("CRYSTAL\n")
            handle.write(f"  NAME {self.name}\n")
            handle.write("  GROUP P1\n")
            cell = " ".join(format_float(value) for value in self.cell_parameters)
            handle.write(f"  CELL {cell}\n")
            for vertex_number, vertex in enumerate(self.topology_vertices, start=1):
                frac = " ".join(format_float(value) for value in vertex.frac_position)
                handle.write(f"  NODE {vertex_number} {vertex.coordination} {frac}\n")
            for edge in self.topology_edges:
                begin = " ".join(format_float(value) for value in edge.begin_frac)
                end = " ".join(format_float(value) for value in edge.end_frac)
                handle.write(f"  EDGE  {begin}   {end}\n")
            for edge in self.topology_edges:
                center = " ".join(format_float(value) for value in edge.center_frac)
                handle.write(f"  # EDGE_CENTER  {center}\n")
            handle.write("END\n")
        return {
            "num_vertices": len(self.topology_vertices),
            "num_edges": len(self.topology_edges),
        }

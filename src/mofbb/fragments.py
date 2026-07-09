"""Fragment construction and PORMAKE-style cut-bond capping."""

from __future__ import annotations

import ase
import networkx as nx
import numpy as np

from .bonding import BondNetwork
from .fingerprint import fragment_fingerprint
from .geometry import center_positions, unwrap_positions
from .models import Fragment, XSite
from .type_aliases import FragmentKind


class FragmentBuilder:
    """Build finite node or linker fragments from source-atom components."""

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

    def build_role_fragments(
        self,
        kind: FragmentKind,
        role_indices: set[int] | frozenset[int],
        connecting_bonds: tuple[tuple[int, int], ...],
    ) -> list[Fragment]:
        """Build all connected fragments for one role: ``node`` or ``linker``."""

        fragments = []
        for index, component in enumerate(
            connected_components(role_indices, self.bonds.pairs), start=1
        ):
            fragments.append(
                self.materialize_fragment(kind, index, component, connecting_bonds)
            )
        return fragments

    def materialize_fragment(
        self,
        kind: FragmentKind,
        index: int,
        component: set[int],
        connecting_bonds: tuple[tuple[int, int], ...],
    ) -> Fragment:
        """Create one capped finite fragment from a source-atom component."""

        source_indices = tuple(sorted(component))
        local_index = {global_index: i for i, global_index in enumerate(source_indices)}
        symbols = [self.symbols[i] for i in source_indices]
        positions = [np.array(self.atoms.positions[i], dtype=float) for i in source_indices]
        local_bonds, unwrap_edges = self._internal_bonds(component, local_index)

        x_sites = self._append_x_caps(
            symbols,
            positions,
            local_bonds,
            unwrap_edges,
            local_index,
            component,
            connecting_bonds,
        )

        unwrapped = unwrap_positions(np.array(positions), self.atoms.cell.array, unwrap_edges)
        centered = center_positions(unwrapped, self.atoms.cell.array)
        fragment_atoms = ase.Atoms(
            symbols=symbols,
            positions=centered,
            cell=self.atoms.cell,
            pbc=self.atoms.pbc,
        )

        return Fragment(
            kind=kind,
            index=index,
            source_indices=source_indices,
            atoms=fragment_atoms,
            bonds=tuple(local_bonds),
            fingerprint=fragment_fingerprint(symbols, local_bonds),
            x_sites=tuple(x_sites),
        )

    def _internal_bonds(
        self, component: set[int], local_index: dict[int, int]
    ) -> tuple[list[tuple[int, int, str]], list[tuple[int, int]]]:
        local_bonds: list[tuple[int, int, str]] = []
        unwrap_edges: list[tuple[int, int]] = []

        for i, j in self.bonds.pairs:
            if i not in component or j not in component:
                continue

            a = local_index[i]
            b = local_index[j]
            local_bonds.append((a, b, self.bonds.bond_type(i, j)))
            unwrap_edges.append((a, b))

        return local_bonds, unwrap_edges

    def _append_x_caps(
        self,
        symbols: list[str],
        positions: list[np.ndarray],
        local_bonds: list[tuple[int, int, str]],
        unwrap_edges: list[tuple[int, int]],
        local_index: dict[int, int],
        component: set[int],
        connecting_bonds: tuple[tuple[int, int], ...],
    ) -> list[XSite]:
        x_sites: list[XSite] = []

        for i, j in connecting_bonds:
            endpoints = self._fragment_cut_endpoints(i, j, component)
            if endpoints is None:
                continue

            internal, external = endpoints
            vector_to_external = self.bonds.vector(internal, external)
            x_position = self.atoms.positions[internal] + 0.5 * vector_to_external
            x_local_index = len(symbols)

            symbols.append(self.x_symbol)
            positions.append(np.array(x_position, dtype=float))
            local_bonds.append((local_index[internal], x_local_index, "S"))
            unwrap_edges.append((local_index[internal], x_local_index))
            x_sites.append(
                XSite(
                    x_index=x_local_index,
                    anchor_index=local_index[internal],
                    source_cut_bond=tuple(sorted((i, j))),
                    internal_source_index=internal,
                    external_source_index=external,
                )
            )

        return x_sites

    @staticmethod
    def _fragment_cut_endpoints(
        i: int, j: int, component: set[int]
    ) -> tuple[int, int] | None:
        if i in component and j not in component:
            return i, j
        if j in component and i not in component:
            return j, i
        return None


def connected_components(
    atom_indices: set[int] | frozenset[int], bonds: list[tuple[int, int]]
) -> list[set[int]]:
    """Return connected components induced by ``atom_indices`` in stable order."""

    graph = nx.Graph()
    graph.add_nodes_from(atom_indices)
    for i, j in bonds:
        if i in atom_indices and j in atom_indices:
            graph.add_edge(i, j)

    return [
        set(component)
        for component in sorted(nx.connected_components(graph), key=lambda item: min(item))
    ]

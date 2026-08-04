"""Decomposition fragment construction and PORMAKE-style cut-bond capping."""

from __future__ import annotations

from typing import Literal

import ase
import networkx as nx
import numpy as np

from .bonding import BondNetwork
from .fingerprint import fragment_fingerprint
from .geometry import center_positions
from .object import Fragment, XSite
from .pbc import (
    ComponentPeriodicity,
    PeriodicBond,
    analyze_component,
    periodic_bonds,
)

FragmentKind = Literal["node", "linker", "metal_node", "organic_node", "edge"]


class FragmentBuilder:
    """Build finite node or linker fragments from source-atom components."""

    def __init__(
        self,
        atoms: ase.Atoms,
        bonds: BondNetwork,
        symbols: list[str],
        x_symbol: str = "X",
        source_records: tuple[PeriodicBond, ...] | None = None,
    ) -> None:
        self.atoms = atoms
        self.bonds = bonds
        self.symbols = symbols
        self.x_symbol = x_symbol
        self.source_records = (
            periodic_bonds(atoms, bonds) if source_records is None else source_records
        )

    def build_role_fragments(
        self,
        kind: FragmentKind,
        role_indices: set[int] | frozenset[int],
        connecting_bonds: tuple[PeriodicBond, ...],
    ) -> list[Fragment]:
        return [
            self._materialize_fragment(
                kind,
                index,
                component,
                self.source_records,
                connecting_bonds,
            )
            for index, component in enumerate(
                connected_components(role_indices, self.bonds.pairs),
                start=1,
            )
        ]

    def materialize_fragment(
        self,
        kind: FragmentKind,
        index: int,
        component: set[int],
        connecting_bonds: tuple[PeriodicBond, ...],
    ) -> Fragment:
        return self._materialize_fragment(
            kind,
            index,
            component,
            self.source_records,
            connecting_bonds,
        )

    def _materialize_fragment(
        self,
        kind: FragmentKind,
        index: int,
        component: set[int],
        source_records: tuple[PeriodicBond, ...],
        connection_records: tuple[PeriodicBond, ...],
    ) -> Fragment:
        source_indices = tuple(sorted(component))
        local_index = {global_index: i for i, global_index in enumerate(source_indices)}
        symbols = [self.symbols[i] for i in source_indices]
        internal_cuts = frozenset(
            record
            for record in connection_records
            if record.i in component and record.j in component
        )
        frame = self._materialization_frame(
            component,
            source_records,
            internal_cuts,
        )
        cell = np.asarray(self.atoms.cell.array, dtype=float)
        positions = [
            np.asarray(self.atoms.positions[source_index], dtype=float)
            + np.asarray(frame.offset(source_index), dtype=int) @ cell
            for source_index in source_indices
        ]
        local_bonds = self._internal_bonds(
            component,
            local_index,
            source_records,
            internal_cuts,
        )
        x_sites = self._append_x_caps(
            symbols,
            positions,
            local_bonds,
            local_index,
            component,
            connection_records,
        )

        centered = center_positions(np.asarray(positions), cell)
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
            source_image_offsets=frame.offsets,
        )

    @staticmethod
    def _materialization_frame(
        component: set[int],
        source_records: tuple[PeriodicBond, ...],
        internal_cuts: frozenset[PeriodicBond],
    ) -> ComponentPeriodicity:
        return analyze_component(
            component,
            source_records,
            excluded=internal_cuts,
        )

    def _internal_bonds(
        self,
        component: set[int],
        local_index: dict[int, int],
        source_records: tuple[PeriodicBond, ...],
        internal_cuts: frozenset[PeriodicBond],
    ) -> list[tuple[int, int, str]]:
        local_bonds: list[tuple[int, int, str]] = []

        for record in source_records:
            i, j = record.pair
            if i not in component or j not in component:
                continue
            if record in internal_cuts:
                continue

            a = local_index[i]
            b = local_index[j]
            local_bonds.append((a, b, self.bonds.bond_type(i, j)))

        return local_bonds

    def _append_x_caps(
        self,
        symbols: list[str],
        positions: list[np.ndarray],
        local_bonds: list[tuple[int, int, str]],
        local_index: dict[int, int],
        component: set[int],
        connection_records: tuple[PeriodicBond, ...],
    ) -> list[XSite]:
        x_sites: list[XSite] = []

        for record in connection_records:
            endpoints = self._fragment_cut_endpoints(record, component)
            if not endpoints:
                continue

            for internal, external in endpoints:
                anchor_index = local_index[internal]
                vector_to_external = self.bonds.vector(internal, external)
                x_position = positions[anchor_index] + 0.5 * vector_to_external
                x_local_index = len(symbols)

                symbols.append(self.x_symbol)
                positions.append(np.asarray(x_position, dtype=float))
                local_bonds.append((anchor_index, x_local_index, "S"))
                x_sites.append(
                    XSite(
                        x_index=x_local_index,
                        anchor_index=anchor_index,
                        source_cut_bond=record.pair,
                        internal_source_index=internal,
                        external_source_index=external,
                        source_cut_shift=record.shift,
                        external_image_shift=record.oriented_shift(internal, external),
                    )
                )
        return x_sites

    @staticmethod
    def _fragment_cut_endpoints(
        record: PeriodicBond,
        component: set[int],
    ) -> tuple[tuple[int, int], ...]:
        i, j = record.pair
        if i in component and j in component:
            return ((i, j), (j, i))
        if i in component and j not in component:
            return ((i, j),)
        if j in component and i not in component:
            return ((j, i),)
        return ()


def connected_components(
    atom_indices: set[int] | frozenset[int], bonds: list[tuple[int, int]]
) -> list[set[int]]:
    graph = nx.Graph()
    graph.add_nodes_from(atom_indices)
    for i, j in bonds:
        if i in atom_indices and j in atom_indices:
            graph.add_edge(i, j)

    return [
        set(component)
        for component in sorted(
            nx.connected_components(graph),
            key=lambda item: min(item),
        )
    ]

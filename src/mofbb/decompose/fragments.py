"""Decomposition fragment construction and PORMAKE-style cut-bond capping."""

from __future__ import annotations

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
from .type_aliases import FragmentKind

ConnectionInput = tuple[int, int] | PeriodicBond


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
            periodic_bonds(atoms, bonds)
            if source_records is None
            else source_records
        )
        self.source_by_pair = {
            record.pair: record for record in self.source_records
        }

    def build_role_fragments(
        self,
        kind: FragmentKind,
        role_indices: set[int] | frozenset[int],
        connecting_bonds: tuple[ConnectionInput, ...],
    ) -> list[Fragment]:
        """Build all connected fragments for one role: ``node`` or ``linker``."""

        source_records, connection_records = self._normalize_connections(
            connecting_bonds
        )
        fragments = []
        for index, component in enumerate(
            connected_components(role_indices, self.bonds.pairs), start=1
        ):
            fragments.append(
                self._materialize_fragment(
                    kind,
                    index,
                    component,
                    source_records,
                    connection_records,
                )
            )
        return fragments

    def materialize_fragment(
        self,
        kind: FragmentKind,
        index: int,
        component: set[int],
        connecting_bonds: tuple[ConnectionInput, ...],
    ) -> Fragment:
        """Create one capped finite fragment from a source-atom component."""

        source_records, connection_records = self._normalize_connections(
            connecting_bonds
        )
        return self._materialize_fragment(
            kind,
            index,
            component,
            source_records,
            connection_records,
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
            kind,
            index,
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
        self._validate_retained_bonds(
            kind,
            index,
            component,
            local_index,
            positions,
            source_records,
            internal_cuts,
        )

        x_sites = self._append_x_caps(
            kind,
            index,
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

    def _normalize_connections(
        self,
        connections: tuple[ConnectionInput, ...],
    ) -> tuple[tuple[PeriodicBond, ...], tuple[PeriodicBond, ...]]:
        normalized: list[PeriodicBond] = []
        seen: set[PeriodicBond] = set()

        for connection in connections:
            if isinstance(connection, PeriodicBond):
                supplied = connection
                pair = supplied.pair
            else:
                pair = tuple(sorted(connection))
                supplied = None

            try:
                source = self.source_by_pair[pair]
            except KeyError as exc:
                raise ValueError(
                    f"Connection {pair} is not present in the source bond network."
                ) from exc
            if supplied is not None and supplied.shift != source.shift:
                raise ValueError(
                    f"Connection {pair} supplies periodic shift {supplied.shift}, "
                    f"but the source bond vector requires {source.shift}."
                )
            if source not in seen:
                normalized.append(source)
                seen.add(source)

        return self.source_records, tuple(normalized)

    @staticmethod
    def _materialization_frame(
        kind: FragmentKind,
        index: int,
        component: set[int],
        source_records: tuple[PeriodicBond, ...],
        internal_cuts: frozenset[PeriodicBond],
    ) -> ComponentPeriodicity:
        cut_pairs = [record.pair for record in sorted(internal_cuts)]
        try:
            frame = analyze_component(
                component,
                source_records,
                excluded=internal_cuts,
            )
        except ValueError as exc:
            raise ValueError(
                f"Fragment {kind} {index} has a disconnected residual bond graph "
                f"after excluding internal cuts {cut_pairs}: {exc}"
            ) from exc
        if frame.rank != 0:
            raise ValueError(
                f"Fragment {kind} {index} residual bond graph has non-zero holonomy "
                f"with rank {frame.rank} and generators {frame.generators} after "
                f"excluding internal cuts {cut_pairs}."
            )
        return frame

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

    def _validate_retained_bonds(
        self,
        kind: FragmentKind,
        index: int,
        component: set[int],
        local_index: dict[int, int],
        positions: list[np.ndarray],
        source_records: tuple[PeriodicBond, ...],
        internal_cuts: frozenset[PeriodicBond],
    ) -> None:
        for record in source_records:
            i, j = record.pair
            if i not in component or j not in component or record in internal_cuts:
                continue
            actual = positions[local_index[j]] - positions[local_index[i]]
            expected = self.bonds.vector(i, j)
            if not np.allclose(actual, expected, rtol=0.0, atol=1.0e-6):
                raise ValueError(
                    f"Fragment {kind} {index} retained distorted source bond "
                    f"{record.pair}: materialized vector {actual.tolist()} does not "
                    f"match the exact oriented bond vector {expected.tolist()}."
                )

    def _append_x_caps(
        self,
        kind: FragmentKind,
        index: int,
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
                self._validate_cap(
                    kind,
                    index,
                    record,
                    internal,
                    external,
                    positions[anchor_index],
                    positions[x_local_index],
                )

        return x_sites

    def _validate_cap(
        self,
        kind: FragmentKind,
        index: int,
        record: PeriodicBond,
        internal: int,
        external: int,
        anchor_position: np.ndarray,
        cap_position: np.ndarray,
    ) -> None:
        actual = cap_position - anchor_position
        expected = 0.5 * self.bonds.vector(internal, external)
        if not np.allclose(actual, expected, rtol=0.0, atol=1.0e-6):
            raise ValueError(
                f"Fragment {kind} {index} cap for source bond {record.pair} at "
                f"anchor {internal} has the wrong direction or length: "
                f"{actual.tolist()} != {expected.tolist()}."
            )

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

"""Post-processing operations on decomposed fragment collections."""

from __future__ import annotations

from collections import defaultdict

from mofbb.decompose.fragments import FragmentBuilder
from mofbb.decompose.object import Fragment
from mofbb.decompose.pbc import PeriodicBond

Key = tuple[str, int]
Result = tuple[
    list[Fragment],
    list[Fragment],
    list[Fragment],
    tuple[PeriodicBond, ...],
]


class TerminalAbsorber:
    """Merge one-neighbor chelating fragments into their adjacent node."""

    def __init__(
        self,
        builder: FragmentBuilder,
        metal: list[Fragment],
        organic: list[Fragment],
        edges: list[Fragment],
        bound: list[Fragment],
        bonds: tuple[PeriodicBond, ...],
    ) -> None:
        self.builder = builder
        self.metal = metal
        self.organic = organic
        self.edges = edges
        self.bound = bound
        self.bonds = bonds

    def run(self) -> Result:
        by_target, removed, cut_bonds = self._plan()
        if not removed:
            return (
                self.metal,
                self.organic,
                self.edges,
                self.bonds,
            )

        kept_bonds = tuple(
            bond for bond in self.bonds if bond not in cut_bonds
        )
        return (
            self._rebuild(
                self.metal,
                by_target,
                kept_bonds,
            ),
            self._rebuild(
                self.organic,
                by_target,
                kept_bonds,
            ),
            [
                frag
                for frag in self.edges
                if self._key(frag) not in removed
            ],
            kept_bonds,
        )

    def _plan(
        self,
    ) -> tuple[
        dict[Key, list[Fragment]],
        set[Key],
        set[PeriodicBond],
    ]:
        available = {
            self._key(frag)
            for frag in (
                *self.metal,
                *self.organic,
                *self.edges,
                *self.bound,
            )
        }
        by_target: dict[Key, list[Fragment]] = defaultdict(list)
        removed = set()
        cut_bonds = set()
        for frag in self.edges:
            target = self._target(frag, available)
            if target is None:
                continue
            by_target[target].append(frag)
            removed.add(self._key(frag))
            cut_bonds.update(
                PeriodicBond(
                    site.source_cut_bond[0],
                    site.source_cut_bond[1],
                    site.source_cut_shift,
                )
                for site in frag.x_sites
            )
        return dict(by_target), removed, cut_bonds

    @staticmethod
    def _target(
        frag: Fragment,
        available: set[Key],
    ) -> Key | None:
        if len(frag.logical_ports) != 1:
            return None
        port = frag.logical_ports[0]
        target = port.neighbor_kind, port.neighbor_index
        if target not in available or target[0] == "edge":
            return None
        return target

    def _rebuild(
        self,
        nodes: list[Fragment],
        by_target: dict[Key, list[Fragment]],
        bonds: tuple[PeriodicBond, ...],
    ) -> list[Fragment]:
        rebuilt = []
        for node in nodes:
            absorbed = by_target.get(self._key(node), ())
            if not absorbed:
                rebuilt.append(node)
                continue
            component = set(node.source_indices)
            for frag in absorbed:
                component.update(frag.source_indices)
            rebuilt.append(
                self.builder.materialize_fragment(
                    node.kind,
                    node.index,
                    component,
                    bonds,
                )
            )
        return rebuilt

    @staticmethod
    def _key(frag: Fragment) -> Key:
        return frag.kind, frag.index

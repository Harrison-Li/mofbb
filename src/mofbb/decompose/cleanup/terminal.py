"""Post-processing operations on decomposed fragment collections."""

from __future__ import annotations

from collections import defaultdict, deque

from mofbb.decompose.fragments import FragmentBuilder
from mofbb.decompose.object import Fragment
from mofbb.decompose.pbc import PeriodicBond, analyze_component

Key = tuple[str, int]
Result = tuple[
    list[Fragment],
    list[Fragment],
    list[Fragment],
    tuple[PeriodicBond, ...],
]


class TerminalAbsorber:
    """Contract finite, one-neighbor fragments, including metal-containing ones.

    Bound solvent remains separate. A periodic neighbor's logical ports may
    quotient distinct images together, so every proposed union also needs an
    explicit periodic-rank check before its cut bonds can be restored.
    """

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
        groups, cut_bonds = self._plan()
        if not cut_bonds:
            return (
                self.metal,
                self.organic,
                self.edges,
                self.bonds,
            )

        kept_bonds = tuple(bond for bond in self.bonds if bond not in cut_bonds)
        rebuilt: dict[str, list[Fragment]] = defaultdict(list)
        for members in groups.values():
            target = members[0]
            if len(members) == 1:
                fragment = target
            else:
                # Reuse a metal member's identity if an organic host absorbs
                # it. Keys stay unique because every source fragment belongs
                # to exactly one group; topology/ports are reassigned afterward.
                metals = [f for f in members if f.kind == "metal_node"]
                if metals:
                    target = min(metals, key=lambda f: f.index)
                component = {i for f in members for i in f.source_indices}
                fragment = self.builder.materialize_fragment(
                    target.kind, target.index, component, kept_bonds
                )
            rebuilt[fragment.kind].append(fragment)
        return (
            sorted(rebuilt["metal_node"], key=lambda f: f.index),
            sorted(rebuilt["organic_node"], key=lambda f: f.index),
            sorted(rebuilt["edge"], key=lambda f: f.index),
            kept_bonds,
        )

    def _plan(
        self,
    ) -> tuple[
        dict[Key, list[Fragment]],
        set[PeriodicBond],
    ]:
        fragments = {self._key(f): f for f in (*self.metal, *self.organic, *self.edges)}
        groups = {key: [fragment] for key, fragment in fragments.items()}
        ports = {
            key: {
                ((p.neighbor_kind, p.neighbor_index), p.translation)
                for p in fragment.logical_ports
            }
            for key, fragment in fragments.items()
        }
        pending = deque(sorted(key for key in groups if len(ports[key]) == 1))
        if not pending:
            return groups, set()
        cuts_by_fragment: dict[Key, set[PeriodicBond]] = {
            key: {
                PeriodicBond(*s.source_cut_bond, s.source_cut_shift)
                for s in fragment.x_sites
            }
            for key, fragment in fragments.items()
        }
        # Index source bonds once; finite-union checks then visit only the
        # proposed group's bonds rather than rescanning the whole crystal.
        records_by_atom: dict[int, set[PeriodicBond]] = defaultdict(set)
        for record in self.builder.source_records:
            records_by_atom[record.i].add(record)
            records_by_atom[record.j].add(record)
        remaining = set(self.bonds)
        restored: set[PeriodicBond] = set()
        while pending:
            key = pending.popleft()
            if key not in groups or len(ports[key]) != 1:
                continue
            target, _ = next(iter(ports[key]))
            if target == key or target not in groups:
                continue
            members = [*groups[target], *groups[key]]
            component = {i for f in members for i in f.source_indices}
            cuts = cuts_by_fragment[key] & remaining
            records = {r for i in component for r in records_by_atom[i]}
            frame = analyze_component(
                component, tuple(records), excluded=(records & remaining) - cuts
            )
            if frame.rank:
                continue
            groups[target] = members
            del groups[key]
            remaining.difference_update(cuts)
            restored.update(cuts)
            ports[target] = {p for p in ports[target] if p[0] != key}
            cuts_by_fragment[target].update(cuts_by_fragment[key])
            if len(ports[target]) == 1:
                pending.append(target)
        return groups, restored

    @staticmethod
    def _key(frag: Fragment) -> Key:
        return frag.kind, frag.index

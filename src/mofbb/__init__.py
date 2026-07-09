"""MOFid SingleNode-inspired decomposition with PORMAKE-style X caps."""

from __future__ import annotations

from typing import Any

from .constants import MOFID_NONMETALS, is_mofid_metal

__all__ = [
    "AtomAssignment",
    "BondNetwork",
    "DecompositionResult",
    "Fragment",
    "FragmentBuilder",
    "MOFID_NONMETALS",
    "SingleNodeAtomAssigner",
    "SingleNodePorDecomposer",
    "TopologyBuilder",
    "TopologyEdge",
    "TopologyVertex",
    "XSite",
    "build_bond_network",
    "is_mofid_metal",
    "largest_connected_component",
]

_LAZY_EXPORTS = {
    "AtomAssignment": (".assignment", "AtomAssignment"),
    "SingleNodeAtomAssigner": (".assignment", "SingleNodeAtomAssigner"),
    "BondNetwork": (".bonding", "BondNetwork"),
    "build_bond_network": (".bonding", "build_bond_network"),
    "largest_connected_component": (".bonding", "largest_connected_component"),
    "SingleNodePorDecomposer": (".decomposer", "SingleNodePorDecomposer"),
    "FragmentBuilder": (".fragments", "FragmentBuilder"),
    "Fragment": (".models", "Fragment"),
    "TopologyEdge": (".models", "TopologyEdge"),
    "TopologyVertex": (".models", "TopologyVertex"),
    "XSite": (".models", "XSite"),
    "DecompositionResult": (".results", "DecompositionResult"),
    "TopologyBuilder": (".topology", "TopologyBuilder"),
}


def __getattr__(name: str) -> Any:
    """Load optional heavy exports lazily.

    This keeps lightweight imports such as ``mofbb.constants``
    usable even in environments where scientific runtime dependencies have not
    been installed yet.
    """

    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attribute_name = _LAZY_EXPORTS[name]
    from importlib import import_module

    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value

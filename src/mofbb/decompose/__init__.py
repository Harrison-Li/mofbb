"""MOF decomposition workflows with PORMAKE-style X caps."""

from __future__ import annotations

from typing import Any

from .constants import MOFID_NONMETALS, is_mofid_metal

__all__ = [
    "MOFID_NONMETALS",
    "AtomAssignment",
    "BondNetwork",
    "DecompositionResult",
    "Fragment",
    "FragmentBuilder",
    "LogicalPort",
    "MofidSolventRemover",
    "NodeEdgeDecomposer",
    "NodeEdgeDecompositionResult",
    "OrganicLinkerSplit",
    "OrganicLinkerSplitter",
    "SingleNodeAtomAssigner",
    "SingleNodeDecomposer",
    "SolventMode",
    "SolventRemovalResult",
    "TopologyBuilder",
    "TopologyEdge",
    "TopologyVertex",
    "XSite",
    "build_bond_network",
    "classify_organic_fragment_kind",
    "decompose_mof_with_edges",
    "is_mofid_metal",
]

_LAZY_EXPORTS = {
    "AtomAssignment": ".assignment",
    "SingleNodeAtomAssigner": ".assignment",
    "BondNetwork": ".bonding",
    "build_bond_network": ".bonding",
    "SingleNodeDecomposer": ".decomposer",
    "MofidSolventRemover": ".solvent",
    "SolventMode": ".solvent",
    "SolventRemovalResult": ".solvent",
    "NodeEdgeDecompositionResult": ".results",
    "DecompositionResult": ".results",
    "NodeEdgeDecomposer": ".edge_decomposer",
    "decompose_mof_with_edges": ".edge_decomposer",
    "OrganicLinkerSplit": ".linker",
    "OrganicLinkerSplitter": ".linker",
    "classify_organic_fragment_kind": ".linker",
    "FragmentBuilder": ".fragments",
    "Fragment": ".object",
    "LogicalPort": ".object",
    "TopologyEdge": ".object",
    "TopologyVertex": ".object",
    "XSite": ".object",
    "TopologyBuilder": ".topology",
}


def __getattr__(name: str) -> Any:
    """Load scientific-runtime exports lazily."""

    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    value = getattr(import_module(_LAZY_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value

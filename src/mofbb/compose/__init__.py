"""MOF composition from topology and building-block inputs."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "Builder",
    "BuildingBlock",
    "AbstractTopologyInput",
    "Framework",
    "Locator",
    "NodeLatticeOptimizer",
    "Reconstructor",
    "Scaler",
    "Topology",
]

_LAZY_EXPORTS = {
    "AbstractTopologyInput": ("mofbb.compose.reconstructor", "AbstractTopologyInput"),
    "Builder": ("mofbb.compose.builder", "Builder"),
    "BuildingBlock": ("mofbb.compose.building_block", "BuildingBlock"),
    "Framework": ("mofbb.compose.framework", "Framework"),
    "Locator": ("mofbb.compose.locator", "Locator"),
    "NodeLatticeOptimizer": (
        "mofbb.compose.node_lattice",
        "NodeLatticeOptimizer",
    ),
    "Reconstructor": ("mofbb.compose.reconstructor", "Reconstructor"),
    "Scaler": ("mofbb.compose.scaler", "Scaler"),
    "Topology": ("mofbb.compose.topology", "Topology"),
}


def __getattr__(name: str) -> Any:
    """Load composition classes only when requested."""

    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attribute_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value

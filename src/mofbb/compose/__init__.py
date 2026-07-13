"""MOF composition from topology and building-block inputs."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "Builder",
    "BuildingBlock",
    "Database",
    "Framework",
    "LocalStructure",
    "Locator",
    "Neighbor",
    "NeighborList",
    "Reconstructor",
    "Scaler",
    "Topology",
]

_LAZY_EXPORTS = {
    "Builder": (".builder", "Builder"),
    "BuildingBlock": (".building_block", "BuildingBlock"),
    "Database": (".database", "Database"),
    "Framework": (".framework", "Framework"),
    "LocalStructure": (".local_structure", "LocalStructure"),
    "Locator": (".locator", "Locator"),
    "Neighbor": (".neighbor", "Neighbor"),
    "NeighborList": (".neighbor", "NeighborList"),
    "Reconstructor": (".reconstructor", "Reconstructor"),
    "Scaler": (".scaler", "Scaler"),
    "Topology": (".topology", "Topology"),
}


def __getattr__(name: str) -> Any:
    """Load composition classes only when requested."""

    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attribute_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value

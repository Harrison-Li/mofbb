"""MOF building-block decomposition and composition tools."""

from __future__ import annotations

from typing import Any

from mofbb import compose, decompose

__all__ = ["compose", "decompose", *decompose.__all__]


def __getattr__(name: str) -> Any:
    """Preserve the original top-level decomposition API lazily."""

    if name not in decompose.__all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(decompose, name)
    globals()[name] = value
    return value

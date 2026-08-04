"""Coordinate utilities for decomposition structures and fragments."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def minimum_image_delta(
    cell: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> np.ndarray:
    """Return the minimum-image Cartesian vector from ``start`` to ``end``."""

    cell = np.asarray(cell, dtype=float)
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    if abs(np.linalg.det(cell)) < 1.0e-12:
        return end - start
    frac_delta = np.linalg.solve(cell.T, end - start)
    frac_delta -= np.round(frac_delta)
    return frac_delta @ cell


def cart_to_frac(cell: np.ndarray, cart: np.ndarray) -> np.ndarray:
    """Convert Cartesian coordinates to fractional coordinates."""

    cell = np.asarray(cell, dtype=float)
    cart = np.asarray(cart, dtype=float)
    if abs(np.linalg.det(cell)) < 1.0e-12:
        return np.array(cart, dtype=float)
    return np.linalg.solve(cell.T, cart)


def unwrap_fractional_near(
    frac_position: Iterable[float], reference_frac: Iterable[float]
) -> tuple[float, float, float]:
    """Translate ``frac_position`` to the periodic image nearest ``reference_frac``."""

    frac = np.array(tuple(frac_position), dtype=float)
    reference = np.array(tuple(reference_frac), dtype=float)
    return tuple(float(value) for value in frac - np.round(frac - reference))


def center_positions(positions: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Move a fragment centroid to the center of the unit cell."""

    if len(positions) == 0:
        return positions
    centroid = positions.mean(axis=0)
    cell_center = np.sum(cell, axis=0) * 0.5
    return positions - centroid + cell_center


def format_float(value: float) -> str:
    """Compact float formatting for CGD files."""

    return f"{value:.10g}"

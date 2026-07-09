import numpy as np

from mofbb.geometry import (
    cart_to_frac,
    minimum_image_delta,
    unwrap_fractional_near,
)


def test_minimum_image_delta_wraps_periodic_boundary() -> None:
    cell = np.eye(3) * 10.0
    start = np.array([9.5, 0.0, 0.0])
    end = np.array([0.5, 0.0, 0.0])

    delta = minimum_image_delta(cell, start, end)

    np.testing.assert_allclose(delta, np.array([1.0, 0.0, 0.0]))


def test_cart_to_frac_handles_orthogonal_cell() -> None:
    cell = np.diag([2.0, 4.0, 5.0])
    cart = np.array([1.0, 2.0, 2.5])

    frac = cart_to_frac(cell, cart)

    np.testing.assert_allclose(frac, np.array([0.5, 0.5, 0.5]))


def test_unwrap_fractional_near_uses_nearest_image() -> None:
    unwrapped = unwrap_fractional_near((0.95, 0.1, 0.1), (0.05, 0.1, 0.1))

    assert unwrapped == (-0.050000000000000044, 0.1, 0.1)

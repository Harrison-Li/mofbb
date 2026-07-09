import pytest

ase = pytest.importorskip("ase")

from mofbb.fingerprint import fragment_fingerprint


def test_fragment_fingerprint_is_stable_for_isomorphic_ordering() -> None:
    symbols_a = ["C", "O", "X"]
    bonds_a = [(0, 1, "S"), (1, 2, "S")]
    symbols_b = ["X", "O", "C"]
    bonds_b = [(0, 1, "S"), (1, 2, "S")]

    assert fragment_fingerprint(symbols_a, bonds_a) == fragment_fingerprint(
        symbols_b, bonds_b
    )

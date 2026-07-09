from mofbb.constants import is_mofid_metal


def test_mofid_metal_classification() -> None:
    assert is_mofid_metal("Zn")
    assert is_mofid_metal("Cu")
    assert not is_mofid_metal("C")
    assert not is_mofid_metal("O")

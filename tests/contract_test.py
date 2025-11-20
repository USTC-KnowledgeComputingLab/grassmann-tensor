import torch
import pytest

from grassmann_tensor import GrassmannTensor


def test_contract() -> None:
    a = GrassmannTensor(
        (False, False, False, True),
        ((2, 2), (4, 4), (8, 8), (8, 8)),
        torch.randn(4, 8, 16, 16, dtype=torch.float64),
    )
    b = GrassmannTensor(
        (False, True, True, True),
        ((8, 8), (4, 4), (4, 4), (32, 32)),
        torch.randn(16, 8, 8, 64, dtype=torch.float64),
    )
    _ = a.contract(b, 3, 0)
    _ = a.contract(b, (0, 2), 3)
    _ = a.contract(b, (0, 2), (1, 2))


def test_contract_assertion() -> None:
    a = GrassmannTensor((False, True), ((1, 0), (1, 0)), torch.randn(1, 1, dtype=torch.float64))
    b = GrassmannTensor(
        (False, True, False, True),
        ((2, 2), (4, 4), (8, 8), (16, 16)),
        torch.randn(4, 8, 16, 32, dtype=torch.float64),
    )
    with pytest.raises(AssertionError, match="Contract requires arrow"):
        _ = a.contract(b, 0, 0)
    with pytest.raises(
        AssertionError, match="All the legs that need to be contracted must have the same arrow"
    ):
        _ = a.contract(b, 0, (0, 1))

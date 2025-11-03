import torch
import pytest

from grassmann_tensor import GrassmannTensor


def test_exponential() -> None:
    a = GrassmannTensor(
        (True, True, True, True),
        ((4, 4), (8, 8), (4, 4), (8, 8)),
        torch.randn(8, 16, 8, 16, dtype=torch.float64),
    )
    a.exponential((0, 3))


def test_exponential_with_empty_parity_block() -> None:
    a = GrassmannTensor((False, True), ((1, 0), (1, 0)), torch.randn(1, 1))
    a.exponential((0,))
    b = GrassmannTensor((False, True), ((0, 1), (0, 1)), torch.randn(1, 1))
    b.exponential((0,))


def test_exponential_assertation() -> None:
    a = GrassmannTensor(
        (True, True, True, True),
        ((2, 2), (4, 4), (8, 8), (16, 16)),
        torch.randn(4, 8, 16, 32, dtype=torch.float64),
    )
    with pytest.raises(AssertionError, match="Exponential requires a square operator"):
        a.exponential((0, 2))

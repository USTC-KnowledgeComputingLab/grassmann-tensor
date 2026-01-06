import torch
import pytest

from grassmann_tensor import GrassmannTensor, NamedGrassmannTensor


@pytest.mark.parametrize(
    "x",
    [
        GrassmannTensor(
            (True, True), ((2, 2), (4, 4)), torch.randn(4, 8, dtype=torch.float64)
        ).update_mask(),
        GrassmannTensor(
            (False, False), ((2, 2), (4, 4)), torch.randn(4, 8, dtype=torch.float64)
        ).update_mask(),
        GrassmannTensor(
            (True, True, True),
            ((2, 2), (4, 4), (8, 8)),
            torch.randn(4, 8, 16, dtype=torch.float64),
        ).update_mask(),
        GrassmannTensor(
            (True, True, True, True),
            ((2, 2), (4, 4), (8, 8), (16, 16)),
            torch.randn(4, 8, 16, 32, dtype=torch.float64),
        ).update_mask(),
    ],
)
def test_reciprocal(x: NamedGrassmannTensor) -> None:
    tensor = x.reciprocal()
    assert tensor.arrow == x.arrow
    assert tensor.edges == x.edges
    assert tensor.tensor.shape == x.tensor.shape
    assert tensor.tensor.dtype == x.tensor.dtype
    assert tensor.tensor.device == x.tensor.device

    assert not torch.isinf(tensor.tensor).any()

    zero = x.tensor == 0
    assert torch.equal(tensor.tensor[zero], x.tensor[zero])

    non_zero = ~zero
    assert torch.allclose(tensor.tensor[non_zero], (1 / x.tensor[non_zero]))


@pytest.mark.parametrize(
    "x",
    [
        NamedGrassmannTensor(
            ("a", "b"), (True, True), ((2, 2), (4, 4)), torch.randn(4, 8, dtype=torch.float64)
        ).update_mask(),
        NamedGrassmannTensor(
            ("a", "b"), (False, False), ((2, 2), (4, 4)), torch.randn(4, 8, dtype=torch.float64)
        ).update_mask(),
        NamedGrassmannTensor(
            ("a", "b", "c"),
            (True, True, True),
            ((2, 2), (4, 4), (8, 8)),
            torch.randn(4, 8, 16, dtype=torch.float64),
        ).update_mask(),
        NamedGrassmannTensor(
            ("a", "b", "c", "d"),
            (True, True, True, True),
            ((2, 2), (4, 4), (8, 8), (16, 16)),
            torch.randn(4, 8, 16, 32, dtype=torch.float64),
        ).update_mask(),
    ],
)
def test_named_reciprocal(x: NamedGrassmannTensor) -> None:
    tensor = x.reciprocal()
    assert tensor.names == x.names
    assert tensor.arrow == x.arrow
    assert tensor.edges == x.edges
    assert tensor.tensor.shape == x.tensor.shape
    assert tensor.tensor.dtype == x.tensor.dtype
    assert tensor.tensor.device == x.tensor.device

    assert not torch.isinf(tensor.tensor).any()

    zero = x.tensor == 0
    assert torch.equal(tensor.tensor[zero], x.tensor[zero])

    non_zero = ~zero
    assert torch.allclose(tensor.tensor[non_zero], (1 / x.tensor[non_zero]))

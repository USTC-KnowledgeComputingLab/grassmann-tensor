import pytest
import torch
from grassmann_tensor import GrassmannTensor

Initialization = tuple[tuple[bool, ...], tuple[tuple[int, int], ...], torch.Tensor]


@pytest.mark.parametrize(
    "x",
    [
        ((False, False), ((2, 2), (1, 3)), torch.randn([4, 4])),
        ((True, False), ((2, 2), (3, 1)), torch.randn([4, 4])),
        ((False, True, False), ((1, 1), (2, 2), (1, 1)), torch.randn([2, 4, 2])),
    ],
)
def test_creation_success(x: Initialization) -> None:
    GrassmannTensor(*x)


@pytest.mark.parametrize(
    "x",
    [
        ((False,), ((2, 2), (1, 3)), torch.randn([4, 4])),
        ((True, False, True), ((2, 2), (3, 1)), torch.randn([4, 4])),
        ((False, True), ((1, 1), (2, 2), (1, 1)), torch.randn([2, 4, 2])),
    ],
)
def test_creation_invalid_arrow(x: Initialization) -> None:
    with pytest.raises(AssertionError, match="Arrow length"):
        GrassmannTensor(*x)


@pytest.mark.parametrize(
    "x",
    [
        ((False, False), ((2, 2),), torch.randn([4, 4])),
        ((True, False), ((2, 2), (1, 1), (3, 1)), torch.randn([4, 4])),
        ((False, True, False), ((1, 1), (1, 1)), torch.randn([2, 4, 2])),
    ],
)
def test_creation_invalid_edges(x: Initialization) -> None:
    with pytest.raises(AssertionError, match="Edges length"):
        GrassmannTensor(*x)


@pytest.mark.parametrize(
    "x",
    [
        ((False, False), ((2, 2), (1, 3)), torch.randn([4, 2])),
        ((True, False), ((2, 2), (3, 1)), torch.randn([2, 4])),
        ((False, True, False), ((1, 1), (2, 2), (1, 1)), torch.randn([4, 4, 2])),
    ],
)
def test_creation_invalid_shape(x: Initialization) -> None:
    with pytest.raises(AssertionError, match="must equal sum of"):
        GrassmannTensor(*x)

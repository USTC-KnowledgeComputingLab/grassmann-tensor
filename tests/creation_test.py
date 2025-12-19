import pytest
import torch
from grassmann_tensor import GrassmannTensor, NamedGrassmannTensor

Initialization = tuple[tuple[bool, ...], tuple[tuple[int, int], ...], torch.Tensor]
NamedInitialization = tuple[
    tuple[str, ...], tuple[bool, ...], tuple[tuple[int, int], ...], torch.Tensor
]


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
        (("a", "b"), (False, False), ((2, 2), (1, 3)), torch.randn([4, 4])),
        (("a", "b"), (True, False), ((2, 2), (3, 1)), torch.randn([4, 4])),
        (("a", "b", "c"), (False, True, False), ((1, 1), (2, 2), (1, 1)), torch.randn([2, 4, 2])),
    ],
)
def test_named_tensor_creation_success(x: NamedInitialization) -> None:
    NamedGrassmannTensor(*x)


@pytest.mark.parametrize(
    "x",
    [
        (("a", "a"), (False, False), ((2, 2), (1, 3)), torch.randn([4, 4])),
        (("a", "b", "a"), (False, True, False), ((1, 1), (2, 2), (1, 1)), torch.randn([2, 4, 2])),
    ],
)
def test_named_tensor_not_unique_names(x: NamedInitialization) -> None:
    with pytest.raises(AssertionError, match="Names must be unique"):
        NamedGrassmannTensor(*x)


@pytest.mark.parametrize(
    "x",
    [
        (("a",), (False, False), ((2, 2), (1, 3)), torch.randn([4, 4])),
        (("a", "b", "c"), (True, False), ((2, 2), (3, 1)), torch.randn([4, 4])),
        (("a", "b"), (False, True, False), ((1, 1), (2, 2), (1, 1)), torch.randn([2, 4, 2])),
    ],
)
def test_named_tensor_invalid_names(x: NamedInitialization) -> None:
    with pytest.raises(AssertionError, match="Names length"):
        NamedGrassmannTensor(*x)


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
        (("a", "b"), (False,), ((2, 2), (1, 3)), torch.randn([4, 4])),
        (("a", "b"), (True, False, True), ((2, 2), (3, 1)), torch.randn([4, 4])),
        (("a", "b", "c"), (False, True), ((1, 1), (2, 2), (1, 1)), torch.randn([2, 4, 2])),
    ],
)
def test_named_tensor_creation_invalid_arrow(x: NamedInitialization) -> None:
    with pytest.raises(AssertionError, match="Arrow length"):
        NamedGrassmannTensor(*x)


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
        (("a", "b"), (False, False), ((2, 2),), torch.randn([4, 4])),
        (("a", "b"), (True, False), ((2, 2), (1, 1), (3, 1)), torch.randn([4, 4])),
        (("a", "b", "c"), (False, True, False), ((1, 1), (1, 1)), torch.randn([2, 4, 2])),
    ],
)
def test_named_tensor_creation_invalid_edges(x: NamedInitialization) -> None:
    with pytest.raises(AssertionError, match="Edges length"):
        NamedGrassmannTensor(*x)


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


@pytest.mark.parametrize(
    "x",
    [
        (("a", "b"), (False, False), ((2, 2), (1, 3)), torch.randn([4, 2])),
        (("a", "b"), (True, False), ((2, 2), (3, 1)), torch.randn([2, 4])),
        (("a", "b", "c"), (False, True, False), ((1, 1), (2, 2), (1, 1)), torch.randn([4, 4, 2])),
    ],
)
def test_named_tensor_creation_invalid_shape(x: NamedInitialization) -> None:
    with pytest.raises(AssertionError, match="must equal sum of"):
        NamedGrassmannTensor(*x)

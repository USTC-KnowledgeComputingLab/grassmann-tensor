import pytest
import torch
import typing

from grassmann_tensor import GrassmannTensor, NamedGrassmannTensor

Tensor: typing.TypeAlias = GrassmannTensor
NamedTensor: typing.TypeAlias = NamedGrassmannTensor
Pairs: typing.TypeAlias = tuple[tuple[int, ...], tuple[int, ...]]
NamedPairs: typing.TypeAlias = set[tuple[str, str]]


def test_identity_assertation() -> None:
    a = GrassmannTensor(
        (True, True, True, True),
        ((2, 2), (4, 4), (8, 8), (16, 16)),
        torch.randn(4, 8, 16, 32, dtype=torch.float64),
    )
    with pytest.raises(AssertionError, match="Identity requires arrow"):
        a.identity(((0, 2), (1, 3)))

    b = GrassmannTensor(
        (False, True, False, True),
        ((2, 2), (4, 4), (8, 8), (16, 16)),
        torch.randn(4, 8, 16, 32, dtype=torch.float64),
    )
    with pytest.raises(AssertionError, match="Identity requires a square operator"):
        b.identity(((0, 2), (1, 3)))

    c = GrassmannTensor(
        (False, True, False, True),
        ((1, 3), (3, 1), (3, 1), (3, 1)),
        torch.randn(4, 4, 4, 4, dtype=torch.float64),
    )
    with pytest.raises(AssertionError, match="Parity blocks must be square"):
        c.identity(((0, 2), (1, 3)))


@pytest.mark.parametrize(
    "tensor, pairs",
    [
        (
            GrassmannTensor(
                (False, True), ((4, 4), (4, 4)), torch.randn(8, 8, dtype=torch.float64)
            ),
            ((0,), (1,)),
        ),
        (
            GrassmannTensor(
                (True, False), ((4, 4), (4, 4)), torch.randn(8, 8, dtype=torch.float64)
            ),
            ((0,), (1,)),
        ),
        (
            GrassmannTensor(
                (False, False, True),
                ((4, 4), (4, 4), (32, 32)),
                torch.randn(8, 8, 64, dtype=torch.float64),
            ),
            ((0, 1), (2,)),
        ),
        (
            GrassmannTensor(
                (False, False, True, True),
                ((4, 4), (8, 8), (4, 4), (8, 8)),
                torch.randn(8, 16, 8, 16, dtype=torch.float64),
            ),
            ((0, 1), (2, 3)),
        ),
    ],
)
def test_identity_via_self_multiplication(
    tensor: Tensor,
    pairs: Pairs,
) -> None:
    identity = tensor.identity(pairs)
    identity, _, _ = identity._group_edges(pairs)
    tensor, _, _ = tensor._group_edges(pairs)
    tensor_reverse_flag = tensor.arrow != (False, True)
    if tensor_reverse_flag:
        identity = identity.reverse((0, 1))
        tensor = tensor.reverse((0, 1))
    assert torch.allclose((identity @ identity).tensor, identity.tensor)
    assert torch.allclose((identity @ tensor).tensor, tensor.tensor)
    assert torch.allclose((tensor @ identity).tensor, tensor.tensor)


def test_named_tensor_identity_assertation() -> None:
    a = NamedGrassmannTensor(
        ("a", "b", "c", "d"),
        (True, True, True, True),
        ((2, 2), (4, 4), (8, 8), (16, 16)),
        torch.randn(4, 8, 16, 32, dtype=torch.float64),
    )
    with pytest.raises(AssertionError, match="Identity requires arrow"):
        a.identity({("a", "b"), ("c", "d")})

    b = NamedGrassmannTensor(
        ("a", "b", "c", "d"),
        (False, True, False, True),
        ((2, 2), (4, 4), (8, 8), (16, 16)),
        torch.randn(4, 8, 16, 32, dtype=torch.float64),
    )
    with pytest.raises(AssertionError, match="Identity requires a square operator"):
        b.identity({("a", "b"), ("c", "d")})

    c = NamedGrassmannTensor(
        ("a", "b", "c", "d"),
        (False, True, False, True),
        ((1, 3), (3, 1), (3, 1), (3, 1)),
        torch.randn(4, 4, 4, 4, dtype=torch.float64),
    )
    with pytest.raises(AssertionError, match="Parity blocks must be square"):
        c.identity({("a", "b"), ("c", "d")})


@pytest.mark.parametrize(
    "tensor, pairs",
    [
        (
            NamedGrassmannTensor(
                ("a", "b"), (False, True), ((4, 4), (4, 4)), torch.randn(8, 8, dtype=torch.float64)
            ),
            {("a", "b")},
        ),
        (
            NamedGrassmannTensor(
                ("a", "b"), (True, False), ((4, 4), (4, 4)), torch.randn(8, 8, dtype=torch.float64)
            ),
            {("a", "b")},
        ),
        (
            NamedGrassmannTensor(
                ("a", "b", "c", "d"),
                (False, False, True, True),
                ((4, 4), (8, 8), (4, 4), (8, 8)),
                torch.randn(8, 16, 8, 16, dtype=torch.float64),
            ),
            {("a", "c"), ("b", "d")},
        ),
        (
            NamedGrassmannTensor(
                ("a", "b", "c", "d"),
                (False, True, False, True),
                ((4, 4), (4, 4), (8, 8), (8, 8)),
                torch.randn(8, 8, 16, 16, dtype=torch.float64),
            ),
            {("a", "b"), ("c", "d")},
        ),
    ],
)
def test_named_tensor_identity_via_self_multiplication(
    tensor: NamedTensor,
    pairs: NamedPairs,
) -> None:
    tensor = tensor.update_mask()
    identity = tensor.identity(pairs)
    contract_pairs = typing.cast(set[tuple[str, str]], {item[::-1] for item in pairs})
    assert identity.contract(identity, contract_pairs).allclose(identity)
    assert (identity.contract(tensor, contract_pairs)).allclose(tensor)
    assert (tensor.contract(identity, contract_pairs)).allclose(tensor)

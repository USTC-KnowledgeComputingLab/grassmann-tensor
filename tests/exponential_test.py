import torch
import pytest
import typing

from grassmann_tensor import GrassmannTensor, NamedGrassmannTensor

Tensor: typing.TypeAlias = GrassmannTensor
Pairs: typing.TypeAlias = tuple[tuple[int, ...], tuple[int, ...]]
NamedTensor: typing.TypeAlias = NamedGrassmannTensor
NamedPairs: typing.TypeAlias = set[tuple[str, str]]


def test_exponential_with_empty_parity_block() -> None:
    a = GrassmannTensor((False, True), ((1, 0), (1, 0)), torch.randn(1, 1, dtype=torch.float64))
    a.exponential(((0,), (1,)))
    b = GrassmannTensor((False, True), ((0, 1), (0, 1)), torch.randn(1, 1, dtype=torch.float64))
    b.exponential(((0,), (1,)))


def test_exponential_assertation() -> None:
    a = GrassmannTensor(
        (True, True, True, True),
        ((2, 2), (4, 4), (8, 8), (16, 16)),
        torch.randn(4, 8, 16, 32, dtype=torch.float64),
    )
    with pytest.raises(AssertionError, match="Exponentiation requires arrow"):
        a.exponential(((0, 2), (1, 3)))

    b = GrassmannTensor(
        (False, True, False, True),
        ((2, 2), (4, 4), (8, 8), (16, 16)),
        torch.randn(4, 8, 16, 32, dtype=torch.float64),
    )
    with pytest.raises(AssertionError, match="Exponentiation requires a square operator"):
        b.exponential(((0, 2), (1, 3)))

    c = GrassmannTensor(
        (False, True, False, True),
        ((1, 3), (3, 1), (3, 1), (3, 1)),
        torch.randn(4, 4, 4, 4, dtype=torch.float64),
    )
    with pytest.raises(AssertionError, match="Parity blocks must be square"):
        c.exponential(((0, 2), (1, 3)))


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
def test_exponential_via_taylor_expansion(
    tensor: Tensor,
    pairs: Pairs,
) -> None:
    tensor_exp = tensor.exponential(pairs)
    iter_tensor = tensor.identity(pairs)
    iter_tensor, _, _ = iter_tensor._group_edges(pairs)
    iter_tensor = iter_tensor.update_mask()
    tensor_group_edges, left_legs, right_legs = tensor._group_edges(pairs)
    tensor_group_edges = tensor_group_edges.update_mask()

    tensor_taylor_expansion = iter_tensor
    for i in range(1, 50):
        iter_tensor @= tensor_group_edges / i
        tensor_taylor_expansion += iter_tensor

    order = left_legs + right_legs
    edges_after_permute = tuple(tensor.edges[i] for i in order)
    tensor_taylor_expansion = tensor_taylor_expansion.reshape(edges_after_permute)
    inv_order = tensor.get_inv_order(order)
    tensor_taylor_expansion = tensor_taylor_expansion.permute(inv_order)

    assert torch.allclose(tensor_taylor_expansion.tensor, tensor_exp.tensor)


def test_named_tensor_exponential_with_empty_parity_block() -> None:
    a = NamedGrassmannTensor(
        ("a", "b"), (False, True), ((1, 0), (1, 0)), torch.randn(1, 1, dtype=torch.float64)
    )
    a.exponential({("a", "b")})
    b = NamedGrassmannTensor(
        ("a", "b"), (False, True), ((0, 1), (0, 1)), torch.randn(1, 1, dtype=torch.float64)
    )
    b.exponential({("a", "b")})


def test_named_tensor_exponential_assertation() -> None:
    a = NamedGrassmannTensor(
        ("a", "b", "c", "d"),
        (True, True, True, True),
        ((2, 2), (4, 4), (8, 8), (16, 16)),
        torch.randn(4, 8, 16, 32, dtype=torch.float64),
    )
    with pytest.raises(AssertionError, match="Exponentiation requires arrow"):
        a.exponential({("a", "b"), ("c", "d")})

    b = NamedGrassmannTensor(
        ("a", "b", "c", "d"),
        (False, True, False, True),
        ((2, 2), (4, 4), (8, 8), (16, 16)),
        torch.randn(4, 8, 16, 32, dtype=torch.float64),
    )
    with pytest.raises(AssertionError, match="Exponentiation requires a square operator"):
        b.exponential({("a", "b"), ("c", "d")})

    c = NamedGrassmannTensor(
        ("a", "b", "c", "d"),
        (False, True, False, True),
        ((1, 3), (3, 1), (3, 1), (3, 1)),
        torch.randn(4, 4, 4, 4, dtype=torch.float64),
    )
    with pytest.raises(AssertionError, match="Parity blocks must be square"):
        c.exponential({("a", "b"), ("c", "d")})

    d = NamedGrassmannTensor(
        ("a", "b", "c", "d"),
        (True, True, True, True),
        ((2, 2), (4, 4), (8, 8), (16, 16)),
        torch.randn(4, 8, 16, 32, dtype=torch.float64),
    )
    with pytest.raises(AssertionError, match="Input pairs must cover all dimension and disjoint"):
        d.exponential({("a", "b"), ("a", "c"), ("a", "d")})


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
def test_named_tensor_exponential_via_taylor_expansion(
    tensor: NamedTensor,
    pairs: NamedPairs,
) -> None:
    tensor = tensor.update_mask()
    tensor_exp = tensor.exponential(pairs)
    iter_tensor = tensor.identity(pairs)
    contract_pairs = typing.cast(set[tuple[str, str]], {item[::-1] for item in pairs})

    tensor_taylor_expansion = iter_tensor
    for i in range(1, 50):
        iter_tensor = iter_tensor.contract(tensor, contract_pairs) / i
        tensor_taylor_expansion += iter_tensor

    assert torch.allclose(tensor_taylor_expansion.tensor, tensor_exp.tensor)

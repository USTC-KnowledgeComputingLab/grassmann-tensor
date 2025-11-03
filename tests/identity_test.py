import pytest
import torch
from typing import TypeAlias

from grassmann_tensor import GrassmannTensor

Tensor: TypeAlias = GrassmannTensor
Pairs: TypeAlias = tuple[tuple[int, ...], tuple[int, ...]]


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

import pytest
import torch
from grassmann_tensor import GrassmannTensor


@pytest.mark.parametrize("arrow", [(i, j, k, l, m) for i in [False, True] for j in [False, True] for k in [False, True] for l in [False, True] for m in [False, True]])
@pytest.mark.parametrize("plan_range", [(i, j) for i in range(5) for j in range(5) if j > i])
def test_reshape_consistency(arrow: tuple[bool, ...], plan_range: tuple[int, int]) -> None:
    l, h = plan_range
    if not all(arrow[l:h]) and any(arrow[l:h]):
        pytest.skip("Invalid reshape plan for the given arrow configuration.")
    edge = (2, 2)
    a = GrassmannTensor(arrow, (edge, edge, edge, edge, edge), torch.randn([4, 4, 4, 4, 4]))
    plan = tuple([-1] * l + [4**(h - l)] + [-1] * (5 - h))
    b = a.reshape(plan)
    c = b.reshape(a.edges)
    assert torch.allclose(a.tensor, c.tensor)


def test_reshape_merging_dimension_mismatch_edges() -> None:
    arrow = (True, True, True)
    edges = ((2, 2), (8, 8), (2, 2))
    a = GrassmannTensor(arrow, edges, torch.randn([4, 16, 4]))
    _ = a.reshape((64, -1))
    _ = a.reshape((-1, 64))
    with pytest.raises(AssertionError, match="Dimension mismatch with edges"):
        _ = a.reshape((16, -1, -1))


def test_reshape_merging_new_shape_exceeds() -> None:
    arrow = (True,)
    edges = ((2, 2),)
    a = GrassmannTensor(arrow, edges, torch.randn([4]))
    with pytest.raises(AssertionError, match="exceeds tensor dimensions"):
        _ = a.reshape((16, -1))


def test_reshape_merging_even_odd_mismatch() -> None:
    arrow = (True, True, True)
    edges = ((2, 2), (8, 8), (2, 2))
    a = GrassmannTensor(arrow, edges, torch.randn([4, 16, 4]))
    _ = a.reshape(((32, 32), (2, 2)))
    _ = a.reshape(((2, 2), (32, 32)))
    with pytest.raises(AssertionError, match="New even and odd number mismatch during merging"):
        _ = a.reshape(((30, 34), (2, 2)))


def test_reshape_merging_mixed_arrows() -> None:
    arrow = (True, False, True)
    edges = ((2, 2), (2, 2), (2, 2))
    a = GrassmannTensor(arrow, edges, torch.randn([4, 4, 4]))
    with pytest.raises(AssertionError, match="Cannot merge edges with different arrows"):
        _ = a.reshape((64,))


def test_reshape_splitting_shape_type() -> None:
    arrow = (True,)
    edges = ((8, 8),)
    a = GrassmannTensor(arrow, edges, torch.randn([16]))
    _ = a.reshape(((2, 2), (2, 2)))
    with pytest.raises(AssertionError, match="New shape must be a pair when splitting"):
        _ = a.reshape((2, (2, 2)))


def test_reshape_splitting_dimension_mismatch_edges() -> None:
    arrow = (True,)
    edges = ((8, 8),)
    a = GrassmannTensor(arrow, edges, torch.randn([16]))
    _ = a.reshape(((2, 2), (2, 2)))
    with pytest.raises(AssertionError, match="Dimension mismatch with edges"):
        _ = a.reshape(((4, 4), (2, 2)))


def test_reshape_splitting_shape_exceeds() -> None:
    arrow = (False,)
    edges = ((2, 2),)
    a = GrassmannTensor(arrow, edges, torch.randn([4]))
    with pytest.raises(AssertionError, match="exceeds specified dimensions"):
        _ = a.reshape(((3, 0), (0, 1)))


def test_reshape_splitting_even_odd_mismatch() -> None:
    arrow = (False,)
    edges = ((6, 10),)
    a = GrassmannTensor(arrow, edges, torch.randn([16]))
    _ = a.reshape(((1, 3), (3, 1)))
    with pytest.raises(AssertionError, match="New even and odd number mismatch during splitting"):
        _ = a.reshape(((2, 2), (2, 2)))

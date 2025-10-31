import random
import pytest
import torch
from grassmann_tensor import GrassmannTensor


@pytest.mark.parametrize(
    "arrow",
    [
        (i, j, k, l, m)
        for i in [False, True]
        for j in [False, True]
        for k in [False, True]
        for l in [False, True]  # noqa: E741
        for m in [False, True]
    ],
)
@pytest.mark.parametrize("plan_range", [(i, j) for i in range(5) for j in range(5) if j > i])
def test_reshape_consistency(arrow: tuple[bool, ...], plan_range: tuple[int, int]) -> None:
    l, h = plan_range  # noqa: E741
    if not all(arrow[l:h]) and any(arrow[l:h]):
        pytest.skip("Invalid reshape plan for the given arrow configuration.")
    edge = (2, 2)
    a = GrassmannTensor(arrow, (edge, edge, edge, edge, edge), torch.randn([4, 4, 4, 4, 4]))
    plan = tuple([-1] * l + [4 ** (h - l)] + [-1] * (5 - h))
    b = a.reshape(plan)
    c = b.reshape(a.edges)
    assert torch.allclose(a.tensor, c.tensor)


def insert_trivial_between_elements(
    input_list: list[tuple[int, int] | int], p: float
) -> list[tuple[int, int] | int]:
    result: list[tuple[int, int] | int] = []
    for i in input_list:
        if random.random() < p:
            result.append((1, 0))
        result.append(i)
    return result


@pytest.mark.parametrize(
    "arrow",
    [
        (i, j, k, l, m)
        for i in [False, True]
        for j in [False, True]
        for k in [False, True]
        for l in [False, True]  # noqa: E741
        for m in [False, True]
    ],
)
@pytest.mark.parametrize("plan_range", [(i, j) for i in range(5) for j in range(5) if j > i])
def test_reshape_trivial_edges(arrow: tuple[bool, ...], plan_range: tuple[int, int]) -> None:
    l, h = plan_range  # noqa: E741
    if not all(arrow[l:h]) and any(arrow[l:h]):
        pytest.skip("Invalid reshape plan for the given arrow configuration.")
    edge = (2, 2)
    a = GrassmannTensor(arrow, (edge, edge, edge, edge, edge), torch.randn([4, 4, 4, 4, 4]))
    plan = tuple(insert_trivial_between_elements([-1] * l + [4 ** (h - l)] + [-1] * (5 - h), 0.5))
    b = a.reshape(plan)
    c = b.reshape(a.edges)
    assert a.edges == c.edges


def test_reshape_merging_dimension_mismatch_edges_because_of_unequal() -> None:
    arrow = (True, True, True)
    edges = ((2, 2), (8, 8), (2, 2))
    a = GrassmannTensor(arrow, edges, torch.randn([4, 16, 4]))
    _ = a.reshape((64, -1))
    _ = a.reshape((-1, 64))
    with pytest.raises(AssertionError, match="Dimension mismatch in merging"):
        _ = a.reshape((16, -1, -1))


def test_reshape_merging_dimension_mismatch_edges_because_of_different_even_odd() -> None:
    arrow = (True, True, True, True, True)
    edges = ((0, 1), (1, 3), (1, 3), (0, 1), (2, 2))
    a = GrassmannTensor(arrow, edges, torch.randn([1, 4, 4, 1, 4]))
    _ = a.reshape((16, -1, -1))
    _ = a.reshape(((6, 10), -1, -1))
    _ = a.reshape(((10, 6), -1))
    _ = a.reshape((4, -1, -1, -1))
    _ = a.reshape(((3, 1), -1, -1, -1))
    with pytest.raises(AssertionError, match="Dimension mismatch in merging"):
        _ = a.reshape(((2, 2), -1, -1, -1))
    with pytest.raises(AssertionError, match="Dimension mismatch in merging"):
        _ = a.reshape(((1, 3), -1, -1, -1))


def test_reshape_merging_new_shape_exceeds() -> None:
    arrow = (True,)
    edges = ((2, 2),)
    a = GrassmannTensor(arrow, edges, torch.randn([4]))
    with pytest.raises(AssertionError, match="New shape exceeds in merging"):
        _ = a.reshape((16, -1))


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


def test_reshape_splitting_dimension_mismatch_edges_because_of_unequal() -> None:
    arrow = (True,)
    edges = ((8, 8),)
    a = GrassmannTensor(arrow, edges, torch.randn([16]))
    _ = a.reshape(((2, 2), (2, 2)))
    with pytest.raises(AssertionError, match="Dimension mismatch in splitting"):
        _ = a.reshape(((4, 4), (2, 2)))


def test_reshape_splitting_dimension_mismatch_edges_because_of_different_even_odd() -> None:
    arrow = (True, True)
    edges = ((3, 1), (2, 2))
    a = GrassmannTensor(arrow, edges, torch.randn([4, 4]))
    _ = a.reshape(((0, 1), (3, 1), (0, 1), (2, 2)))
    with pytest.raises(AssertionError, match="Dimension mismatch in splitting"):
        _ = a.reshape(((0, 1), (2, 2), (0, 1), (2, 2)))
    with pytest.raises(AssertionError, match="Dimension mismatch in splitting"):
        _ = a.reshape(((0, 1), (3, 1), (2, 2)))


def test_reshape_splitting_shape_exceeds() -> None:
    arrow = (False,)
    edges = ((8, 8),)
    a = GrassmannTensor(arrow, edges, torch.randn([16]))
    with pytest.raises(AssertionError, match="New shape exceeds in splitting"):
        _ = a.reshape(((1, 1), (1, 1)))


def test_reshape_equal_edges_trivial() -> None:
    arrow = (True,)
    edges = ((2, 2),)
    a = GrassmannTensor(arrow, edges, torch.randn([4]))
    _ = a.reshape((4,))
    _ = a.reshape(((2, 2),))


def test_reshape_equal_edges_nontrivial_splitting() -> None:
    arrow = (True,)
    edges = ((1, 3),)
    a = GrassmannTensor(arrow, edges, torch.randn([4]))
    _ = a.reshape(((3, 1), (1, 0), (0, 1)))


def test_reshape_equal_edges_nontrivial_splitting_with_other_edge() -> None:
    arrow = (True, True)
    edges = ((1, 3), (2, 2))
    a = GrassmannTensor(arrow, edges, torch.randn([4, 4]))
    _ = a.reshape(((3, 1), (1, 0), (0, 1), (2, 2)))


def test_reshape_equal_edges_nontrivial_merging() -> None:
    arrow = (True, True, True)
    edges = ((1, 3), (1, 0), (0, 1))
    a = GrassmannTensor(arrow, edges, torch.randn([4, 1, 1]))
    _ = a.reshape(((3, 1),))


def test_reshape_equal_edges_nontrivial_merging_with_other_edge() -> None:
    arrow = (True, True, True, True)
    edges = ((1, 3), (1, 0), (0, 1), (2, 2))
    a = GrassmannTensor(arrow, edges, torch.randn([4, 1, 1, 4]))
    _ = a.reshape(((3, 1), (2, 2)))


def test_reshape_with_none() -> None:
    a = GrassmannTensor((), (), torch.tensor(2333)).reshape(((1, 0), (1, 0))).reshape(())
    assert len(a.arrow) == 0 and len(a.edges) == 0 and a.tensor.dim() == 0
    b = GrassmannTensor((), (), torch.tensor(2333)).reshape(((1, 0), (1, 0))).reshape(())
    assert len(b.arrow) == 0 and len(b.edges) == 0 and b.tensor.dim() == 0
    c = GrassmannTensor((), (), torch.tensor(2333)).reshape((1, 1))
    assert len(c.arrow) == 2 and len(c.edges) == 2 and c.tensor.dim() == 2


def test_reshape_with_none_edge_assertion() -> None:
    with pytest.raises(AssertionError, match="Only pure even edges can be merged into none edges"):
        _ = GrassmannTensor((True, True), ((0, 1), (1, 0)), torch.tensor([[2333]])).reshape(())
    with pytest.raises(AssertionError, match="Cannot split none edges into illegal edges"):
        _ = GrassmannTensor((), (), torch.tensor(2333)).reshape(((0, 1),))
    with pytest.raises(AssertionError, match="Cannot split none edges into illegal edges"):
        _ = GrassmannTensor((), (), torch.tensor(2333)).reshape(((0, 1), (1, 0)))
    with pytest.raises(AssertionError, match="Cannot use -1 when reshaping from a scalar"):
        _ = GrassmannTensor((), (), torch.tensor(2333)).reshape((1, -1))
    with pytest.raises(AssertionError, match="Ambiguous integer dim"):
        _ = GrassmannTensor((), (), torch.tensor(2333)).reshape((2, 2))


@pytest.mark.parametrize(
    "arrow, edges, tensor",
    [
        ((True, True), ((0, 1), (0, 1)), torch.tensor([[2333]])),
        ((True, True, True), ((0, 1), (1, 0), (0, 1)), torch.tensor([[[2333]]])),
    ],
)
@pytest.mark.parametrize(
    "shape",
    [
        (1,),
        (1, 1),
        (1, 1, 1),
        (1, 1, 1, 1),
    ],
)
def test_reshape_with_one_dimension(
    arrow: tuple[bool, ...],
    edges: tuple[tuple[int, int], ...],
    tensor: torch.Tensor,
    shape: tuple[int, ...],
) -> None:
    a = GrassmannTensor(arrow, edges, tensor).reshape(shape)
    assert (
        len(a.arrow) == len(shape) and len(a.edges) == len(shape) and a.tensor.dim() == len(shape)
    )


def test_reshape_trailing_nontrivial_dim_raises() -> None:
    a = GrassmannTensor((True,), ((2, 2),), torch.randn([4]))
    with pytest.raises(AssertionError, match="New shape exceeds after exhausting self dimensions"):
        _ = a.reshape((-1, (2, 2)))


@pytest.mark.parametrize(
    "tensor",
    [
        GrassmannTensor(
            (True, True, True, True),
            ((1, 0), (1, 0), (2, 2), (8, 8)),
            torch.randn(1, 1, 4, 16),
        ),
    ],
)
@pytest.mark.parametrize(
    "shape",
    [
        (1, 64),
        ((1, 0), 64),
        (-1, 64),
    ],
)
def test_reshape_trivial_head_equivalence(
    tensor: GrassmannTensor,
    shape: tuple[int, ...],
) -> None:
    baseline_tensor = tensor.reshape((1, 64))
    actual_tensor = tensor.reshape(shape)

    assert actual_tensor.edges == ((1, 0), (32, 32))
    assert torch.allclose(actual_tensor.tensor, baseline_tensor.tensor)

    roundtrip_tensor = actual_tensor.reshape(tensor.edges)
    assert torch.allclose(roundtrip_tensor.tensor, tensor.tensor)

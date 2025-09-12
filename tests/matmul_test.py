import pytest
import torch
from grassmann_tensor import GrassmannTensor

Broadcast = tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
MatmulCase = tuple[bool, bool, tuple[int, int], tuple[int, int], tuple[int, int]]


@pytest.mark.parametrize("a_is_vector", [False, True])
@pytest.mark.parametrize("b_is_vector", [False, True])
@pytest.mark.parametrize("normal_arrow_order", [False, True])
@pytest.mark.parametrize(
    "broadcast",
    [
        ((), (), ()),
        ((2,), (), (2,)),
        ((), (3,), (3,)),
        ((1,), (4,), (4,)),
        ((5,), (1,), (5,)),
        ((6,), (6,), (6,)),
        ((7, 8), (7, 8), (7, 8)),
        ((1, 8), (7, 8), (7, 8)),
        ((8,), (7, 8), (7, 8)),
        ((7, 1), (7, 8), (7, 8)),
        ((7, 8), (1, 8), (7, 8)),
        ((7, 8), (8,), (7, 8)),
        ((7, 8), (7, 1), (7, 8)),
    ],
)
@pytest.mark.parametrize(
    "x",
    [
        (False, False, (1, 1), (1, 1), (1, 1)),
        (False, True, (1, 1), (1, 1), (1, 1)),
        (True, False, (1, 1), (1, 1), (1, 1)),
        (True, True, (1, 1), (1, 1), (1, 1)),
        (False, False, (2, 2), (2, 2), (2, 2)),
        (False, True, (2, 2), (2, 2), (2, 2)),
        (True, False, (2, 2), (2, 2), (2, 2)),
        (True, True, (2, 2), (2, 2), (2, 2)),
    ],
)
def test_matmul(
    a_is_vector: bool,
    b_is_vector: bool,
    normal_arrow_order: bool,
    broadcast: Broadcast,
    x: MatmulCase,
) -> None:
    broadcast_a, broadcast_b, broadcast_result = broadcast
    arrow_a, arrow_b, edge_a, edge_common, edge_b = x
    if a_is_vector and broadcast_a != ():
        pytest.skip("Vector a cannot be broadcasted")
    if b_is_vector and broadcast_b != ():
        pytest.skip("Vector b cannot be broadcasted")
    dim_a = sum(edge_a)
    dim_common = sum(edge_common)
    dim_b = sum(edge_b)
    if a_is_vector:
        a = GrassmannTensor(
            (*(False for _ in broadcast_a), True if normal_arrow_order else False),
            (*((i, 0) for i in broadcast_a), edge_common),
            torch.randn([*broadcast_a, dim_common]),
        ).update_mask()
    else:
        a = GrassmannTensor(
            (*(False for _ in broadcast_a), arrow_a, True if normal_arrow_order else False),
            (*((i, 0) for i in broadcast_a), edge_a, edge_common),
            torch.randn([*broadcast_a, dim_a, dim_common]),
        ).update_mask()
    if b_is_vector:
        b = GrassmannTensor(
            (*(False for _ in broadcast_b), False if normal_arrow_order else True),
            (*((i, 0) for i in broadcast_b), edge_common),
            torch.randn([*broadcast_b, dim_common]),
        ).update_mask()
    else:
        b = GrassmannTensor(
            (*(False for _ in broadcast_b), False if normal_arrow_order else True, arrow_b),
            (*((i, 0) for i in broadcast_b), edge_common, edge_b),
            torch.randn([*broadcast_b, dim_common, dim_b]),
        ).update_mask()
    c = a.matmul(b)
    expected = a.tensor.matmul(b.tensor)
    if not a_is_vector and not b_is_vector and not normal_arrow_order:
        expected[..., edge_a[0] :, edge_b[0] :] *= -1
    if a_is_vector:
        if b_is_vector:
            assert c.arrow == tuple(False for _ in broadcast_result)
            assert c.edges == tuple((i, 0) for i in broadcast_result)
        else:
            assert c.arrow == (*(False for _ in broadcast_result), arrow_b)
            assert c.edges == (*((i, 0) for i in broadcast_result), edge_b)
    else:
        if b_is_vector:
            assert c.arrow == (*(False for _ in broadcast_result), arrow_a)
            assert c.edges == (*((i, 0) for i in broadcast_result), edge_a)
        else:
            assert c.arrow == (*(False for _ in broadcast_result), arrow_a, arrow_b)
            assert c.edges == (*((i, 0) for i in broadcast_result), edge_a, edge_b)
    assert torch.allclose(c.tensor, expected)

import pytest
import torch
from grassmann_tensor import GrassmannTensor

MatmulMatrixCase = tuple[bool, bool, tuple[int, int], tuple[int, int], tuple[int, int]]


@pytest.mark.parametrize("x", [
    (False, False, (1, 1), (1, 1), (1, 1)),
    (False, True, (1, 1), (1, 1), (1, 1)),
    (True, False, (1, 1), (1, 1), (1, 1)),
    (True, True, (1, 1), (1, 1), (1, 1)),
    (False, False, (2, 2), (2, 2), (2, 2)),
    (False, True, (2, 2), (2, 2), (2, 2)),
    (True, False, (2, 2), (2, 2), (2, 2)),
    (True, True, (2, 2), (2, 2), (2, 2)),
])
def test_matmul_matrix_tf(x: MatmulMatrixCase) -> None:
    arrow_a, arrow_b, edge_a, edge_common, edge_b = x
    dim_a = sum(edge_a)
    dim_common = sum(edge_common)
    dim_b = sum(edge_b)
    a = GrassmannTensor((arrow_a, True), (edge_a, edge_common), torch.randn([dim_a, dim_common])).update_mask()
    b = GrassmannTensor((False, arrow_b), (edge_common, edge_b), torch.randn([dim_common, dim_b])).update_mask()
    c = a.matmul(b)
    expected = a.tensor.matmul(b.tensor)
    assert c.arrow == (arrow_a, arrow_b)
    assert c.edges == (edge_a, edge_b)
    assert torch.allclose(c.tensor, expected)


@pytest.mark.parametrize("x", [
    (False, False, (1, 1), (1, 1), (1, 1)),
    (False, True, (1, 1), (1, 1), (1, 1)),
    (True, False, (1, 1), (1, 1), (1, 1)),
    (True, True, (1, 1), (1, 1), (1, 1)),
    (False, False, (2, 2), (2, 2), (2, 2)),
    (False, True, (2, 2), (2, 2), (2, 2)),
    (True, False, (2, 2), (2, 2), (2, 2)),
    (True, True, (2, 2), (2, 2), (2, 2)),
])
def test_matmul_matrix_ft(x: MatmulMatrixCase) -> None:
    arrow_a, arrow_b, edge_a, edge_common, edge_b = x
    dim_a = sum(edge_a)
    dim_common = sum(edge_common)
    dim_b = sum(edge_b)
    a = GrassmannTensor((arrow_a, False), (edge_a, edge_common), torch.randn([dim_a, dim_common])).update_mask()
    b = GrassmannTensor((True, arrow_b), (edge_common, edge_b), torch.randn([dim_common, dim_b])).update_mask()
    c = a.matmul(b)
    expected = a.tensor.matmul(b.tensor)
    expected[edge_a[0]:, edge_b[0]:] *= -1
    assert c.arrow == (arrow_a, arrow_b)
    assert c.edges == (edge_a, edge_b)
    assert torch.allclose(c.tensor, expected)

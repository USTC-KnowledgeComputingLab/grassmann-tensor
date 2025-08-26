import pytest
import torch
from grassmann_tensor import GrassmannTensor

MatmulMatrixMatrixCase = tuple[bool, bool, tuple[int, int], tuple[int, int], tuple[int, int]]


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
def test_matmul_matrix_matrix_tf(x: MatmulMatrixMatrixCase) -> None:
    arrow_a, arrow_b, edge_a, edge_common, edge_b = x
    dim_a = sum(edge_a)
    dim_common = sum(edge_common)
    dim_b = sum(edge_b)
    a = GrassmannTensor(
        (arrow_a, True), (edge_a, edge_common), torch.randn([dim_a, dim_common])
    ).update_mask()
    b = GrassmannTensor(
        (False, arrow_b), (edge_common, edge_b), torch.randn([dim_common, dim_b])
    ).update_mask()
    c = a.matmul(b)
    expected = a.tensor.matmul(b.tensor)
    assert c.arrow == (arrow_a, arrow_b)
    assert c.edges == (edge_a, edge_b)
    assert torch.allclose(c.tensor, expected)


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
def test_matmul_matrix_matrix_ft(x: MatmulMatrixMatrixCase) -> None:
    arrow_a, arrow_b, edge_a, edge_common, edge_b = x
    dim_a = sum(edge_a)
    dim_common = sum(edge_common)
    dim_b = sum(edge_b)
    a = GrassmannTensor(
        (arrow_a, False), (edge_a, edge_common), torch.randn([dim_a, dim_common])
    ).update_mask()
    b = GrassmannTensor(
        (True, arrow_b), (edge_common, edge_b), torch.randn([dim_common, dim_b])
    ).update_mask()
    c = a.matmul(b)
    expected = a.tensor.matmul(b.tensor)
    expected[edge_a[0] :, edge_b[0] :] *= -1
    assert c.arrow == (arrow_a, arrow_b)
    assert c.edges == (edge_a, edge_b)
    assert torch.allclose(c.tensor, expected)


MatmulMatrixVectorCase = tuple[bool, tuple[int, int], tuple[int, int]]


@pytest.mark.parametrize(
    "x",
    [
        (False, (1, 1), (1, 1)),
        (True, (1, 1), (1, 1)),
        (False, (2, 2), (2, 2)),
        (True, (2, 2), (2, 2)),
    ],
)
def test_matmul_matrix_vector_tf(x: MatmulMatrixVectorCase) -> None:
    arrow_a, edge_a, edge_common = x
    dim_a = sum(edge_a)
    dim_common = sum(edge_common)
    a = GrassmannTensor(
        (arrow_a, True), (edge_a, edge_common), torch.randn([dim_a, dim_common])
    ).update_mask()
    b = GrassmannTensor((False,), (edge_common,), torch.randn([dim_common])).update_mask()
    c = a.matmul(b)
    expected = a.tensor.matmul(b.tensor)
    assert c.arrow == (arrow_a,)
    assert c.edges == (edge_a,)
    assert torch.allclose(c.tensor, expected)


@pytest.mark.parametrize(
    "x",
    [
        (False, (1, 1), (1, 1)),
        (True, (1, 1), (1, 1)),
        (False, (2, 2), (2, 2)),
        (True, (2, 2), (2, 2)),
    ],
)
def test_matmul_matrix_vector_ft(x: MatmulMatrixVectorCase) -> None:
    arrow_a, edge_a, edge_common = x
    dim_a = sum(edge_a)
    dim_common = sum(edge_common)
    a = GrassmannTensor(
        (arrow_a, False), (edge_a, edge_common), torch.randn([dim_a, dim_common])
    ).update_mask()
    b = GrassmannTensor((True,), (edge_common,), torch.randn([dim_common])).update_mask()
    c = a.matmul(b)
    expected = a.tensor.matmul(b.tensor)
    assert c.arrow == (arrow_a,)
    assert c.edges == (edge_a,)
    assert torch.allclose(c.tensor, expected)


@pytest.mark.parametrize(
    "x",
    [
        (False, (1, 1), (1, 1)),
        (True, (1, 1), (1, 1)),
        (False, (2, 2), (2, 2)),
        (True, (2, 2), (2, 2)),
    ],
)
def test_matmul_vector_matrix_ft(x: MatmulMatrixVectorCase) -> None:
    arrow_b, edge_common, edge_b = x
    dim_common = sum(edge_common)
    dim_b = sum(edge_b)
    a = GrassmannTensor((False,), (edge_common,), torch.randn([dim_common])).update_mask()
    b = GrassmannTensor(
        (True, arrow_b), (edge_common, edge_b), torch.randn([dim_common, dim_b])
    ).update_mask()
    c = a.matmul(b)
    expected = a.tensor.matmul(b.tensor)
    assert c.arrow == (arrow_b,)
    assert c.edges == (edge_b,)
    assert torch.allclose(c.tensor, expected)


@pytest.mark.parametrize(
    "x",
    [
        (False, (1, 1), (1, 1)),
        (True, (1, 1), (1, 1)),
        (False, (2, 2), (2, 2)),
        (True, (2, 2), (2, 2)),
    ],
)
def test_matmul_vector_matrix_tf(x: MatmulMatrixVectorCase) -> None:
    arrow_b, edge_common, edge_b = x
    dim_common = sum(edge_common)
    dim_b = sum(edge_b)
    a = GrassmannTensor((True,), (edge_common,), torch.randn([dim_common])).update_mask()
    b = GrassmannTensor(
        (False, arrow_b), (edge_common, edge_b), torch.randn([dim_common, dim_b])
    ).update_mask()
    c = a.matmul(b)
    expected = a.tensor.matmul(b.tensor)
    assert c.arrow == (arrow_b,)
    assert c.edges == (edge_b,)
    assert torch.allclose(c.tensor, expected)


MatmulVectorVectorCase = tuple[bool, tuple[int, int]]


@pytest.mark.parametrize(
    "x",
    [
        (False, (1, 1)),
        (True, (1, 1)),
        (False, (2, 2)),
        (True, (2, 2)),
    ],
)
def test_matmul_vector_vector_ft(x: MatmulVectorVectorCase) -> None:
    arrow_b, edge_common = x
    dim_common = sum(edge_common)
    a = GrassmannTensor((False,), (edge_common,), torch.randn([dim_common])).update_mask()
    b = GrassmannTensor((True,), (edge_common,), torch.randn([dim_common])).update_mask()
    c = a.matmul(b)
    expected = a.tensor.matmul(b.tensor)
    assert c.arrow == ()
    assert c.edges == ()
    assert torch.allclose(c.tensor, expected)


@pytest.mark.parametrize(
    "x",
    [
        (False, (1, 1)),
        (True, (1, 1)),
        (False, (2, 2)),
        (True, (2, 2)),
    ],
)
def test_matmul_vector_vector_tf(x: MatmulVectorVectorCase) -> None:
    arrow_b, edge_common = x
    dim_common = sum(edge_common)
    a = GrassmannTensor((True,), (edge_common,), torch.randn([dim_common])).update_mask()
    b = GrassmannTensor((False,), (edge_common,), torch.randn([dim_common])).update_mask()
    c = a.matmul(b)
    expected = a.tensor.matmul(b.tensor)
    assert c.arrow == ()
    assert c.edges == ()
    assert torch.allclose(c.tensor, expected)

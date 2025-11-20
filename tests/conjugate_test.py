import torch
import pytest
from typing import TypeAlias

from grassmann_tensor import GrassmannTensor

Arrow_Edges: TypeAlias = tuple[tuple[bool, ...], tuple[tuple[int, int], ...]]


@pytest.fixture(
    params=[
        (
            (True, False),
            ((2, 2), (2, 2)),
        ),
        (
            (True, False, False),
            ((2, 2), (2, 2), (2, 2)),
        ),
    ]
)
def arrow_edges(request: pytest.FixtureRequest) -> Arrow_Edges:
    return request.param


@pytest.fixture(
    params=[
        (
            (True, False),
            ((4, 0), (4, 0)),
        ),
        (
            (True, False, False),
            ((4, 0), (4, 0), (4, 0)),
        ),
    ]
)
def arrow_edges_without_symmetry(request: pytest.FixtureRequest) -> Arrow_Edges:
    return request.param


def create_random_tensor(
    arrow_edges: Arrow_Edges,
    *,
    dtype: torch.dtype,
) -> GrassmannTensor:
    arrow, edges = arrow_edges
    shape = [even + odd for even, odd in edges]

    if dtype == torch.float64:
        tensor = torch.rand(*shape, dtype=dtype)
    else:
        tensor = torch.randn(*shape) + 1j * torch.randn(*shape).to(dtype)
    return GrassmannTensor(arrow, edges, tensor)


@pytest.fixture
def random_real_tensor(arrow_edges: Arrow_Edges) -> GrassmannTensor:
    return create_random_tensor(arrow_edges, dtype=torch.float64)


@pytest.fixture
def random_complex_tensor(arrow_edges: Arrow_Edges) -> GrassmannTensor:
    return create_random_tensor(arrow_edges, dtype=torch.complex128)


def test_conjugate_involution_with_complex_tensor(random_complex_tensor: GrassmannTensor) -> None:
    contrast_a = random_complex_tensor
    contrast_b = contrast_a.conj().conj()
    assert contrast_a.arrow == contrast_b.arrow
    assert torch.allclose(contrast_a.tensor, contrast_b.tensor)


def test_conjugate_involution_with_real_tensor(random_real_tensor: GrassmannTensor) -> None:
    contrast_a = random_real_tensor
    contrast_b = contrast_a.conj().conj()
    assert contrast_a.arrow == contrast_b.arrow
    assert torch.allclose(contrast_a.tensor, contrast_b.tensor)


def test_conjugate_reverse_order_of_contraction(
    arrow_edges: Arrow_Edges,
) -> None:
    a = create_random_tensor(arrow_edges, dtype=torch.complex128).update_mask()
    b = create_random_tensor(arrow_edges, dtype=torch.complex128).update_mask()

    contrast_a = GrassmannTensor.contract(a, b, a.tensor.dim() - 1, 0)
    contrast_a = contrast_a.conj()

    a_conj = a.conj()
    b_conj = b.conj()

    contrast_b = GrassmannTensor.contract(a_conj, b_conj, a_conj.tensor.dim() - 1, 0)

    assert contrast_a.arrow == contrast_b.arrow
    assert torch.allclose(contrast_a.tensor, contrast_b.tensor)


def test_conjugate_without_symmetry_equality(
    arrow_edges_without_symmetry: Arrow_Edges,
) -> None:
    a = create_random_tensor(arrow_edges_without_symmetry, dtype=torch.float64)
    b = create_random_tensor(arrow_edges_without_symmetry, dtype=torch.complex128)

    assert torch.allclose(a.tensor, a.conj().tensor)
    assert torch.allclose(b.tensor.conj(), b.conj().tensor)

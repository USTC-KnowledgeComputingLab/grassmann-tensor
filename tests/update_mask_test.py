import pytest
import torch
from grassmann_tensor import GrassmannTensor

Initialization = tuple[tuple[bool, ...], tuple[tuple[int, int], ...], torch.Tensor]


@pytest.fixture(params=[
    (
        (),
        (),
        torch.randn([], dtype=torch.float32),
    ),
    (
        (False, True),
        ((2, 2), (0, 2)),
        torch.randn([4, 2], dtype=torch.float32),
    ),
    (
        (False, True),
        ((2, 2), (0, 0)),
        torch.randn([4, 0], dtype=torch.float32),
    ),
    (
        (True, False),
        ((2, 2), (3, 1)),
        torch.randn([4, 4], dtype=torch.float32),
    ),
    (
        (False, True, False),
        ((1, 1), (2, 2), (1, 1)),
        torch.randn([2, 4, 2], dtype=torch.float32),
    ),
])
def x(request: pytest.FixtureRequest) -> Initialization:
    return request.param


def test_update_mask(x: Initialization) -> None:
    tensor = GrassmannTensor(*x)
    updated_tensor = tensor.update_mask()
    assert torch.all(updated_tensor.tensor == torch.where(updated_tensor.mask, 0, x[2]))

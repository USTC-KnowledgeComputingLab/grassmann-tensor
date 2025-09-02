import pytest
import torch
from grassmann_tensor import GrassmannTensor

Initialization = tuple[tuple[bool, ...], tuple[tuple[int, int], ...], torch.Tensor]


@pytest.fixture(params=[
    ((False, False), ((2, 2), (2, 2)), torch.randn([4, 4])),
    ((False, True), ((2, 2), (1, 3)), torch.randn([4, 4])),
    ((False, True), ((2, 0), (1, 3)), torch.randn([2, 4])),
    ((True, False), ((0, 2), (1, 3)), torch.randn([2, 4])),
    ((True, False), ((0, 0), (1, 3)), torch.randn([0, 4])),
    ((True,), ((2, 0),), torch.randn([2])),
    ((False,), ((0, 2),), torch.randn([2])),
    ((), (), torch.randn([])),
    ((False, False, True), ((2, 2), (1, 3), (4, 0)), torch.randn([4, 4, 4])),
])
def x(request: pytest.FixtureRequest) -> Initialization:
    return request.param


def test_arrow(x: Initialization) -> None:
    tensor = GrassmannTensor(*x)
    assert tensor.arrow == x[0]


def test_edges(x: Initialization) -> None:
    tensor = GrassmannTensor(*x)
    assert tensor.edges == x[1]


def test_tensor(x: Initialization) -> None:
    tensor = GrassmannTensor(*x)
    assert torch.equal(tensor.tensor, x[2])


def test_parity(x: Initialization) -> None:
    tensor = GrassmannTensor(*x)
    assert len(tensor.parity) == tensor.tensor.dim()
    for [even, odd], parity in zip(x[1], tensor.parity):
        total = even + odd
        assert parity.shape == (total,)
        assert parity.dtype == torch.bool
        for i in range(total):
            assert parity[i] == (i >= even)


def test_mask(x: Initialization) -> None:
    tensor = GrassmannTensor(*x)
    assert tensor.mask.shape == tensor.tensor.shape
    assert tensor.mask.dtype == torch.bool
    for indices in zip(*torch.unravel_index(torch.arange(tensor.tensor.numel()), tensor.tensor.shape)):
        mask = tensor.mask[indices]
        expect = False
        for rank, parity in enumerate(tensor.parity):
            expect ^= bool(parity[indices[rank]])
        assert mask == expect

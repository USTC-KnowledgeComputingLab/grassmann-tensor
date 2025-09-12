import torch
from grassmann_tensor.tensor import GrassmannTensor


def test_trace() -> None:
    gt = GrassmannTensor(
        (False, False, False),
        ((1, 1), (1, 1), (1, 1)),
        torch.tensor([[[1, 0], [0, 2]], [[0, 3], [4, 0]]]),
    )
    gt.trace((0, 1))

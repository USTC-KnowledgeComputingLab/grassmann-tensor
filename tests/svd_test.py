import torch
import math
from grassmann_tensor import GrassmannTensor


def test_svd() -> None:
    gt = GrassmannTensor(
        (True, True, True, True),
        ((8, 8), (4, 4), (2, 2), (1, 1)),
        torch.randn([16, 8, 4, 2], dtype=torch.float64),
    )
    U, S, Vh = gt.svd((0, 3))

    # reshape U
    # left_arrow = U.arrow[:-1]
    left_dim = math.prod(U.tensor.shape[:-1])
    left_edge = list(U.edges[:-1])
    U = U.reshape((left_dim, -1))

    # reshape Vh
    # right_arrow = Vh.arrow[1:]
    right_dim = math.prod(Vh.tensor.shape[1:])
    right_edge = list(Vh.edges[1:])
    Vh = Vh.reshape((-1, right_dim))

    US = GrassmannTensor.matmul(U, S)
    USV = GrassmannTensor.matmul(US, Vh)

    USV = USV.reshape(tuple(left_edge + right_edge))
    USV = USV.permute((0, 2, 3, 1))

    assert torch.allclose(gt.tensor, USV.tensor)

import pytest
import torch
from grassmann_tensor.tensor import GrassmannTensor

ReverseCase = tuple[
    tuple[bool, ...], tuple[tuple[int, int], ...], torch.Tensor, tuple[int, ...], torch.Tensor
]


@pytest.mark.parametrize(
    "x",
    [
        ((), (), torch.tensor(6), (), torch.tensor(6)),
        ((False, False), ((1, 1), (0, 0)), torch.zeros([2, 0]), (0,), torch.zeros([2, 0])),
        (
            (False, False),
            ((1, 1), (0, 1)),
            torch.tensor([[0], [4]]),
            (0,),
            torch.tensor([[0], [4]]),
        ),
        (
            (False, False),
            ((1, 1), (1, 1)),
            torch.tensor([[1, 0], [0, 4]]),
            (),
            torch.tensor([[1, 0], [0, 4]]),
        ),
        (
            (False, False),
            ((1, 1), (1, 1)),
            torch.tensor([[1, 0], [0, 4]]),
            (0,),
            torch.tensor([[1, 0], [0, 4]]),
        ),
        (
            (True, False),
            ((1, 1), (1, 1)),
            torch.tensor([[1, 0], [0, 4]]),
            (0,),
            torch.tensor([[1, 0], [0, -4]]),
        ),
        (
            (False, False),
            ((1, 1), (1, 1)),
            torch.tensor([[1, 0], [0, 4]]),
            (1,),
            torch.tensor([[1, 0], [0, 4]]),
        ),
        (
            (False, True),
            ((1, 1), (1, 1)),
            torch.tensor([[1, 0], [0, 4]]),
            (1,),
            torch.tensor([[1, 0], [0, -4]]),
        ),
        (
            (False, False, False),
            ((1, 1), (1, 1), (1, 1)),
            torch.tensor([[[1, 0], [0, 2]], [[0, 3], [4, 0]]]),
            (0,),
            torch.tensor([[[1, 0], [0, 2]], [[0, 3], [4, 0]]]),
        ),
        (
            (True, False, False),
            ((1, 1), (1, 1), (1, 1)),
            torch.tensor([[[1, 0], [0, 2]], [[0, 3], [4, 0]]]),
            (0,),
            torch.tensor([[[1, 0], [0, 2]], [[0, -3], [-4, 0]]]),
        ),
        (
            (False, True, True),
            ((1, 1), (1, 1), (1, 1)),
            torch.tensor([[[1, 0], [0, 2]], [[0, 3], [4, 0]]]),
            (2,),
            torch.tensor([[[1, 0], [0, -2]], [[0, -3], [4, 0]]]),
        ),
        (
            (True, False, True),
            ((1, 1), (1, 1), (1, 1)),
            torch.tensor([[[1, 0], [0, 2]], [[0, 3], [4, 0]]]),
            (0, 1),
            torch.tensor([[[1, 0], [0, 2]], [[0, -3], [-4, 0]]]),
        ),
        (
            (True, True, True),
            ((1, 1), (1, 1), (1, 1)),
            torch.tensor([[[1, 0], [0, 2]], [[0, 3], [4, 0]]]),
            (0, 1),
            torch.tensor([[[1, 0], [0, -2]], [[0, -3], [4, 0]]]),
        ),
    ],
)
def test_reverse(x: ReverseCase) -> None:
    arrow, edges, tensor, reverse_by, expected = x
    grassmann_tensor = GrassmannTensor(arrow, edges, tensor)
    result = grassmann_tensor.reverse(reverse_by)
    assert torch.allclose(result.tensor, expected)


ReverseFailCase = tuple[
    tuple[bool, ...], tuple[tuple[int, int], ...], torch.Tensor, tuple[int, ...], str
]


@pytest.mark.parametrize(
    "x",
    [
        (
            (False, False),
            ((1, 1), (1, 1)),
            torch.tensor([[1, 0], [0, 4]]),
            (0, 0),
            "Indices must be unique",
        ),
        (
            (False, False),
            ((1, 1), (1, 1)),
            torch.tensor([[1, 0], [0, 4]]),
            (2,),
            "Indices must be within tensor dimensions",
        ),
    ],
)
def test_reverse_fail(x: ReverseFailCase) -> None:
    arrow, edges, tensor, reverse_by, message = x
    grassmann_tensor = GrassmannTensor(arrow, edges, tensor)
    with pytest.raises(AssertionError, match=message):
        grassmann_tensor.reverse(reverse_by)

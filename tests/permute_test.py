import pytest
import torch
from grassmann_tensor import GrassmannTensor, NamedGrassmannTensor

PermuteCase = tuple[
    tuple[bool, ...], tuple[tuple[int, int], ...], torch.Tensor, tuple[int, ...], torch.Tensor
]


@pytest.mark.parametrize(
    "x",
    [
        ((), (), torch.tensor(6), (), torch.tensor(6)),
        ((False,), ((1, 1),), torch.tensor([1, 2]), (0,), torch.tensor([1, 2])),
        ((False, True), ((1, 1), (0, 0)), torch.zeros([2, 0]), (1, 0), torch.zeros([0, 2])),
        (
            (False, True),
            ((1, 1), (0, 1)),
            torch.tensor([[0], [4]]),
            (1, 0),
            torch.tensor([[0, -4]]),
        ),
        (
            (False, False),
            ((1, 1), (1, 1)),
            torch.tensor([[1, 0], [0, 4]]),
            (0, 1),
            torch.tensor([[1, 0], [0, 4]]),
        ),
        (
            (True, False),
            ((1, 1), (1, 1)),
            torch.tensor([[1, 0], [0, 4]]),
            (1, 0),
            torch.tensor([[1, 0], [0, -4]]),
        ),
        (
            (False, True, True),
            ((1, 1), (1, 1), (1, 1)),
            torch.tensor([[[1, 0], [0, 2]], [[0, 3], [4, 0]]]),
            (0, 2, 1),
            torch.tensor([[[1, 0], [0, -2]], [[0, 4], [3, 0]]]),
        ),
        (
            (True, True, True),
            ((1, 1), (1, 1), (1, 1)),
            torch.tensor([[[1, 0], [0, 2]], [[0, 3], [4, 0]]]),
            (1, 0, 2),
            torch.tensor([[[1, 0], [0, 3]], [[0, 2], [-4, 0]]]),
        ),
        (
            (True, False, False),
            ((1, 1), (1, 1), (1, 1)),
            torch.tensor([[[1, 0], [0, 2]], [[0, 3], [4, 0]]]),
            (2, 1, 0),
            torch.tensor([[[1, 0], [0, -4]], [[0, -3], [-2, 0]]]),
        ),
        (
            (False, False, False),
            ((1, 1), (1, 1), (1, 1)),
            torch.tensor([[[1, 0], [0, 2]], [[0, 3], [4, 0]]]),
            (2, 0, 1),
            torch.tensor([[[1, 0], [0, 4]], [[0, -2], [-3, 0]]]),
        ),
    ],
)
def test_permute(x: PermuteCase) -> None:
    arrow, edges, tensor, before_by_after, expected = x
    grassmann_tensor = GrassmannTensor(arrow, edges, tensor)
    result = grassmann_tensor.permute(before_by_after)
    assert torch.allclose(result.tensor, expected)


PermuteFailCase = tuple[
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
            "Permutation indices must be unique",
        ),
        (
            (False, False),
            ((1, 1), (1, 1)),
            torch.tensor([[1, 0], [0, 4]]),
            (2, 0),
            "Permutation indices must cover all dimensions",
        ),
        (
            (False, False),
            ((1, 1), (1, 1)),
            torch.tensor([[1, 0], [0, 4]]),
            (0, 0, 1),
            "Permutation indices must be unique",
        ),
    ],
)
def test_permute_fail(x: PermuteFailCase) -> None:
    arrow, edges, tensor, before_by_after, message = x
    grassmann_tensor = GrassmannTensor(arrow, edges, tensor)
    with pytest.raises(AssertionError, match=message):
        grassmann_tensor.permute(before_by_after)


def test_permute_high_order() -> None:
    edge = (2, 2)
    a = GrassmannTensor(
        (False, False, False, False, False, False),
        (edge, edge, edge, edge, edge, edge),
        torch.randn(4, 4, 4, 4, 4, 4),
    ).update_mask()
    # a[i, j, k, l, m, n] -> b[l, j, i, n, k, m]
    b = a.permute((3, 1, 0, 5, 2, 4))
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for l in range(4):  # noqa: E741
                    for m in range(4):
                        for n in range(4):
                            p = [bool(x & 2) for x in (i, j, k, l, m, n)]
                            if sum(p) % 2 != 0:
                                continue
                            # i j k l m n
                            # (l) (i j k) m n
                            # l (j) (i) k m n
                            # l j i (n) (k m)
                            sign = (
                                (p[3] & (p[0] ^ p[1] ^ p[2]))
                                ^ (p[1] & p[0])
                                ^ (p[5] & (p[2] ^ p[4]))
                            )
                            if sign:
                                assert b.tensor[l, j, i, n, k, m] == -a.tensor[i, j, k, l, m, n]
                            else:
                                assert b.tensor[l, j, i, n, k, m] == a.tensor[i, j, k, l, m, n]


NamedPermuteCase = tuple[
    tuple[str, ...],
    tuple[bool, ...],
    tuple[tuple[int, int], ...],
    torch.Tensor,
    tuple[str, ...],
    torch.Tensor,
]


@pytest.mark.parametrize(
    "x",
    [
        ((), (), (), torch.tensor(6), (), torch.tensor(6)),
        (("a",), (False,), ((1, 1),), torch.tensor([1, 2]), ("a",), torch.tensor([1, 2])),
        (
            ("a", "b"),
            (False, True),
            ((1, 1), (0, 0)),
            torch.zeros([2, 0]),
            ("b", "a"),
            torch.zeros([0, 2]),
        ),
        (
            ("a", "b"),
            (False, True),
            ((1, 1), (0, 1)),
            torch.tensor([[0], [4]]),
            ("b", "a"),
            torch.tensor([[0, -4]]),
        ),
        (
            ("a", "b"),
            (False, False),
            ((1, 1), (1, 1)),
            torch.tensor([[1, 0], [0, 4]]),
            ("a", "b"),
            torch.tensor([[1, 0], [0, 4]]),
        ),
        (
            ("a", "b"),
            (True, False),
            ((1, 1), (1, 1)),
            torch.tensor([[1, 0], [0, 4]]),
            ("b", "a"),
            torch.tensor([[1, 0], [0, -4]]),
        ),
        (
            ("a", "b", "c"),
            (False, True, True),
            ((1, 1), (1, 1), (1, 1)),
            torch.tensor([[[1, 0], [0, 2]], [[0, 3], [4, 0]]]),
            ("a", "c", "b"),
            torch.tensor([[[1, 0], [0, -2]], [[0, 4], [3, 0]]]),
        ),
        (
            ("a", "b", "c"),
            (True, True, True),
            ((1, 1), (1, 1), (1, 1)),
            torch.tensor([[[1, 0], [0, 2]], [[0, 3], [4, 0]]]),
            ("b", "a", "c"),
            torch.tensor([[[1, 0], [0, 3]], [[0, 2], [-4, 0]]]),
        ),
        (
            ("a", "b", "c"),
            (True, False, False),
            ((1, 1), (1, 1), (1, 1)),
            torch.tensor([[[1, 0], [0, 2]], [[0, 3], [4, 0]]]),
            ("c", "b", "a"),
            torch.tensor([[[1, 0], [0, -4]], [[0, -3], [-2, 0]]]),
        ),
        (
            ("a", "b", "c"),
            (False, False, False),
            ((1, 1), (1, 1), (1, 1)),
            torch.tensor([[[1, 0], [0, 2]], [[0, 3], [4, 0]]]),
            ("c", "a", "b"),
            torch.tensor([[[1, 0], [0, 4]], [[0, -2], [-3, 0]]]),
        ),
    ],
)
def test_named_tensor_permute(x: NamedPermuteCase) -> None:
    names, arrow, edges, tensor, before_by_after, expected = x
    grassmann_tensor = NamedGrassmannTensor(names, arrow, edges, tensor)
    result = grassmann_tensor.permute(before_by_after)
    assert torch.allclose(result.tensor, expected)


NamedPermuteFailCase = tuple[
    tuple[str, ...],
    tuple[bool, ...],
    tuple[tuple[int, int], ...],
    torch.Tensor,
    tuple[str, ...],
    str,
]


@pytest.mark.parametrize(
    "x",
    [
        (
            ("a", "b"),
            (False, False),
            ((1, 1), (1, 1)),
            torch.tensor([[1, 0], [0, 4]]),
            ("a", "a"),
            "Permutation indices must be unique",
        ),
        (
            ("a", "b"),
            (False, False),
            ((1, 1), (1, 1)),
            torch.tensor([[1, 0], [0, 4]]),
            ("a",),
            "Permutation indices must cover all dimensions",
        ),
        (
            ("a", "b"),
            (False, False),
            ((1, 1), (1, 1)),
            torch.tensor([[1, 0], [0, 4]]),
            ("a", "a", "b"),
            "Permutation indices must be unique",
        ),
    ],
)
def test_named_tensor_permute_fail(x: NamedPermuteFailCase) -> None:
    names, arrow, edges, tensor, before_by_after, message = x
    grassmann_tensor = NamedGrassmannTensor(names, arrow, edges, tensor)
    with pytest.raises(AssertionError, match=message):
        grassmann_tensor.permute(before_by_after)


def test_named_tensor_permute_high_order() -> None:
    edge = (2, 2)
    a = NamedGrassmannTensor(
        ("i", "j", "k", "l", "m", "n"),
        (False, False, False, False, False, False),
        (edge, edge, edge, edge, edge, edge),
        torch.randn(4, 4, 4, 4, 4, 4),
    ).update_mask()
    # a[i, j, k, l, m, n] -> b[l, j, i, n, k, m]
    b = a.permute(("l", "j", "i", "n", "k", "m"))
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for l in range(4):  # noqa: E741
                    for m in range(4):
                        for n in range(4):
                            p = [bool(x & 2) for x in (i, j, k, l, m, n)]
                            if sum(p) % 2 != 0:
                                continue
                            # i j k l m n
                            # (l) (i j k) m n
                            # l (j) (i) k m n
                            # l j i (n) (k m)
                            sign = (
                                (p[3] & (p[0] ^ p[1] ^ p[2]))
                                ^ (p[1] & p[0])
                                ^ (p[5] & (p[2] ^ p[4]))
                            )
                            if sign:
                                assert b.tensor[l, j, i, n, k, m] == -a.tensor[i, j, k, l, m, n]
                            else:
                                assert b.tensor[l, j, i, n, k, m] == a.tensor[i, j, k, l, m, n]


def test_mask_cache_should_not_survive_permute() -> None:
    a = NamedGrassmannTensor(
        ("a", "b", "c"),
        (False, False, False),
        ((2, 2), (1, 0), (1, 0)),
        torch.randn(4, 1, 1, dtype=torch.float64),
    )

    _ = a.mask

    b = a.permute(("c", "a", "b"))

    gt_fresh = GrassmannTensor(
        _arrow=b.arrow,
        _edges=b.edges,
        _tensor=b.tensor,
        _parity=None,
        _mask=None,
    )

    assert torch.equal(b.mask, gt_fresh.mask)

import torch
import pytest
from _pytest.mark.structures import ParameterSet
import math
import itertools
from typing import TypeAlias, Iterable, Any

from grassmann_tensor import GrassmannTensor

Arrow: TypeAlias = tuple[bool, ...]
Edges: TypeAlias = tuple[tuple[int, int], ...]
Tensor: TypeAlias = torch.Tensor
Cutoff: TypeAlias = int | tuple[int, int] | None
Tau: TypeAlias = float
FreeNamesU: TypeAlias = tuple[int, ...]

SVDCases = Iterable[ParameterSet]


def get_total_singular(edges: Edges, free_names_u: FreeNamesU) -> tuple[int, int]:
    even_singular = min(GrassmannTensor.calculate_even_odd(tuple(edges[i] for i in free_names_u)))
    set_all = set(range(len(edges)))
    rest_idx = sorted(set_all - set(free_names_u))
    odd_singular = min(GrassmannTensor.calculate_even_odd(tuple(edges[i] for i in rest_idx)))
    return even_singular, odd_singular


def tau_for_cutoff(c: int, total: int, alpha: float = 0.8, slack: float = 1.05) -> float:
    cut = 0
    if isinstance(c, int):
        cut = c
    lo, hi = 1e-8, 1e-1
    x = (total - cut) / max(1, total - 1)
    return (lo + (hi - lo) * (x**alpha)) * slack


def choose_free_names(n_edges: int, limit: int = 8) -> list[FreeNamesU]:
    combos = [
        tuple(c) for r in range(1, n_edges) for c in itertools.combinations(range(n_edges), r)
    ]
    return combos[:limit]


BASE_GT_CASES: list[tuple[Arrow, Edges, Tensor]] = [
    ((True, True), ((2, 2), (4, 4)), torch.randn(4, 8, dtype=torch.float64)),
    ((True, True, True), ((2, 2), (4, 4), (8, 8)), torch.randn(4, 8, 16, dtype=torch.float64)),
    (
        (True, True, True, True),
        ((2, 2), (4, 4), (8, 8), (16, 16)),
        torch.randn(4, 8, 16, 32, dtype=torch.float64),
    ),
]


def svd_cases() -> SVDCases:
    params = []
    for arrow, edges, tensor in BASE_GT_CASES:
        for fnu in choose_free_names(len(edges)):
            even_singular, odd_singular = get_total_singular(edges, fnu)
            max_singular = max(even_singular, odd_singular)
            total = even_singular + odd_singular
            cutoff_list = [
                None,
                max_singular,
                max_singular - 1,
                (even_singular, odd_singular),
            ]
            for cutoff in cutoff_list:
                if cutoff is None:
                    kept = total
                elif isinstance(cutoff, int):
                    k = cutoff
                    kept = min(k, even_singular) + min(k, odd_singular)
                else:
                    ke = min(int(cutoff[0]), even_singular)
                    ko = min(int(cutoff[1]), odd_singular)
                    kept = ke + ko
                tau = tau_for_cutoff(kept, total)
                params.append(
                    pytest.param(
                        arrow,
                        edges,
                        tensor,
                        cutoff,
                        tau,
                        fnu,
                        id=f"edges={tuple(edges)}|fnu={fnu}|cut={cutoff}|tau={tau:.2e}",
                    )
                )
    return params


@pytest.mark.parametrize(
    "arrow, edges, tensor, cutoff, tau, free_names_u",
    svd_cases(),
)
@pytest.mark.repeat(20)
def test_svd(
    arrow: Arrow,
    edges: Edges,
    tensor: Tensor,
    cutoff: Cutoff,
    tau: Tau,
    free_names_u: FreeNamesU,
) -> None:
    gt = GrassmannTensor(arrow, edges, tensor)
    U, S, Vh = gt.svd(free_names_u, cutoff=cutoff)

    # reshape U
    left_dim = math.prod(U.tensor.shape[:-1])
    left_edge = list(U.edges[:-1])
    U = U.reshape((left_dim, -1))

    # reshape Vh
    right_dim = math.prod(Vh.tensor.shape[1:])
    right_edge = list(Vh.edges[1:])
    Vh = Vh.reshape((-1, right_dim))

    US = GrassmannTensor.matmul(U, S)
    USV = GrassmannTensor.matmul(US, Vh)

    set_all = set(range(len(edges)))
    set_u = set(free_names_u)
    set_v = sorted(set_all - set_u)
    perm_order = list(free_names_u) + list(set_v)
    inv_perm = [perm_order.index(i) for i in range(len(edges))]

    USV = USV.reshape(tuple(left_edge + right_edge))
    USV = USV.permute(tuple(inv_perm))

    masked = gt.update_mask().tensor
    den = masked.norm()
    eps = torch.finfo(masked.dtype).eps
    rel_err = (masked - USV.tensor).norm() / max(den, eps)
    assert rel_err <= tau


@pytest.mark.parametrize(
    "arrow, edges, tensor, cutoff , tau, free_names_u",
    svd_cases(),
)
@pytest.mark.parametrize(
    "incompatible_cutoff",
    [
        -1,
        0,
        (
            1,
            2,
            3,
        ),
        "string",
        {"key", "value"},
        [1, 2, 3],
        {1, 2},
        object(),
    ],
)
def test_svd_with_incompatible_cutoff(
    arrow: Arrow,
    edges: Edges,
    tensor: Tensor,
    cutoff: Cutoff,
    tau: Tau,
    free_names_u: FreeNamesU,
    incompatible_cutoff: Any,
) -> None:
    gt = GrassmannTensor(arrow, edges, tensor)
    if isinstance(incompatible_cutoff, int):
        with pytest.raises(AssertionError, match="Cutoff must be greater than 0"):
            _, _, _ = gt.svd(free_names_u, cutoff=incompatible_cutoff)
    elif isinstance(incompatible_cutoff, tuple):
        with pytest.raises(
            AssertionError, match="The length of cutoff must be 2 if cutoff is a tuple"
        ):
            _, _, _ = gt.svd(free_names_u, cutoff=incompatible_cutoff)
    else:
        with pytest.raises(
            ValueError, match="Cutoff must be an integer or a tuple of two integers"
        ):
            _, _, _ = gt.svd(free_names_u, cutoff=incompatible_cutoff)


@pytest.mark.parametrize("a,b", [(3, 5), (1, 1), (8, 2)])
def test_svd_both_blocks_empty_raises_with_int_cutoff(a: int, b: int) -> None:
    # edges: left=(even_left=0, odd_left=a), right=(even_right=b, odd_right=0)
    # tensor shape must be (a, b)
    arrow = (True, True)
    edges = ((0, a), (b, 0))
    tensor = torch.randn(a, b, dtype=torch.float64)

    gt = GrassmannTensor(arrow, edges, tensor)

    free_names_u = (0,)
    with pytest.raises(RuntimeError, match="Both parity block are empty. Can not form SVD."):
        _ = gt.svd(free_names_u, cutoff=1)


@pytest.mark.parametrize("a,b", [(3, 5), (2, 4), (7, 3)])
def test_svd_both_blocks_empty_raises_with_tuple_cutoff(a: int, b: int) -> None:
    arrow = (True, True)
    edges = ((0, a), (b, 0))
    tensor = torch.randn(a, b, dtype=torch.float64)

    gt = GrassmannTensor(arrow, edges, tensor)

    free_names_u = (0,)
    with pytest.raises(RuntimeError, match="Both parity block are empty. Can not form SVD."):
        _ = gt.svd(free_names_u, cutoff=(1, 1))


@pytest.mark.parametrize(
    "a,b,c,k",
    [
        (3, 5, 7, 2),
        (4, 1, 2, 3),
    ],
)
def test_svd_int_cutoff_even_block_empty_select_from_odd_only(
    a: int, b: int, c: int, k: int
) -> None:
    arrow = (True, True)
    edges = ((0, a), (b, c))
    tensor = torch.randn(a, b + c, dtype=torch.float64)

    gt = GrassmannTensor(arrow, edges, tensor)
    U, S, Vh = gt.svd((0,), cutoff=k)

    expected_k = min(k, min(a, c))
    assert U.edges[-1] == (0, expected_k)
    assert Vh.edges[0] == (0, expected_k)
    assert S.edges == ((0, expected_k), (0, expected_k))


@pytest.mark.parametrize(
    "a,b,k",
    [
        (5, 4, 2),
        (7, 3, 5),
    ],
)
def test_svd_int_cutoff_odd_block_empty_select_from_even_only(a: int, b: int, k: int) -> None:
    arrow = (True, True)
    edges = ((a, 0), (b, 0))
    tensor = torch.randn(a, b, dtype=torch.float64)

    gt = GrassmannTensor(arrow, edges, tensor)
    U, S, Vh = gt.svd((0,), cutoff=k)

    expected_k = min(k, min(a, b))
    assert U.edges[-1] == (expected_k, 0)
    assert Vh.edges[0] == (expected_k, 0)
    assert S.edges == ((expected_k, 0), (expected_k, 0))

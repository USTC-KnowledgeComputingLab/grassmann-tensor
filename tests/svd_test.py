import torch
import pytest
from _pytest.mark.structures import ParameterSet
import math
import itertools
from typing import TypeAlias, Iterable

from grassmann_tensor import GrassmannTensor

Arrow: TypeAlias = tuple[bool, ...]
Edges: TypeAlias = tuple[tuple[int, int], ...]
Tensor: TypeAlias = torch.Tensor
Cutoff: TypeAlias = int
Tau: TypeAlias = float
FreeNamesU: TypeAlias = tuple[int, ...]

SVDCases = Iterable[ParameterSet]


def get_total_singular(edges: Edges, free_names_u: FreeNamesU) -> int:
    even, odd = edges[free_names_u[0]]
    for i in range(1, len(free_names_u)):
        e, o = edges[free_names_u[i]]
        even, odd = even * e + odd * o, even * o + odd * e
    total_singular = min(even, odd)

    set_all = set(range(len(edges)))
    right_idx = sorted(set_all - set(free_names_u))
    even, odd = edges[right_idx[0]]
    for i in range(1, len(right_idx)):
        e, o = edges[right_idx[i]]
        even, odd = even * e + odd * o, even * o + odd * e
    total_singular += min(even, odd)
    return total_singular


def tau_for_cutoff(c: int, total: int, alpha: float = 0.8) -> float:
    lo, hi = 1e-8, 1e-1
    x = (total - c) / max(1, total - 1)
    return lo + (hi - lo) * (x**alpha)


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
            total = get_total_singular(edges, fnu)
            for cutoff in [None, total, total - 1, total - 2]:
                if cutoff is not None and cutoff < 1:
                    continue
                tau = tau_for_cutoff(cutoff or total, total)
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
    "arrow, edges, tensor, cutoff, tau, free_names_u",
    svd_cases(),
)
def test_svd_with_zero_cutoff(
    arrow: Arrow,
    edges: Edges,
    tensor: Tensor,
    cutoff: Cutoff,
    tau: Tau,
    free_names_u: FreeNamesU,
) -> None:
    gt = GrassmannTensor(arrow, edges, tensor)
    with pytest.raises(AssertionError, match="Cutoff must be greater than 0"):
        _, _, _ = gt.svd(free_names_u, cutoff=0)

import torch
import pytest
from _pytest.mark.structures import ParameterSet
import random
from typing import Iterable

from grassmann_tensor import GrassmannTensor

ContractCases = Iterable[ParameterSet]


def contract_cases() -> ContractCases:
    edge_unit = (4, 4)
    max_dim = 4
    num_cases = 10

    rng = random.Random(0)
    gen = torch.Generator().manual_seed(0)

    cases = []

    for case_idx in range(num_cases):
        dim = rng.randint(1, max_dim)

        edges = tuple(edge_unit for _ in range(dim))

        shape = (sum(edge_unit),) * dim

        arrow = tuple(bool(rng.getrandbits(1)) for _ in range(dim))

        tensor = torch.randn(*shape, dtype=torch.float64, generator=gen)

        a = GrassmannTensor(
            arrow,
            edges,
            tensor,
        )

        contract_length = rng.randint(1, dim)
        if contract_length == 1:
            leg_a: int | tuple[int, ...] = (
                (sorted(rng.sample(range(dim), contract_length)))[0]
                if rng.random() < 0.5
                else tuple(sorted(rng.sample(range(dim), contract_length)))
            )
            leg_b: int | tuple[int, ...] = (
                (sorted(rng.sample(range(dim), contract_length)))[0]
                if rng.random() < 0.5
                else tuple(sorted(rng.sample(range(dim), contract_length)))
            )
        else:
            leg_a = tuple(sorted(rng.sample(range(dim), contract_length)))
            leg_b = tuple(sorted(rng.sample(range(dim), contract_length)))

        cases.append(
            pytest.param(
                a,
                leg_a,
                leg_b,
                id=f"arrow={arrow}-dim={dim}-leg_a={leg_a}-leg_b={leg_b}",
            )
        )

    return cases


@pytest.mark.parametrize("a, leg_a, leg_b", contract_cases())
def test_contract(
    a: GrassmannTensor,
    leg_a: int | tuple[int, ...],
    leg_b: int | tuple[int, ...],
) -> None:
    _ = a.contract(a, leg_a, leg_b)


def test_contract_assertion() -> None:
    a = GrassmannTensor((False, True), ((4, 4), (4, 4)), torch.randn(8, 8, dtype=torch.float64))
    b = GrassmannTensor(
        (False, True, False, True),
        ((4, 4), (4, 4), (4, 4), (4, 4)),
        torch.randn(8, 8, 8, 8, dtype=torch.float64),
    )
    with pytest.raises(AssertionError, match="Indices must be unique"):
        _ = a.contract(b, (0, 0), 0)
    with pytest.raises(AssertionError, match="Indices must be within tensor dimensions"):
        _ = a.contract(b, 0, (0, 4))


def test_contract_full_legs() -> None:
    a = GrassmannTensor(
        (False, False, False, True),
        ((2, 2), (4, 4), (8, 8), (8, 8)),
        torch.randn(4, 8, 16, 16, dtype=torch.float64),
    )
    b = GrassmannTensor(
        (False, True, True, True),
        ((8, 8), (8, 8), (4, 4), (2, 2)),
        torch.randn(16, 16, 8, 4, dtype=torch.float64),
    )
    c = a.contract(b, (0, 1, 2, 3), (0, 1, 2, 3))
    assert c.tensor.dim() == 0

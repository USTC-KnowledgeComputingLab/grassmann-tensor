import torch
import pytest
from _pytest.mark.structures import ParameterSet
import random
from typing import Iterable

from grassmann_tensor import GrassmannTensor, NamedGrassmannTensor

ContractCases = Iterable[ParameterSet]
NamedContractCases = Iterable[ParameterSet]


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


def named_contract_cases() -> NamedContractCases:
    edge_unit = (4, 4)
    max_dim = 4
    num_cases = 10

    rng = random.Random(0)
    gen = torch.Generator().manual_seed(0)

    cases = []

    for case_idx in range(num_cases):
        dim = rng.randint(1, max_dim)

        edges = tuple(edge_unit for _ in range(dim))
        arrow = tuple(bool(rng.getrandbits(1)) for _ in range(dim))
        shape = (sum(edge_unit),) * dim
        tensor = torch.randn(*shape, dtype=torch.float64, generator=gen)
        a = NamedGrassmannTensor(
            tuple(f"a{i}" for i in range(dim)),
            arrow,
            edges,
            tensor,
        )

        b = NamedGrassmannTensor(
            tuple(f"b{i}" for i in range(dim)),
            arrow,
            edges,
            tensor,
        )

        contract_length = rng.randint(1, dim)

        leg_a = tuple(sorted(rng.sample(range(dim), contract_length)))
        leg_b = tuple(sorted(rng.sample(range(dim), contract_length)))

        pairs: set[tuple[str, str]] = {(a.names[i], b.names[j]) for i, j in zip(leg_a, leg_b)}
        contracted_a = set(a.names[i] for i in leg_a)
        contracted_b = set(b.names[j] for j in leg_b)
        result_names = tuple(n for n in a.names if n not in contracted_a) + tuple(
            n for n in b.names if n not in contracted_b
        )

        cases.append(
            pytest.param(
                a,
                b,
                pairs,
                result_names,
                id=f"arrow={arrow}-dim={dim}-pairs={sorted(pairs)}-result={result_names}",
            )
        )

    return cases


@pytest.mark.parametrize("a, b, pairs, result_names", named_contract_cases())
def test_named_tensor_contract(
    a: NamedGrassmannTensor,
    b: NamedGrassmannTensor,
    pairs: set[tuple[str, str]],
    result_names: set[tuple[str, str]],
) -> None:
    out = a.contract(b, pairs)
    assert out.names == result_names


def test_named_tensor_contract_full_legs() -> None:
    a = NamedGrassmannTensor(
        ("a", "b", "c", "d"),
        (False, False, False, True),
        ((2, 2), (4, 4), (8, 8), (8, 8)),
        torch.randn(4, 8, 16, 16, dtype=torch.float64),
    )
    b = NamedGrassmannTensor(
        ("a", "b", "c", "d"),
        (False, True, True, True),
        ((8, 8), (8, 8), (4, 4), (2, 2)),
        torch.randn(16, 16, 8, 4, dtype=torch.float64),
    )
    c = a.contract(b, {("a", "d"), ("b", "c"), ("c", "b"), ("d", "a")})
    assert c.tensor.dim() == 0


def test_named_tensor_contract_different_order() -> None:
    a = NamedGrassmannTensor(
        ("a0", "a1"), (False, False), ((2, 0), (2, 0)), torch.tensor([[1, 2], [3, 4]])
    )
    b = NamedGrassmannTensor(
        ("b0", "b1"), (True, True), ((2, 0), (2, 0)), torch.tensor([[10, 20], [30, 40]])
    )
    c = a.contract(b, {("a0", "b0"), ("a1", "b1")})
    d = a.contract(b, {("a1", "b1"), ("a0", "b0")})
    e = a.contract(b, {("a0", "b1"), ("a1", "b0")})
    f = a.contract(b, {("a1", "b0"), ("a0", "b1")})
    assert torch.allclose(c.tensor, d.tensor)
    assert torch.allclose(e.tensor, f.tensor)

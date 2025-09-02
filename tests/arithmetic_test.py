from __future__ import annotations
import typing
import pytest
import torch
from grassmann_tensor import GrassmannTensor


@pytest.fixture(
    params=[
        (
            GrassmannTensor((False, False), ((2, 2), (1, 3)), torch.randn([4, 4])),
            GrassmannTensor((False, False), ((2, 2), (1, 3)), torch.randn([4, 4])),
        ),
        (
            GrassmannTensor((True, False, True), ((1, 1), (2, 2), (3, 1)), torch.randn([2, 4, 4])),
            GrassmannTensor((True, False, True), ((1, 1), (2, 2), (3, 1)), torch.randn([2, 4, 4])),
        ),
        (
            GrassmannTensor(
                (True, True, False, False),
                ((1, 2), (2, 2), (1, 1), (3, 1)),
                torch.randn([3, 4, 2, 4]),
            ),
            GrassmannTensor(
                (True, True, False, False),
                ((1, 2), (2, 2), (1, 1), (3, 1)),
                torch.randn([3, 4, 2, 4]),
            ),
        ),
    ]
)
def tensors(request: pytest.FixtureRequest) -> tuple[GrassmannTensor, GrassmannTensor]:
    return request.param


@pytest.fixture(
    params=[
        (
            GrassmannTensor((False, False), ((2, 2), (1, 3)), torch.randn([4, 4])),
            GrassmannTensor((False,), ((2, 2),), torch.randn([4])),
        ),
        (
            GrassmannTensor((True, False, True), ((1, 1), (2, 2), (3, 1)), torch.randn([2, 4, 4])),
            GrassmannTensor((True, False, True), ((1, 2), (2, 2), (3, 1)), torch.randn([3, 4, 4])),
        ),
        (
            GrassmannTensor((True, True, False), ((1, 2), (2, 2), (3, 1)), torch.randn([3, 4, 4])),
            GrassmannTensor(
                (True, True, False, False),
                ((3, 2), (2, 2), (1, 1), (3, 1)),
                torch.randn([5, 4, 2, 4]),
            ),
        ),
    ]
)
def mismatch_tensors(request: pytest.FixtureRequest) -> tuple[GrassmannTensor, GrassmannTensor]:
    return request.param


@pytest.fixture(
    params=[
        torch.randn([]),
        torch.randn([]).item(),
    ]
)
def scalar(request: pytest.FixtureRequest) -> torch.Tensor | float:
    return request.param


class FakeTensor:
    def __init__(self) -> None:
        pass

    def __add__(self, other: typing.Any) -> FakeTensor:
        if isinstance(other, torch.Tensor):
            return self
        else:
            return NotImplemented

    def __radd__(self, other: typing.Any) -> FakeTensor:
        if isinstance(other, torch.Tensor):
            return self
        else:
            return NotImplemented

    def __sub__(self, other: typing.Any) -> FakeTensor:
        if isinstance(other, torch.Tensor):
            return self
        else:
            return NotImplemented

    def __rsub__(self, other: typing.Any) -> FakeTensor:
        if isinstance(other, torch.Tensor):
            return self
        else:
            return NotImplemented

    def __mul__(self, other: typing.Any) -> FakeTensor:
        if isinstance(other, torch.Tensor):
            return self
        else:
            return NotImplemented

    def __rmul__(self, other: typing.Any) -> FakeTensor:
        if isinstance(other, torch.Tensor):
            return self
        else:
            return NotImplemented

    def __truediv__(self, other: typing.Any) -> FakeTensor:
        if isinstance(other, torch.Tensor):
            return self
        else:
            return NotImplemented

    def __rtruediv__(self, other: typing.Any) -> FakeTensor:
        if isinstance(other, torch.Tensor):
            return self
        else:
            return NotImplemented


@pytest.mark.parametrize(
    "unsupported_type",
    [
        "string",  # string
        None,  # NoneType
        {"key", "value"},  # dict
        [1, 2, 3],  # list
        {1, 2},  # set
        object(),  # arbitrary object
        FakeTensor(),  # an ill defined tensor-like object
    ],
)
def test_arithmetic(
    unsupported_type: typing.Any,
    tensors: tuple[GrassmannTensor, GrassmannTensor],
    scalar: torch.Tensor | float,
) -> None:
    tensor_a, tensor_b = tensors

    # Test __pos__ method.
    assert torch.equal((+tensor_a).tensor, +tensor_a.tensor)

    # Test __neg__ method.
    assert torch.equal((-tensor_a).tensor, -tensor_a.tensor)

    # Test __add__ method.
    assert torch.equal((tensor_a + scalar).tensor, tensor_a.tensor + scalar)
    assert torch.equal((tensor_a + tensor_b).tensor, tensor_a.tensor + tensor_b.tensor)
    assert torch.equal((scalar + tensor_a).tensor, scalar + tensor_a.tensor)

    with pytest.raises(TypeError):
        tensor_a + unsupported_type

    with pytest.raises(TypeError):
        unsupported_type + tensor_a

    # Test __iadd__ method.
    tensor_c = tensor_a.clone()
    tensor_c += scalar
    assert torch.equal(tensor_c.tensor, tensor_a.tensor + scalar)
    tensor_c = tensor_a.clone()
    tensor_c += tensor_b
    assert torch.equal(tensor_c.tensor, tensor_a.tensor + tensor_b.tensor)

    with pytest.raises(TypeError):
        tensor_c = tensor_a.clone()
        tensor_c += unsupported_type

    # Test __sub__ method.
    assert torch.equal((tensor_a - scalar).tensor, tensor_a.tensor - scalar)
    assert torch.equal((tensor_a - tensor_b).tensor, tensor_a.tensor - tensor_b.tensor)
    assert torch.equal((scalar - tensor_a).tensor, scalar - tensor_a.tensor)

    with pytest.raises(TypeError):
        tensor_a - unsupported_type

    with pytest.raises(TypeError):
        unsupported_type - tensor_a

    # Test __isub__ method.
    tensor_c = tensor_a.clone()
    tensor_c -= scalar
    assert torch.equal(tensor_c.tensor, tensor_a.tensor - scalar)
    tensor_c = tensor_a.clone()
    tensor_c -= tensor_b
    assert torch.equal(tensor_c.tensor, tensor_a.tensor - tensor_b.tensor)

    with pytest.raises(TypeError):
        tensor_c = tensor_a.clone()
        tensor_c -= unsupported_type

    # Test __mul__ method.
    assert torch.allclose((tensor_a * scalar).tensor, tensor_a.tensor * scalar)
    assert torch.allclose((tensor_a * tensor_b).tensor, tensor_a.tensor * tensor_b.tensor)
    assert torch.allclose((scalar * tensor_a).tensor, scalar * tensor_a.tensor)

    with pytest.raises(TypeError):
        tensor_a * unsupported_type

    with pytest.raises(TypeError):
        unsupported_type * tensor_a

    # Test __imul__ method.
    tensor_c = tensor_a.clone()
    tensor_c *= scalar
    assert torch.allclose(tensor_c.tensor, tensor_a.tensor * scalar)
    tensor_c = tensor_a.clone()
    tensor_c *= tensor_b
    assert torch.allclose(tensor_c.tensor, tensor_a.tensor * tensor_b.tensor)

    with pytest.raises(TypeError):
        tensor_c = tensor_a.clone()
        tensor_c *= unsupported_type

    # Test __truediv__ method.
    assert torch.allclose((tensor_a / scalar).tensor, tensor_a.tensor / scalar)
    assert torch.allclose((tensor_a / tensor_b).tensor, tensor_a.tensor / tensor_b.tensor)
    assert torch.allclose((scalar / tensor_a).tensor, scalar / tensor_a.tensor)

    with pytest.raises(TypeError):
        tensor_a / unsupported_type

    with pytest.raises(TypeError):
        unsupported_type / tensor_a

    # Test __itruediv__ method.
    tensor_c = tensor_a.clone()
    tensor_c /= scalar
    assert torch.allclose(tensor_c.tensor, tensor_a.tensor / scalar)
    tensor_c = tensor_a.clone()
    tensor_c /= tensor_b
    assert torch.allclose(tensor_c.tensor, tensor_a.tensor / tensor_b.tensor)

    with pytest.raises(TypeError):
        tensor_c = tensor_a.clone()
        tensor_c /= unsupported_type


def test_arithmetic_fail(mismatch_tensors: tuple[GrassmannTensor, GrassmannTensor]) -> None:
    tensor_a, tensor_b = mismatch_tensors

    # Test __add__ method.
    with pytest.raises(AssertionError, match="must match for arithmetic operations"):
        tensor_a + tensor_b
    with pytest.raises(AssertionError, match="must match for arithmetic operations"):
        tensor_c = tensor_a.clone()
        tensor_c += tensor_b

    # Test __sub__ method.
    with pytest.raises(AssertionError, match="must match for arithmetic operations"):
        tensor_a - tensor_b
    with pytest.raises(AssertionError, match="must match for arithmetic operations"):
        tensor_c = tensor_a.clone()
        tensor_c -= tensor_b

    # Test __mul__ method.
    with pytest.raises(AssertionError, match="must match for arithmetic operations"):
        tensor_a * tensor_b
    with pytest.raises(AssertionError, match="must match for arithmetic operations"):
        tensor_c = tensor_a.clone()
        tensor_c *= tensor_b

    # Test __truediv__ method.
    with pytest.raises(AssertionError, match="must match for arithmetic operations"):
        tensor_a / tensor_b
    with pytest.raises(AssertionError, match="must match for arithmetic operations"):
        tensor_c = tensor_a.clone()
        tensor_c /= tensor_b

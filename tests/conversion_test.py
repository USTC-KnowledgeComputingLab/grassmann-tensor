import typing
import pytest
import torch
from grassmann_tensor import GrassmannTensor


@pytest.fixture()
def x() -> GrassmannTensor:
    return GrassmannTensor((False, False), ((2, 2), (1, 3)), torch.randn([4, 4], device="cpu:0"))


@pytest.mark.parametrize("dtype_arg", ["position", "keyword", "none"])
@pytest.mark.parametrize("device_arg", ["position", "keyword", "none"])
@pytest.mark.parametrize("device_format", ["object", "string"])
def test_conversion(
    x: GrassmannTensor,
    dtype_arg: typing.Literal["position", "keyword", "none"],
    device_arg: typing.Literal["position", "keyword", "none"],
    device_format: typing.Literal["object", "string"],
) -> None:
    args: list[typing.Any] = []
    kwargs: dict[str, typing.Any] = {}

    device_str = "cuda:0" if torch.cuda.is_available() else "cpu:0"
    device = torch.device(device_str) if device_format == "object" else device_str
    match device_arg:
        case "position":
            args.append(device)
        case "keyword":
            kwargs["device"] = device
        case _:
            pass

    match dtype_arg:
        case "position":
            args.append(torch.complex128)
        case "keyword":
            kwargs["dtype"] = torch.complex128
        case _:
            pass

    if len(args) > 1:
        pytest.skip("Cannot pass both dtype and device as positional arguments")

    y = x.to(*args, **kwargs)
    assert isinstance(y, GrassmannTensor)
    assert y.arrow == x.arrow
    assert y.edges == x.edges
    assert y.tensor.dtype == torch.complex128 if dtype_arg != "none" else torch.float32
    assert (
        y.tensor.device.type
        == (torch.device(device_str) if device_arg != "none" else torch.device("cpu:0")).type
    )
    assert torch.allclose(y.tensor, x.tensor.to(dtype=y.tensor.dtype, device=y.tensor.device))


def test_conversion_duplicated_value(x: GrassmannTensor) -> None:
    with pytest.raises(AssertionError, match="Duplicate device specification"):
        x.to(torch.device("cpu"), device=torch.device("cpu"))
    with pytest.raises(AssertionError, match="Duplicate dtype specification"):
        x.to(torch.complex128, dtype=torch.complex128)
    with pytest.raises(AssertionError, match="Duplicate device specification"):
        x.to("cpu", device=torch.device("cpu"))

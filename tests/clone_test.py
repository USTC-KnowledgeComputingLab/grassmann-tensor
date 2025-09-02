import typing
import copy
import pytest
import torch
from grassmann_tensor import GrassmannTensor


@pytest.mark.parametrize("parity,mask", [(True, True), (True, False), (False, False)])
@pytest.mark.parametrize("which", ["clone", "copy", "deepcopy"])
def test_clone(
    parity: bool,
    mask: bool,
    which: typing.Literal["clone", "copy", "deepcopy"],
) -> None:
    original_tensor = GrassmannTensor(
        _arrow=(False, True),
        _edges=((2, 2), (1, 3)),
        _tensor=torch.randn([4, 4]),
    )

    if parity:
        _ = original_tensor.parity
    if mask:
        _ = original_tensor.mask

    match which:
        case "clone":
            cloned_tensor = original_tensor.clone()
        case "copy":
            cloned_tensor = copy.copy(original_tensor)
        case "deepcopy":
            cloned_tensor = copy.deepcopy(original_tensor)

    assert cloned_tensor._arrow == original_tensor._arrow
    assert cloned_tensor._edges == original_tensor._edges
    assert torch.equal(cloned_tensor._tensor, original_tensor._tensor)
    if parity:
        assert cloned_tensor._parity is not None
        assert original_tensor._parity is not None
        assert all(torch.equal(c, o) for c, o in zip(cloned_tensor._parity, original_tensor._parity))
    else:
        assert cloned_tensor._parity is original_tensor._parity
    if mask:
        assert cloned_tensor._mask is not None
        assert original_tensor._mask is not None
        assert torch.equal(cloned_tensor._mask, original_tensor._mask)
    else:
        assert cloned_tensor._mask is original_tensor._mask

    assert id(original_tensor.tensor) != id(cloned_tensor.tensor)

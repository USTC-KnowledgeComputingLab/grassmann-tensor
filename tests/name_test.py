import torch
import pytest

from grassmann_tensor import NamedGrassmannTensor


@pytest.mark.parametrize("rename_range", [(i, j) for i in range(5) for j in range(5) if j >= i])
def test_rename(rename_range: tuple[int, int]) -> None:
    l, h = rename_range  # noqa: E741
    edge = (2, 2)
    a = NamedGrassmannTensor(
        tuple(f"o{i}" for i in range(5)),
        tuple([False] * 5),
        tuple(edge for _ in range(5)),
        torch.randn(*[4] * 5),
    )
    new_names = tuple(f"n{i}" for i in range(h - l))
    name_map: dict[str, str] = {old: new for old, new in zip(a.names[l:h], new_names)}
    b = a.rename(name_map)
    assert b.names == a.names[:l] + new_names + a.names[h:]


def test_rename_duplicate_names() -> None:
    edge = (2, 2)
    a = NamedGrassmannTensor(
        tuple(f"o{i}" for i in range(5)),
        tuple([False] * 5),
        tuple(edge for _ in range(5)),
        torch.randn(*[4] * 5),
    )
    new_names = tuple("n" for i in range(5))
    name_map: dict[str, str] = {old: new for old, new in zip(a.names, new_names)}
    with pytest.raises(ValueError, match="Duplicate name"):
        _ = a.rename(name_map)


def test_get_name_index() -> None:
    edge = (2, 2)
    a = NamedGrassmannTensor(
        tuple(f"o{i}" for i in range(5)),
        tuple([False] * 5),
        tuple(edge for _ in range(5)),
        torch.randn(*[4] * 5),
    )
    with pytest.raises(KeyError, match="not in names"):
        _ = a.get_name_index("a")

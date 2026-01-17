import torch
import pytest
import itertools

from grassmann_tensor import GrassmannTensor, NamedGrassmannTensor


def generate_filled_data(
    edges: tuple[tuple[int, int], ...], data: torch.Tensor | None = None
) -> torch.Tensor:
    shape = tuple(even + odd for even, odd in edges)

    if data is None:
        tensor = torch.zeros(shape)
        filled = torch.arange(filled_count(edges))
    else:
        assert data is not None
        filled = data.reshape(-1)
        tensor = torch.zeros(shape, dtype=data.dtype, device=data.device)
    i = 0
    ranges = [range(s) for s in shape]

    for idx in itertools.product(*ranges):
        total_parity = 0
        for k, (even, _) in enumerate(edges):
            total_parity ^= 1 if idx[k] >= even else 0
        if total_parity == 0:
            tensor[idx] = filled[i]
            i += 1
    return tensor


def filled_count(edges: tuple[tuple[int, int], ...]) -> int:
    total = 1
    diff = 1
    for even, odd in edges:
        total *= even + odd
        diff *= even - odd
    return (total + diff) // 2


@pytest.mark.parametrize(
    "edges",
    [
        ((1, 1),),
        ((2, 2),),
        ((2, 2), (2, 2)),
        ((2, 2), (2, 2), (2, 2)),
        ((2, 2), (2, 2), (2, 2), (2, 2)),
    ],
)
def test_norm(edges: tuple[tuple[int, int], ...]) -> None:
    arrow = tuple([False] * len(edges))
    filled = filled_count(edges)
    half = filled // 2
    if filled % 2 == 0:
        data = torch.arange(-half, half, dtype=torch.float64)
    else:
        data = torch.arange(-half, half + 1, dtype=torch.float64)
    tensor_data = generate_filled_data(edges, data=data)
    max_val = tensor_data.abs().max().item()
    min_val = tensor_data.abs().min().item()

    tensor = GrassmannTensor(arrow, edges, tensor_data)
    assert tensor.norm(p=torch.inf) == max_val
    assert tensor.norm(p=-torch.inf) == min_val
    assert tensor.norm(p=0) == filled - 1
    assert tensor.norm(p=2) == torch.linalg.vector_norm(data, ord=2)


@pytest.mark.parametrize(
    "edges",
    [
        ((1, 1),),
        ((2, 2),),
        ((2, 2), (2, 2)),
        ((2, 2), (2, 2), (2, 2)),
        ((2, 2), (2, 2), (2, 2), (2, 2)),
    ],
)
def test_named_norm(edges: tuple[tuple[int, int], ...]) -> None:
    names = tuple(chr(96 + i) for i in range(len(edges)))
    arrow = tuple([False] * len(edges))
    filled = filled_count(edges)
    half = filled // 2
    if filled % 2 == 0:
        data = torch.arange(-half, half, dtype=torch.float64)
    else:
        data = torch.arange(-half, half + 1, dtype=torch.float64)
    tensor_data = generate_filled_data(edges, data=data)
    max_val = tensor_data.abs().max().item()
    min_val = tensor_data.abs().min().item()

    tensor = NamedGrassmannTensor(names, arrow, edges, tensor_data)
    assert tensor.norm(p=torch.inf) == max_val
    assert tensor.norm(p=-torch.inf) == min_val
    assert tensor.norm(p=0) == filled - 1
    assert tensor.norm(p=2) == torch.linalg.vector_norm(data, ord=2)


def test_sqrt() -> None:
    tensor = GrassmannTensor((False, False), ((1, 1), (1, 1)), torch.Tensor([[-4, 9], [0, -1]]))
    assert torch.allclose(tensor.sqrt().tensor, torch.Tensor(([[2, 3], [0, 1]])))


def test_named_sqrt() -> None:
    tensor = NamedGrassmannTensor(
        ("a", "b"), (False, False), ((1, 1), (1, 1)), torch.Tensor([[-4, 9], [0, -1]])
    )
    assert torch.allclose(tensor.sqrt().tensor, torch.Tensor(([[2, 3], [0, 1]])))


def test_rank() -> None:
    tensor = NamedGrassmannTensor(
        ("a", "b"), (False, False), ((1, 1), (1, 1)), torch.Tensor([[-4, 9], [0, -1]])
    )
    assert tensor.rank() == 2


def test_allclose() -> None:
    data1 = torch.Tensor([[0, 1], [2, 3]]).to(dtype=torch.float64, device="cpu")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data2 = torch.Tensor([[0, -1], [-2, 3]]).to(dtype=torch.float64, device=device)
    tensor1 = NamedGrassmannTensor(("a", "b"), (False, True), ((1, 1), (1, 1)), data1)
    tensor2 = NamedGrassmannTensor(("a", "b"), (False, True), ((1, 1), (1, 1)), data2)
    assert tensor1.allclose(tensor2)

    tensor3 = NamedGrassmannTensor(("a", "b"), (False, True), ((1, 1), (1, 1)), torch.randn(2, 2))
    tensor4 = NamedGrassmannTensor(("a", "b"), (False, True), ((1, 1), (1, 1)), torch.randn(2, 2))
    assert not tensor3.allclose(tensor4)

    data = generate_filled_data(((1, 1), (1, 1)))
    tensor5 = NamedGrassmannTensor(("a", "b"), (False, True), ((1, 1), (1, 1)), data)
    tensor6 = NamedGrassmannTensor(("a", "b"), (False, True), ((1, 1), (1, 1)), data)
    tensor6 = tensor6.permute(("b", "a"))
    assert tensor5.allclose(tensor6)


def test_allclose_other_type() -> None:
    data = generate_filled_data(((1, 1), (1, 1)))
    tensor1 = NamedGrassmannTensor(("a", "b"), (False, True), ((1, 1), (1, 1)), data)
    tensor2 = data
    with pytest.raises(TypeError, match="Expected NamedGrassmannTensor"):
        tensor1.allclose(tensor2)  # type: ignore[arg-type]


def test_allclose_mismatch_names() -> None:
    data = generate_filled_data(((1, 1), (1, 1)))
    tensor1 = NamedGrassmannTensor(("a", "b"), (False, True), ((1, 1), (1, 1)), data)
    tensor2 = NamedGrassmannTensor(("c", "d"), (False, True), ((1, 1), (1, 1)), data)
    with pytest.raises(TypeError, match="Expected same name"):
        tensor1.allclose(tensor2)


def test_allclose_mismatch_arrows() -> None:
    data = generate_filled_data(((1, 1), (1, 1)))
    tensor1 = NamedGrassmannTensor(("a", "b"), (False, True), ((1, 1), (1, 1)), data)
    tensor2 = NamedGrassmannTensor(("a", "b"), (False, False), ((1, 1), (1, 1)), data)
    with pytest.raises(TypeError, match="Expected same arrow"):
        tensor1.allclose(tensor2)


def test_allclose_mismatch_edges() -> None:
    data = generate_filled_data(((1, 1), (1, 1)))
    tensor1 = NamedGrassmannTensor(("a", "b"), (False, True), ((1, 1), (1, 1)), data)
    tensor2 = NamedGrassmannTensor(("a", "b"), (False, True), ((2, 0), (0, 2)), data)
    with pytest.raises(TypeError, match="Expected same edges"):
        tensor1.allclose(tensor2)

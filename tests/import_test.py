def test_import() -> None:
    from grassmann_tensor import GrassmannTensor, NamedGrassmannTensor

    assert isinstance(GrassmannTensor, type)
    assert isinstance(NamedGrassmannTensor, type)

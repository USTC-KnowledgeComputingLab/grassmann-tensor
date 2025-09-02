"""
A Grassmann tensor class.
"""

from __future__ import annotations

__all__ = ["GrassmannTensor"]

import dataclasses
import functools
import typing
import torch


@dataclasses.dataclass
class GrassmannTensor:
    """
    A Grassmann tensor class, which stores a tensor along with information about its edges.
    Each dimension of the tensor is composed of an even and an odd part, represented as a pair of integers.
    """

    _arrow: tuple[bool, ...]
    _edges: tuple[tuple[int, int], ...]
    _tensor: torch.Tensor
    _parity: tuple[torch.Tensor, ...] | None = None
    _mask: torch.Tensor | None = None

    @property
    def arrow(self) -> tuple[bool, ...]:
        """
        The arrow of the tensor, represented as a tuple of booleans indicating the order of the fermion operators.
        """
        return self._arrow

    @property
    def edges(self) -> tuple[tuple[int, int], ...]:
        """
        The edges of the tensor, represented as a tuple of pairs (even, odd).
        """
        return self._edges

    @property
    def tensor(self) -> torch.Tensor:
        """
        The underlying tensor data.
        """
        return self._tensor

    @property
    def parity(self) -> tuple[torch.Tensor, ...]:
        """
        The parity of each edge, represented as a tuple of tensors.
        """
        if self._parity is None:
            self._parity = tuple(self._edge_mask(even, odd) for (even, odd) in self._edges)
        return self._parity

    @property
    def mask(self) -> torch.Tensor:
        """
        The mask of the tensor, which has the same shape as the tensor and indicates which elements could be non-zero based on the parity.
        """
        if self._mask is None:
            self._mask = self._tensor_mask()
        return self._mask

    def to(self, whatever: torch.device | torch.dtype | str | None = None, *, device: torch.device | None = None, dtype: torch.dtype | None = None) -> GrassmannTensor:
        """
        Copy the tensor to a specified device or copy it to a specified data type.
        """
        match whatever:
            case torch.device():
                assert device is None, "Duplicate device specification."
                device = whatever
            case torch.dtype():
                assert dtype is None, "Duplicate dtype specification."
                dtype = whatever
            case str():
                assert device is None, "Duplicate device specification."
                device = torch.device(whatever)
            case _:
                pass
        match (device, dtype):
            case (None, None):
                return self
            case (None, _):
                return dataclasses.replace(
                    self,
                    _tensor=self._tensor.to(dtype=dtype),
                )
            case (_, None):
                return dataclasses.replace(
                    self,
                    _tensor=self._tensor.to(device=device),
                    _parity=tuple(p.to(device) for p in self._parity) if self._parity is not None else None,
                    _mask=self._mask.to(device) if self._mask is not None else None,
                )
            case _:
                return dataclasses.replace(
                    self,
                    _tensor=self._tensor.to(device=device, dtype=dtype),
                    _parity=tuple(p.to(device=device) for p in self._parity) if self._parity is not None else None,
                    _mask=self._mask.to(device=device) if self._mask is not None else None,
                )

    def update_mask(self) -> GrassmannTensor:
        """
        Update the mask of the tensor based on its parity.
        """
        self._tensor = torch.where(self.mask, 0, self._tensor)
        return self

    def permute(self, before_by_after: tuple[int, ...]) -> GrassmannTensor:
        """
        Permute the indices of the Grassmann tensor.
        """
        assert len(before_by_after) == len(set(before_by_after)), "Permutation indices must be unique."
        assert set(before_by_after) == set(range(self.tensor.dim())), "Permutation indices must cover all dimensions."

        arrow = tuple(self.arrow[i] for i in before_by_after)
        edges = tuple(self.edges[i] for i in before_by_after)
        tensor = self.tensor.permute(before_by_after)
        parity = tuple(self.parity[i] for i in before_by_after)
        mask = self.mask.permute(before_by_after)

        total_parity = functools.reduce(
            torch.logical_xor,
            (
                torch.logical_and(self._unsqueeze(parity[i], i, self.tensor.dim()), self._unsqueeze(parity[j], j, self.tensor.dim()))
                for j in range(self.tensor.dim())
                for i in range(0, j)  # all 0 <= i < j < dim
                if before_by_after[i] > before_by_after[j]),
            torch.zeros([], dtype=torch.bool, device=self.tensor.device),
        )
        tensor = torch.where(total_parity, -tensor, +tensor)

        return dataclasses.replace(
            self,
            _arrow=arrow,
            _edges=edges,
            _tensor=tensor,
            _parity=parity,
            _mask=mask,
        )

    def reverse(self, indices: tuple[int, ...]) -> GrassmannTensor:
        """
        Reverse the specified indices of the Grassmann tensor.

        A single sign is generated during reverse, which should be applied to one of the connected two tensors.
        This package always applies it to the tensor with arrow as True.
        """
        assert len(set(indices)) == len(indices), f"Indices must be unique. Got {indices}."
        assert all(0 <= i < self.tensor.dim() for i in indices), f"Indices must be within tensor dimensions. Got {indices}."

        arrow = tuple(self.arrow[i] ^ i in indices for i in range(self.tensor.dim()))
        tensor = self.tensor

        total_parity = functools.reduce(
            torch.logical_xor,
            (self._unsqueeze(parity, index, self.tensor.dim()) for index, parity in enumerate(self.parity) if index in indices and self.arrow[index]),
            torch.zeros([], dtype=torch.bool, device=self.tensor.device),
        )
        tensor = torch.where(total_parity, -tensor, +tensor)

        return dataclasses.replace(
            self,
            _arrow=arrow,
            _tensor=tensor,
        )

    def _reorder_indices(self, edges: tuple[tuple[int, int], ...]) -> tuple[int, int, torch.Tensor, torch.Tensor]:
        parity = functools.reduce(
            torch.logical_xor,
            (self._unsqueeze(self._edge_mask(even, odd), index, len(edges)) for index, (even, odd) in enumerate(edges)),
            torch.zeros([], dtype=torch.bool, device=self.tensor.device),
        )
        flatten_parity = parity.flatten()
        even = (~flatten_parity).nonzero().squeeze()
        odd = flatten_parity.nonzero().squeeze()
        reorder = torch.cat([even, odd], dim=0)

        total = functools.reduce(
            torch.add,
            (self._unsqueeze(self._edge_mask(even, odd), index, len(edges)).to(dtype=torch.int16) for index, (even, odd) in enumerate(edges)),
            torch.zeros([], dtype=torch.int16, device=self.tensor.device),
        )
        count = total * (total - 1)
        sign = (count & 2).to(dtype=torch.bool)
        return len(even), len(odd), reorder, sign.flatten()

    def reshape(self, new_shape: tuple[int | tuple[int, int], ...]) -> GrassmannTensor:
        """
        Reshape the Grassmann tensor, which may split or merge edges.

        The new shape must be compatible with the original shape.
        This operation does not change the arrow and it cannot merge two edges with different arrows.

        The new shape should be a tuple of each new dimension, which is represented as either a single integer or a pair of two integers.
        When a dimension is not changed, user could pass -1 to indicate that the dimension remains the same.
        When a dimension is merged, user only needs to pass a single integer to indicate the new dimension size.
        When a dimension is split, user must pass several pairs of two integers (even, odd) to indicate the new even and odd parts.

        A single sign is generated during merging or splitting two edges, which should be applied to one of the connected two tensors.
        This package always applies it to the tensor with arrow as True.
        """
        # This function reshapes the Grassmann tensor according to the new shape, including the following steps:
        # 1. Generate new arrow, edges, and shape for tensor
        # 2. Reorder the indices for splitting
        # 3. Apply the sign for splitting
        # 4. reshape the core tensor according to the new shape
        # 5. Apply the sign for merging
        # 6. Reorder the indices for merging

        # pylint: disable=too-many-branches, too-many-locals, too-many-statements

        arrow: list[bool] = []
        edges: list[tuple[int, int]] = []
        shape: list[int] = []

        splitting_sign: list[tuple[int, torch.Tensor]] = []
        splitting_reorder: list[tuple[int, torch.Tensor]] = []
        merging_reorder: list[tuple[int, torch.Tensor]] = []
        merging_sign: list[tuple[int, torch.Tensor]] = []

        cursor_plan: int = 0
        cursor_self: int = 0
        while True:
            if new_shape[cursor_plan] == -1:
                # Does not change
                arrow.append(self.arrow[cursor_self])
                edges.append(self.edges[cursor_self])
                shape.append(self.tensor.shape[cursor_self])
                cursor_self += 1
                cursor_plan += 1
            else:
                cursor_new_shape = new_shape[cursor_plan]
                total = cursor_new_shape if isinstance(cursor_new_shape, int) else cursor_new_shape[0] + cursor_new_shape[1]
                if total >= self.tensor.shape[cursor_self]:
                    # Merging
                    new_cursor_self = cursor_self
                    self_total = 1
                    while True:
                        self_total *= self.tensor.shape[new_cursor_self]
                        new_cursor_self += 1
                        if self_total == total:
                            break
                        assert self_total < total, f"Dimension mismatch with edges {self.edges} and new shape {new_shape}."
                        assert new_cursor_self < self.tensor.dim(), f"New shape {new_shape} exceeds tensor dimensions {self.tensor.dim()}."
                    even, odd, reorder, sign = self._reorder_indices(self.edges[cursor_self:new_cursor_self])
                    if isinstance(cursor_new_shape, tuple):
                        assert (even, odd) == cursor_new_shape, f"New even and odd number mismatch during merging {self.edges} to {new_shape}."
                    arrow.append(self.arrow[cursor_self])
                    assert all(
                        self_arrow == arrow[-1] for self_arrow in self.arrow[cursor_self:new_cursor_self]), f"Cannot merge edges with different arrows {self.arrow[cursor_self:new_cursor_self]}."
                    edges.append((even, odd))
                    shape.append(total)
                    if cursor_self + 1 != new_cursor_self:
                        # Really something merged
                        merging_sign.append((cursor_plan, sign))
                        merging_reorder.append((cursor_plan, reorder))
                    cursor_self = new_cursor_self
                    cursor_plan += 1
                else:
                    # Splitting
                    new_cursor_plan = cursor_plan
                    plan_total = 1
                    while True:
                        new_cursor_new_shape = new_shape[new_cursor_plan]
                        assert isinstance(new_cursor_new_shape, tuple), f"New shape must be a pair when splitting, got {new_cursor_new_shape}."
                        plan_total *= new_cursor_new_shape[0] + new_cursor_new_shape[1]
                        new_cursor_plan += 1
                        if plan_total == self.tensor.shape[cursor_self]:
                            break
                        assert plan_total < self.tensor.shape[cursor_self], f"Dimension mismatch with edges {self.edges} and new shape {new_shape}."
                        assert new_cursor_plan < len(new_shape), f"New shape {new_shape} exceeds specified dimensions {len(new_shape)}."
                    # new_shape has been verified to be tuple[int, int] in the loop
                    even, odd, reorder, sign = self._reorder_indices(typing.cast(tuple[tuple[int, int], ...], new_shape[cursor_plan:new_cursor_plan]))
                    assert (even, odd) == self.edges[cursor_self], f"New even and odd number mismatch during splitting {self.edges[cursor_self]} to {new_shape[cursor_plan:new_cursor_plan]}."
                    for i in range(cursor_plan, new_cursor_plan):
                        # new_shape has been verified to be tuple[int, int] in the loop
                        new_cursor_new_shape = typing.cast(tuple[int, int], new_shape[i])
                        arrow.append(self.arrow[cursor_self])
                        edges.append(new_cursor_new_shape)
                        shape.append(new_cursor_new_shape[0] + new_cursor_new_shape[1])
                    splitting_reorder.append((cursor_self, reorder))
                    splitting_sign.append((cursor_self, sign))
                    cursor_self += 1
                    cursor_plan = new_cursor_plan

            if cursor_plan == len(new_shape) and cursor_self == self.tensor.dim():
                break

        tensor = self.tensor

        for index, reorder in splitting_reorder:
            inverse_reorder = torch.empty_like(reorder)
            inverse_reorder[reorder] = torch.arange(reorder.size(0), device=reorder.device)
            tensor = tensor.index_select(index, inverse_reorder)

        splitting_parity = functools.reduce(
            torch.logical_xor,
            (self._unsqueeze(sign, index, self.tensor.dim()) for index, sign in splitting_sign if self.arrow[index]),
            torch.zeros([], dtype=torch.bool, device=self.tensor.device),
        )
        tensor = torch.where(splitting_parity, -tensor, +tensor)

        tensor = tensor.reshape(shape)

        merging_parity = functools.reduce(
            torch.logical_xor,
            (self._unsqueeze(sign, index, tensor.dim()) for index, sign in merging_sign if arrow[index]),
            torch.zeros([], dtype=torch.bool, device=self.tensor.device),
        )
        tensor = torch.where(merging_parity, -tensor, +tensor)

        for index, reorder in merging_reorder:
            tensor = tensor.index_select(index, reorder)

        return GrassmannTensor(_arrow=tuple(arrow), _edges=tuple(edges), _tensor=tensor)

    def matmul(self, other: GrassmannTensor) -> GrassmannTensor:
        """
        Perform matrix multiplication with another Grassmann tensor.
        Both of them should be rank 2 tensors, except some pure even edges could exist before the last two edges.
        """
        # The creation operator order from arrow is (False True)
        # So (x, True) * (False, y) = (x, y)
        tensor_a = self
        tensor_b = other

        vector_a = False
        if tensor_a.tensor.dim() == 1:
            tensor_a = tensor_a.reshape(((1, 0), -1))
            vector_a = True
        vector_b = False
        if tensor_b.tensor.dim() == 1:
            tensor_b = tensor_b.reshape((-1, (1, 0)))
            vector_b = True

        assert all(odd == 0 for (even, odd) in tensor_a.edges[:-2]), f"All edges except the last two must be pure even. Got {tensor_a.edges[:-2]}."
        assert all(odd == 0 for (even, odd) in tensor_b.edges[:-2]), f"All edges except the last two must be pure even. Got {tensor_b.edges[:-2]}."

        if tensor_a.arrow[-1] is not True:
            tensor_a = tensor_a.reverse((tensor_a.tensor.dim() - 1,))
        if tensor_b.arrow[-2] is not False:
            tensor_b = tensor_b.reverse((tensor_b.tensor.dim() - 2,))

        broadcast_a = tensor_a.tensor.dim() - 2
        broadcast_b = tensor_b.tensor.dim() - 2

        arrow = []
        edges = []
        for i in range(-max(broadcast_a, broadcast_b), 0):
            arrow.append(False)
            candidate_a = candidate_b = 1
            if i >= -broadcast_a:
                candidate_a = tensor_a.edges[i - 2][0]
            if i >= -broadcast_b:
                candidate_b = tensor_b.edges[i - 2][0]
            assert candidate_a == candidate_b or candidate_a == 1 or candidate_b == 1, f"Cannot broadcast edges {tensor_a.edges[i - 2]} and {tensor_b.edges[i - 2]}."
            edges.append((max(candidate_a, candidate_b), 0))
        if not vector_a:
            arrow.append(tensor_a.arrow[-2])
            edges.append(tensor_a.edges[-2])
        if not vector_b:
            arrow.append(tensor_b.arrow[-1])
            edges.append(tensor_b.edges[-1])
        tensor = torch.matmul(tensor_a.tensor, tensor_b.tensor)
        if vector_a:
            tensor = tensor.squeeze(-2)
        if vector_b:
            tensor = tensor.squeeze(-1)

        return GrassmannTensor(
            _arrow=tuple(arrow),
            _edges=tuple(edges),
            _tensor=tensor,
        )

    def __post_init__(self) -> None:
        assert len(self._arrow) == self._tensor.dim(), f"Arrow length ({len(self._arrow)}) must match tensor dimensions ({self._tensor.dim()})."
        assert len(self._edges) == self._tensor.dim(), f"Edges length ({len(self._edges)}) must match tensor dimensions ({self._tensor.dim()})."
        for dim, (even, odd) in zip(self._tensor.shape, self._edges):
            assert even >= 0 and odd >= 0 and dim == even + odd, f"Dimension {dim} must equal sum of even ({even}) and odd ({odd}) parts, and both must be non-negative."

    def _unsqueeze(self, tensor: torch.Tensor, index: int, dim: int) -> torch.Tensor:
        return tensor.view([-1 if i == index else 1 for i in range(dim)])

    def _edge_mask(self, even: int, odd: int) -> torch.Tensor:
        return torch.cat([torch.zeros(even, dtype=torch.bool, device=self.tensor.device), torch.ones(odd, dtype=torch.bool, device=self.tensor.device)])

    def _tensor_mask(self) -> torch.Tensor:
        return functools.reduce(
            torch.logical_xor,
            (self._unsqueeze(parity, index, self._tensor.dim()) for index, parity in enumerate(self.parity)),
            torch.zeros_like(self._tensor, dtype=torch.bool),
        )

    def _validate_edge_compatibility(self, other: GrassmannTensor) -> None:
        """
        Validate that the edges of two ParityTensor instances are compatible for arithmetic operations.
        """
        assert self._arrow == other.arrow, f"Arrows must match for arithmetic operations. Got {self._arrow} and {other.arrow}."
        assert self._edges == other.edges, f"Edges must match for arithmetic operations. Got {self._edges} and {other.edges}."

    def __pos__(self) -> GrassmannTensor:
        return dataclasses.replace(
            self,
            _tensor=+self._tensor,
        )

    def __neg__(self) -> GrassmannTensor:
        return dataclasses.replace(
            self,
            _tensor=-self._tensor,
        )

    def __add__(self, other: typing.Any) -> GrassmannTensor:
        if isinstance(other, GrassmannTensor):
            self._validate_edge_compatibility(other)
            return dataclasses.replace(
                self,
                _tensor=self._tensor + other._tensor,
            )
        try:
            result = self._tensor + other
        except TypeError:
            return NotImplemented
        if isinstance(result, torch.Tensor):
            return dataclasses.replace(
                self,
                _tensor=result,
            )
        return NotImplemented

    def __radd__(self, other: typing.Any) -> GrassmannTensor:
        try:
            result = other + self._tensor
        except TypeError:
            return NotImplemented
        if isinstance(result, torch.Tensor):
            return dataclasses.replace(
                self,
                _tensor=result,
            )
        return NotImplemented

    def __iadd__(self, other: typing.Any) -> GrassmannTensor:
        if isinstance(other, GrassmannTensor):
            self._validate_edge_compatibility(other)
            self._tensor += other._tensor
            return self
        try:
            self._tensor += other
        except TypeError:
            return NotImplemented
        if isinstance(self._tensor, torch.Tensor):
            return self
        return NotImplemented

    def __sub__(self, other: typing.Any) -> GrassmannTensor:
        if isinstance(other, GrassmannTensor):
            self._validate_edge_compatibility(other)
            return dataclasses.replace(
                self,
                _tensor=self._tensor - other._tensor,
            )
        try:
            result = self._tensor - other
        except TypeError:
            return NotImplemented
        if isinstance(result, torch.Tensor):
            return dataclasses.replace(
                self,
                _tensor=result,
            )
        return NotImplemented

    def __rsub__(self, other: typing.Any) -> GrassmannTensor:
        try:
            result = other - self._tensor
        except TypeError:
            return NotImplemented
        if isinstance(result, torch.Tensor):
            return dataclasses.replace(
                self,
                _tensor=result,
            )
        return NotImplemented

    def __isub__(self, other: typing.Any) -> GrassmannTensor:
        if isinstance(other, GrassmannTensor):
            self._validate_edge_compatibility(other)
            self._tensor -= other._tensor
            return self
        try:
            self._tensor -= other
        except TypeError:
            return NotImplemented
        if isinstance(self._tensor, torch.Tensor):
            return self
        return NotImplemented

    def __mul__(self, other: typing.Any) -> GrassmannTensor:
        if isinstance(other, GrassmannTensor):
            self._validate_edge_compatibility(other)
            return dataclasses.replace(
                self,
                _tensor=self._tensor * other._tensor,
            )
        try:
            result = self._tensor * other
        except TypeError:
            return NotImplemented
        if isinstance(result, torch.Tensor):
            return dataclasses.replace(
                self,
                _tensor=result,
            )
        return NotImplemented

    def __rmul__(self, other: typing.Any) -> GrassmannTensor:
        try:
            result = other * self._tensor
        except TypeError:
            return NotImplemented
        if isinstance(result, torch.Tensor):
            return dataclasses.replace(
                self,
                _tensor=result,
            )
        return NotImplemented

    def __imul__(self, other: typing.Any) -> GrassmannTensor:
        if isinstance(other, GrassmannTensor):
            self._validate_edge_compatibility(other)
            self._tensor *= other._tensor
            return self
        try:
            self._tensor *= other
        except TypeError:
            return NotImplemented
        if isinstance(self._tensor, torch.Tensor):
            return self
        return NotImplemented

    def __truediv__(self, other: typing.Any) -> GrassmannTensor:
        if isinstance(other, GrassmannTensor):
            self._validate_edge_compatibility(other)
            return dataclasses.replace(
                self,
                _tensor=self._tensor / other._tensor,
            )
        try:
            result = self._tensor / other
        except TypeError:
            return NotImplemented
        if isinstance(result, torch.Tensor):
            return dataclasses.replace(
                self,
                _tensor=result,
            )
        return NotImplemented

    def __rtruediv__(self, other: typing.Any) -> GrassmannTensor:
        try:
            result = other / self._tensor
        except TypeError:
            return NotImplemented
        if isinstance(result, torch.Tensor):
            return dataclasses.replace(
                self,
                _tensor=result,
            )
        return NotImplemented

    def __itruediv__(self, other: typing.Any) -> GrassmannTensor:
        if isinstance(other, GrassmannTensor):
            self._validate_edge_compatibility(other)
            self._tensor /= other._tensor
            return self
        try:
            self._tensor /= other
        except TypeError:
            return NotImplemented
        if isinstance(self._tensor, torch.Tensor):
            return self
        return NotImplemented

    def clone(self) -> GrassmannTensor:
        """
        Create a deep copy of the Grassmann tensor.
        """
        return dataclasses.replace(
            self,
            _tensor=self._tensor.clone(),
            _parity=tuple(parity.clone() for parity in self._parity) if self._parity is not None else None,
            _mask=self._mask.clone() if self._mask is not None else None,
        )

    def __copy__(self) -> GrassmannTensor:
        return self.clone()

    def __deepcopy__(self, memo: dict) -> GrassmannTensor:
        return self.clone()

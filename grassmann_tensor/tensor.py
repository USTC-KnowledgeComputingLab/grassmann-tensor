"""
A Grassmann tensor class.
"""

from __future__ import annotations

__all__ = ["GrassmannTensor", "NamedGrassmannTensor"]

import dataclasses
import functools
import typing
import math
import operator

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

    def to(
        self,
        whatever: torch.device | torch.dtype | str | None = None,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> GrassmannTensor:
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
                    _parity=tuple(p.to(device) for p in self._parity)
                    if self._parity is not None
                    else None,
                    _mask=self._mask.to(device) if self._mask is not None else None,
                )
            case _:
                return dataclasses.replace(
                    self,
                    _tensor=self._tensor.to(device=device, dtype=dtype),
                    _parity=tuple(p.to(device=device) for p in self._parity)
                    if self._parity is not None
                    else None,
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
        assert len(before_by_after) == len(set(before_by_after)), (
            "Permutation indices must be unique."
        )
        assert set(before_by_after) == set(range(self.tensor.dim())), (
            "Permutation indices must cover all dimensions."
        )

        arrow = tuple(self.arrow[i] for i in before_by_after)
        edges = tuple(self.edges[i] for i in before_by_after)
        tensor = self.tensor.permute(before_by_after)
        parity = tuple(self.parity[i] for i in before_by_after)
        mask = self.mask.permute(before_by_after)

        total_parity = functools.reduce(
            torch.logical_xor,
            (
                torch.logical_and(
                    self._unsqueeze(parity[i], i, self.tensor.dim()),
                    self._unsqueeze(parity[j], j, self.tensor.dim()),
                )
                for j in range(self.tensor.dim())
                for i in range(0, j)  # all 0 <= i < j < dim
                if before_by_after[i] > before_by_after[j]
            ),
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

    def reverse(self, indices: tuple[int, ...], apply_parity: bool = True) -> GrassmannTensor:
        """
        Reverse the specified indices of the Grassmann tensor.

        A single sign is generated during reverse, which should be applied to one of the connected two tensors.
        This package always applies it to the tensor with arrow as True.
        """
        assert len(set(indices)) == len(indices), f"Indices must be unique. Got {indices}."
        assert all(0 <= i < self.tensor.dim() for i in indices), (
            f"Indices must be within tensor dimensions. Got {indices}."
        )

        arrow = tuple(self.arrow[i] ^ (i in indices) for i in range(self.tensor.dim()))
        tensor = self.tensor

        total_parity = functools.reduce(
            torch.logical_xor,
            (
                self._unsqueeze(parity, index, self.tensor.dim())
                for index, parity in enumerate(self.parity)
                if index in indices and self.arrow[index] is apply_parity
            ),
            torch.zeros([], dtype=torch.bool, device=self.tensor.device),
        )
        tensor = torch.where(total_parity, -tensor, +tensor)

        return dataclasses.replace(
            self,
            _arrow=arrow,
            _tensor=tensor,
        )

    def _reorder_indices(
        self, edges: tuple[tuple[int, int], ...]
    ) -> tuple[int, int, torch.Tensor, torch.Tensor]:
        parity = functools.reduce(
            torch.logical_xor,
            (
                self._unsqueeze(self._edge_mask(even, odd), index, len(edges))
                for index, (even, odd) in enumerate(edges)
            ),
            torch.zeros([], dtype=torch.bool, device=self.tensor.device),
        )
        flatten_parity = parity.flatten()
        even = (~flatten_parity).nonzero().squeeze(-1)
        odd = flatten_parity.nonzero().squeeze(-1)
        reorder = torch.cat([even, odd], dim=0)

        total = functools.reduce(
            torch.add,
            (
                self._unsqueeze(self._edge_mask(even, odd), index, len(edges)).to(dtype=torch.int16)
                for index, (even, odd) in enumerate(edges)
            ),
            torch.zeros([], dtype=torch.int16, device=self.tensor.device),
        )
        count = total * (total - 1)
        sign = (count & 2).to(dtype=torch.bool)
        return len(even), len(odd), reorder, sign.flatten()

    def _calculate_even_odd(self) -> tuple[int, int]:
        return self.calculate_even_odd(self.edges)

    @staticmethod
    def calculate_even_odd(edges: tuple[tuple[int, int], ...]) -> tuple[int, int]:
        return functools.reduce(
            lambda accumulator, even_odd_pair: (
                accumulator[0] * even_odd_pair[0] + accumulator[1] * even_odd_pair[1],
                accumulator[0] * even_odd_pair[1] + accumulator[1] * even_odd_pair[0],
            ),
            edges,
            (1, 0),
        )

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

        arrow: list[bool] = []
        edges: list[tuple[int, int]] = []
        shape: list[int] = []

        splitting_sign: list[tuple[int, torch.Tensor]] = []
        splitting_reorder: list[tuple[int, torch.Tensor]] = []
        merging_reorder: list[tuple[int, torch.Tensor]] = []
        merging_sign: list[tuple[int, torch.Tensor]] = []

        original_self_is_scalar = self.tensor.dim() == 0
        if original_self_is_scalar:
            new_shape_list: list[tuple[int, int]] = []
            for item in new_shape:
                if item == -1:
                    raise AssertionError("Cannot use -1 when reshaping from a scalar")
                if isinstance(item, int):
                    if item != 1:
                        raise AssertionError(
                            f"Ambiguous integer dim {item} from scalar. "
                            "Use explicit (even, odd) pairs, or only use 1 for trivial edges."
                        )
                    new_shape_list.append((1, 0))
                else:
                    new_shape_list.append(item)
            new_shape = tuple(new_shape_list)
            edges_only = typing.cast(tuple[tuple[int, int], ...], new_shape)
            assert self.calculate_even_odd(edges_only) == (1, 0), (
                "Cannot split none edges into illegal edges"
            )

        if len(new_shape) == 0:
            assert self._calculate_even_odd() == (1, 0), (
                "Only pure even edges can be merged into none edges"
            )
            tensor = self.tensor.reshape(())
            return GrassmannTensor(_arrow=(), _edges=(), _tensor=tensor)

        if new_shape == (1,) and int(self.tensor.numel()) == 1:
            even_self, odd_self = self._calculate_even_odd()
            new_shape = ((even_self, odd_self),)

        cursor_plan: int = 0
        cursor_self: int = 0
        while cursor_plan != len(new_shape) or cursor_self != self.tensor.dim():
            if cursor_self == self.tensor.dim() and cursor_plan != len(new_shape):
                new_shape_check = new_shape[cursor_plan]
                if (isinstance(new_shape_check, int) and new_shape_check == 1) or (
                    new_shape_check == (1, 0)
                ):
                    if cursor_plan < len(self.arrow):
                        arrow.append(self.arrow[cursor_plan])
                    else:
                        arrow.append(False)
                    edges.append((1, 0))
                    shape.append(1)
                    cursor_plan += 1
                    continue
                raise AssertionError(
                    "New shape exceeds after exhausting self dimensions: "
                    f"edges={self.edges}, new_shape={new_shape}"
                )

            if cursor_plan != len(new_shape):
                new_shape_check = new_shape[cursor_plan]
                if (
                    isinstance(new_shape_check, int)
                    and new_shape_check == 1
                    and self.tensor.shape[cursor_self] != 1
                ):
                    arrow.append(False)
                    edges.append((1, 0))
                    shape.append(1)
                    cursor_plan += 1
                    continue

            if cursor_plan != len(new_shape) and new_shape[cursor_plan] == -1:
                # Does not change
                arrow.append(self.arrow[cursor_self])
                edges.append(self.edges[cursor_self])
                shape.append(self.tensor.shape[cursor_self])
                cursor_self += 1
                cursor_plan += 1
                continue
            elif (
                cursor_plan != len(new_shape)
                and new_shape[cursor_plan] == (1, 0)
                and cursor_plan < len(new_shape) - 1
            ):
                # A trivial plan edge
                arrow.append(False)
                edges.append((1, 0))
                shape.append(1)
                cursor_plan += 1
                continue
            elif cursor_self != self.tensor.dim() and self.edges[cursor_self] == (1, 0):
                # A trivial self edge
                cursor_self += 1
                continue
            cursor_new_shape = new_shape[cursor_plan]
            total = (
                cursor_new_shape
                if isinstance(cursor_new_shape, int)
                else cursor_new_shape[0] + cursor_new_shape[1]
            )
            # one of total and shape[cursor_self] is not trivial, otherwise it should be handled before
            if total == self.tensor.shape[cursor_self]:
                # We do not know whether it is merging or splitting, check more
                if isinstance(cursor_new_shape, int) or cursor_new_shape == self.edges[cursor_self]:
                    # If the new shape is exactly the same as the current edge, we treat it as no change
                    arrow.append(self.arrow[cursor_self])
                    edges.append(self.edges[cursor_self])
                    shape.append(self.tensor.shape[cursor_self])
                    cursor_self += 1
                    cursor_plan += 1
                    continue
                # Let's see if there are (0, 1) edges in the remaining self edges, if yes, we treat it as merging, otherwise splitting
                cursor_self_finding = cursor_self
                cursor_self_found = False
                while True:
                    cursor_self_finding += 1
                    if cursor_self_finding == self.tensor.dim():
                        break
                    if self.edges[cursor_self_finding] == (1, 0):
                        continue
                    if self.edges[cursor_self_finding] == (0, 1):
                        cursor_self_found = True
                        break
                    break
                merging = cursor_self_found
            elif total > self.tensor.shape[cursor_self]:
                merging = True
            elif total < self.tensor.shape[cursor_self]:
                merging = False
            if merging:
                # Merging between [cursor_self, new_cursor_self) and the another side contains dimension as self_total
                new_cursor_self = cursor_self
                self_total = 1
                while True:
                    # Try to include more dimension from self
                    self_total *= self.tensor.shape[new_cursor_self]
                    new_cursor_self += 1
                    # One dimension included, check if we can stop
                    if self_total == total:
                        even, odd, reorder, sign = self._reorder_indices(
                            self.edges[cursor_self:new_cursor_self]
                        )
                        if isinstance(cursor_new_shape, tuple):
                            if (even, odd) == cursor_new_shape:
                                break
                        else:
                            break
                    # For some reason we cannot stop here, continue to include more dimension, check something before continue
                    assert self_total <= total, (
                        f"Dimension mismatch in merging with edges {self.edges} and new shape {new_shape}."
                    )
                    assert new_cursor_self < self.tensor.dim(), (
                        f"New shape exceeds in merging with edges {self.edges} and new shape {new_shape}."
                    )
                # The merging block [cursor_self, new_cursor_self) has been determined
                arrow.append(self.arrow[cursor_self])
                assert all(
                    self_arrow == arrow[-1]
                    for self_arrow in self.arrow[cursor_self:new_cursor_self]
                ), (
                    f"Cannot merge edges with different arrows {self.arrow[cursor_self:new_cursor_self]}."
                )
                edges.append((even, odd))
                shape.append(total)
                merging_sign.append((cursor_plan, sign))
                merging_reorder.append((cursor_plan, reorder))
                cursor_self = new_cursor_self
                cursor_plan += 1
            else:
                # Splitting between [cursor_plan, new_cursor_plan) and the another side contains dimension as plan_total
                new_cursor_plan = cursor_plan
                plan_total = 1
                while True:
                    # Try to include more dimension from new_shape
                    new_cursor_new_shape = new_shape[new_cursor_plan]
                    assert isinstance(new_cursor_new_shape, tuple), (
                        f"New shape must be a pair when splitting, got {new_cursor_new_shape}."
                    )
                    plan_total *= new_cursor_new_shape[0] + new_cursor_new_shape[1]
                    new_cursor_plan += 1
                    # One dimension included, check if we can stop
                    if plan_total == self.tensor.shape[cursor_self]:
                        # new_shape block has been verified to be always tuple[int, int] before
                        even, odd, reorder, sign = self._reorder_indices(
                            typing.cast(
                                tuple[tuple[int, int], ...],
                                new_shape[cursor_plan:new_cursor_plan],
                            )
                        )
                        if (even, odd) == self.edges[cursor_self]:
                            break
                    # For some reason we cannot stop here, continue to include more dimension, check something before continue
                    assert plan_total <= self.tensor.shape[cursor_self], (
                        f"Dimension mismatch in splitting with edges {self.edges} and new shape {new_shape}."
                    )
                    assert new_cursor_plan < len(new_shape), (
                        f"New shape exceeds in splitting with edges {self.edges} and new shape {new_shape}."
                    )
                # The splitting block [cursor_plan, new_cursor_plan) has been determined
                for i in range(cursor_plan, new_cursor_plan):
                    # new_shape block has been verified to be always tuple[int, int] in the loop
                    new_cursor_new_shape = typing.cast(tuple[int, int], new_shape[i])
                    arrow.append(self.arrow[cursor_self])
                    edges.append(new_cursor_new_shape)
                    shape.append(new_cursor_new_shape[0] + new_cursor_new_shape[1])
                splitting_reorder.append((cursor_self, reorder))
                splitting_sign.append((cursor_self, sign))
                if self.tensor.dim() != 0:
                    cursor_self += 1
                cursor_plan = new_cursor_plan

        tensor = self.tensor

        for index, reorder in splitting_reorder:
            inverse_reorder = torch.empty_like(reorder)
            inverse_reorder[reorder] = torch.arange(reorder.size(0), device=reorder.device)
            tensor = tensor.index_select(index, inverse_reorder)

        splitting_parity = functools.reduce(
            torch.logical_xor,
            (
                self._unsqueeze(sign, index, self.tensor.dim())
                for index, sign in splitting_sign
                if self.tensor.dim() != 0 and self.arrow[index]
            ),
            torch.zeros([], dtype=torch.bool, device=self.tensor.device),
        )
        tensor = torch.where(splitting_parity, -tensor, +tensor)

        tensor = tensor.reshape(shape)

        merging_parity = functools.reduce(
            torch.logical_xor,
            (
                self._unsqueeze(sign, index, tensor.dim())
                for index, sign in merging_sign
                if arrow[index]
            ),
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

        assert all(odd == 0 for (even, odd) in tensor_a.edges[:-2]), (
            f"All edges except the last two must be pure even. Got {tensor_a.edges[:-2]}."
        )
        assert all(odd == 0 for (even, odd) in tensor_b.edges[:-2]), (
            f"All edges except the last two must be pure even. Got {tensor_b.edges[:-2]}."
        )

        if tensor_a.arrow[-1] is not True:
            tensor_a = tensor_a.reverse((tensor_a.tensor.dim() - 1,))
        if tensor_b.arrow[-2] is not False:
            tensor_b = tensor_b.reverse((tensor_b.tensor.dim() - 2,))

        arrow = []
        edges = []
        for i in range(-max(tensor_a.tensor.dim(), tensor_b.tensor.dim()), -2):
            arrow.append(False)
            candidate_a = candidate_b = 1
            if i >= -tensor_a.tensor.dim():
                candidate_a, _ = tensor_a.edges[i]
            if i >= -tensor_b.tensor.dim():
                candidate_b, _ = tensor_b.edges[i]
            assert candidate_a == candidate_b or candidate_a == 1 or candidate_b == 1, (
                f"Cannot broadcast edges {tensor_a.edges[i]} and {tensor_b.edges[i]}."
            )
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

    def _group_edges(
        self,
        pairs: tuple[int, ...] | tuple[tuple[int, ...], tuple[int, ...]],
    ) -> tuple[GrassmannTensor, tuple[int, ...], tuple[int, ...]]:
        return self.group_edges(self, pairs)

    @staticmethod
    def group_edges(
        tensor: GrassmannTensor,
        pairs: tuple[int, ...] | tuple[tuple[int, ...], tuple[int, ...]],
    ) -> tuple[GrassmannTensor, tuple[int, ...], tuple[int, ...]]:
        left_legs, right_legs = GrassmannTensor.get_legs_pair(tensor.tensor.dim(), pairs)

        order = left_legs + right_legs

        tensor = tensor.permute(order)

        left_dim = math.prod(tensor.tensor.shape[: len(left_legs)])
        right_dim = math.prod(tensor.tensor.shape[len(left_legs) :])

        tensor = tensor.reshape((left_dim, right_dim))

        return tensor, left_legs, right_legs

    @staticmethod
    def get_legs_pair(
        dim: int, pairs: tuple[int, ...] | tuple[tuple[int, ...], tuple[int, ...]]
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        def check_pairs_coverage(dim: int, pairs: tuple[tuple[int, ...], tuple[int, ...]]) -> bool:
            set0 = set(pairs[0])
            set1 = set(pairs[1])

            are_disjoint = set0.isdisjoint(set1)

            is_complete_union = (set0 | set1) == set(range(dim))

            no_duplicates = len(pairs[0]) + len(pairs[1]) == dim

            return are_disjoint and is_complete_union and no_duplicates

        if (isinstance(pairs, tuple) and len(pairs)) and all(
            isinstance(x, tuple) and all(isinstance(i, int) for i in x) for x in pairs
        ):
            left_legs = typing.cast(tuple[int, ...], pairs[0])
            right_legs = typing.cast(tuple[int, ...], pairs[1])
        else:
            left_legs = typing.cast(tuple[int, ...], pairs)
            right_legs = tuple(i for i in range(dim) if i not in left_legs)

        assert check_pairs_coverage(dim, (left_legs, right_legs)), (
            f"Input pairs must cover all dimension and disjoint, but got {(left_legs, right_legs)}"
        )

        return left_legs, right_legs

    def _get_legs_pair(
        self, pairs: tuple[int, ...] | tuple[tuple[int, ...], tuple[int, ...]]
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return self.get_legs_pair(self.tensor.dim(), pairs)

    def svd(
        self,
        free_names_u: tuple[int, ...],
        *,
        cutoff: int | None | tuple[int, int] = None,
    ) -> tuple[GrassmannTensor, GrassmannTensor, GrassmannTensor]:
        """
        This function is used to computes the singular value decomposition of a grassmann tensor.
        The SVD are implemented by follow steps:
        1. Split the legs into left and right;
        2. Merge the tensor with two groups.
        3. Split the block tensor into two parts.
        4. Compute the singular value decomposition.
        5. Use cutoff to keep the largest cutoff singular values (globally across even/odd blocks).
        6. Contract U, S and Vh.
        7. Split the legs into original left and right.
        The returned tensors U and V are not unique, nor are they continuous with respect to self.
        Due to this lack of uniqueness, different hardware and software may compute different singular vectors.
        Gradients computed using U or Vh will only be finite when A does not have repeated singular values.
        Furthermore, if the distance between any two singular values is close to zero, the gradient
        will be numerically unstable, as it depends on the singular values
        """
        if isinstance(cutoff, tuple):
            assert len(cutoff) == 2, "The length of cutoff must be 2 if cutoff is a tuple."

        left_legs, right_legs = self._get_legs_pair(free_names_u)
        order = left_legs + right_legs
        tensor = self.permute(order)

        arrow_reverse = tuple(i for i, current in enumerate(tensor.arrow) if current)
        if arrow_reverse:
            tensor = tensor.reverse(arrow_reverse, apply_parity=False)

        left_dim = math.prod(tensor.tensor.shape[: len(left_legs)])
        right_dim = math.prod(tensor.tensor.shape[len(left_legs) :])
        tensor = tensor.reshape((left_dim, right_dim))

        origin_arrow_left = tuple(self.arrow[i] for i in left_legs)
        origin_arrow_right = tuple(self.arrow[i] for i in right_legs)

        arrow_reverse_left = tuple(i for i, current in enumerate(origin_arrow_left) if current)
        arrow_reverse_right = tuple(
            i + 1 for i, current in enumerate(origin_arrow_right) if current
        )

        (even_left, odd_left) = tensor.edges[0]
        (even_right, odd_right) = tensor.edges[1]
        even_tensor = tensor.tensor[:even_left, :even_right]
        odd_tensor = tensor.tensor[even_left:, even_right:]

        if even_tensor.numel() > 0:
            U_even, S_even, Vh_even = torch.linalg.svd(even_tensor, full_matrices=False)
        else:
            U_even = even_tensor.new_zeros((even_left, 0))
            S_even = even_tensor.new_zeros((0,))
            Vh_even = even_tensor.new_zeros((0, even_right))

        if odd_tensor.numel() > 0:
            U_odd, S_odd, Vh_odd = torch.linalg.svd(odd_tensor, full_matrices=False)
        else:
            U_odd = odd_tensor.new_zeros((odd_left, 0))
            S_odd = odd_tensor.new_zeros((0,))
            Vh_odd = odd_tensor.new_zeros((0, odd_right))

        n_even, n_odd = S_even.shape[0], S_odd.shape[0]

        if cutoff is None:
            k_even, k_odd = n_even, n_odd
        elif isinstance(cutoff, int):
            if n_even == 0 and n_odd == 0:
                raise RuntimeError("Both parity block are empty. Can not form SVD.")
            assert cutoff > 0, f"Cutoff must be greater than 0, but got {cutoff}"
            k_even = min(cutoff, n_even)
            k_odd = min(cutoff, n_odd)
        elif isinstance(cutoff, tuple):
            assert len(cutoff) == 2, "The length of cutoff must be 2 if cutoff is a tuple."
            if n_even == 0 and n_odd == 0:
                raise RuntimeError("Both parity block are empty. Can not form SVD.")
            k_even = max(0, min(int(cutoff[0]), n_even))
            k_odd = max(0, min(int(cutoff[1]), n_odd))
        else:
            raise ValueError(
                f"Cutoff must be an integer or a tuple of two integers, but got {cutoff}"
            )

        keep_even = torch.zeros(n_even, dtype=torch.bool, device=S_even.device)
        keep_odd = torch.zeros(n_odd, dtype=torch.bool, device=S_odd.device)
        if k_even > 0:
            keep_even[:k_even] = True
        if k_odd > 0:
            keep_odd[:k_odd] = True

        U_even_trunc = U_even[:, keep_even]
        S_even_trunc = S_even[keep_even]
        Vh_even_trunc = Vh_even[keep_even, :]

        U_odd_trunc = U_odd[:, keep_odd]
        S_odd_trunc = S_odd[keep_odd]
        Vh_odd_trunc = Vh_odd[keep_odd, :]

        U_tensor = torch.block_diag(U_even_trunc, U_odd_trunc)  # type: ignore[no-untyped-call]
        S_tensor = torch.cat([S_even_trunc, S_odd_trunc], dim=0)
        Vh_tensor = torch.block_diag(Vh_even_trunc, Vh_odd_trunc)  # type: ignore[no-untyped-call]

        U_tensor = U_tensor.to(dtype=self.tensor.dtype, device=self.tensor.device)
        S_tensor = S_tensor.to(dtype=self.tensor.dtype, device=self.tensor.device)
        Vh_tensor = Vh_tensor.to(dtype=self.tensor.dtype, device=self.tensor.device)

        U_edges = (
            (U_even_trunc.shape[0], U_odd_trunc.shape[0]),
            (U_even_trunc.shape[1], U_odd_trunc.shape[1]),
        )
        S_edges = (
            (U_even_trunc.shape[1], U_odd_trunc.shape[1]),
            (Vh_even_trunc.shape[0], Vh_odd_trunc.shape[0]),
        )
        Vh_edges = (
            (Vh_even_trunc.shape[0], Vh_odd_trunc.shape[0]),
            (Vh_even_trunc.shape[1], Vh_odd_trunc.shape[1]),
        )

        U = GrassmannTensor(_arrow=(False, True), _edges=U_edges, _tensor=U_tensor)
        S = GrassmannTensor(
            _arrow=(
                False,
                True,
            ),
            _edges=S_edges,
            _tensor=torch.diag(S_tensor),
        )
        Vh = GrassmannTensor(_arrow=(False, False), _edges=Vh_edges, _tensor=Vh_tensor)

        left_edges = [self.edges[i] for i in left_legs]
        right_edges = [self.edges[i] for i in right_legs]

        U = U.reshape((*left_edges, U_edges[1]))
        U = U.reverse(arrow_reverse_left)

        Vh = Vh.reshape((Vh_edges[0], *right_edges))
        Vh = Vh.reverse(arrow_reverse_right)

        return U, S, Vh

    @staticmethod
    def get_inv_order(order: tuple[int, ...]) -> tuple[int, ...]:
        inv = [0] * len(order)
        for new_position, origin_idx in enumerate(order):
            inv[origin_idx] = new_position
        return tuple(inv)

    def contract(
        self,
        b: GrassmannTensor,
        leg_a: int | tuple[int, ...],
        leg_b: int | tuple[int, ...],
    ) -> GrassmannTensor:
        """
        This function is used to contract the GrassmannTensor with other GrassmannTensors.
        For two fermion tensors A and B, their edges are divided into two groups: common edges and free edges.
        Common edges connect each other, while the other edges are free edges.
        The following operations are performed in sequence:
        1. `Permutation`
        Place all free edges in A on the left and common edges on the right, while place all common edges in B on
        left and free edges on the right.
        2. `Reverse`
        Reverse the fermionic arrows of all free edges in A to False. If a sign is generated, the sign is not placed in
        this tensor. Reverse the fermionic arrows of all common edges in A to True. If a sign is generated, the sign is
        placed in this tensor. Reverse the fermionic arrows of all free edges in B to False. If a sign is generated, the
        sign is not placed in this tensor. Reverse the fermionic arrows of all common edges in B to False. If a sign is
        generated, the sign is not placed in this tensor.
        3. `Merge`
        If merging free edges of A generates a sign, it is not included in this tensor; If merging common edges of A
        generates a sign, it is included in this tensor; If merging free edges of B generates a sign, it is not included
        in this tensor; If merging common edges of B generates a sign, it is not included in this tensor.
        4. `Matrix multiplication`
        5. `Split`
        Split the remaining two free edges to restore the original shape of the tensor, and do not place any of the
        generated signs in this tensor.
        6. `Reverse`
        Reverse the fermionic arrows back to their original orientation in A and B, and the resulting sign is not placed
        in this tensor.
        """
        a = self

        leg_tuple_a = (leg_a,) if isinstance(leg_a, int) else leg_a
        leg_tuple_b = (leg_b,) if isinstance(leg_b, int) else leg_b

        for tensor, leg in ((a, leg_tuple_a), (b, leg_tuple_b)):
            assert len(set(leg)) == len(leg), f"Indices must be unique. Got {leg}."
            assert all(0 <= i < tensor.tensor.dim() for i in leg), (
                f"Indices must be within tensor dimensions. Got {leg}."
            )

        contract_length_a, contract_length_b = (
            1 if isinstance(leg, int) else len(leg) for leg in (leg_a, leg_b)
        )

        dim_a = a.tensor.dim()
        dim_b = b.tensor.dim()

        right_leg_a, left_leg_a = self.get_legs_pair(dim_a, leg_tuple_a)
        left_leg_b, right_leg_b = self.get_legs_pair(dim_b, leg_tuple_b)

        order_a = left_leg_a + right_leg_a
        order_b = left_leg_b + right_leg_b

        # 1. Permutation
        a = a.permute(order_a)
        b = b.permute(order_b)

        arrow = a.arrow[:-contract_length_a] + b.arrow[contract_length_b:]
        edges = a.edges[:-contract_length_a] + b.edges[contract_length_b:]
        shape = a.tensor.shape[:-contract_length_a] + b.tensor.shape[contract_length_b:]

        # 2. Reverse
        # Reverse tensor `a`
        arrow_after_reverse_a = tuple(
            [False] * (dim_a - contract_length_a) + [True] * contract_length_a
        )

        tensor_after_reverse = a.tensor
        # Only calculate the sign of common edges of `a`
        parity_index = tuple(range(dim_a - contract_length_a, dim_a))
        reversing_parity = functools.reduce(
            torch.logical_xor,
            (
                self._unsqueeze(parity, index, self.tensor.dim())
                for index, parity in enumerate(a.parity)
                if index in parity_index and not a.arrow[index]
            ),
            torch.zeros([], dtype=torch.bool, device=self.tensor.device),
        )
        tensor_after_reverse = torch.where(
            reversing_parity, -tensor_after_reverse, +tensor_after_reverse
        )

        a = dataclasses.replace(
            a,
            _arrow=arrow_after_reverse_a,
            _tensor=tensor_after_reverse,
        )

        # 3. Merge
        # Merge tensor `a` free edges
        arrow_merge_free_edges = tuple([False] + [True] * contract_length_a)
        edges_merge_free_edges = typing.cast(
            tuple[tuple[int, int], ...],
            (a.calculate_even_odd(a.edges[:-contract_length_a]), *a.edges[-contract_length_a:]),
        )
        shape_merge_free_edges = (
            math.prod(a.tensor.shape[:-contract_length_a]),
            *a.tensor.shape[-contract_length_a:],
        )
        tensor_merge_free_edges = a.tensor.reshape(shape_merge_free_edges)
        a = dataclasses.replace(
            a,
            _arrow=arrow_merge_free_edges,
            _edges=edges_merge_free_edges,
            _tensor=tensor_merge_free_edges,
        )

        # Merge tensor `a` common edges
        arrow_merge_common_edges = (False, True)
        shape_merge_common_edges = (a.tensor.shape[0], math.prod(a.tensor.shape[1:]))
        even, odd, _, sign = self._reorder_indices(a.edges[1:])
        edges_merge_common_edges = typing.cast(
            tuple[tuple[int, int], ...],
            (a.edges[0], (even, odd)),
        )
        tensor_merge_common_edges = a.tensor.reshape(shape_merge_common_edges)
        merging_parity = functools.reduce(
            torch.logical_xor,
            (self._unsqueeze(sign, 1, tensor_merge_common_edges.dim())),
            torch.zeros([], dtype=torch.bool, device=self.tensor.device),
        )
        tensor_merge_common_edges = torch.where(
            merging_parity, -tensor_merge_common_edges, +tensor_merge_common_edges
        )

        a = dataclasses.replace(
            a,
            _arrow=arrow_merge_common_edges,
            _edges=edges_merge_common_edges,
            _tensor=tensor_merge_common_edges,
        )

        # Reverse and merge tensor `b`
        arrow_after_reverse_merge_b = (False, True)
        edges_after_reverse_merge_b = (
            self.calculate_even_odd(b.edges[:contract_length_b]),
            self.calculate_even_odd(b.edges[contract_length_b:]),
        )
        tensor_after_reverse_merge_b = b.tensor.reshape(
            (
                math.prod(b.tensor.shape[:contract_length_b]),
                math.prod(b.tensor.shape[contract_length_b:]),
            )
        )

        b = dataclasses.replace(
            b,
            _arrow=arrow_after_reverse_merge_b,
            _edges=edges_after_reverse_merge_b,
            _tensor=tensor_after_reverse_merge_b,
        )

        # 4. Matrix multiplication
        c = a @ b

        # Split and reverse back
        c = dataclasses.replace(c, _arrow=arrow, _edges=edges, _tensor=c.tensor.reshape(shape))
        return c

    def exponential(
        self, pairs: tuple[tuple[int, ...], tuple[int, ...]], *, permute_back: bool = True
    ) -> GrassmannTensor:
        tensor, left_legs, right_legs = self._group_edges(pairs)

        assert tensor.arrow in ((False, True), (True, False)), (
            f"Exponentiation requires arrow (False, True) or (True, False), but got {tensor.arrow}"
        )

        tensor_reverse_flag = tensor.arrow != (False, True)
        if tensor_reverse_flag:
            tensor = tensor.reverse((0, 1), False)

        left_dim, right_dim = tensor.tensor.shape

        assert left_dim == right_dim, (
            f"Exponentiation requires a square operator, but got {left_dim} x {right_dim}."
        )

        (even_left, odd_left) = tensor.edges[0]
        (even_right, odd_right) = tensor.edges[1]

        assert even_left == even_right and odd_left == odd_right, (
            f"Parity blocks must be square, but got L=({even_left},{odd_left}), R=({even_right},{odd_right})"
        )

        even_tensor = tensor.tensor[:even_left, :even_right]
        odd_tensor = tensor.tensor[even_left:, even_right:]

        even_tensor_exp = torch.linalg.matrix_exp(even_tensor)
        odd_tensor_exp = torch.linalg.matrix_exp(odd_tensor)

        tensor_exp = torch.block_diag(even_tensor_exp, odd_tensor_exp)  # type: ignore[no-untyped-call]

        tensor_exp = dataclasses.replace(tensor, _tensor=tensor_exp)

        if tensor_reverse_flag:
            tensor_exp = tensor_exp.reverse((0, 1))

        order = left_legs + right_legs
        edges_after_permute = tuple(self.edges[i] for i in order)
        tensor_exp = tensor_exp.reshape(edges_after_permute)

        if permute_back:
            inv_order = self.get_inv_order(order)

            tensor_exp = tensor_exp.permute(inv_order)

        return tensor_exp

    def identity(
        self, pairs: tuple[tuple[int, ...], tuple[int, ...]], *, permute_back: bool = True
    ) -> GrassmannTensor:
        tensor, left_legs, right_legs = self._group_edges(pairs)

        assert tensor.arrow in ((False, True), (True, False)), (
            f"Identity requires arrow (False, True) or (True, False), but got {tensor.arrow}"
        )

        tensor_reverse_flag = tensor.arrow != (False, True)
        if tensor_reverse_flag:
            tensor = tensor.reverse((0, 1), False)

        left_dim, right_dim = tensor.tensor.shape

        assert left_dim == right_dim, (
            f"Identity requires a square operator, but got {left_dim} x {right_dim}."
        )

        (even_left, odd_left) = tensor.edges[0]
        (even_right, odd_right) = tensor.edges[1]

        assert even_left == even_right and odd_left == odd_right, (
            f"Parity blocks must be square, but got L=({even_left},{odd_left}), R=({even_right},{odd_right})"
        )

        I = torch.eye(left_dim, dtype=tensor.tensor.dtype, device=tensor.tensor.device)  # noqa: E741

        tensor_identity = dataclasses.replace(tensor, _tensor=I)

        if tensor_reverse_flag:
            tensor_identity = tensor_identity.reverse((0, 1))

        order = left_legs + right_legs
        edges_after_permute = tuple(self.edges[i] for i in order)
        tensor_identity = tensor_identity.reshape(edges_after_permute)

        if permute_back:
            inv_order = self.get_inv_order(order)

            tensor_identity = tensor_identity.permute(inv_order)

        return tensor_identity

    def conjugate(self) -> GrassmannTensor:
        tensor_conj = self.tensor.conj()

        dim = self.tensor.dim()
        parity = self.parity

        total_parity = functools.reduce(
            torch.logical_xor,
            (
                torch.logical_and(
                    self._unsqueeze(parity[i], i, dim),
                    self._unsqueeze(parity[j], j, dim),
                )
                for j in range(dim)
                for i in range(0, j)
            ),
            torch.zeros([], dtype=torch.bool, device=self.tensor.device),
        )

        tensor_conj = torch.where(total_parity, -tensor_conj, tensor_conj)

        return dataclasses.replace(
            self,
            _arrow=tuple(not arrow for arrow in self.arrow),
            _tensor=tensor_conj,
        )

    def conj(self) -> GrassmannTensor:
        return self.conjugate()

    def __post_init__(self) -> None:
        assert len(self._arrow) == self._tensor.dim(), (
            f"Arrow length ({len(self._arrow)}) must match tensor dimensions ({self._tensor.dim()})."
        )
        assert len(self._edges) == self._tensor.dim(), (
            f"Edges length ({len(self._edges)}) must match tensor dimensions ({self._tensor.dim()})."
        )
        for dim, (even, odd) in zip(self._tensor.shape, self._edges):
            assert even >= 0 and odd >= 0 and dim == even + odd, (
                f"Dimension {dim} must equal sum of even ({even}) and odd ({odd}) parts, and both must be non-negative."
            )

    def _unsqueeze(self, tensor: torch.Tensor, index: int, dim: int) -> torch.Tensor:
        return tensor.view([-1 if i == index else 1 for i in range(dim)])

    def _edge_mask(self, even: int, odd: int) -> torch.Tensor:
        return torch.cat(
            [
                torch.zeros(even, dtype=torch.bool, device=self.tensor.device),
                torch.ones(odd, dtype=torch.bool, device=self.tensor.device),
            ]
        )

    def _tensor_mask(self) -> torch.Tensor:
        return functools.reduce(
            torch.logical_xor,
            (
                self._unsqueeze(parity, index, self._tensor.dim())
                for index, parity in enumerate(self.parity)
            ),
            torch.zeros_like(self._tensor, dtype=torch.bool),
        )

    def reciprocal(self) -> GrassmannTensor:
        return dataclasses.replace(
            self, _tensor=torch.where(self.tensor == 0, self.tensor, 1 / self.tensor)
        )

    def norm(self, p: typing.Any) -> float:
        return float(torch.linalg.vector_norm(self.tensor.masked_select(~self.mask), ord=p))

    def sqrt(self) -> GrassmannTensor:
        return dataclasses.replace(self, _tensor=torch.sqrt(torch.abs(self.tensor)))

    def _validate_edge_compatibility(self, other: GrassmannTensor) -> None:
        """
        Validate that the edges of two ParityTensor instances are compatible for arithmetic operations.
        """
        assert self._arrow == other.arrow, (
            f"Arrows must match for arithmetic operations. Got {self._arrow} and {other.arrow}."
        )
        assert self._edges == other.edges, (
            f"Edges must match for arithmetic operations. Got {self._edges} and {other.edges}."
        )

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

    def __matmul__(self, other: typing.Any) -> GrassmannTensor:
        if isinstance(other, GrassmannTensor):
            return self.matmul(other)
        return NotImplemented

    def __rmatmul__(self, other: typing.Any) -> GrassmannTensor:
        return NotImplemented

    def __imatmul__(self, other: typing.Any) -> GrassmannTensor:
        if isinstance(other, GrassmannTensor):
            return self.matmul(other)
        return NotImplemented

    def clone(self) -> GrassmannTensor:
        """
        Create a deep copy of the Grassmann tensor.
        """
        return dataclasses.replace(
            self,
            _tensor=self._tensor.clone(),
            _parity=tuple(parity.clone() for parity in self._parity)
            if self._parity is not None
            else None,
            _mask=self._mask.clone() if self._mask is not None else None,
        )

    def __copy__(self) -> GrassmannTensor:
        return self.clone()

    def __deepcopy__(self, memo: dict) -> GrassmannTensor:
        return self.clone()


@dataclasses.dataclass
class NamedGrassmannTensor:
    _names: tuple[str, ...]
    _arrow: tuple[bool, ...]
    _edges: tuple[tuple[int, int], ...]
    _tensor: torch.Tensor
    _parity: tuple[torch.Tensor, ...] | None = None
    _mask: torch.Tensor | None = None

    _name_dict: dict[str, int] = dataclasses.field(init=False, repr=False)
    _gt: GrassmannTensor = dataclasses.field(init=False, repr=False)

    @property
    def names(self) -> tuple[str, ...]:
        """
        The names of the tensor, represented as a tuple of str.
        """
        return self._names

    @property
    def arrow(self) -> tuple[bool, ...]:
        return self._arrow

    @property
    def edges(self) -> tuple[tuple[int, int], ...]:
        return self._edges

    @property
    def tensor(self) -> torch.Tensor:
        return self._tensor

    @property
    def gt(self) -> GrassmannTensor:
        return self._gt

    @property
    def parity(self) -> tuple[torch.Tensor, ...]:
        if self._parity is None:
            self._parity = self.gt.parity
        return self._parity

    @property
    def mask(self) -> torch.Tensor:
        if self._mask is None:
            self._mask = self.gt.mask
        return self._mask

    def update_mask(self) -> NamedGrassmannTensor:
        tensor = self.gt.update_mask()
        return dataclasses.replace(
            self,
            _tensor=tensor.tensor,
        )

    def rename(self, name_map: dict[str, str]) -> NamedGrassmannTensor:
        if not name_map:
            return self

        names = tuple(name_map.get(name, name) for name in self.names)

        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate names after rename: {names}")

        return dataclasses.replace(self, _names=names)

    def __post_init__(self) -> None:
        assert len(self._names) == len(set(self._names)), (
            f"Names must be unique, but got {self._names}"
        )
        assert len(self._names) == self._tensor.dim(), (
            f"Names length ({len(self._names)}) must match tensor dimensions ({self._tensor.dim()})."
        )
        object.__setattr__(self, "_name_dict", {name: i for i, name in enumerate(self._names)})
        gt = GrassmannTensor(
            _arrow=self._arrow,
            _edges=self._edges,
            _tensor=self._tensor,
            _parity=self._parity,
            _mask=self._mask,
        )
        object.__setattr__(self, "_gt", gt)

    def to(
        self,
        whatever: torch.device | torch.dtype | str | None = None,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> NamedGrassmannTensor:
        tensor = self.gt.to(whatever, device=device, dtype=dtype)
        return dataclasses.replace(
            self,
            _tensor=tensor.tensor,
            _parity=None,
            _mask=None,
        )

    def get_name_index(self, name: str) -> int:
        try:
            return self._name_dict[name]
        except KeyError:
            raise KeyError(f"{name!r} not in names list {self._names!r}") from None

    def permute(self, before_by_after: tuple[str, ...]) -> NamedGrassmannTensor:
        order = tuple(self.get_name_index(name) for name in before_by_after)
        tensor = self.gt.permute(order)
        return dataclasses.replace(
            self,
            _names=before_by_after,
            _arrow=tensor.arrow,
            _edges=tensor.edges,
            _tensor=tensor.tensor,
            _parity=None,
            _mask=None,
        )

    def reverse(self, reversed_names: set[str], apply_parity: bool = True) -> NamedGrassmannTensor:
        assert len(reversed_names) == len(set(reversed_names)), (
            f"Indices must be unique, but got {reversed_names}"
        )
        indices = tuple(self.get_name_index(name) for name in reversed_names)
        tensor = self.gt.reverse(indices, apply_parity=apply_parity)
        return dataclasses.replace(
            self,
            _arrow=tensor.arrow,
            _edges=tensor.edges,
            _tensor=tensor.tensor,
        )

    def _merge_edge_get_names(self, merge_map: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
        reserved_names: list[str] = []
        for name in self.names:
            found = next(
                (
                    (new_name, old_names)
                    for new_name, old_names in merge_map.items()
                    if name in old_names
                ),
                None,
            )
            if found is None:
                reserved_names.append(name)
            else:
                new_name, old_names = found
                if name == old_names[0]:
                    reserved_names.append(new_name)
        return tuple(reserved_names)

    @staticmethod
    def _merge_edge_get_name_group(
        name: str, merge_map: dict[str, tuple[str, ...]]
    ) -> tuple[str, ...]:
        merge_group = merge_map.get(name, None)
        return (name,) if merge_group is None else merge_group

    def merge_edge(
        self,
        merge_map: dict[str, tuple[str, ...]],
    ) -> NamedGrassmannTensor:
        all_old_names = [old_name for group in merge_map.values() for old_name in group]
        assert len(all_old_names) == len(set(all_old_names)), (
            f"Names must be unique, but got {all_old_names}"
        )
        assert all(len(old_names) > 0 for old_names in merge_map.values()), (
            "Merge edge does not support empty old_names."
        )
        assert all(
            all(old_name in self.names for old_name in old_names)
            for old_names in merge_map.values()
        ), f"Old names must be in names list, but got {merge_map.values()}"

        merge_map = {
            new_name: tuple(sorted(group, key=self.get_name_index))
            for new_name, group in merge_map.items()
        }

        names = self._merge_edge_get_names(merge_map)

        permuted_names: list[str] = functools.reduce(
            operator.add,
            (list(self._merge_edge_get_name_group(name, merge_map)) for name in names),
            [],
        )

        permuted_tensor = self.permute(tuple(permuted_names))

        new_edges: list[tuple[int, int]] = []

        for new_name in names:
            if new_name in merge_map:
                old_names = merge_map[new_name]
                merge_edges = tuple(
                    permuted_tensor.edges[permuted_tensor.get_name_index(old_name)]
                    for old_name in old_names
                )
                even, odd = permuted_tensor.gt.calculate_even_odd(merge_edges)
                new_edges.append((even, odd))
            else:
                index = permuted_tensor.get_name_index(new_name)
                new_edges.append(permuted_tensor.edges[index])

        merged_tensor = permuted_tensor.gt.reshape(tuple(new_edges))

        return dataclasses.replace(
            self,
            _names=names,
            _arrow=merged_tensor.arrow,
            _edges=merged_tensor.edges,
            _tensor=merged_tensor.tensor,
            _parity=None,
            _mask=None,
        )

    def to_scalar(self) -> NamedGrassmannTensor:
        tensor = self.gt.reshape(())
        return dataclasses.replace(
            self, _names=(), _arrow=(), _edges=(), _tensor=tensor.tensor, _parity=None, _mask=None
        )

    @staticmethod
    def _split_edge_get_name_group(
        name: str,
        split_map: dict[str, tuple[tuple[str, tuple[int, int]], ...]],
    ) -> list[str]:
        split_group = split_map.get(name, None)
        return [name] if split_group is None else [new_name for new_name, _ in split_group]

    @staticmethod
    def _split_edge_get_edge_group(
        name: str,
        edge: tuple[int, int],
        split_map: dict[str, tuple[tuple[str, tuple[int, int]], ...]],
    ) -> list[tuple[int, int]]:
        split_group = split_map.get(name, None)
        return [edge] if split_group is None else [new_edge for _, new_edge in split_group]

    def split_edge(
        self, split_map: dict[str, tuple[tuple[str, tuple[int, int]], ...]]
    ) -> NamedGrassmannTensor:
        new_names: tuple[str, ...]
        new_edges: tuple[tuple[int, int], ...]
        if len(self.names) == 0:
            assert set(split_map.keys()) == {""}, (
                "For scalar tensor, split_map must have only key ''."
            )
            new_group = split_map[""]
            new_names = tuple(new_name for new_name, _ in new_group)
            new_edges = tuple(new_edge for _, new_edge in new_group)

            split_tensor = self.gt.reshape(new_edges)
            return dataclasses.replace(
                self,
                _names=new_names,
                _arrow=split_tensor.arrow,
                _edges=split_tensor.edges,
                _tensor=split_tensor.tensor,
                _parity=None,
                _mask=None,
            )

        assert all(old_name in self.names for old_name in split_map.keys()), (
            f"Old name must be in names {self.names}"
        )

        new_names = tuple(
            functools.reduce(
                operator.add,
                (self._split_edge_get_name_group(name, split_map) for name in self.names),
                [],
            )
        )

        new_edges = tuple(
            functools.reduce(
                operator.add,
                (
                    self._split_edge_get_edge_group(name, edge, split_map)
                    for name, edge in zip(self.names, self.edges)
                ),
                [],
            )
        )

        split_tensor = self.gt.reshape(new_edges)

        return dataclasses.replace(
            self,
            _names=tuple(new_names),
            _arrow=split_tensor.arrow,
            _edges=split_tensor.edges,
            _tensor=split_tensor.tensor,
            _parity=None,
            _mask=None,
        )

    def matmul(self, other: NamedGrassmannTensor) -> NamedGrassmannTensor:
        tensor_a = self
        tensor_b = other

        vector_a = tensor_a.tensor.dim() == 1
        vector_b = tensor_b.tensor.dim() == 1

        names: list[str] = []
        for i in range(-max(max(tensor_a.tensor.dim(), 2), max(tensor_b.tensor.dim(), 2)), -2):
            candidate_a = candidate_b = 1
            name_a = name_b = None
            if i >= -tensor_a.tensor.dim():
                candidate_a, _ = tensor_a.edges[i]
                name_a = tensor_a.names[i]
            if i >= -tensor_b.tensor.dim():
                candidate_b, _ = tensor_b.edges[i]
                name_b = tensor_b.names[i]
            if candidate_a >= candidate_b:
                picked_name = name_a if name_a is not None else name_b
            else:
                picked_name = name_b if name_b is not None else name_a
            names.append(typing.cast(str, picked_name))

        if not vector_a:
            names.append(tensor_a.names[-2])
        if not vector_b:
            names.append(tensor_b.names[-1])

        tensor = tensor_a.gt @ tensor_b.gt

        return NamedGrassmannTensor(
            _names=tuple(names),
            _arrow=tensor.arrow,
            _edges=tensor.edges,
            _tensor=tensor.tensor,
            _parity=None,
            _mask=None,
        )

    def conjugate(self) -> NamedGrassmannTensor:
        tensor = self.gt.conj()
        return dataclasses.replace(
            self,
            _arrow=tensor.arrow,
            _edges=tensor.edges,
            _tensor=tensor.tensor,
            _parity=None,
            _mask=None,
        )

    def conj(self) -> NamedGrassmannTensor:
        return self.conjugate()

    def _get_left_right_indices(
        self, pairs: set[tuple[str, str]]
    ) -> tuple[tuple[str, ...], tuple[int, ...], tuple[int, ...]]:
        pairs_map = {set_0: set_1 for set_0, set_1 in pairs}
        left_set = set(pairs_map)
        right_set = set(pairs_map.values())

        are_disjoint = left_set.isdisjoint(right_set)
        is_complete_union = (left_set | right_set) == set(self.names)
        no_duplicates = len(left_set) + len(right_set) == len(self.names)

        assert are_disjoint and is_complete_union and no_duplicates, (
            f"Input pairs must cover all dimension and disjoint, but got {pairs_map}"
        )

        left_names = tuple(name for name in self.names if name in left_set)
        right_names = tuple(pairs_map[name] for name in left_names)

        names = left_names + right_names

        left_idx = tuple(self.get_name_index(name) for name in left_names)
        right_idx = tuple(self.get_name_index(name) for name in right_names)

        return names, left_idx, right_idx

    def exponential(self, pairs: set[tuple[str, str]]) -> NamedGrassmannTensor:
        names, left_idx, right_idx = self._get_left_right_indices(pairs)

        exp = self.gt.exponential((left_idx, right_idx), permute_back=False)

        return dataclasses.replace(
            self,
            _names=names,
            _arrow=exp.arrow,
            _edges=exp.edges,
            _tensor=exp.tensor,
        )

    def identity(self, pairs: set[tuple[str, str]]) -> NamedGrassmannTensor:
        names, left_idx, right_idx = self._get_left_right_indices(pairs)

        identity = self.gt.identity((left_idx, right_idx), permute_back=False)

        return dataclasses.replace(
            self,
            _names=names,
            _arrow=identity.arrow,
            _edges=identity.edges,
            _tensor=identity.tensor,
        )

    def svd(
        self,
        free_names_u: set[str],
        common_name_u: str,
        common_name_v: str,
        singular_name_u: str,
        singular_name_v: str,
        *,
        cutoff: int | None | tuple[int, int] = None,
    ) -> tuple[NamedGrassmannTensor, NamedGrassmannTensor, NamedGrassmannTensor]:
        free_names_u_indices = tuple(self.get_name_index(name) for name in free_names_u)
        u, s, vh = self.gt.svd(free_names_u_indices, cutoff=cutoff)

        left_names = tuple(self.names[i] for i in free_names_u_indices)
        right_names = tuple(
            name for i, name in enumerate(self.names) if i not in set(free_names_u_indices)
        )

        U = NamedGrassmannTensor(
            _names=left_names + (common_name_u,),
            _arrow=u.arrow,
            _edges=u.edges,
            _tensor=u.tensor,
        )
        S = NamedGrassmannTensor(
            _names=(singular_name_u, singular_name_v),
            _arrow=s.arrow,
            _edges=s.edges,
            _tensor=s.tensor,
        )
        Vh = NamedGrassmannTensor(
            _names=(common_name_v,) + right_names,
            _arrow=vh.arrow,
            _edges=vh.edges,
            _tensor=vh.tensor,
        )

        return U, S, Vh

    def contract(
        self,
        other: NamedGrassmannTensor,
        contract_pairs: set[tuple[str, str]],
    ) -> NamedGrassmannTensor:
        assert contract_pairs, "contract_pairs must be non-empty"

        for pair in contract_pairs:
            assert (
                isinstance(pair, tuple) and len(pair) == 2 and all(isinstance(x, str) for x in pair)
            ), f"Each contract pair must be (str, str), got: {pair!r}"

        names_a = [a for a, _ in contract_pairs]
        names_b = [b for _, b in contract_pairs]
        assert len(names_a) == len(set(names_a)), f"Duplicate names on A side: {names_a}"
        assert len(names_b) == len(set(names_b)), f"Duplicate names on B side: {names_b}"

        assert all(a_name in self.names for a_name in names_a), (
            "Some names of self side not in name list."
        )
        assert all(b_name in other.names for b_name in names_b), (
            "Some names of other side not in name list."
        )

        name_set_a = set(names_a)
        name_set_b = set(names_b)

        if self.tensor.numel() >= other.tensor.numel():
            dict_map = {a: b for a, b in contract_pairs}
            ordered_a = [name for name in self.names if name in name_set_a]
            ordered_b = [dict_map[a] for a in ordered_a]
        else:
            dict_map = {b: a for a, b in contract_pairs}
            ordered_b = [name for name in other.names if name in name_set_b]
            ordered_a = [dict_map[b] for b in ordered_b]

        assert all(
            self.edges[self.get_name_index(name_a)] == other.edges[other.get_name_index(name_b)]
            for name_a, name_b in zip(ordered_a, ordered_b)
        ), "Contract edges must be same."

        leg_a = tuple(self.get_name_index(name) for name in ordered_a)
        leg_b = tuple(other.get_name_index(name) for name in ordered_b)

        c = self.gt.contract(other.gt, leg_a, leg_b)

        contract_set_a = set(ordered_a)
        contract_set_b = set(ordered_b)
        names = tuple(name for name in self.names if name not in contract_set_a) + tuple(
            name for name in other.names if name not in contract_set_b
        )

        return NamedGrassmannTensor(
            _names=names,
            _arrow=c.arrow,
            _edges=c.edges,
            _tensor=c.tensor,
            _parity=None,
            _mask=None,
        )

    def reciprocal(self) -> NamedGrassmannTensor:
        return dataclasses.replace(
            self, _tensor=torch.where(self.tensor == 0, self.tensor, 1 / self.tensor)
        )

    def norm(self, p: typing.Any) -> float:
        return float(torch.linalg.vector_norm(self.tensor.reshape(-1), ord=p))

    def sqrt(self) -> NamedGrassmannTensor:
        return dataclasses.replace(self, _tensor=torch.sqrt(torch.abs(self.tensor)))

    def rank(self) -> int:
        return len(self.names)

    def allclose(
        self, other: NamedGrassmannTensor, rtol: float = 1e-05, atol: float = 1e-8
    ) -> bool:
        if not isinstance(other, NamedGrassmannTensor):
            raise TypeError(f"Expected NamedGrassmannTensor, got {type(other)}")
        if set(self._names) != set(other._names):
            raise TypeError(
                f"Expected same name, but got self: {set(self._names)}, other: {set(other._names)}"
            )

        if tuple(self._names) != tuple(other._names):
            other = other.permute(self._names)

        if self._arrow != other._arrow:
            raise TypeError(
                f"Expected same arrow, but got self: {self._arrow}, other: {other._arrow}"
            )

        if self._edges != other._edges:
            raise TypeError(
                f"Expected same edges, but got self: {self._edges}, other: {other._edges}"
            )

        tensor_a = self.update_mask()._tensor
        tensor_b = other.update_mask()._tensor

        tensor_b = tensor_b.to(tensor_a.device)

        return tensor_a.allclose(tensor_b, rtol=rtol, atol=atol)

    def _validate_edge_compatibility(self, other: NamedGrassmannTensor) -> None:
        assert self._names == other.names, (
            f"Names must match for arithmetic operations. Got {self._names} and {other.names}."
        )
        assert self._arrow == other.arrow, (
            f"Arrows must match for arithmetic operations. Got {self._arrow} and {other.arrow}."
        )
        assert self._edges == other.edges, (
            f"Edges must match for arithmetic operations. Got {self._edges} and {other.edges}."
        )

    def __pos__(self) -> NamedGrassmannTensor:
        return dataclasses.replace(
            self,
            _tensor=+self._tensor,
        )

    def __neg__(self) -> NamedGrassmannTensor:
        return dataclasses.replace(
            self,
            _tensor=-self._tensor,
        )

    def __add__(self, other: typing.Any) -> NamedGrassmannTensor:
        if isinstance(other, NamedGrassmannTensor):
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

    def __radd__(self, other: typing.Any) -> NamedGrassmannTensor:
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

    def __iadd__(self, other: typing.Any) -> NamedGrassmannTensor:
        if isinstance(other, NamedGrassmannTensor):
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

    def __sub__(self, other: typing.Any) -> NamedGrassmannTensor:
        if isinstance(other, NamedGrassmannTensor):
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

    def __rsub__(self, other: typing.Any) -> NamedGrassmannTensor:
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

    def __isub__(self, other: typing.Any) -> NamedGrassmannTensor:
        if isinstance(other, NamedGrassmannTensor):
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

    def __mul__(self, other: typing.Any) -> NamedGrassmannTensor:
        if isinstance(other, NamedGrassmannTensor):
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

    def __rmul__(self, other: typing.Any) -> NamedGrassmannTensor:
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

    def __imul__(self, other: typing.Any) -> NamedGrassmannTensor:
        if isinstance(other, NamedGrassmannTensor):
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

    def __truediv__(self, other: typing.Any) -> NamedGrassmannTensor:
        if isinstance(other, NamedGrassmannTensor):
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

    def __rtruediv__(self, other: typing.Any) -> NamedGrassmannTensor:
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

    def __itruediv__(self, other: typing.Any) -> NamedGrassmannTensor:
        if isinstance(other, NamedGrassmannTensor):
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

    def __matmul__(self, other: typing.Any) -> NamedGrassmannTensor:
        if isinstance(other, NamedGrassmannTensor):
            return self.matmul(other)
        return NotImplemented

    def __rmatmul__(self, other: typing.Any) -> NamedGrassmannTensor:
        return NotImplemented

    def __imatmul__(self, other: typing.Any) -> NamedGrassmannTensor:
        if isinstance(other, NamedGrassmannTensor):
            return self.matmul(other)
        return NotImplemented

    def clone(self) -> NamedGrassmannTensor:
        """
        Create a deep copy of the Grassmann tensor.
        """
        return dataclasses.replace(
            self,
            _tensor=self._tensor.clone(),
            _parity=tuple(parity.clone() for parity in self._parity)
            if self._parity is not None
            else None,
            _mask=self._mask.clone() if self._mask is not None else None,
        )

    def __copy__(self) -> NamedGrassmannTensor:
        return self.clone()

    def __deepcopy__(self, memo: dict) -> NamedGrassmannTensor:
        return self.clone()

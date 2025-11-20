"""
A Grassmann tensor class.
"""

from __future__ import annotations

__all__ = ["GrassmannTensor"]

import dataclasses
import functools
import typing
import math

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

    def reverse(self, indices: tuple[int, ...]) -> GrassmannTensor:
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
                if index in indices and self.arrow[index]
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
            tensor = tensor.reverse(arrow_reverse).reverse(arrow_reverse).reverse(arrow_reverse)

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

        assert (k_even > 0 or n_even == 0) and (k_odd > 0 or n_odd == 0), (
            "Per-block cutoff must be compatible with available singulars"
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
    def get_inv_order(dim: int, order: tuple[int, ...]) -> tuple[int, ...]:
        inv = [0] * dim
        for new_position, origin_idx in enumerate(order):
            inv[origin_idx] = new_position
        return tuple(inv)

    def _get_inv_order(self, order: tuple[int, ...]) -> tuple[int, ...]:
        return self.get_inv_order(self.tensor.dim(), order)

    def contract(
        self,
        b: GrassmannTensor,
        a_leg: int | tuple[int, ...],
        b_leg: int | tuple[int, ...],
    ) -> GrassmannTensor:
        a = self

        contract_lengths = []
        for leg, tensor in ((a_leg, a), (b_leg, b)):
            if isinstance(leg, int):
                contract_lengths.append(1)
            else:
                contract_lengths.append(len(leg))
                assert all(tensor.arrow[i] == tensor.arrow[leg[0]] for i in leg), (
                    "All the legs that need to be contracted must have the same arrow"
                )

        contract_length_a, contract_length_b = contract_lengths

        a_leg_tuple = (a_leg,) if isinstance(a_leg, int) else a_leg
        b_leg_tuple = (b_leg,) if isinstance(b_leg, int) else b_leg

        a_range_list = tuple(range(a.tensor.dim()))
        b_range_list = tuple(range(b.tensor.dim()))

        a_contract_set = set(a_leg_tuple)
        b_contract_set = set(b_leg_tuple)

        order_a = tuple(i for i in a_range_list if i not in a_contract_set) + a_leg_tuple
        order_b = b_leg_tuple + tuple(i for i in b_range_list if i not in b_contract_set)

        tensor_a = a.permute(order_a)
        tensor_b = b.permute(order_b)

        assert (tensor_a.arrow[-1], tensor_b.arrow[0]) in ((False, True), (True, False)), (
            f"Contract requires arrow (False, True) or (True, False), but got {tensor_a.arrow[-1], tensor_b.arrow[0]}"
        )

        arrow_after_permute_a = tensor_a.arrow
        arrow_after_permute_b = tensor_b.arrow

        edge_after_permute_a = tensor_a.edges
        edge_after_permute_b = tensor_b.edges

        arrow_expected_a = [i >= a.tensor.dim() - contract_length_a for i in range(a.tensor.dim())]
        arrow_expected_b = [i >= contract_length_b for i in range(b.tensor.dim())]

        arrow_reverse_a = tuple(
            i
            for i, (cur, exp) in enumerate(zip(arrow_after_permute_a, arrow_expected_a))
            if cur != exp
        )
        arrow_reverse_b = tuple(
            i
            for i, (cur, exp) in enumerate(zip(arrow_after_permute_b, arrow_expected_b))
            if cur != exp
        )

        if arrow_reverse_a:
            tensor_a = (
                tensor_a.reverse(arrow_reverse_a).reverse(arrow_reverse_a).reverse(arrow_reverse_a)
            )
        if arrow_reverse_b:
            tensor_b = (
                tensor_b.reverse(arrow_reverse_b).reverse(arrow_reverse_b).reverse(arrow_reverse_b)
            )

        tensor_a = tensor_a.reshape(
            (
                math.prod(tensor_a.tensor.shape[:-contract_length_a]),
                math.prod(tensor_a.tensor.shape[-contract_length_a:]),
            )
        )
        tensor_b = tensor_b.reshape(
            (
                math.prod(tensor_b.tensor.shape[:contract_length_b]),
                math.prod(tensor_b.tensor.shape[contract_length_b:]),
            )
        )

        c = tensor_a @ tensor_b

        c = c.reshape(
            (edge_after_permute_a[:-contract_length_a] + edge_after_permute_b[contract_length_b:])
        )

        arrow_reverse_c = tuple(
            [i for i in arrow_reverse_a if i < a.tensor.dim() - contract_length_a]
            + [
                (a.tensor.dim() - contract_length_a) + (i - contract_length_b)
                for i in arrow_reverse_b
                if i >= contract_length_b
            ]
        )
        c = c.reverse(arrow_reverse_c)
        return c

    def exponential(self, pairs: tuple[tuple[int, ...], tuple[int, ...]]) -> GrassmannTensor:
        tensor, left_legs, right_legs = self._group_edges(pairs)

        assert tensor.arrow in ((False, True), (True, False)), (
            f"Exponentiation requires arrow (False, True) or (True, False), but got {tensor.arrow}"
        )

        tensor_reverse_flag = tensor.arrow != (False, True)
        if tensor_reverse_flag:
            tensor = tensor.reverse((0, 1))

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

        inv_order = self._get_inv_order(order)

        tensor_exp = tensor_exp.permute(inv_order)

        return tensor_exp

    def identity(self, pairs: tuple[tuple[int, ...], tuple[int, ...]]) -> GrassmannTensor:
        tensor, left_legs, right_legs = self._group_edges(pairs)

        assert tensor.arrow in ((False, True), (True, False)), (
            f"Identity requires arrow (False, True) or (True, False), but got {tensor.arrow}"
        )

        tensor_reverse_flag = tensor.arrow != (False, True)
        if tensor_reverse_flag:
            tensor = tensor.reverse((0, 1))

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

        inv_order = self._get_inv_order(order)

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

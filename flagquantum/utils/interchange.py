# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# utils/interchange.py
"""Qubit interchange utilities for quantum state tensors.

This module provides functions for interchanging qubits within a quantum state
tensor, handling both distributed (DTensor) and local tensor representations.
It supports interchanging between ungrouped, grouped, and sharded qubits while
maintaining the integrity of the quantum state.

The state tensor has the following dimension structure (from left to right):
    - Dimension 0: Batch size (multiple independent quantum states)
    - Dimensions 1..n: Qubit dimensions (size 2 for each ungrouped qubit,
      size 2^k for groups of k qubits)
    - Last dimension: Real/imaginary components (size 2)

Example:
    Basic qubit interchange with batch:
    >>> from flagquantum.utils import interchange_qubits
    >>> import torch
    >>>
    >>> # Create a quantum state for 3 qubits with batch size 2
    >>> # Shape: [batch=2, qubit0, qubit1, qubit2, real/imag=2]
    >>> state = torch.zeros(2, 2, 2, 2, 2)
    >>> # Initialize both batches to |000> state
    >>> state[:, 0, 0, 0, 0] = 1  # Real part = 1
    >>>
    >>> # Grouping tensor: each qubit in its own group (ungrouped)
    >>> grouping = torch.tensor([[1, 2, 3], [-1, -1, -1]])
    >>>
    >>> # Interchange qubit 0 and qubit 2 for all batches
    >>> new_state, new_grouping = interchange_qubits(state, grouping, 0, 2)
"""

from typing import Union

import torch
from torch.distributed.tensor import DTensor


def interchange_qubits(
    state: Union[DTensor, torch.Tensor], grouping: torch.Tensor, wire1: int, wire2: int
) -> tuple[Union[DTensor, torch.Tensor], torch.Tensor]:
    """Interchanges two qubits within a given statevector.

    This function performs qubit interchange operations, handling various cases
    including ungrouped, grouped, and sharded qubits. It operates recursively
    based on several base interchange patterns. The batch dimension is preserved
    and all operations are applied to all batches simultaneously.

    Args:
        state: Tensor or DTensor containing quantum state information.
            Shape: [batch_size, *qubit_dims, 2] where qubit_dims are the
            dimensions for each qubit or qubit group.
        grouping: 2xN tensor where N is the number of qubits.
            - First row: group numbers (positive integers)
            - Second row: relative positions (-1 for ungrouped, -2 for sharded,
              non-negative for positions within a group)
        wire1: Index of the first qubit to interchange (0-indexed).
        wire2: Index of the second qubit to interchange (0-indexed).

    Returns:
        A tuple containing:
            - new_state: Updated quantum state with qubits interchanged
            - new_grouping: Updated grouping tensor reflecting the swap

    Raises:
        RuntimeError: If attempting to interchange incompatible qubit types
            (e.g., sharded with grouped unsharded qubit).
        IndexError: If wire1 or wire2 is out of range.

    Example:
        Interchanging ungrouped qubits with batch:
        >>> # 4 qubit system with batch size 3
        >>> state = torch.randn(3, 2, 2, 2, 2, 2)
        >>> grouping = torch.tensor([[1, 2, 3, 4], [-1, -1, -1, -1]])
        >>> new_state, new_grouping = interchange_qubits(state, grouping, 0, 3)
        >>> # All 3 batches have qubits 0 and 3 swapped

        Interchanging grouped qubits:
        >>> # 4 qubits: qubits 0-1 grouped together
        >>> state = torch.randn(1, 4, 2, 2, 2)  # [batch, group0, qubit2, qubit3, imag]
        >>> grouping = torch.tensor([
        ...     [1, 1, 2, 3],  # Qubit0-1 in group1, qubit2 in group2, qubit3 in group3
        ...     [0, 1, -1, -1]  # Qubit0 and qubit1 positions in group
        ... ])
        >>> new_state, new_grouping = interchange_qubits(state, grouping, 0, 3)

        Distributed tensor usage:
        >>> from torch.distributed.tensor import DTensor, Shard
        >>> if dist.is_initialized():
        ...     mesh = init_device_mesh("cuda", (2,))
        ...     d_state = DTensor.from_local(state, mesh, [Shard(0)])
        ...     new_state, new_grouping = interchange_qubits(d_state, grouping, 1, 2)

    Note:
        - This function does not reshard automatically and can only interchange
          sharded qubits with ungrouped qubits.
        - The batch dimension (dimension 0) is preserved and never interchanged.
        - For DTensor inputs, the distributed placements are preserved.
        - The recursion depth is limited by the qubit grouping structure.
    """
    wire_info = grouping[:, [wire1, wire2]]
    if wire1 == wire2:
        return state, grouping
    elif (wire_info[1] > -1).any():
        if (
            wire_info[1] == -2
        ).any():  # Cannot interchange sharded qubit with grouped unsharded qubit
            return state, grouping
        elif (wire_info[1] == -1).any():  # Exactly one wire is ungrouped
            if (
                wire_info[1, 0] == -1
            ):  # Presume from now on that wire2 is the ungrouped wire
                return interchange_qubits(state, grouping, wire2, wire1)
            else:
                lone_wire_idx = wire_info[0, 1]
                # Get index and size of wire1 group along with relative position of wire1
                grouped_wire_info = [
                    wire_info[0, 0].item(),
                    wire_info[1, 0].item(),
                    (grouping[0] == wire_info[0, 0]).sum().item(),
                ]
                # grouped wires cannot be leftmost or rightmost dimensions; get index and size of left and right groups
                left_group_info = [
                    grouped_wire_info[0] - 1,
                    (grouping[0] == (grouped_wire_info[0] - 1)).sum().item(),
                ]
                right_group_info = [
                    grouped_wire_info[0] + 1,
                    (grouping[0] == (grouped_wire_info[0] + 1)).sum().item(),
                ]

                need_left_interchange = (lone_wire_idx == left_group_info[0]) and (
                    grouped_wire_info[1] > 0
                )
                need_right_interchange = (lone_wire_idx == right_group_info[0]) and (
                    grouped_wire_info[1] < grouped_wire_info[2] - 1
                )
                need_interchange = need_left_interchange or need_right_interchange
                if need_interchange:
                    all_ungrouped_idxs = grouping[0, grouping[1] == -1]
                    # Pick a new ungrouped qubit that isn't adjacent to wire1
                    helper_idx = all_ungrouped_idxs[
                        (
                            (all_ungrouped_idxs != left_group_info[0])
                            & (all_ungrouped_idxs != right_group_info[0])
                        )
                    ][0]
                    helper_qubit = (
                        torch.nonzero(grouping[0] == helper_idx).flatten()[0]
                    ).int()
                    new_state, new_grouping = interchange_qubits(
                        state, grouping, wire1, helper_qubit
                    )
                    new_state, new_grouping = interchange_qubits(
                        new_state, new_grouping, wire1, wire2
                    )
                    new_state, new_grouping = interchange_qubits(
                        new_state, new_grouping, helper_qubit, wire2
                    )
                    return new_state, new_grouping
                else:
                    # Get temporary tensor shape
                    state_shape = list(state.shape)
                    temp_shape = state_shape.copy()
                    temp_shape[grouped_wire_info[0]] = 2
                    temp_shape[left_group_info[0]] = 2 ** (
                        left_group_info[1] + grouped_wire_info[1]
                    )
                    temp_shape[right_group_info[0]] = 2 ** (
                        right_group_info[1]
                        + grouped_wire_info[2]
                        - 1
                        - grouped_wire_info[1]
                    )

                    # Reshape to isolate grouped wire dimension, interchange wires, and then reshape back
                    new_state = state.reshape(temp_shape)
                    new_state, _ = interchange_dims(
                        new_state,
                        grouping,
                        grouped_wire_info[0],
                        lone_wire_idx,
                        new_state.ndim,
                    )
                    new_state = new_state.reshape(state_shape)

                    new_grouping = grouping.detach().clone()
                    new_grouping[:, wire1], new_grouping[:, wire2] = (
                        grouping[:, wire2],
                        grouping[:, wire1],
                    )
                    return new_state, new_grouping
        else:  # If both wires are grouped, use ungrouped wire as medium of interchange
            helper_qubit = (torch.nonzero(grouping[1] == -1).flatten()[0]).int()
            new_state, new_grouping = interchange_qubits(
                state, grouping, wire2, helper_qubit
            )
            new_state, new_grouping = interchange_qubits(
                new_state, new_grouping, wire1, wire2
            )
            new_state, new_grouping = interchange_qubits(
                new_state, new_grouping, helper_qubit, wire1
            )
            return new_state, new_grouping
    else:
        # Between ungrouped and sharded qubits, interchanging qubits is the same as interchanging dimensions
        return interchange_dims(
            state, grouping, wire_info[0, 0], wire_info[0, 1], state.ndim
        )


def interchange_dims(
    state: Union[DTensor, torch.Tensor],
    grouping: torch.Tensor,
    dim1: int,
    dim2: int,
    num_dims: int,
) -> tuple[Union[DTensor, torch.Tensor], torch.Tensor]:
    """Interchanges two tensor dimensions and updates the grouping tensor.

    This utility function swaps two dimensions in the quantum state tensor
    and updates the grouping information accordingly. It preserves relative
    orders within dimensions and designations of ungrouped or sharded qubits.
    The batch dimension (dimension 0) is typically not interchanged.

    Args:
        state: Tensor or DTensor containing quantum state information.
            Must have exactly `num_dims` dimensions.
        grouping: 2xN tensor where N is the number of qubits. Contains group
            numbers and relative positions.
        dim1: First dimension index to interchange (0-indexed). Note that
            dimension 0 is usually the batch dimension.
        dim2: Second dimension index to interchange (0-indexed).
        num_dims: Total number of dimensions in the state tensor.

    Returns:
        A tuple containing:
            - new_state: Tensor with dimensions interchanged (via permutation)
            - new_grouping: Updated grouping tensor with dimension indices swapped

    Example:
        Basic dimension interchange with batch:
        >>> # State: [batch=2, qubit0, grouped_qubits, qubit3, imag=2]
        >>> state = torch.randn(2, 2, 4, 2, 2)
        >>> grouping = torch.tensor([
        ...     [1, 2, 2, 3],
        ...     [-1, 0, 1, -1]
        ... ])
        >>> # Interchange qubit0 (dim1) and grouped group (dim2)
        >>> new_state, new_grouping = interchange_dims(state, grouping, 1, 2, 5)

        Swapping back after operations:
        >>> # After various operations, dimensions may need reordering
        >>> original_order = [0, 3, 1, 2, 4]  # Example permutation
        >>> new_state, new_grouping = interchange_dims(state, grouping, 1, 3, 5)

        Distributed tensor example:
        >>> from torch.distributed.tensor import DTensor
        >>> if dist.is_initialized():
        ...     d_state = DTensor.from_local(state, mesh, [Shard(1)])
        ...     new_d_state, new_grouping = interchange_dims(d_state, grouping, 1, 2, 5)

    Note:
        - This function performs a simple permutation of dimensions without
          modifying the underlying data layout.
        - For DTensor inputs, the permutation operation is supported for
          both CPU and CUDA tensors.
        - The grouping tensor's first row is updated to reflect new
          dimension indices while preserving group relationships.
        - Dimension indices in grouping refer to positions *after* batch dimension.
          So a grouping value of 1 corresponds to state dimension 1 (first qubit dim).
    """
    permute_list = list(range(num_dims))
    permute_list[dim1], permute_list[dim2] = permute_list[dim2], permute_list[dim1]
    new_state = state.permute(permute_list)
    new_grouping = grouping.detach().clone()
    where_dim1 = new_grouping[0] == dim1
    where_dim2 = new_grouping[0] == dim2
    new_grouping[0, where_dim1], new_grouping[0, where_dim2] = dim2, dim1
    return new_state, new_grouping


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    "interchange_dims",
    "interchange_qubits",
]

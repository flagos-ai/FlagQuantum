# utils/maybe_dtensor.py
"""Utilities for working with distributed tensors (DTensor) and local tensors.

This module provides helper functions for safely handling both local tensors
(torch.Tensor) and distributed tensors (DTensor). All functions are designed
to work transparently with both types, falling back to identity operations
when working with local tensors.

The main use case is in quantum device implementations that need to support
both single-GPU and distributed multi-GPU execution without code duplication.

Example:
    Basic usage with local tensor:
    >>> import torch
    >>> from flagquantum.utils import is_dtensor, maybe_to_local
    >>>
    >>> local_tensor = torch.randn(4, 4)
    >>> is_dtensor(local_tensor)
    False
    >>> maybe_to_local(local_tensor)  # Returns same tensor
    tensor([...])

    Usage with distributed tensor:
    >>> from torch.distributed.tensor import DTensor, Shard
    >>> from torch.distributed.device_mesh import init_device_mesh
    >>>
    >>> mesh = init_device_mesh("cuda", (2,))
    >>> d_tensor = DTensor.from_local(local_tensor, mesh, [Shard(0)])
    >>> is_dtensor(d_tensor)
    True
    >>> local_copy = maybe_to_local(d_tensor)  # Gets local shard
"""

from typing import Optional, Union

import torch
from torch.distributed.tensor import DeviceMesh, DTensor, Shard, distribute_tensor

# ============================================================================
# Type Checking
# ============================================================================


def is_dtensor(tensor: Union[torch.Tensor, DTensor]) -> bool:
    """Check if a tensor is a distributed tensor (DTensor).

    This function safely checks whether the input tensor is a distributed
    tensor instance.

    Args:
        tensor: Input tensor (either local torch.Tensor or distributed DTensor).

    Returns:
        True if the tensor is a DTensor, False otherwise.

    Example:
        >>> import torch
        >>> from flagquantum.utils import is_dtensor
        >>>
        >>> local_tensor = torch.randn(3, 3)
        >>> is_dtensor(local_tensor)
        False
        >>>
        >>> from torch.distributed.tensor import DTensor
        >>> d_tensor = DTensor.from_local(local_tensor, mesh, [Shard(0)])
        >>> is_dtensor(d_tensor)
        True

    Note:
        This function is useful for writing code that needs to handle both
        local and distributed tensors differently.
    """
    return isinstance(tensor, DTensor)


# ============================================================================
# Conversions
# ============================================================================


def maybe_to_local(
    tensor: Union[torch.Tensor, DTensor],
) -> Union[torch.Tensor, DTensor]:
    """Convert a DTensor to its local shard, or return the tensor unchanged.

    If the input is a DTensor, this function returns the local portion of
    the distributed tensor. If the input is a local tensor, it returns the
    tensor unchanged.

    Args:
        tensor: Input tensor (local or distributed).

    Returns:
        - If DTensor: The local shard as a torch.Tensor
        - If torch.Tensor: The original tensor unchanged

    Example:
        Local tensor (no conversion):
        >>> local_tensor = torch.randn(4, 4)
        >>> result = maybe_to_local(local_tensor)
        >>> result is local_tensor
        True

        Distributed tensor (extract local shard):
        >>> from torch.distributed.device_mesh import init_device_mesh
        >>> mesh = init_device_mesh("cuda", (2,))
        >>> d_tensor = DTensor.from_local(local_tensor, mesh, [Shard(0)])
        >>> local_shard = maybe_to_local(d_tensor)
        >>> local_shard.shape
        torch.Size([2, 4])  # Half of original on each rank

    Note:
        This function is useful when you need to perform operations that are
        only supported on local tensors.
    """
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def maybe_full_tensor(tensor: Union[torch.Tensor, DTensor]) -> torch.Tensor:
    """Gather the full tensor from all distributed processes.

    For distributed tensors, this function gathers the shards from all
    processes to reconstruct the full tensor. For local tensors, it returns
    the tensor unchanged.

    Args:
        tensor: Input tensor (local or distributed).

    Returns:
        - If DTensor: The full gathered tensor as a torch.Tensor
        - If torch.Tensor: The original tensor unchanged

    Warning:
        This function should be used with caution as it can cause memory issues
        when working with large tensors, since the full tensor is replicated
        on all processes.

    Example:
        >>> # With distributed tensor
        >>> d_tensor = DTensor.from_local(local_tensor, mesh, [Shard(0)])
        >>> full_tensor = maybe_full_tensor(d_tensor)
        >>> full_tensor.shape == local_tensor.shape
        True
        >>>
        >>> # With local tensor (no change)
        >>> local_tensor = torch.randn(4, 4)
        >>> result = maybe_full_tensor(local_tensor)
        >>> result is local_tensor
        True

    Note:
        This function is typically used for debugging, logging, or when
        processing the final output of a quantum circuit.
    """
    return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor


# ============================================================================
# Info Extraction
# ============================================================================


def maybe_get_dtensor_info(
    tensor: Union[torch.Tensor, DTensor],
) -> tuple[Optional[DeviceMesh], tuple[Shard, ...]]:
    """Extract distribution information from a tensor.

    For distributed tensors, returns the device mesh and placements.
    For local tensors, returns (None, empty tuple).

    Args:
        tensor: Input tensor (local or distributed).

    Returns:
        A tuple containing:
            - device_mesh: The DeviceMesh of the DTensor, or None if local
            - placements: A tuple of Shard placements, or empty tuple if local

    Example:
        Extract info from distributed tensor:
        >>> d_tensor = DTensor.from_local(tensor, mesh, [Shard(0), Shard(1)])
        >>> mesh, placements = maybe_get_dtensor_info(d_tensor)
        >>> print(mesh)
        DeviceMesh(...)
        >>> print(placements)
        (Shard(dim=0), Shard(dim=1))

        Info from local tensor:
        >>> local_tensor = torch.randn(4, 4)
        >>> mesh, placements = maybe_get_dtensor_info(local_tensor)
        >>> mesh is None
        True
        >>> placements
        ()

    Note:
        This function is useful when you need to preserve distribution
        information while performing operations that temporarily convert
        to local tensors.
    """
    if isinstance(tensor, DTensor):
        return tensor.device_mesh, tuple(tensor.placements)
    return None, ()


# ============================================================================
# Creation / Distribution
# ============================================================================


def maybe_from_local(
    tensor: Union[torch.Tensor, DTensor],
    device_mesh: Optional[DeviceMesh] = None,
    placements: Optional[tuple[Shard, ...]] = None,
) -> Union[torch.Tensor, DTensor]:
    """Convert a local tensor to a DTensor if distribution info is provided.

    This function creates a DTensor from a local tensor only when both
    device_mesh and placements are provided. Otherwise, returns the tensor
    unchanged.

    Args:
        tensor: Local tensor to potentially distribute.
        device_mesh: DeviceMesh for the distributed tensor. If None, tensor
            is returned unchanged.
        placements: Tuple of Shard placements. If empty or None, tensor is
            returned unchanged.

    Returns:
        - If device_mesh and placements provided: DTensor
        - Otherwise: Original local tensor

    Example:
        Convert local tensor to distributed:
        >>> from torch.distributed.device_mesh import init_device_mesh
        >>> mesh = init_device_mesh("cuda", (2,))
        >>> local_tensor = torch.randn(4, 4)
        >>> d_tensor = maybe_from_local(local_tensor, mesh, [Shard(0)])
        >>> isinstance(d_tensor, DTensor)
        True

        Keep as local tensor (no distribution info):
        >>> result = maybe_from_local(local_tensor)  # No mesh or placements
        >>> result is local_tensor
        True

        Invalid inputs (returns original):
        >>> result = maybe_from_local(local_tensor, mesh)  # Missing placements
        >>> result is local_tensor
        True

    Note:
        This function is the inverse of maybe_to_local() when distribution
        information is available.
    """
    placements = placements or ()
    return (
        DTensor.from_local(tensor, device_mesh=device_mesh, placements=placements)
        if device_mesh and placements
        else tensor
    )


def maybe_distribute_tensor(
    tensor: Union[torch.Tensor, DTensor],
    device_mesh: Optional[DeviceMesh] = None,
    placements: Optional[tuple[Shard, ...]] = None,
) -> Union[torch.Tensor, DTensor]:
    """Distribute a tensor using the given device mesh and placements.

    Similar to maybe_from_local, but uses `distribute_tensor` which can handle
    already distributed tensors gracefully. If the tensor is already a DTensor
    and the distribution matches, it may return the same tensor.

    Args:
        tensor: Input tensor (local or already distributed).
        device_mesh: DeviceMesh for distribution. If None, tensor is returned
            unchanged.
        placements: Tuple of Shard placements. If empty or None, tensor is
            returned unchanged.

    Returns:
        - If device_mesh and placements provided: Distributed DTensor
        - Otherwise: Original tensor unchanged

    Example:
        Distribute a local tensor:
        >>> mesh = init_device_mesh("cuda", (2,))
        >>> local_tensor = torch.randn(8, 8)
        >>> d_tensor = maybe_distribute_tensor(local_tensor, mesh, [Shard(0)])
        >>> d_tensor.shape
        torch.Size([8, 8])
        >>> d_tensor.placements
        (Shard(dim=0),)

        Redistribute a tensor (if needed):
        >>> # Already distributed with different sharding
        >>> d_tensor = DTensor.from_local(local_tensor, mesh, [Shard(0)])
        >>> # Redistribute with different placements
        >>> redistributed = maybe_distribute_tensor(d_tensor, mesh, [Shard(1)])
        >>> redistributed.placements
        (Shard(dim=1),)

        No distribution (returns original):
        >>> result = maybe_distribute_tensor(local_tensor)  # No mesh
        >>> result is local_tensor
        True

    Note:
        This function is preferred over maybe_from_local when working with
        tensors that might already be distributed, as it uses the more
        flexible `distribute_tensor` function.
    """
    placements = placements or ()
    return (
        distribute_tensor(tensor, device_mesh=device_mesh, placements=placements)
        if device_mesh and placements
        else tensor
    )


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    # Type checking
    "is_dtensor",
    # Conversions
    "maybe_to_local",
    "maybe_full_tensor",
    # Info extraction
    "maybe_get_dtensor_info",
    # Creation
    "maybe_from_local",
    "maybe_distribute_tensor",
]

# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 _ControlledBase._tensor_data.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/_infra/controlled.py

"""Insert a child tensor into the active block of a controlled identity."""

from collections.abc import Callable, Mapping, Sequence

import numpy as np
import torch
from numpy.typing import NDArray

from ...core.ir.program.signatures import Signature
from ...core.value_parameters import is_symbolic_value
from ..value_dimensions import ArrayValue
from .models import TensorNetworkNode
from .register_axis_groups import register_tensor_label_groups
from .register_tensor import register_identity_tensor, register_tensor_shape
from .register_tensor_binding import bind_register_tensor
from .register_tensor_data import register_active_indices


def controlled_register_tensor(
    controls: Signature,
    child_signature: Signature,
    patterns: Sequence[ArrayValue],
    child_tensor: Callable[[], NDArray[np.complex128]],
) -> NDArray[np.complex128]:
    """Build the pinned source's finite THRU-only dense controlled tensor.

    Control declarations precede child declarations; output axes precede input
    axes, with one axis per row-major word element. The child callback is called
    once, after identity allocation and before symbolic-pattern rejection.
    Raw integer pattern values index both control sides as in the source,
    including NumPy negative indexing. No signed-value codec is applied.

    This is an explicit exponential-memory inspection path, not a decomposition
    or a gate/resource estimator. Inputs remain unchanged. Invalid layout,
    symbolic patterns and incompatible child tensors raise ValueError; raw
    out-of-range indices retain NumPy's IndexError.
    """
    if not controls.thru_registers_only or not child_signature.thru_registers_only:
        raise ValueError("controlled dense tensors require THRU registers")
    if len(controls) != len(patterns):
        raise ValueError("one activation pattern is required per control register")
    signature = Signature((*controls, *child_signature))
    data = register_identity_tensor(signature)
    child_shape = register_tensor_shape(child_signature)
    child = child_tensor()
    if child_shape:
        child = child.reshape(child_shape)
    if is_symbolic_value(*patterns):
        raise ValueError("cannot compute active tensor space for symbolic patterns")
    for register, pattern in zip(controls, patterns, strict=True):
        if not isinstance(pattern, np.ndarray) or pattern.shape != register.shape:
            raise ValueError("activation pattern shape must match its control register")
    indices = register_active_indices(signature, patterns)
    if not isinstance(data, np.ndarray):
        # The pinned scalar identity is a NumPy scalar, which cannot be assigned.
        raise TypeError("scalar controlled identity does not support item assignment")
    data[indices] = child
    return data


def controlled_register_tensor_node(
    controls: Signature,
    child_signature: Signature,
    patterns: Sequence[ArrayValue],
    child_tensor: Callable[[], NDArray[np.complex128]],
    incoming: Mapping[str, tuple[tuple[int, ...], ...]],
    outgoing: Mapping[str, tuple[tuple[int, ...], ...]],
    like: torch.Tensor,
) -> TensorNetworkNode:
    """Attach the existing controlled tensor in source output/input bit order.

    Quantum or classical THRU controls retain their explicit activation patterns.
    The caller supplies the phaseful child; no classical-action approximation or
    function-name dispatch occurs. Reuse controlled_register_tensor's source
    callback order, active-block semantics and refusal boundaries. Bindings and
    distinct external integer labels are checked before invoking the child.

    Dtype/device follow like. Real prototypes reject nonzero imaginary data
    instead of dropping phase; NumPy child callbacks themselves are not trainable.
    Subsequent torch contractions retain input-state gradients. Dense storage
    grows exponentially as in the existing supplier. No measurement, dephasing,
    physical control wire, reset, resource model or instruction is emitted.
    """
    signature = Signature((*controls, *child_signature))
    groups = register_tensor_label_groups(signature, incoming, outgoing)
    labels = tuple(label for group in groups for label in group)
    if len(set(labels)) != len(labels):
        raise ValueError("controlled tensor external labels must be distinct")
    data = controlled_register_tensor(controls, child_signature, patterns, child_tensor)
    if not like.is_complex() and np.any(data.imag != 0):
        raise ValueError("complex controlled tensor requires a complex prototype")
    return bind_register_tensor(
        torch.as_tensor(
            data if like.is_complex() else data.real,
            dtype=like.dtype,
            device=like.device,
        ),
        signature,
        incoming,
        outgoing,
        name="controlled_register",
    )

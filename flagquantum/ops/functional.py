# ops/functional.py
"""Quantum gate operations for statevector simulation."""

import functools
import logging
from typing import List, Optional, Union

import numpy as np
import torch
from torch.autograd import Function, backward
from torch.distributed.tensor import DTensor

from ..utils.interchange import interchange_dims, interchange_qubits
from ..utils.maybe_dtensor import (
    maybe_from_local,
    maybe_get_dtensor_info,
    maybe_to_local,
)
from .matrices import GATE_MAT_DICT

logger = logging.getLogger(__name__)

# Constants
GROUP_UNSHARDED = -1
EPS = 1e-8


# _ensure_tensor 保持简单
def _ensure_tensor(params):
    if params is None:
        return None
    if isinstance(params, torch.Tensor):
        return params
    return torch.tensor(params, requires_grad=True)  # 统一转换


class InvertibleUnitaryBMM(Function):
    """
    Implements an unitary batched matrix multiply as an invertible computation to save activation memory
    """

    @staticmethod
    def forward(ctx, matrix, state, dummy):
        # `dummy` allows passing activations thru backwards pass (in place of actual gradients)
        ctx.save_for_backward(matrix)
        return matrix.bmm(state), dummy

    @staticmethod
    def backward(ctx, go, output):
        # Recompute input from output
        (matrix,) = ctx.saved_tensors
        inp = matrix.mH.bmm(output)
        del output

        inp = inp.detach().requires_grad_()
        mtx = matrix.detach().requires_grad_()
        with torch.enable_grad():
            out = mtx.bmm(inp)
        backward(out, go)
        return mtx.grad, inp.grad, inp


class InvertiblePostUnitaryStep(Function):
    """
    No-op in forwards, but starts to pass the output back for the invertible computation
    """

    @staticmethod
    def forward(ctx, inp, dummy):
        # `dummy` allows passing activations thru backwards pass (in place of actual gradients)
        out = inp  # no-op, but easier to follow logic
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, go):
        # Hijacking `dummy.grad` for activations here
        (out,) = ctx.saved_tensors
        return go, out


def _ensure_wires_list(wires: Union[int, List[int]]) -> List[int]:
    """Convert single wire to list for consistent handling."""
    return [wires] if isinstance(wires, int) else wires


def _ungroup_wires(
    state: Union[DTensor, torch.Tensor],
    groupings: torch.Tensor,
    wires: List[int],
    invertible_dummy: Optional[Union[DTensor, torch.Tensor]],
) -> tuple:
    """Move wires to ungrouped positions if they are grouped."""
    eligible_ungroups = list(
        set(torch.nonzero(groupings[1] == GROUP_UNSHARDED).flatten().tolist())
        - set(wires)
    )

    for wire in wires:
        if groupings[1, wire] > GROUP_UNSHARDED:
            if not eligible_ungroups:
                raise RuntimeError(
                    f"No eligible ungrouped wires to swap with wire {wire}"
                )
            ungroup = eligible_ungroups.pop(0)

            if invertible_dummy is not None:
                invertible_dummy, _ = interchange_qubits(
                    invertible_dummy, groupings, wire, ungroup
                )
            state, groupings = interchange_qubits(state, groupings, wire, ungroup)

    return state, groupings, invertible_dummy


def _reorder_wire_dims(
    state: Union[DTensor, torch.Tensor],
    groupings: torch.Tensor,
    wires: List[int],
    target_dims: List[int],
    invertible_dummy: Optional[Union[DTensor, torch.Tensor]],
    num_dims: int,
) -> tuple:
    """Reorder wire dimensions to consecutive positions."""
    for i, wire in enumerate(wires):
        current_dim = groupings[0, wire].item()
        target_dim = i + 1
        if current_dim != target_dim:
            if invertible_dummy is not None:
                invertible_dummy, _ = interchange_dims(
                    invertible_dummy, groupings, current_dim, target_dim, num_dims
                )
            state, groupings = interchange_dims(
                state, groupings, current_dim, target_dim, num_dims
            )
    return state, groupings, invertible_dummy


def _prepare_matrix(
    mat: torch.Tensor,
    bsz: int,
    device: torch.device,
) -> torch.Tensor:
    """Prepare matrix for batch multiplication."""
    if len(mat.shape) == 2:
        # No batch dimension, expand to batch
        expand_shape = [bsz] + list(mat.shape)
        mat = mat.expand(expand_shape)
    return mat.to(device)


def _process_params(
    params: Optional[Union[torch.Tensor, float, List[float]]],
    bsz: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Process parameters into proper tensor format."""
    if params is None:
        return None

    if not isinstance(params, torch.Tensor):
        params = torch.tensor(params, dtype=torch.float32)

    params = params.to(device)

    if params.dim() == 0:
        # Scalar parameter
        params = params.expand(bsz, 1)
    elif params.dim() == 1:
        # 1D parameter
        params = params.unsqueeze(-1).expand(bsz, -1)

    return params


def apply_unitary_bmm(
    state: Union[DTensor, torch.Tensor],
    mat: torch.Tensor,
    wires: Union[int, List[int]],
    groupings: torch.Tensor,
    invertible_dummy: Optional[Union[DTensor, torch.Tensor]] = None,
) -> tuple:
    """
    Apply the unitary to the statevector using local batch matrix multiply.
    Note: Assumes that none of the sharding dimensions are affected by wires.

    Args:
        state: The batched statevectors as a real DTensor. in last dim, 0th index is real part, 1st index is imaginary part
        mat: The batched unitary matrix of the operation as a complex Tensor.
        wires: Which qubit(s) the operation is applied to.
        groupings: a grouping tensor indicating the positions of all wires inside state
        invertible_dummy: use invertible computation to save memory? if yes, needs to be a dummy tensor to store gradients. if no, use None

    Returns:
        torch.Tensor: The new batch of statevectors.
    """
    mat = mat.to(state.device)
    wires = _ensure_wires_list(wires)

    # First ensure wires are ungrouped
    eligible_ungroups = list(
        set(torch.nonzero(groupings[1] == GROUP_UNSHARDED).flatten().tolist())
        - set(wires)
    )
    for wire in wires:
        if groupings[1, wire] > GROUP_UNSHARDED:
            ungroup = eligible_ungroups.pop(0)
            if invertible_dummy is not None:
                invertible_dummy, _ = interchange_qubits(
                    invertible_dummy, groupings, wire, ungroup
                )
            state, groupings = interchange_qubits(state, groupings, wire, ungroup)

    num_dims = state.ndim
    wire_dims = groupings[0, wires].tolist()
    # Move wire dimensions into first qubit dimensions
    for i in range(len(wires)):
        current_dim = groupings[0, wires[i]].item()
        if current_dim != i + 1:
            if invertible_dummy is not None:
                invertible_dummy, _ = interchange_dims(
                    invertible_dummy, groupings, current_dim, i + 1, num_dims
                )
            state, groupings = interchange_dims(
                state, groupings, current_dim, i + 1, num_dims
            )

    local_shape = maybe_to_local(state).shape
    bsz = local_shape[0]
    dtensor_mesh, dtensor_placements = maybe_get_dtensor_info(state)
    if invertible_dummy is not None:
        invertible_dummy = torch.view_as_complex(
            maybe_to_local(invertible_dummy).contiguous()
        ).reshape([bsz, 2 ** len(wires), -1])
    permuted = torch.view_as_complex(maybe_to_local(state)).reshape(
        [bsz, 2 ** len(wires), -1]
    )

    # permuted (b, m, k)
    # mat ([b,] n, m)
    if len(mat.shape) == 2:
        # matrix no batch, state in batch mode
        expand_shape = [bsz] + list(mat.shape)
        mat = mat.expand(expand_shape)
    # both matrix and state are in batch mode
    if invertible_dummy is not None:
        new_state, invertible_dummy = InvertibleUnitaryBMM.apply(
            mat, permuted, invertible_dummy
        )
    else:
        new_state = mat.bmm(permuted)

    new_state, invertible_dummy = [
        (
            x
            if x is None
            else maybe_from_local(
                torch.view_as_real(x).reshape(local_shape),
                device_mesh=dtensor_mesh,
                placements=dtensor_placements,
            )
        )
        for x in (new_state, invertible_dummy)
    ]

    new_grouping = groupings
    # Rearrange dims back in place
    for i in range(len(wires)):
        current_dim = groupings[0, wires[i]].item()
        if current_dim != wire_dims[i]:
            if invertible_dummy is not None:
                invertible_dummy, _ = interchange_dims(
                    invertible_dummy, groupings, wire_dims[i], current_dim, num_dims
                )
            new_state, new_grouping = interchange_dims(
                new_state, new_grouping, wire_dims[i], current_dim, num_dims
            )

    return new_state, new_grouping, invertible_dummy


def gate(
    name_or_mat: Union[str, torch.Tensor],
    q_device,
    wires: Union[int, List[int]],
    params: Optional[Union[torch.Tensor, float, List[float]]] = None,
    inverse: bool = False,
) -> None:
    """
    Gate operation accepts either a known gate name and retrieves the corresponding matrix,
    or directly accepts an unnamed gate matrix.
    Automatically checks unnamed gate matrices for sizing but does NOT check for unitarity.
    """
    params = _ensure_tensor(params)

    if isinstance(name_or_mat, str):
        mat = GATE_MAT_DICT[name_or_mat]
    elif isinstance(name_or_mat, torch.Tensor):
        wires_list = _ensure_wires_list(wires)
        if not (
            name_or_mat.ndim == 2
            and name_or_mat.shape[0] == name_or_mat.shape[1]
            and name_or_mat.shape[0] == 2 ** len(wires_list)
        ):
            raise ValueError("Invalid gate matrix provided")
        mat = name_or_mat
    else:
        raise TypeError("Invalid gate type provided.")

    # 多参数门列表（需要保持参数形状不变）
    multi_param_gates = ["u2", "u3"]

    # 在 gate 函数中处理形状
    if params is not None:
        if not isinstance(params, torch.Tensor):
            params = torch.tensor(params, requires_grad=True)

        if name_or_mat in multi_param_gates:
            # 确保是 2D
            if params.dim() == 1:
                params = params.unsqueeze(0)

            # 扩展到 batch size
            if params.shape[0] == 1 and q_device.bsz > 1:
                params = params.expand(q_device.bsz, -1)
            # 确保最后一维大小正确
            expected = 2 if name_or_mat == "u2" else 3
            if params.shape[-1] != expected:
                raise ValueError(
                    f"{name_or_mat} 期望每组 {expected} 个参数, 但现在是 {params.shape[-1]} 个。"
                )
        else:
            # 单参数门：形状 [batch, 1]
            if params.dim() == 0:
                params = params.reshape(1, 1)  # 标量 -> [1, 1]
            elif params.dim() == 1:
                params = params.reshape(-1, 1)  # [n] -> [n, 1]
            # 扩展到 batch size
            if params.shape[0] == 1 and q_device.bsz > 1:
                params = params.expand(q_device.bsz, -1)

    wires = _ensure_wires_list(wires)

    if q_device.record_op:
        q_device.op_history.append(
            {
                "name_or_mat": name_or_mat,
                "wires": np.array(wires).squeeze().tolist(),
                "params": (
                    params.squeeze().detach().cpu().numpy().tolist()
                    if params is not None
                    else None
                ),
                "trainable": params.requires_grad if params is not None else False,
            }
        )

    # in dynamic mode, the function is computed instantly
    if callable(mat):
        matrix = mat(params)
    else:
        matrix = mat

    if inverse:
        matrix = matrix.mH

    assert np.log2(matrix.shape[-1]) == len(wires)

    # handle resharding here so that applying unitary on the state operates in parallel
    q_device.maybe_reshard(wires)

    state, groupings = q_device.noncanonical_states
    state, groupings, invertible_dummy = apply_unitary_bmm(
        state, matrix, wires, groupings, invertible_dummy=q_device._invertible_dummy
    )
    q_device._states = state
    q_device._groupings = groupings
    q_device._invertible_dummy = invertible_dummy


# Dynamically create gate functions
def _create_gate_function(name: str, inverse: bool = False):
    """Create a gate function for the given gate name."""
    return functools.partial(gate, name, inverse=inverse)


# populate namespace with functionals
for gate_name in GATE_MAT_DICT.keys():
    vars()[gate_name] = functools.partial(gate, gate_name)
    vars()[f"{gate_name}_inv"] = functools.partial(gate, gate_name, inverse=True)


# Explicitly define exports
__all__ = (
    list(GATE_MAT_DICT.keys())
    + [f"{name}_inv" for name in GATE_MAT_DICT.keys()]
    + [
        "apply_unitary_bmm",
        "gate",
    ]
)

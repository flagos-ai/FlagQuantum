# measurement/measure.py
"""Quantum measurement operations.

This module provides measurement functions for quantum devices,
including Pauli Z basis measurements with support for:
- Differentiable approximate sampling (training mode)
- Non-differentiable exact sampling (inference mode)
- Post-selection on specific qubit states
- Distributed tensor support

Examples
--------
Basic measurement:

>>> from flagquantum.measure import measure_allZ
>>> from flagquantum.devices import DistributedQuantumDevice
>>>
>>> device = DistributedQuantumDevice(n_wires=4, bsz=2)
>>> # Apply some gates
>>> device.h(wires=[0])
>>> device.cx(wires=[0, 1])
>>>
>>> # Measure all qubits
>>> expectations = measure_allZ(device)
>>> print(expectations.shape)  # (2, 4)

With shots (sampling):

>>> outcomes = measure_allZ(device, shots=1024)
>>> print(outcomes.shape)  # (2, 4)

With post-selection:

>>> # Only keep runs where qubit 0 is in |1>
>>> expectations, retained = measure_allZ(device, postselect_cond={0: 1})
>>> print(retained.shape)  # (batch_size_with_condition,)

Training mode (differentiable):

>>> expectations = measure_allZ(device, shots=100, training=True)
>>> # Gradients can flow through the measurement
"""

import math
from typing import Dict, Optional, Tuple, Union

import torch
from torch.distributed.tensor import DTensor

from ..utils.maybe_dtensor import (
    is_dtensor,
    maybe_from_local,
    maybe_full_tensor,
    maybe_get_dtensor_info,
    maybe_to_local,
)

# ============================================================================
# Sampling Functions
# ============================================================================


def sampler_diff_approx(
    state_mag: Union[torch.Tensor, DTensor], shots: int, global_rank: int, world_sz: int
) -> Union[torch.Tensor, DTensor]:
    """Differentiable approximate sampling."""
    p = state_mag
    p_2 = torch.sqrt(p + 1e-16)

    z = torch.randn(maybe_to_local(p).size(), device=p.device)
    e_d = torch.zeros(maybe_to_local(p).size(), device=p.device)
    if global_rank == world_sz - 1:
        z[-1] = 0
        e_d[-1] = 1
    p_device_mesh, p_placements = maybe_get_dtensor_info(p)
    z = maybe_from_local(z, device_mesh=p_device_mesh, placements=p_placements)
    e_d = maybe_from_local(e_d, device_mesh=p_device_mesh, placements=p_placements)

    reduce_dims = tuple(range(1, p.ndim))
    u = (p_2 - e_d) / (
        torch.linalg.vector_norm(p_2 - e_d, dim=reduce_dims, keepdim=True) + 1e-8
    )
    mu = p * shots
    qz = z - 2 * u * (z * u).sum(reduce_dims, keepdim=True)
    v = p_2 * qz * (shots**0.5)

    state_mag_noisy = torch.nn.functional.relu(v + mu)
    state_mag_noisy = state_mag_noisy / (
        state_mag_noisy.sum(reduce_dims, keepdim=True) + 1e-8
    )
    return state_mag_noisy


def sampler_nondiff_exact(
    state_mag: Union[torch.Tensor, DTensor], shots: int, global_rank: int, *unused_args
) -> Union[torch.Tensor, DTensor]:
    """Non-differentiable exact sampling."""
    orig_shape_local = maybe_to_local(state_mag).shape
    state_mag_device_mesh, state_mag_placements = maybe_get_dtensor_info(state_mag)
    shard_dims = [s_.dim for s_ in state_mag_placements if hasattr(s_, "dim")]
    excluded_dims = set([0] + shard_dims)
    reduce_dims = tuple(
        dim for dim in range(state_mag.ndim) if dim not in excluded_dims
    )
    gpu_probs = state_mag.sum(list(reduce_dims))
    gpu_probs = maybe_full_tensor(gpu_probs).view((orig_shape_local[0], -1))

    if is_dtensor(state_mag):
        local_shots = torch.distributions.multinomial.Multinomial(
            shots, gpu_probs
        ).sample()
        local_shots = local_shots[:, global_rank]
    else:
        local_shots = shots * torch.ones(orig_shape_local[0], device=state_mag.device)

    state_mag_noisy = []
    state_mag_local = maybe_to_local(state_mag)
    for i in range(orig_shape_local[0]):
        local_shots_ = int(local_shots[i].item())
        if local_shots_ > 0:
            state_mag_local_ = state_mag_local[i].view(-1)
            state_mag_local_norm_ = state_mag_local_ / (state_mag_local_.sum() + 1e-8)
            state_mag_noisy_ = (
                torch.distributions.multinomial.Multinomial(
                    local_shots_, state_mag_local_norm_
                ).sample()
                / shots
            )
            state_mag_noisy_ = state_mag_noisy_.reshape(orig_shape_local[1:])
        else:
            state_mag_noisy_ = torch.zeros(
                orig_shape_local[1:], device=state_mag.device
            )
        state_mag_noisy.append(state_mag_noisy_)
    state_mag_noisy = torch.stack(state_mag_noisy)
    state_mag_noisy = maybe_from_local(
        state_mag_noisy,
        device_mesh=state_mag_device_mesh,
        placements=state_mag_placements,
    )
    return state_mag_noisy


# ============================================================================
# Main Measurement Function
# ============================================================================


def measure_allZ(  # noqa: N802
    q_device,
    shots: int = 0,
    postselect_cond: Optional[Dict[int, int]] = None,
    training: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Measure all qubits in Pauli Z basis."""

    if postselect_cond is None:
        postselect_cond = {}

    # ========== Add this section: record measurement operation ==========
    # Add all qubits to op_history

    if q_device.record_op:
        all_wires = list(range(q_device.n_wires))
        measurement_op = {
            "name_or_mat": "measure_allZ",
            "wires": all_wires,
            "params": None,
            "trainable": False,
            "shots": shots,
            "postselect_cond": postselect_cond,
        }
        q_device.op_history.append(measurement_op)

    if not len(postselect_cond) < q_device.n_wires - q_device.log2_devices:
        raise RuntimeError(
            "Postselected qubits exceeds number of available unsharded slots."
        )
    q_device.maybe_reshard(list(postselect_cond.keys()))
    states, groupings = q_device.noncanonical_states
    sharded_wires = torch.nonzero(groupings[1] == -2).flatten()
    grouped_wires = torch.nonzero(groupings[1] >= 0).flatten()
    ungrouped_wires = torch.nonzero(groupings[1] == -1).flatten()

    state_mag = (states**2).sum(-1)

    # Postselection
    if postselect_cond:
        if not set(postselect_cond.values()).issubset({0, 1}):
            raise RuntimeError("Postselection criteria are ill-defined.")
        local_mask_size = maybe_to_local(state_mag).size()
        local_mask_size = [
            local_mask_size[i] if i > 0 else 1 for i in range(len(local_mask_size))
        ]
        local_mask = torch.ones(local_mask_size, device=states.device, dtype=bool)
        states_device_mesh, states_placements = maybe_get_dtensor_info(states)
        post_wires = list(postselect_cond.keys())
        post_bits = list(postselect_cond.values())
        post_groups = groupings[:, post_wires]
        for i in range(len(post_wires)):
            wire_info = post_groups[:, i].flatten()
            group_idx = wire_info[0].item()
            qubit_idx = wire_info[1].item()
            if qubit_idx == -1:
                slice_idx = (
                    (slice(None),) * group_idx
                    + (1 - post_bits[i],)
                    + (slice(None),) * (len(local_mask_size) - group_idx - 1)
                )
            else:
                group_size = state_mag.shape[group_idx]
                n_group_qubits = int(math.log2(group_size))
                group_bool = torch.zeros(
                    group_size, device=local_mask.device, dtype=bool
                ).reshape((2,) * n_group_qubits)
                group_bool[
                    (slice(None),) * wire_info[1].item()
                    + (1 - post_bits[i],)
                    + (slice(None),) * (n_group_qubits - qubit_idx - 1)
                ] = True
                slice_idx = (
                    (slice(None),) * wire_info[0].item()
                    + (group_bool.flatten(),)
                    + (slice(None),) * (len(local_mask_size) - group_idx - 1)
                )
            local_mask[slice_idx] = False

        full_mask = maybe_from_local(
            local_mask, device_mesh=states_device_mesh, placements=states_placements
        )
        state_mag = torch.where(full_mask, state_mag, torch.zeros_like(state_mag))
        norm_square = state_mag.sum(
            [d for d in range(full_mask.ndim) if d > 0]
        ).reshape(-1, 1)
        retained = norm_square.squeeze(-1) > 1e-16
        if not torch.any(retained):
            raise RuntimeError("Postselection did not retain any batch elements.")
        elif not torch.all(retained):
            state_mag = state_mag[retained]
            norm_square = norm_square[retained]
        state_mag = state_mag / norm_square.reshape(
            norm_square.shape[0], *([1] * (state_mag.ndim - 1))
        )
    else:
        retained = None

    if shots > 0:
        if not training:
            torch.manual_seed(q_device.shared_seed)
            q_device.shared_seed += 1
            sampler = sampler_nondiff_exact
        else:
            sampler = sampler_diff_approx
    else:

        def sampler(x, _0, _1, _2):
            return x

    state_mag_noisy = sampler(state_mag, shots, q_device.global_rank, q_device.world_sz)

    probs = torch.zeros(
        (state_mag_noisy.shape[0], q_device.n_wires, 2), device=state_mag_noisy.device
    )

    # Keep the original gradient propagation method
    sharded_reduce_list = list(groupings[0, sharded_wires])
    if sharded_reduce_list:
        shard_reduced_state_mag = state_mag_noisy.sum(list(groupings[0, sharded_wires]))
    else:
        shard_reduced_state_mag = state_mag_noisy
    remaining_dims = torch.arange(
        1, shard_reduced_state_mag.ndim, device=state_mag_noisy.device
    )
    for wire in ungrouped_wires:
        reduce_list = remaining_dims[
            remaining_dims != groupings[0, wire].item()
        ].tolist()
        if reduce_list:
            prob_ = shard_reduced_state_mag.sum(reduce_list)
        else:
            prob_ = shard_reduced_state_mag
        prob_ = maybe_full_tensor(prob_)
        probs[:, wire, :] = prob_

    if len(grouped_wires) > 0 and len(ungrouped_wires) > 0:
        prev_wire = ungrouped_wires[0]
        reduction_dims = remaining_dims[
            remaining_dims != groupings[0, prev_wire].item()
        ].tolist()
        shard_reduced_groupings = groupings.detach().clone()
        for wire in grouped_wires:
            shard_reduced_state_mag, shard_reduced_groupings = (
                q_device.interchange_qubits(
                    shard_reduced_state_mag, shard_reduced_groupings, wire, prev_wire
                )
            )
            prob_ = shard_reduced_state_mag.sum(reduction_dims)
            if q_device.world_sz > 1:
                prob_ = prob_.full_tensor()
            probs[:, wire, :] = prob_
            prev_wire = wire

    if is_dtensor(state_mag_noisy):
        remaining_dims = torch.arange(
            1, state_mag_noisy.ndim, device=state_mag_noisy.device
        )
        unshard_mask = torch.ones(
            remaining_dims.shape, dtype=bool, device=remaining_dims.device
        )
        for wire in sharded_wires:
            unshard_mask &= remaining_dims != groupings[0, wire].item()
        only_shard_state_mag = state_mag_noisy.sum(
            remaining_dims[unshard_mask].tolist()
        )
        remaining_dims = torch.arange(
            1, q_device.log2_devices + 1, device=remaining_dims.device
        )
        only_shard_groupings = groupings.detach().clone()
        only_shard_groupings[0, sharded_wires] -= (
            min(only_shard_groupings[0, sharded_wires]) - 1
        )
        for wire in sharded_wires:
            reduce_list = remaining_dims[
                remaining_dims != only_shard_groupings[0, wire].item()
            ].tolist()
            if reduce_list:
                prob_ = only_shard_state_mag.sum(reduce_list)
            else:
                prob_ = only_shard_state_mag
            if q_device.world_sz > 1:
                prob_ = prob_.full_tensor()
            probs[:, wire, :] = prob_

    y = probs @ torch.tensor([1.0, -1.0], device=probs.device)

    if postselect_cond:
        return y, retained
    return y


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    "measure_allZ",
    "sampler_diff_approx",
    "sampler_nondiff_exact",
]

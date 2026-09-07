"""Shape-bucketed compiled kernels for batched nearest-neighbour MPS sweeps."""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from typing import Sequence

import torch
import torch.nn.functional as functional

from .mps.models import MPSConfig
from .mps_state import MPSState

# Compiling tiny edge buckets costs more than it saves and, for long chains,
# creates many dynamic-shape specializations.  Interior buckets contain almost
# all sites/bonds and remain on the compiled fast path.
_MIN_COMPILED_BUCKET_SIZE = 8


def _ry_bucket_eager(tensors: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    # tensors: [sites, batch, left, physical, right, real_imag]
    sites, batch = tensors.shape[:2]
    half = angles.reshape(sites, 1) / 2
    cos = torch.cos(half).to(tensors.dtype).reshape(sites, 1, 1, 1, 1)
    sin = torch.sin(half).to(tensors.dtype).reshape(sites, 1, 1, 1, 1)
    zero = tensors[:, :, :, 0, :, :]
    one = tensors[:, :, :, 1, :, :]
    return torch.stack((cos * zero - sin * one, sin * zero + cos * one), dim=3)


def _rxx_bucket_eager(
    left: torch.Tensor,
    right: torch.Tensor,
    angles: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Spatially disjoint bonds with equal shapes become an additional batch
    # dimension, producing one large contraction and one exact split kernel.
    bonds, batch, left_dim = left.shape[:3]
    right_dim = right.shape[-2]
    lr, li = left[..., 0], left[..., 1]
    rr, ri = right[..., 0], right[..., 1]
    real = torch.einsum("kblsm,kbmtr->kblstr", lr, rr) - torch.einsum(
        "kblsm,kbmtr->kblstr", li, ri
    )
    imag = torch.einsum("kblsm,kbmtr->kblstr", lr, ri) + torch.einsum(
        "kblsm,kbmtr->kblstr", li, rr
    )
    half = angles.reshape(bonds, 1, 1, 1, 1, 1) / 2
    cos = torch.cos(half)
    sin = torch.sin(half)
    flipped_real = torch.flip(real, dims=(3, 4))
    flipped_imag = torch.flip(imag, dims=(3, 4))
    merged = torch.stack(
        (cos * real + sin * flipped_imag, cos * imag - sin * flipped_real),
        dim=-1,
    )
    matrix = merged.reshape(bonds * batch, left_dim * 2, 2 * right_dim, 2)
    # The final dimension is the real/imag channel, not a matrix dimension.
    rows, cols = matrix.shape[-3], matrix.shape[-2]
    if rows <= cols:
        eye = torch.zeros(
            bonds * batch, rows, rows, 2, dtype=matrix.dtype, device=matrix.device
        )
        eye[..., 0] = torch.eye(rows, dtype=matrix.dtype, device=matrix.device)
        left_out = eye.reshape(bonds, batch, left_dim, 2, rows, 2)
        right_out = matrix.reshape(bonds, batch, rows, 2, right_dim, 2)
    else:
        left_out = matrix.reshape(bonds, batch, left_dim, 2, cols, 2)
        eye = torch.zeros(
            bonds * batch, cols, cols, 2, dtype=matrix.dtype, device=matrix.device
        )
        eye[..., 0] = torch.eye(cols, dtype=matrix.dtype, device=matrix.device)
        right_out = eye.reshape(bonds, batch, cols, 2, right_dim, 2)
    return left_out, right_out


@lru_cache(maxsize=1)
def _compiled_kernels():
    if not hasattr(torch, "compile"):
        return _ry_bucket_eager, _rxx_bucket_eager
    # Exact MPS sweeps visit a bounded family of bond shapes. Each shape gets
    # an autotuned Triton specialization; the default limit of eight is too
    # small even for a two-step brickwork circuit.
    torch._dynamo.config.recompile_limit = max(64, torch._dynamo.config.recompile_limit)
    torch.set_float32_matmul_precision("highest")
    return (
        torch.compile(
            _ry_bucket_eager,
            fullgraph=True,
            dynamic=True,
            mode="max-autotune-no-cudagraphs",
        ),
        torch.compile(
            _rxx_bucket_eager,
            fullgraph=True,
            dynamic=True,
            mode="max-autotune-no-cudagraphs",
        ),
    )


def _apply_ry_layer(
    tensors: list[torch.Tensor], field: torch.Tensor, *, compiled: bool
) -> None:
    buckets: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for wire, tensor in enumerate(tensors):
        buckets[tuple(tensor.shape[1:])].append(wire)
    for wires in buckets.values():
        use_compiled = compiled and len(wires) >= _MIN_COMPILED_BUCKET_SIZE
        kernel = _compiled_kernels()[0] if use_compiled else _ry_bucket_eager
        packed = torch.stack([tensors[wire] for wire in wires], dim=0)
        updated = kernel(packed, field[wires])
        for position, wire in enumerate(wires):
            tensors[wire] = updated[position]


def _apply_rxx_parity(
    tensors: list[torch.Tensor],
    coupling: torch.Tensor,
    parity: int,
    *,
    compiled: bool,
    max_bond: int | None,
) -> None:
    buckets: dict[tuple[tuple[int, ...], tuple[int, ...]], list[int]] = defaultdict(
        list
    )
    for wire in range(parity, len(tensors) - 1, 2):
        key = (tuple(tensors[wire].shape[1:]), tuple(tensors[wire + 1].shape[1:]))
        buckets[key].append(wire)
    for wires in buckets.values():
        use_compiled = compiled and len(wires) >= _MIN_COMPILED_BUCKET_SIZE
        kernel = _compiled_kernels()[1] if use_compiled else _rxx_bucket_eager
        left = torch.stack([tensors[wire] for wire in wires], dim=0)
        right = torch.stack([tensors[wire + 1] for wire in wires], dim=0)
        full_rank = min(left.shape[2] * 2, right.shape[-2] * 2)
        if max_bond is not None and max_bond < full_rank:
            raise ValueError(
                "compiled brickwork exact split requires max_bond >= active full rank"
            )
        left_out, right_out = kernel(left, right, coupling[wires])
        for position, wire in enumerate(wires):
            tensors[wire] = left_out[position]
            tensors[wire + 1] = right_out[position]


def run_batched_brickwork_mps(
    coupling: torch.Tensor,
    field: torch.Tensor,
    flipped_sites: Sequence[Sequence[int]],
    *,
    time_steps: int,
    dt: float,
    max_bond: int | None,
    compiled: bool = True,
) -> MPSState:
    """Run a fused spatial-bucket brickwork sweep without Circuit interpretation."""

    n_wires = int(field.numel())
    batch = len(flipped_sites)
    if batch < 1 or coupling.shape != (n_wires - 1,):
        raise ValueError("invalid batch or coupling shape")
    # Build the complete product-state batch in one allocation.  Creating one
    # tensor and one host-to-device bit copy per site is especially expensive
    # for the 1024-qubit, shallow-circuit workload, where those tiny launches
    # are a material fraction of a step.
    product = torch.zeros(
        n_wires, batch, 1, 2, 1, 2, dtype=field.dtype, device=field.device
    )
    bits = torch.zeros(n_wires, batch, dtype=torch.long, device=field.device)
    flip_pairs = [
        (int(wire), probe_index)
        for probe_index, sites in enumerate(flipped_sites)
        for wire in sites
    ]
    if flip_pairs:
        flip_index = torch.tensor(flip_pairs, dtype=torch.long, device=field.device)
        bits[flip_index[:, 0], flip_index[:, 1]] = 1
    wires = torch.arange(n_wires, device=field.device).unsqueeze(1)
    batches = torch.arange(batch, device=field.device).unsqueeze(0)
    product[wires, batches, 0, bits, 0, 0] = 1
    tensors = list(product.unbind(0))
    scaled_field = 2.0 * float(dt) * field
    scaled_coupling = 2.0 * float(dt) * coupling
    for _ in range(int(time_steps)):
        _apply_ry_layer(tensors, scaled_field, compiled=compiled)
        _apply_rxx_parity(
            tensors, scaled_coupling, 0, compiled=compiled, max_bond=max_bond
        )
        _apply_rxx_parity(
            tensors, scaled_coupling, 1, compiled=compiled, max_bond=max_bond
        )
    complex_tensors = [torch.view_as_complex(tensor.contiguous()) for tensor in tensors]
    return MPSState(
        complex_tensors,
        config=MPSConfig(max_bond=max_bond, cutoff=0.0),
    )


__all__ = ("run_batched_brickwork_mps",)


def _environment_step_real(
    env: torch.Tensor, tensor: torch.Tensor, ket_tensor: torch.Tensor | None = None
) -> torch.Tensor:
    er, ei = env[..., 0], env[..., 1]
    ar, ai = tensor[..., 0], tensor[..., 1]
    if ket_tensor is None:
        ket_tensor = tensor
    kr, ki = ket_tensor[..., 0], ket_tensor[..., 1]
    cr = torch.einsum("bij,bipr->bjpr", er, ar) + torch.einsum("bij,bipr->bjpr", ei, ai)
    ci = -torch.einsum("bij,bipr->bjpr", er, ai) + torch.einsum(
        "bij,bipr->bjpr", ei, ar
    )
    out_r = torch.einsum("bjpr,bjps->brs", cr, kr) - torch.einsum(
        "bjpr,bjps->brs", ci, ki
    )
    out_i = torch.einsum("bjpr,bjps->brs", cr, ki) + torch.einsum(
        "bjpr,bjps->brs", ci, kr
    )
    return torch.stack((out_r, out_i), dim=-1)


def _real_observable_scan_eager(
    padded_tensors: torch.Tensor, targets: tuple[int, ...]
) -> tuple[torch.Tensor, torch.Tensor]:
    from torch._higher_order_ops import scan

    sites, batch, bond = padded_tensors.shape[:3]
    initial = torch.zeros(
        batch, bond, bond, 2, dtype=padded_tensors.dtype, device=padded_tensors.device
    )
    initial[:, 0, 0, 0] = 1

    def step(carry, tensor):
        updated = _environment_step_real(carry, tensor)
        return updated, updated.clone()

    _, left_after = scan(step, initial, padded_tensors, dim=0)
    right_oriented = padded_tensors.permute(0, 1, 4, 3, 2, 5)
    _, right_before = scan(step, initial, right_oriented, dim=0, reverse=True)
    # Environment before site i and after site i.
    left = torch.cat((initial.unsqueeze(0), left_after[:-1]), dim=0)
    right = torch.cat((right_before[1:], initial.unsqueeze(0)), dim=0)
    z_sign = padded_tensors.new_tensor((1.0, -1.0))
    z_values = []
    zz_values = []
    for wire in targets:
        tensor = padded_tensors[wire]
        # Insert Z by multiplying the ket physical channel.
        z_tensor = tensor * z_sign.reshape(1, 1, 2, 1, 1)
        inserted = _environment_step_real(left[wire], tensor, z_tensor)
        overlap = (
            inserted[..., 0] * right[wire][..., 0]
            - inserted[..., 1] * right[wire][..., 1]
        )
        z_values.append(overlap.sum(dim=(-2, -1)))
        if wire + 1 < sites:
            next_z = padded_tensors[wire + 1] * z_sign.reshape(1, 1, 2, 1, 1)
            pair = _environment_step_real(inserted, padded_tensors[wire + 1], next_z)
            pair_overlap = (
                pair[..., 0] * right[wire + 1][..., 0]
                - pair[..., 1] * right[wire + 1][..., 1]
            )
            zz_values.append(pair_overlap.sum(dim=(-2, -1)))
    return torch.stack(z_values, dim=-1), torch.stack(zz_values, dim=-1)


@lru_cache(maxsize=8)
def _compiled_observable_scan(targets: tuple[int, ...]):
    torch._dynamo.config.recompile_limit = max(64, torch._dynamo.config.recompile_limit)

    def fixed_targets(padded_tensors):
        return _real_observable_scan_eager(padded_tensors, targets)

    return torch.compile(
        fixed_targets,
        fullgraph=True,
        dynamic=False,
        # A scan contains many repeated small BMMs.  Max-autotune currently
        # tunes each lowered instance independently, making compile time grow
        # with chain length without improving the recurrence's critical path.
        mode="default",
    )


def compiled_local_z_zz(
    state: MPSState, wires: Sequence[int], *, compiled: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate local Z/ZZ through one padded real-channel environment scan."""

    real_tensors = [torch.view_as_real(tensor) for tensor in state.tensors]
    max_bond = max(max(tensor.shape[1], tensor.shape[3]) for tensor in real_tensors)
    padded = []
    for tensor in real_tensors:
        left_pad = max_bond - tensor.shape[1]
        right_pad = max_bond - tensor.shape[3]
        padded.append(functional.pad(tensor, (0, 0, 0, right_pad, 0, 0, 0, left_pad)))
    stacked = torch.stack(padded, dim=0)
    targets = tuple(int(wire) for wire in wires)
    if compiled:
        return _compiled_observable_scan(targets)(stacked)
    return _real_observable_scan_eager(stacked, targets)


__all__ = ("compiled_local_z_zz", "run_batched_brickwork_mps")

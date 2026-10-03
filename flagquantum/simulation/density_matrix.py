"""Exact local density-matrix simulation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import torch

from ..core.ir import CircuitIR
from ..core.runtime_config import get_runtime_config
from ..kernels.catalog import KERNEL_DEVICES
from ..noise import KrausChannel
from .gate_matrix import gate_matrix


def _complex_dtype(dtype: torch.dtype | None = None) -> torch.dtype:
    return dtype or getattr(torch, get_runtime_config().complex_dtype)


def _resolve_device(
    device: torch.device | str | None, *, fallback: torch.device
) -> torch.device:
    """Resolve a requested device against the declared execution device axis.

    The axis is declared once, by the kernel catalog, so this simulator cannot
    drift into accepting a device no kernel record names. A device requested as
    a string must already spell a declared device: ``torch.device`` would accept
    ``"CUDA"``, while the catalog matches record fields by exact spelling.
    """
    if device is None:
        return fallback
    name = device.split(":", maxsplit=1)[0] if isinstance(device, str) else device.type
    if name not in KERNEL_DEVICES:
        raise ValueError(
            f"undeclared density-matrix device: {name}; declared devices are "
            + ", ".join(sorted(KERNEL_DEVICES))
        )
    return torch.device(device)


def _resolve_density_dtype(
    dtype: torch.dtype | None, *, fallback: torch.dtype
) -> torch.dtype:
    """Resolve a requested precision, refusing a silent downgrade to real."""
    if dtype is None:
        return fallback
    if not dtype.is_complex:
        raise ValueError(
            f"density matrix dtype must be complex, got {dtype}; a real dtype "
            "would discard the phase rather than record a downgrade"
        )
    return dtype


def density_matrix(circuit_or_state: Any) -> torch.Tensor:
    """Return a batched density matrix from a circuit or statevector."""

    state = (
        circuit_or_state.state()
        if hasattr(circuit_or_state, "state")
        else circuit_or_state
    )
    state = torch.as_tensor(state)
    if state.ndim == 1:
        state = state.reshape(1, -1)
    return state.unsqueeze(-1) * torch.conj(state).unsqueeze(-2)


def _basis_bits(index: int, n_wires: int) -> list[int]:
    return [(index >> (n_wires - 1 - wire)) & 1 for wire in range(n_wires)]


def _bits_to_index(bits: Sequence[int]) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def _validate_wires(wires: Sequence[int], n_wires: int) -> tuple[int, ...]:
    """Refuse a wire the Hilbert space does not have, and normalize the rest.

    Every function here addresses wires by indexing a size-``2**n_wires`` axis,
    and a negative index is a valid Python index, so ``-1`` silently addressed
    the last wire: ``apply_unitary_density`` returned a density matrix identical
    to the one for the wire the caller did not name. An index past the last wire
    raised a bare ``IndexError`` from the indexing expression instead of naming
    the offending wire.
    """
    normalized = tuple(int(wire) for wire in wires)
    if any(wire < 0 or wire >= n_wires for wire in normalized):
        raise ValueError("wire index out of range")
    return normalized


def expand_operator(
    matrix: torch.Tensor,
    wires: Sequence[int],
    n_wires: int,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Expand a k-wire operator into the full Hilbert space."""

    wires = _validate_wires(wires, n_wires)
    matrix = torch.as_tensor(matrix, dtype=_complex_dtype(dtype), device=device)
    if matrix.ndim == 3:
        return torch.stack(
            [
                expand_operator(
                    item, wires, n_wires, dtype=matrix.dtype, device=matrix.device
                )
                for item in matrix
            ]
        )
    dim = 2**n_wires
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("operator matrix must be square")
    if matrix.shape[0] != 2 ** len(wires):
        raise ValueError("operator dimension does not match the target wires")
    full = torch.zeros(dim, dim, dtype=matrix.dtype, device=matrix.device)
    for col in range(dim):
        bits = _basis_bits(col, n_wires)
        sub_col = _bits_to_index(tuple(bits[wire] for wire in wires))
        for sub_row in range(2 ** len(wires)):
            row_bits = list(bits)
            replacement = _basis_bits(sub_row, len(wires))
            for offset, wire in enumerate(wires):
                row_bits[wire] = replacement[offset]
            row = _bits_to_index(row_bits)
            full[row, col] = matrix[sub_row, sub_col]
    return full


def apply_unitary_density(
    rho: torch.Tensor,
    matrix: torch.Tensor,
    wires: Sequence[int],
    n_wires: int,
) -> torch.Tensor:
    """Apply a unitary matrix to a batched density matrix."""

    if rho.ndim == 2:
        rho = rho.reshape(1, *rho.shape)
    full = expand_operator(matrix, wires, n_wires, dtype=rho.dtype, device=rho.device)
    if full.ndim == 2:
        full = full.expand(rho.shape[0], -1, -1)
    return torch.bmm(torch.bmm(full, rho), torch.conj(full).transpose(-1, -2))


def apply_kraus_density(
    rho: torch.Tensor,
    kraus: KrausChannel | Sequence[torch.Tensor],
    wires: Sequence[int],
    n_wires: int,
) -> torch.Tensor:
    """Apply Kraus operators to a batched density matrix."""

    if rho.ndim == 2:
        rho = rho.reshape(1, *rho.shape)
    ops = kraus.kraus if isinstance(kraus, KrausChannel) else tuple(kraus)
    if not ops:
        # `out` starts at zero, so an empty channel returned the zero map: a
        # trace-0 "state" that silently zeroed every later expectation.
        raise ValueError("Kraus channel must contain at least one operator")
    out = torch.zeros_like(rho)
    for op in ops:
        full = expand_operator(op, wires, n_wires, dtype=rho.dtype, device=rho.device)
        if full.ndim == 2:
            full = full.expand(rho.shape[0], -1, -1)
        out = out + torch.bmm(torch.bmm(full, rho), torch.conj(full).transpose(-1, -2))
    return out


def density_matrix_from_ir(
    circuit_or_ir: Any,
    *,
    bsz: int | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Execute unitary and channel IR instructions as a density matrix.

    ``bsz``, ``device``, and ``dtype`` request the batch size, execution device,
    and precision of the result. A request is either honoured or refused; it is
    never ignored. Omitting one inherits what the input already declares: a
    ``Circuit`` supplies its own batch size, device, and dtype, and a
    :class:`CircuitIR` uses the runtime complex default. A ``Circuit`` whose
    state carries a different number of batch rows than a requested ``bsz`` is
    refused, because a fresh state of that size would not be that circuit's
    state.
    """

    if bsz is not None and bsz < 1:
        raise ValueError(f"density matrix batch size must be positive, got {bsz}")
    if hasattr(circuit_or_ir, "to_ir"):
        circuit = circuit_or_ir
        ir = circuit.to_ir()
        state = circuit.initial_state()
        target_device = _resolve_device(device, fallback=state.device)
        target_dtype = _resolve_density_dtype(dtype, fallback=state.dtype)
        if state.device != target_device or state.dtype != target_dtype:
            state = state.to(device=target_device, dtype=target_dtype)
        if bsz is not None and state.shape[0] != bsz:
            raise ValueError(
                "density matrix batch size mismatch: the circuit state carries "
                f"{state.shape[0]} batch rows but bsz={bsz} was requested"
            )
    elif isinstance(circuit_or_ir, CircuitIR):
        ir = circuit_or_ir
        target_device = _resolve_device(device, fallback=torch.device("cpu"))
        target_dtype = _resolve_density_dtype(
            dtype, fallback=getattr(torch, get_runtime_config().complex_dtype)
        )
        state = torch.zeros(
            bsz or 1,
            2**ir.n_wires,
            dtype=target_dtype,
            device=target_device,
        )
        state[:, 0] = 1
    else:
        raise TypeError("density_matrix_from_ir expects a Circuit or CircuitIR.")

    rho = density_matrix(state)
    for instruction in ir:
        if instruction.metadata.get("is_channel"):
            rho = apply_kraus_density(
                rho,
                instruction.matrix,
                instruction.wires,
                ir.n_wires,
            )
            continue
        matrix = gate_matrix(
            instruction,
            bsz=rho.shape[0],
            device=rho.device,
            dtype=rho.dtype,
        )
        rho = apply_unitary_density(rho, matrix, instruction.wires, ir.n_wires)
    return rho


def expectation_z_density(
    rho: torch.Tensor,
    wires: Iterable[int] | int | None = None,
) -> torch.Tensor:
    """Compute Z expectations from a density matrix."""

    if rho.ndim == 2:
        rho = rho.reshape(1, *rho.shape)
    n_wires = int(torch.log2(torch.tensor(rho.shape[-1], dtype=torch.float32)).item())
    if wires is None:
        target_wires = tuple(range(n_wires))
    elif isinstance(wires, int):
        target_wires = (wires,)
    else:
        target_wires = tuple(int(wire) for wire in wires)

    probs = torch.real(torch.diagonal(rho, dim1=-2, dim2=-1))
    shaped = probs.reshape((rho.shape[0],) + (2,) * n_wires)
    values = []
    for wire in _validate_wires(target_wires, n_wires):
        axes = tuple(axis for axis in range(1, n_wires + 1) if axis != wire + 1)
        marginal = shaped.sum(dim=axes) if axes else shaped
        values.append(marginal[:, 0] - marginal[:, 1])
    return torch.stack(values, dim=-1)


__all__ = (
    "apply_kraus_density",
    "apply_unitary_density",
    "density_matrix",
    "density_matrix_from_ir",
    "expectation_z_density",
    "expand_operator",
)

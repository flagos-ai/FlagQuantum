"""Exact local density-matrix simulation."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

import torch

from ..core.ir import CircuitIR
from ..core.runtime_config import get_runtime_config
from ..kernels.catalog import KERNEL_DEVICES
from ..noise import KrausChannel
from .density_matrix_dispatch import _density_bmm
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
    """Return a batched density matrix from a circuit or statevector.

    A program that carries a channel instruction is executed as a density matrix
    rather than read off a statevector: the outer product of a statevector is
    rank one, so it cannot represent the mixed state a channel produces, and the
    statevector route refuses a channel instead of approximating it. A program
    without one keeps the cheaper outer-product route.
    """

    to_ir = getattr(circuit_or_state, "to_ir", None)
    if to_ir is not None:
        ir = to_ir()
        if any(item.metadata.get("is_channel") for item in ir):
            return density_matrix_from_ir(circuit_or_state)
    state = (
        circuit_or_state.state()
        if hasattr(circuit_or_state, "state")
        else circuit_or_state
    )
    state = torch.as_tensor(state)
    if state.ndim == 1:
        state = state.reshape(1, -1)
    return state.unsqueeze(-1) * torch.conj(state).unsqueeze(-2)


def _basis_bits(index: int, n_qubits: int) -> list[int]:
    return [(index >> (n_qubits - 1 - qubit)) & 1 for qubit in range(n_qubits)]


def _bits_to_index(bits: Sequence[int]) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def _validate_qubits(qubits: Sequence[int], n_qubits: int) -> tuple[int, ...]:
    """Refuse a qubit the Hilbert space does not have, and normalize the rest.

    Every function here addresses qubits by indexing a size-``2**n_qubits`` axis,
    and a negative index is a valid Python index, so ``-1`` silently addressed
    the last qubit: ``apply_unitary_density`` returned a density matrix identical
    to the one for the qubit the caller did not name. An index past the last qubit
    raised a bare ``IndexError`` from the indexing expression instead of naming
    the offending qubit.
    """
    normalized = tuple(int(qubit) for qubit in qubits)
    if any(qubit < 0 or qubit >= n_qubits for qubit in normalized):
        raise ValueError("qubit index out of range")
    return normalized


def expand_operator(
    matrix: torch.Tensor,
    qubits: Sequence[int],
    n_qubits: int,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Expand a k-qubit operator into the full Hilbert space."""

    qubits = _validate_qubits(qubits, n_qubits)
    matrix = torch.as_tensor(matrix, dtype=_complex_dtype(dtype), device=device)
    if matrix.ndim == 3:
        return torch.stack(
            [
                expand_operator(
                    item, qubits, n_qubits, dtype=matrix.dtype, device=matrix.device
                )
                for item in matrix
            ]
        )
    dim = 2**n_qubits
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("operator matrix must be square")
    if matrix.shape[0] != 2 ** len(qubits):
        raise ValueError("operator dimension does not match the target qubits")
    full = torch.zeros(dim, dim, dtype=matrix.dtype, device=matrix.device)
    for col in range(dim):
        bits = _basis_bits(col, n_qubits)
        sub_col = _bits_to_index(tuple(bits[qubit] for qubit in qubits))
        for sub_row in range(2 ** len(qubits)):
            row_bits = list(bits)
            replacement = _basis_bits(sub_row, len(qubits))
            for offset, qubit in enumerate(qubits):
                row_bits[qubit] = replacement[offset]
            row = _bits_to_index(row_bits)
            full[row, col] = matrix[sub_row, sub_col]
    return full


def apply_unitary_density(
    rho: torch.Tensor,
    matrix: torch.Tensor,
    qubits: Sequence[int],
    n_qubits: int,
) -> torch.Tensor:
    """Apply a unitary matrix to a batched density matrix."""

    if rho.ndim == 2:
        rho = rho.reshape(1, *rho.shape)
    full = expand_operator(matrix, qubits, n_qubits, dtype=rho.dtype, device=rho.device)
    if full.ndim == 2:
        full = full.expand(rho.shape[0], -1, -1)
    return _density_bmm(_density_bmm(full, rho), torch.conj(full).transpose(-1, -2))


def apply_kraus_density(
    rho: torch.Tensor,
    kraus: KrausChannel | Sequence[torch.Tensor],
    qubits: Sequence[int],
    n_qubits: int,
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
        full = expand_operator(op, qubits, n_qubits, dtype=rho.dtype, device=rho.device)
        if full.ndim == 2:
            full = full.expand(rho.shape[0], -1, -1)
        out = out + _density_bmm(
            _density_bmm(full, rho), torch.conj(full).transpose(-1, -2)
        )
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


def reduced_density_matrix(
    rho: torch.Tensor,
    qubits: Iterable[int] | int | None = None,
) -> torch.Tensor:
    """Trace out every qubit the caller did not name.

    ``rho`` is a batched density matrix over ``n_qubits`` qubits whose axes are
    big-endian per qubit, the same convention ``density_matrix`` produces and
    ``expectation_z_density`` reads. The result is the reduced matrix over the
    named qubits, ordered as the caller named them: asking for ``(1, 0)``
    returns a matrix in that basis order rather than in ascending qubit order,
    which is a permutation of the same numbers and would otherwise be returned
    silently wrong rather than refused.

    The reduction is a trace, not a marginal. Summing a density matrix's row or
    column index instead of contracting its row index against its column index
    would return a matrix whose trace is the state's purity rather than one, so
    the two indices are split by ``n_qubits`` and consumed in pairs, from the
    highest traced qubit down so that no surviving ``dim1``/``dim2`` pair has
    already shifted under an earlier removal.
    """

    if rho.ndim == 2:
        rho = rho.reshape(1, *rho.shape)
    if rho.ndim != 3 or rho.shape[-1] != rho.shape[-2]:
        raise ValueError("a density matrix must be a square matrix per batch item")
    n_qubits = int(round(math.log2(rho.shape[-1])))
    if qubits is None:
        target_qubits = tuple(range(n_qubits))
    elif isinstance(qubits, int):
        target_qubits = (qubits,)
    else:
        target_qubits = tuple(int(qubit) for qubit in qubits)
    target_qubits = _validate_qubits(target_qubits, n_qubits)
    if len(set(target_qubits)) != len(target_qubits):
        raise ValueError("reduced density-matrix qubits must be unique")

    surviving = sorted(target_qubits)
    traced = [qubit for qubit in range(n_qubits - 1, -1, -1) if qubit not in surviving]
    tensor = rho.reshape((rho.shape[0],) + (2,) * (2 * n_qubits))
    remaining = n_qubits
    for qubit in traced:
        # The row axis of qubit ``q`` is ``1 + q`` and its column axis is
        # ``1 + n_qubits + q``; ``torch.diagonal`` removes both and appends the
        # shared index, so summing that last axis contracts them.
        tensor = torch.diagonal(tensor, dim1=1 + qubit, dim2=1 + remaining + qubit).sum(
            dim=-1
        )
        remaining -= 1
    order = (
        (0,)
        + tuple(1 + surviving.index(qubit) for qubit in target_qubits)
        + tuple(1 + len(surviving) + surviving.index(qubit) for qubit in target_qubits)
    )
    tensor = tensor.permute(order)
    return tensor.reshape(
        rho.shape[0], 2 ** len(target_qubits), 2 ** len(target_qubits)
    )


def expectation_z_density(
    rho: torch.Tensor,
    qubits: Iterable[int] | int | None = None,
) -> torch.Tensor:
    """Compute Z expectations from a density matrix."""

    if rho.ndim == 2:
        rho = rho.reshape(1, *rho.shape)
    n_qubits = int(torch.log2(torch.tensor(rho.shape[-1], dtype=torch.float32)).item())
    if qubits is None:
        target_qubits = tuple(range(n_qubits))
    elif isinstance(qubits, int):
        target_qubits = (qubits,)
    else:
        target_qubits = tuple(int(qubit) for qubit in qubits)

    probs = torch.real(torch.diagonal(rho, dim1=-2, dim2=-1))
    shaped = probs.reshape((rho.shape[0],) + (2,) * n_qubits)
    values = []
    for qubit in _validate_qubits(target_qubits, n_qubits):
        axes = tuple(axis for axis in range(1, n_qubits + 1) if axis != qubit + 1)
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
    "reduced_density_matrix",
)

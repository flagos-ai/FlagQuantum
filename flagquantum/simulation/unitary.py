"""Dense unitary materialization for small canonical programs.

The counterpart of ``cudaq.get_unitary``. This module belongs to Simulation
because materializing a matrix is numerical work: Compiler may name the entry
point but must not execute a kernel, so the compiler-side callers re-export
rather than reimplement.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from ..core.ir import CircuitIR, ensure_circuit_ir
from ..core.runtime_config import get_runtime_config

if TYPE_CHECKING:
    from ..circuit import Circuit

MAX_UNITARY_QUBITS = 12
# Largest qubit count a dense unitary is materialized for.
#
# The input basis and the matrix it produces are each ``4**n_qubits`` complex entries,
# so 12 qubits is already 268 MB of complex64 held at once. The ceiling matches the one
# the dense Hamiltonian diagnostic uses, and it exists so that an oversized request
# fails before anything is allocated instead of during it.

_NON_UNITARY_OPCODES = frozenset({"measure", "reset"})


def _unitary_program(program: Any) -> CircuitIR:
    """Return the IR of *program* after refusing everything non-unitary."""

    ir = ensure_circuit_ir(program)
    if ir.n_wires > MAX_UNITARY_QUBITS:
        raise ValueError(
            f"a dense {ir.n_wires}-qubit unitary is not materialized; the limit is "
            f"{MAX_UNITARY_QUBITS} qubits"
        )
    if ir.observables:
        raise ValueError(
            "a unitary belongs to the circuit alone, not to the observable "
            "expectations asked of it"
        )
    if ir.measurements:
        raise ValueError(
            "a unitary belongs to the circuit alone; remove the measurement "
            "request before materializing it"
        )
    for index, instruction in enumerate(ir.instructions):
        metadata = instruction.metadata
        if (
            instruction.name in _NON_UNITARY_OPCODES
            or metadata.get("is_channel")
            or metadata.get("is_dynamic")
            or "conditions" in metadata
            or "condition_clauses" in metadata
        ):
            raise ValueError(
                f"instruction {index} ({instruction.name!r}) is not a unitary "
                "operation, so the program has no unitary matrix"
            )
    return ir


def get_unitary(
    program: Any,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Return the dense unitary matrix of a program's gates.

    The matrix is produced by running the program's own exact statevector kernel
    once per computational basis state, so it agrees with
    :meth:`flagquantum.Circuit.state` by construction rather than through a
    second, independently written matrix-composition path.

    The result follows ``U[i, j] = <i| U |j>``: column ``j`` is the image of
    computational basis state ``j``. Qubit 0 is the most significant index bit,
    as everywhere else in FlagQuantum, so on two qubits basis index ``i`` carries
    qubit 0 in bit 1 and qubit 1 in bit 0.

    Args:
        program: A ``Circuit`` or canonical ``CircuitIR`` holding only unitary
            operations. A noise channel, a mid-circuit measurement, a reset, or
            a classical condition has no unitary matrix and is refused by index
            and opcode. Unbound named parameters are refused; call
            ``Circuit.bind_parameters`` first.
        device: The device the kernel runs on. The default is the runtime
            configuration's device.
        dtype: The matrix's complex dtype. The default is the program's own
            declared dtype, which is complex64 unless the program was built with
            complex128.

    Returns:
        A ``(2**n_qubits, 2**n_qubits)`` complex ``torch.Tensor``.

    Raises:
        ValueError: If the program is wider than :data:`MAX_UNITARY_QUBITS`, if
            it declares measurements or observables, if any instruction is not
            unitary, or if a gate parameter is unbound or not a finite real
            number.

    Examples:
        >>> import torch
        >>> import flagquantum as fq
        >>> from flagquantum.simulation import get_unitary
        >>> unitary = get_unitary(fq.Circuit(1).x(0))
        >>> bool(torch.allclose(unitary, torch.tensor([[0, 1], [1, 0]], dtype=unitary.dtype)))
        True
    """

    ir = _unitary_program(program)
    config = get_runtime_config()
    resolved_device = torch.device(device if device is not None else config.device)
    resolved_dtype = dtype if dtype is not None else getattr(torch, ir.dtype)
    if not resolved_dtype.is_complex:
        raise ValueError("a unitary matrix must have a complex dtype")

    dimension = 1 << ir.n_wires
    basis = torch.eye(dimension, dtype=resolved_dtype)
    circuit = _basis_circuit(ir, basis, resolved_device, resolved_dtype)
    return circuit.state().transpose(0, 1)


def _unitary_with_workspace(
    program: CircuitIR, logical_width: int, borrowed_wires: frozenset[int]
) -> torch.Tensor:
    """Inspect a bounded through matrix with verified workspace restoration."""
    matrix = get_unitary(program, device="cpu", dtype=torch.complex128)
    admitted, rejected = _workspace_basis(
        program.n_wires, logical_width, borrowed_wires
    )
    columns = matrix.index_select(1, admitted) if rejected.numel() else matrix
    return _restore_workspace(columns, admitted, rejected, len(borrowed_wires))


def _workspace_basis(
    width: int, logical_width: int, borrowed_wires: frozenset[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0 < logical_width <= width or any(
        not logical_width <= wire < width for wire in borrowed_wires
    ):
        raise ValueError("Invalid workspace layout for unitary inspection")

    clean_mask = sum(
        1 << (width - wire - 1)
        for wire in range(logical_width, width)
        if wire not in borrowed_wires
    )
    indices = torch.arange(1 << width, device="cpu")
    clean = (indices & clean_mask) == 0
    return indices[clean], indices[~clean]


def _restore_workspace(
    columns: torch.Tensor,
    admitted: torch.Tensor,
    rejected: torch.Tensor,
    borrowed_width: int,
) -> torch.Tensor:
    """Check a branch's admitted columns and reduce over restored workspace."""
    if rejected.numel():
        leakage = columns.index_select(0, rejected)
        if not torch.all(torch.abs(leakage) <= 2e-12):
            raise ValueError("Unitary body does not restore clean workspace")

    projected = columns.index_select(0, admitted) if rejected.numel() else columns
    if not borrowed_width:
        return projected
    stride = 1 << borrowed_width
    reduced = projected[::stride, ::stride].contiguous()
    identity = torch.eye(stride, dtype=projected.dtype, device=projected.device)
    if not torch.allclose(projected, torch.kron(reduced, identity), atol=2e-12, rtol=0):
        raise ValueError("Unitary body does not restore borrowed workspace")
    return reduced


def _basis_circuit(
    ir: CircuitIR,
    basis: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> Circuit:
    """Build the program whose batched state is the unitary, column by column."""

    from ..circuit import Circuit

    return Circuit.from_ir(
        ir,
        bsz=basis.shape[0],
        device=device,
        dtype=dtype,
        inputs=basis,
    )


__all__ = ("MAX_UNITARY_QUBITS", "get_unitary")

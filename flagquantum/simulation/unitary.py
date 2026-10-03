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

MAX_UNITARY_WIRES = 12
"""Largest wire count a dense unitary is materialized for.

The input basis and the matrix it produces are each ``4**n_wires`` complex
entries, so 12 wires is already 268 MB of complex64 held at once. The ceiling
matches the one the dense Hamiltonian diagnostic uses, and it exists so that an
oversized request fails before anything is allocated instead of during it.
"""

_NON_UNITARY_OPCODES = frozenset({"measure", "reset"})


def _unitary_program(program: Any) -> CircuitIR:
    """Return the IR of *program* after refusing everything non-unitary."""

    ir = ensure_circuit_ir(program)
    if ir.n_wires > MAX_UNITARY_WIRES:
        raise ValueError(
            f"a dense {ir.n_wires}-wire unitary is not materialized; the limit is "
            f"{MAX_UNITARY_WIRES} wires"
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
    computational basis state ``j``. Wire 0 is the most significant index bit,
    as everywhere else in FlagQuantum, so on two wires basis index ``i`` carries
    wire 0 in bit 1 and wire 1 in bit 0.

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
        A ``(2**n_wires, 2**n_wires)`` complex ``torch.Tensor``.

    Raises:
        ValueError: If the program is wider than :data:`MAX_UNITARY_WIRES`, if
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


__all__ = ("MAX_UNITARY_WIRES", "get_unitary")

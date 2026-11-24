# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Calls follow locked Qualtran v0.7.0 block_encoding/sparse_matrix.py.

"""Compose uniform and QROM-driven entry rotations on caller-owned indices."""

from math import pi

from ....core.ir import CircuitIR, Instruction
from ....core.ir.program import FunctionRecord, ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import opaque_register_type, quantum_bit_type
from ....core.ir.program.registers import Register
from ....core.ir.program.signatures import Signature
from ....core.value_math import SymbolicInt
from ..component_adjoint import adjoint_component_calls
from ..component_builder import ComponentProgramBuilder
from ..component_gate_function import bit_gate_function
from ..primitives.pauli_rotation import RotationAngle, pauli_rotation_ir
from .qrom_component import qrom_program
from .qrom_table import QromTable


def entry_signature(width: SymbolicInt) -> Signature:
    """Keep flag q followed by the source's opaque row and column indices."""
    return Signature.build_from_dtypes(
        q=quantum_bit_type(),
        i=opaque_register_type(width),
        j=opaque_register_type(width),
    )


def uniform_entry_program(
    width: SymbolicInt, angle: RotationAngle, *, adjoint: bool = False
) -> ProgramRecord:
    """Apply the source atomic Ry with both index words as bystanders."""
    if type(adjoint) is not bool:
        raise ValueError("Entry direction must be boolean")

    def rotation(value: RotationAngle) -> ProgramRecord:
        return ProgramRecord(
            (
                bit_gate_function(
                    "ry",
                    (Register("q", quantum_bit_type()),),
                    CircuitIR(1, pauli_rotation_ir("y", 0, value)),
                ),
            )
        )

    forward, inverse = rotation(angle), rotation(-angle)
    builder = ComponentProgramBuilder(entry_signature(width))
    state = dict(builder.inputs)
    state["q"] = builder.call_program(forward, {"q": state["q"]}, entry="ry")["q"]
    record = builder.finalize(state)
    return adjoint_component_calls(record, ((forward, inverse),)) if adjoint else record


def explicit_entry_program(
    width: SymbolicInt, table: QromTable, *, adjoint: bool = False
) -> ProgramRecord:
    """Allocate/load/rotate/unload/free the original MSB-first angle word.

    Positive controlled Ry is an atomic source Controlled node: its source
    decomposition refuses the atomic Ry. Each native cry leaf is independently
    checked as a full complex operator. QROM retains its own predicate work and
    measurements, and returns itself as its source adjoint.
    """
    if type(adjoint) is not bool:
        raise ValueError("Entry direction must be boolean")
    if (
        len(table.target_bitsizes) != 1
        or table.target_shapes != ((),)
        or table.num_controls != 0
        or table.selection_bitsizes != (width, width)
    ):
        raise ValueError(
            "Explicit entry needs one scalar QROM angle target and two indices"
        )
    precision = table.target_bitsizes[0]
    if type(width) is not int or type(precision) is not int or width == 0:
        signature = entry_signature(width)
        return ProgramRecord((FunctionRecord("main", (), registers=tuple(signature)),))
    if precision < 1:
        raise ValueError("Explicit entry precision must be positive")
    lookup = qrom_program(table)
    builder = ComponentProgramBuilder(entry_signature(width))
    state = dict(builder.inputs)
    target = builder.allocate(precision)
    values = builder.call_program(
        lookup, {"selection0": state["i"], "selection1": state["j"], "target0_": target}
    )
    loaded = values["target0_"]
    assert isinstance(loaded, RegisterBinding)
    bits = builder.split(loaded)
    pairs = [(lookup, lookup)]
    for k in range(precision):
        angle = 2 * pi * 2 ** -(k + 1)

        def rotation(value: float) -> ProgramRecord:
            return ProgramRecord(
                (
                    bit_gate_function(
                        "controlled_ry",
                        tuple(Signature.build(ctrl=1, q=1)),
                        CircuitIR(2, (Instruction("cry", (0, 1), {"theta": value}),)),
                    ),
                )
            )

        forward, inverse = rotation(angle), rotation(-angle)
        pairs.append((forward, inverse))
        returned = builder.call_program(
            forward, {"ctrl": bits[k], "q": state["q"]}, entry=forward.functions[0].name
        )
        bits[k], state["q"] = returned["ctrl"], returned["q"]
    target = builder.join(bits)
    values = builder.call_program(
        lookup,
        {
            "selection0": values["selection0"],
            "selection1": values["selection1"],
            "target0_": target,
        },
    )
    unloaded = values["target0_"]
    assert isinstance(unloaded, RegisterBinding)
    builder.free(unloaded)
    state.update(i=values["selection0"], j=values["selection1"])
    record = builder.finalize(state)
    return adjoint_component_calls(record, tuple(pairs)) if adjoint else record

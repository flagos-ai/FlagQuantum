# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 basic_gates/diag_gates.ipynb.

"""Native bindings for the three original diagonal-gate examples."""

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler._component_trace.resource_generalization import (
    CliffordCall,
    PhaseCall,
)
from flagquantum.compiler.registers.primitives.pauli_gate import (
    controlled_pauli_with_controls_ir,
    pauli_gate_ir,
)
from flagquantum.core.ir import Instruction
from flagquantum.core.ir.program.signatures import Signature

DiagonalExample = tuple[str, Signature, FieldOperation, CliffordCall | PhaseCall]


def diagonal_gate_examples() -> tuple[DiagonalExample, ...]:
    """Bind the exact original signatures, native atomic gates and logical keys."""
    single = Signature.build(q=1)
    return (
        ("Z", single, FieldOperation(1, pauli_gate_ir("z", 0)), CliffordCall("z", 1)),
        ("S", single, FieldOperation(1, (Instruction("s", (0,)),)), PhaseCall("s")),
        (
            "CZ",
            Signature.build(q1=1, q2=1),
            FieldOperation(2, controlled_pauli_with_controls_ir("z", 0, 1)),
            CliffordCall("cz", 2),
        ),
    )

# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Source calls follow locked Qualtran v0.7.0 chemistry/sparse/select_bloq.py.

"""Bind the available five-call generic sparse SELECT control graph."""

from dataclasses import replace
from typing import Literal

from ....core.ir import CircuitIR, Instruction
from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.registers import Register
from ....core.ir.program.signatures import Signature
from ..component_control import (
    ControlAdder,
    _stored_control_adder,
    control_component_program,
)
from ..component_definition_match import matches_component_definition
from ..component_gate_function import bit_gate_function
from ..component_graph import _namespace_helpers
from ..primitives.pauli_rotation import pauli_power_ir
from .majorana_component import majorana_program
from .majorana_control_component import controlled_majorana_program
from .sparse_chemistry_select_component import (
    SPARSE_SELECT_STAGES,
    sparse_select_component_pair,
)


def generic_sparse_select_program(
    num_spin_orb: int,
    *,
    control_value: int | None = None,
    adjoint: bool = False,
    cy_profile: Literal["cirq", "scalar"] = "cirq",
) -> ProgramRecord:
    """Retain generic control of the actual forward or adjoint sparse calls.

    The new scalar control precedes the original MSB-first THRU ports. The
    original specialized control, when present, remains a bystander. Four
    independently controlled Majorana graphs and one atomic controlled S/Sdg
    leaf replace the premature whole-root rejection. Controlled phase reuses
    the shared exact Pauli-power realization. Missing source Controlled(And)
    bodies still prevent whole-root physical binding.
    """
    record = sparse_select_component_pair(num_spin_orb, control_value=control_value)[
        int(adjoint)
    ]
    signature = Signature(record.functions[0].registers or ())
    functions = {function.name: function for function in record.functions}
    definitions = list(record.functions)
    adders: dict[str, ControlAdder] = {}
    candidates: list[tuple[ProgramRecord, ProgramRecord]] = []
    for orbital, spin, pauli, flagged in SPARSE_SELECT_STAGES:
        selection = (signature.get_left(orbital), signature.get_left(spin))
        control = Register("control", quantum_bit_type()) if flagged else None
        candidates.append(
            (
                majorana_program(selection, pauli, control=control, adjoint=adjoint),
                controlled_majorana_program(
                    selection,
                    pauli,
                    control=control,
                    adjoint=adjoint,
                    cy_profile=cy_profile,
                ),
            )
        )
    operations = record.functions[0].blocks[0].operations
    for index, operation in enumerate(operations):
        name = operation.callee
        assert name is not None
        module: ProgramRecord | None = None
        for template, controlled in candidates:
            if matches_component_definition(record, name, template):
                module = controlled
                break
        if module is None:
            # The source has no composite body for controlled S/Sdg, but its
            # atomic SDK action is the existing exact controlled phase.
            phase_index = len(operations) - 1 if adjoint else 0
            if index != phase_index or len(operations) != 5:
                raise ValueError("Sparse control encountered an unknown source child")
            child = functions[name]
            assert child.registers is not None
            template = ProgramRecord(
                (
                    bit_gate_function(
                        "phase",
                        tuple(Signature.build(q=1)),
                        CircuitIR(1, (Instruction("sdg" if adjoint else "s", (0,)),)),
                    ),
                )
            )
            if not matches_component_definition(record, name, template):
                raise ValueError("Sparse control phase differs from its source owner")
            module = ProgramRecord(
                (
                    bit_gate_function(
                        "main",
                        tuple(Signature.build(ctrl=1, q=1)),
                        CircuitIR(
                            2,
                            pauli_power_ir("z", 1, -0.5 if adjoint else 0.5, control=0),
                        ),
                    ),
                )
            )
        imported = _namespace_helpers(module, f"__fq_sparse_generic_{index}")
        definitions.extend(imported.functions)
        adders[name] = _stored_control_adder(imported.functions[0])
    return control_component_program(
        replace(record, functions=tuple(definitions)), adders
    )

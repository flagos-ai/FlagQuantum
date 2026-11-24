# Copyright 2023 Google LLC
# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Control selection follows Qualtran v0.7.0 ControlledViaAnd and CtrlSpec.from_cirq_cv.

"""Bridge named positive-control components to the existing predicate owner."""

from collections.abc import Callable
from typing import Any

import numpy as np

from ...compiler.formats.legacy_component_arguments import ControlSpecification
from ...compiler.formats.legacy_controlled_factories import controlled_via_and_factories
from ...compiler.registers.component_builder import ComponentProgramBuilder
from ...compiler.registers.component_control import (
    ControlAdder,
    _stored_control_adder,
    quantum_control_signature,
)
from ...compiler.registers.component_graph import _collections, _elements
from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.register_types import quantum_bit_type
from ...core.ir.program.signatures import Signature


def named_positive_control(
    child: FunctionRecord, positive: ProgramRecord
) -> tuple[ProgramRecord, ControlAdder]:
    """Retain positive specialization while adapting its ordered port grouping."""
    assert child.registers is not None
    signature = quantum_control_signature(
        Signature(child.registers), ((quantum_bit_type(), ()),)
    )
    root = positive.functions[0]
    assert root.registers is not None
    ports = Signature(root.registers)
    builder = ComponentProgramBuilder(signature, name=child.name + "_positive")
    inputs = _collections(ports.lefts(), _elements(builder.inputs.values()))
    outputs = builder.call_program(positive, inputs, entry=root.name)
    program = builder.finalize(
        _collections(signature.rights(), _elements(outputs.values()))
    )
    return program, _stored_control_adder(program.functions[0])


def named_predicate_control(
    name: str,
    child: FunctionRecord,
    values: Any,
    positive: Callable[[FunctionRecord], tuple[ProgramRecord, ControlAdder]],
) -> FunctionRecord | ProgramRecord:
    """Preserve the locked single-conjunction quantum control specification."""
    clauses = tuple(values.expand())
    if len(clauses) != 1:
        raise ValueError(
            "CtrlSpec currently only supports converting SumOfProduct "
            "representation with a single AND clause."
        )
    if not clauses[0]:
        raise ValueError(
            "Sum of bits across the default QBit control shape should match zero controls"
        )
    bits = np.asarray(clauses[0], dtype=int)
    if len(bits) == 1:
        bits = bits.reshape(())
    specification = ControlSpecification(((quantum_bit_type(), bits),))
    factory = controlled_via_and_factories(control_system=positive)[
        "qualtran.bloqs.mcmt.controlled_via_and.ControlledViaAnd"
    ]
    return factory(name, None, {"subbloq": child, "ctrl_spec": specification})

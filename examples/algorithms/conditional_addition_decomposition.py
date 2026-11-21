# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 arithmetic/controlled_add_or_subtract.py.

"""Present full conditional arithmetic through shared finite call wiring."""

from collections.abc import Mapping

from examples.algorithms.controlled_word_decomposition import (
    controlled_complement_program,
)
from flagquantum.compiler.registers.arithmetic.conditional_addition_component import (
    conditional_addition_program,
)
from flagquantum.compiler.registers.arithmetic.conditional_addition_graph import (
    conditional_addition_graph,
)
from flagquantum.core.ir.program import FunctionRecord, ProgramRecord
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.drawer.component_score import component_register_score
from flagquantum.drawer.partition_symbols import partition_wire_symbol
from flagquantum.drawer.register_score_data import MusicalScoreData, register_score_data


def conditional_addition_catalog_program(
    operation: str, ports: Signature, parameters: Mapping[str, object]
) -> ProgramRecord | None:
    """Reuse concrete owners and preserve source-success symbolic child calls."""
    if operation != "ControlledAddOrSubtract":
        return None
    polarity = parameters["add_when_ctrl_is_on"]
    if type(polarity) is not bool:
        raise ValueError("Conditional arithmetic catalog requires a boolean polarity")
    a, b = ports[1].dtype, ports[2].dtype
    if type(a.num_qubits) is int and type(b.num_qubits) is int:
        return conditional_addition_program(a, b, add_when_control_is_on=polarity)
    addition = ProgramRecord(
        (
            FunctionRecord(
                "Add", (), registers=tuple(Signature.build_from_dtypes(a=a, b=b))
            ),
        )
    )
    return conditional_addition_graph(
        addition, controlled_complement_program(b), add_when_control_is_on=polarity
    )


def conditional_addition_score(
    operation: str, ports: Signature, parameters: Mapping[str, object]
) -> MusicalScoreData | None:
    """Draw native calls while retaining the original controlled-title failure."""
    record = conditional_addition_catalog_program(operation, ports, parameters)
    if record is None:
        return None
    # Frozen source drawings reject Controlled(BitwiseNot)'s TextBox title.
    return register_score_data(
        component_register_score(record, record.functions[0].name),
        register_wire_symbol=partition_wire_symbol,
    )

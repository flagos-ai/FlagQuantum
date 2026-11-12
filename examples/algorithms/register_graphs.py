# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Examples adapted from Qualtran v0.7.0 graphviz.ipynb and for_testing/with_decomposition.py.
# https://github.com/quantumlib/Qualtran/tree/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran

"""Compose native register graph, signature and classical-value demonstrations."""

from dataclasses import replace

import numpy as np

from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_gate_function import bit_gate_function
from flagquantum.compiler.registers.component_graph import iter_component_calls
from flagquantum.compiler.registers.control.component_predicate import (
    control_predicate_program,
)
from flagquantum.compiler.registers.control.control_predicate import ControlRegister
from flagquantum.compiler.registers.control.multi_control import multi_and_stages
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import FunctionRecord, ProgramRecord
from flagquantum.core.ir.program.register_bindings import RegisterBinding
from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.drawer.classical_component_graph import (
    export_classical_component_graph_dot,
)
from flagquantum.drawer.component_score import component_register_score
from flagquantum.drawer.register_graph import (
    RegisterConnection,
    _export_register_graph_dot,
)
from flagquantum.drawer.register_graph_table import (
    pretty_connection_attributes,
    signature_instance_table,
)
from flagquantum.drawer.register_score import RegisterStep
from flagquantum.simulation.classical_and import and_classical_values
from flagquantum.simulation.component_classical_state import ComponentClassicalState


def parallel_program() -> ProgramRecord:
    """Split a three-bit word, apply three X atoms in parallel, then join."""
    bit = RegisterType("bit", 1)
    atom = bit_gate_function(
        "atom", (Register("q", bit),), CircuitIR(1, (Instruction("x", (0,)),))
    )
    builder = ComponentProgramBuilder(
        Signature((Register("reg", RegisterType("opaque", 3)),)), (atom,)
    )
    word = builder.inputs["reg"]
    assert isinstance(word, RegisterBinding)
    outputs = []
    for binding in builder.split(word):
        result = builder.call_dict("atom", {"q": binding})["q"]
        assert isinstance(result, RegisterBinding)
        outputs.append(result)
    return builder.finalize({"reg": builder.join(outputs)})


def multi_and_program() -> ProgramRecord:
    """Retain the source notebook's four controls and three owned prefix outputs."""
    bit = RegisterType("bit", 1)
    predicate = control_predicate_program(
        (Register("ctrl", bit, (2,)),), (ControlRegister(1, (1, 1)),), name="and"
    )
    builder = ComponentProgramBuilder(
        Signature(
            (
                Register("ctrl", bit, (4,)),
                Register("junk", bit, (2,), RegisterSide.RIGHT),
                Register("target", bit, side=RegisterSide.RIGHT),
            )
        ),
        predicate.functions,
    )
    wires = dict(enumerate(np.asarray(builder.inputs["ctrl"])))
    for left, right, target, _ in multi_and_stages((1, 1, 1, 1)):
        result = builder.call_dict(
            "and", {"ctrl": np.array([wires[left], wires[right]], dtype=object)}
        )
        ctrl = np.asarray(result["ctrl"])
        wires[left], wires[right], wires[target] = ctrl[0], ctrl[1], result["target"]
    return builder.finalize(
        {
            "ctrl": np.array([wires[i] for i in range(4)], dtype=object),
            "junk": np.array([wires[4], wires[5]], dtype=object),
            "target": wires[6],
        }
    )


def classical_predicate_dot() -> str:
    """Simulation computes values; Drawer consumes the completed SSA mapping."""
    record = multi_and_program()

    def action(function: FunctionRecord, values: dict[str, object]) -> object:
        assert function.name == "and"
        return and_classical_values(np.asarray(values["ctrl"]))

    state = ComponentClassicalState(
        record,
        tuple(i for i, _, _ in iter_component_calls(record)),
        {"ctrl": np.array([1, 1, 0, 1])},
        action,
    )
    state.simulate()
    return export_classical_component_graph_dot(record, state.assignments)


def composite_dot(*, pretty: bool = False) -> str:
    """Draw real split, atom and join calls without executing the program."""

    signatures: list[Signature] = []

    def style(function: FunctionRecord, step: RegisterStep) -> RegisterStep:
        assert function.registers is not None
        signatures.append(Signature(function.registers))
        if len(step.inputs) == 1 and len(step.outputs) == 3:
            return replace(step, name="Split")
        if len(step.inputs) == 3 and len(step.outputs) == 1:
            return replace(step, name="Join")
        return replace(step, name="Atom")

    def attributes(edge: RegisterConnection) -> str:
        return pretty_connection_attributes(edge) if pretty else f'label="{edge.width}"'

    return _export_register_graph_dot(
        component_register_score(parallel_program(), style=style),
        attributes,
        table=lambda ordinal, step, ports: signature_instance_table(
            signatures[ordinal],
            step,
            ports,
            pretty=pretty,
            reshaping=ordinal in (0, 4),
        ),
    )

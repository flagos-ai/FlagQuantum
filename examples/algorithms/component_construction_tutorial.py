# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 _infra/composite_bloq.ipynb.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/_infra/composite_bloq.ipynb

"""Build and inspect native component graphs without foreign runtime objects."""

from dataclasses import dataclass, replace

import numpy as np

from examples.algorithms.register_graphs import parallel_program
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_gate_function import bit_gate_function
from flagquantum.compiler.registers.component_graph import iter_component_calls
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.signatures import Signature


def two_cnot_program(*, crossed: bool) -> ProgramRecord:
    """Keep the source's returned order after two sequential CNOT calls."""
    gate = bit_gate_function(
        "CNOT",
        tuple(Signature.build(ctrl=1, target=1)),
        CircuitIR(2, (Instruction("cx", (0, 1)),)),
    )
    builder = ComponentProgramBuilder(Signature.build(q0=1, q1=1), (gate,))
    q0, q1 = builder.call_tuple(
        "CNOT", {"ctrl": builder.inputs["q0"], "target": builder.inputs["q1"]}
    )
    q0, q1 = builder.call_tuple(
        "CNOT", {"ctrl": q1 if crossed else q0, "target": q0 if crossed else q1}
    )
    return builder.finalize({"q0": q0, "q1": q1})


def compose_parallel(
    *, count: int = 2, inline_at: frozenset[int] = frozenset()
) -> ProgramRecord:
    """Insert full children or their immediate call sequences while building."""
    parallel = parallel_program()
    child = replace(parallel.functions[0], name="parallel")
    builder = ComponentProgramBuilder(
        Signature.build(stuff=3), (child, *parallel.functions[1:])
    )
    stuff = builder.inputs["stuff"]
    for index in range(count):
        bindings = {"reg": stuff}
        outputs = (
            builder.inline_calls(parallel, bindings)
            if index in inline_at
            else builder.call_dict("parallel", bindings)
        )
        stuff = outputs["reg"]
    return builder.finalize({"stuff": stuff})


def wrap_program(record: ProgramRecord, *, name: str) -> ProgramRecord:
    """Keep the supplied decomposition inside one immediate child call."""
    child = replace(record.functions[0], name=name)
    signature = Signature(child.registers or ())
    builder = ComponentProgramBuilder(signature, (child, *record.functions[1:]))
    return builder.finalize(builder.call_dict(name, builder.inputs))


def labelled_connections(
    record: ProgramRecord, *, offset: int = 0
) -> tuple[tuple[str, str], ...]:
    """Show connection identity with explicit occurrence labels for copied graphs."""
    root = record.functions[0]
    block = root.blocks[0]
    producers = {value: f"input[{i}]" for i, value in enumerate(block.arguments)}
    connections = []
    for position, inputs, outputs in iter_component_calls(record):
        for name, bindings in inputs.items():
            for index, value in enumerate(np.asarray(bindings).flat):
                connections.append(
                    (producers[value.value], f"{offset + position}.{name}[{index}]")
                )
        for output in outputs:
            for value in np.asarray(output).flat:
                producers[value.value] = (
                    f"{offset + position}.{value.register.name}{value.index}"
                )
    assert block.return_values is not None
    connections.extend(
        (producers[value], f"output[{i}]")
        for i, value in enumerate(block.return_values)
    )
    return tuple(connections)


@dataclass(frozen=True)
class BoundComponentExample:
    """An explicit bound example operation accepted by Circuit.append."""

    record: ProgramRecord
    inputs: tuple[tuple[int, ...], ...]

    def append_to(self, circuit: Circuit) -> Circuit:
        bound = bind_register_program(self.record, self.inputs)
        return circuit.compose(bound.circuit)

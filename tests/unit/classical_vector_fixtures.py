"""Locked-source example data and real native bodies for cross-check scenarios.

Reference objects stay in this test boundary. No collected test module is
imported; finite-call and private-work construction have separate fixture owners.
"""

from dataclasses import replace
from functools import cache
from typing import Any

import numpy as np

from flagquantum.compiler.registers.arithmetic.addition import add_ir
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.core.ir.program.register_bindings import RegisterBinding
from flagquantum.core.ir.program.registers import Register, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.classical_addition import register_add_classical_values
from flagquantum.simulation.classical_controlled_bits import (
    cnot_classical_values,
    toffoli_classical_values,
)
from flagquantum.simulation.classical_pauli_values import x_classical_value
from tests.unit.register_binding_fixtures import allocated_circuit

KINDS = ("xgate", "cnot", "toffoli", "add")


@cache
def example(kind: str) -> tuple[ProgramRecord, Any, CircuitIR]:
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic import Add
    from qualtran.bloqs.basic_gates import CNOT, Toffoli, XGate

    registers: tuple[Register, ...]
    bit = RegisterType("bit", 1)
    if kind == "add":
        registers = (
            Register("a", RegisterType("uint", 5)),
            Register("b", RegisterType("uint", 5)),
        )
        source = Add(QUInt(5))
        circuit = CircuitIR(
            14,
            add_ir(
                tuple(range(5)),
                tuple(range(5, 10)),
                tuple(range(10, 14)),
                tuple(range(4)),
            ),
        )
    elif kind == "toffoli":
        registers = (Register("ctrl", bit, (2,)), Register("target", bit))
        source = Toffoli()
        circuit = CircuitIR(3, (Instruction("ccx", (0, 1, 2)),))
    elif kind == "cnot":
        registers = (Register("ctrl", bit), Register("target", bit))
        source = CNOT()
        circuit = CircuitIR(2, (Instruction("cx", (0, 1)),))
    else:
        assert kind == "xgate"
        registers = (Register("q", bit),)
        source = XGate()
        circuit = CircuitIR(1, (Instruction("x", (0,)),))
    signature = Signature(registers)
    data_width = sum(int(r.total_bits()) for r in registers)
    private = allocated_circuit(circuit, data_width)
    definitions = tuple(
        replace(f, name="body") if f.name == "main" else f for f in private.functions
    )
    builder = ComponentProgramBuilder(signature, definitions)
    bits: list[RegisterBinding] = []
    for register in registers:
        for word in np.asarray(builder.inputs[register.name]).flat:
            bits.extend(builder.split(word))
    result = builder.call_dict("body", {"data": builder.join(bits)})["data"]
    assert isinstance(result, RegisterBinding)
    returned = builder.split(result)
    outputs = {}
    position = 0
    for register in registers:
        words = np.empty(register.shape, dtype=object)
        for index in register.all_idxs():
            width = int(register.bitsize)
            words[index] = builder.join(
                returned[position : position + width], register.dtype
            )
            position += width
        outputs[register.name] = words if register.shape else words[()]
    record = builder.finalize(outputs)
    assert verify_program_record(record) == ()
    assert [(r.name, r.shape, r.bitsize) for r in registers] == [
        (r.name, r.shape, r.bitsize) for r in source.signature
    ]
    return record, source, circuit


def evaluate(kind: str, values: Any) -> dict[str, Any]:
    if kind == "add":
        a, b = register_add_classical_values(values["a"], values["b"], 5)
        return {"a": a, "b": b}
    if kind == "toffoli":
        ctrl, target = toffoli_classical_values(values["ctrl"], values["target"])
        return {"ctrl": ctrl, "target": target}
    if kind == "cnot":
        control, target = cnot_classical_values(values["ctrl"], values["target"])
        return {"ctrl": control, "target": target}
    return {"q": x_classical_value(values["q"])}


def original_inputs(kind: str) -> dict[str, Any]:
    if kind == "add":
        return {"a": 2, "b": 3}
    if kind == "toffoli":
        return {"ctrl": np.array([1, 1]), "target": 0}
    if kind == "cnot":
        return {"ctrl": 1, "target": 0}
    return {"q": 0}

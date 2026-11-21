"""Cycle wiring binds supplied child implementations without changing its caller."""

from functools import partial

import pytest

from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_gate_function import bit_gate_function
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.compiler.registers.sorting.permutation_graph import (
    permutation_cycle_graph,
)
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.register_bindings import RegisterBinding
from flagquantum.core.ir.program.register_types import (
    bounded_unsigned_register_type,
    quantum_bit_type,
    unsigned_register_type,
)
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature

pytestmark = pytest.mark.unit


def _definition(record: ProgramRecord, value: int) -> ProgramRecord:
    return record


def test_supplied_xor_replacement_reaches_normal_binding() -> None:
    word, bit = unsigned_register_type(2), quantum_bit_type()
    predicate_builder = ComponentProgramBuilder(
        Signature.build_from_dtypes(x=word, target=bit)
    )
    predicate = predicate_builder.finalize(predicate_builder.inputs)
    x_gate = bit_gate_function(
        "replacement_x", (Register("q", bit),), CircuitIR(1, (Instruction("x", (0,)),))
    )
    for flip in (False, True):
        builder = ComponentProgramBuilder(
            Signature.build_from_dtypes(ctrl=bit, x=word), (x_gate,)
        )
        value = builder.inputs["x"]
        assert isinstance(value, RegisterBinding)
        if flip:
            bits = builder.split(value)
            bits[0] = builder.call_dict(x_gate.name, {"q": bits[0]})["q"]
            value = builder.join(bits, word)
        xor = builder.finalize({"ctrl": builder.inputs["ctrl"], "x": value})
        graph = permutation_cycle_graph(
            bounded_unsigned_register_type(2, 4),
            (0,),
            partial(_definition, predicate),
            partial(_definition, xor),
        )
        bound = bind_register_program(graph, ((0, 1),), clean=(2,))
        assert tuple(
            (gate.name, gate.wires) for gate in bound.circuit.instructions
        ) == ((("x", (0,)),) if flip else ())

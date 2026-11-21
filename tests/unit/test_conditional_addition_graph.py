"""Conditional call wiring uses the supplied arithmetic implementation at binding."""

import pytest

from flagquantum.compiler.registers.arithmetic.conditional_addition_graph import (
    conditional_addition_graph,
)
from flagquantum.compiler.registers.bits.bitwise_component import bitwise_word_program
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program.register_types import (
    quantum_bit_type,
    unsigned_register_type,
)
from flagquantum.core.ir.program.signatures import Signature

pytestmark = pytest.mark.unit


def test_replaced_addition_changes_bound_target_without_changing_consumer() -> None:
    word, bit = unsigned_register_type(2), quantum_bit_type()
    control = ComponentProgramBuilder(Signature.build_from_dtypes(ctrl=bit, x=word))
    complement = control.finalize(control.inputs)
    for flip in (False, True):
        builder = ComponentProgramBuilder(Signature.build_from_dtypes(a=word, b=word))
        b = builder.inputs["b"]
        if flip:
            b = builder.call_program(
                bitwise_word_program(word, "xor_constant", pattern=1), {"x": b}
            )["x"]
        addition = builder.finalize({"a": builder.inputs["a"], "b": b})
        record = conditional_addition_graph(
            addition, complement, add_when_control_is_on=False
        )
        result = bind_register_program(record, ((0,), (1, 2), (3, 4)))
        assert tuple((g.name, g.wires) for g in result.circuit.instructions) == (
            (("x", (4,)),) if flip else ()
        )

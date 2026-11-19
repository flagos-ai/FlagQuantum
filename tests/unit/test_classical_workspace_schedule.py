"""Workspace helpers with carried CBits retain ordinary component scheduling."""

import pytest

from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_schedule import local_operation_order
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import (
    BlockRecord,
    FunctionRecord,
    Operation,
    ValueRef,
    verify_program_record,
)
from flagquantum.core.ir.program.register_types import (
    classical_bit_type,
    opaque_register_type,
    quantum_bit_type,
)
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.core.ir.program.signatures import Signature

pytestmark = pytest.mark.unit


def test_classical_passthrough_allocation_free_helpers_and_zero_quantum_edges() -> None:
    control = Register("ctrl", classical_bit_type())
    word = Register("work", opaque_register_type(1), side=RegisterSide.RIGHT)
    c, w = (ValueRef("entry", i, "register") for i in range(2))
    allocation = FunctionRecord(
        "allocate",
        (
            BlockRecord(
                "entry",
                (c,),
                operations=(
                    Operation(
                        "allocate_register",
                        results=(w,),
                        register_type=word.dtype,
                        workspace_dirty=False,
                    ),
                ),
                return_values=(c, w),
            ),
        ),
        registers=(control, word),
    )
    release = FunctionRecord(
        "release",
        (
            BlockRecord(
                "entry",
                (c, w),
                operations=(Operation("free_register", (w,), workspace_dirty=False),),
                return_values=(c,),
            ),
        ),
        registers=(control, word.adjoint()),
    )
    builder = ComponentProgramBuilder(
        Signature((control, Register("q", quantum_bit_type()))), (allocation, release)
    )
    allocated = builder.call_dict("allocate", {"ctrl": builder.inputs["ctrl"]})
    returned = builder.call_dict("release", allocated)
    module = builder.finalize({"ctrl": returned["ctrl"], "q": builder.inputs["q"]})
    assert verify_program_record(module) == ()
    functions = {f.name: f for f in module.functions}
    assert local_operation_order(module.functions[0], functions) == (0, 1)
    assert local_operation_order(
        module.functions[0], functions, logical_graph=True
    ) == (0, 1)
    for value in (0, 1):
        bound = bind_register_program(
            module, ((), (0,)), classical_inputs=(value, None), clean=(1,)
        )
        assert bound.outputs == ((), (0,)) and bound.classical_outputs == ((0,), ())
        assert [e.kind for e in bound.events] == ["allocate", "free"]
        assert [g.name for g in bound.circuit.instructions] == ["classical_xor"]

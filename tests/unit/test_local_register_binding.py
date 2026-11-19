"""Local component binding preserves actual work reuse and pool ownership."""

from dataclasses import replace

import pytest

from flagquantum.compiler.registers._local_register_binding import (
    _bind_local_register_program,
)
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_gate_function import bit_gate_function
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program.register_bindings import RegisterBinding
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("dirty", [False, True])
@pytest.mark.parametrize("reverse_storage", [False, True])
def test_local_binding_reuses_work_and_rejects_forward_references(
    dirty: bool, reverse_storage: bool
) -> None:
    bit = quantum_bit_type()
    cx = bit_gate_function(
        "cx",
        (Register("a", bit), Register("b", bit)),
        CircuitIR(2, (Instruction("cx", (0, 1)),)),
    )
    builder = ComponentProgramBuilder(Signature((Register("data", bit),)), (cx,))
    data = builder.inputs["data"]
    for _ in range(3):
        work = builder.allocate(dtype=bit, dirty=dirty)
        for _ in range(2):
            data, returned_work = builder.call_tuple("cx", {"a": data, "b": work})
            assert isinstance(returned_work, RegisterBinding)
            work = returned_work
        assert isinstance(work, RegisterBinding)
        builder.free(work, dirty=dirty)
    program = builder.finalize({"data": data})
    if reverse_storage:
        program = replace(
            program,
            functions=tuple(
                replace(
                    function,
                    blocks=(
                        replace(
                            function.blocks[0],
                            operations=tuple(reversed(function.blocks[0].operations)),
                        ),
                    ),
                )
                for function in program.functions
            ),
        )
        with pytest.raises(ValueError, match="invalid local register Program"):
            _bind_local_register_program(program, "main")
        return
    actual = _bind_local_register_program(program, "main")
    expected = bind_register_program(
        program,
        ((0,),),
        clean=() if dirty else (1,),
        borrowed=(1,) if dirty else (),
        input_ownership=((0, False),),
    )
    assert actual == expected
    assert actual.circuit.n_wires == 2
    assert len(actual.events) == 6


def test_local_binding_rejects_invalid_entry() -> None:
    builder = ComponentProgramBuilder(
        Signature((Register("data", quantum_bit_type()),))
    )
    program = builder.finalize(builder.inputs)
    with pytest.raises(ValueError, match="unknown Program entry"):
        _bind_local_register_program(program, "missing")
    empty = replace(program, functions=(replace(program.functions[0], blocks=()),))
    with pytest.raises(ValueError, match="finite function bodies"):
        _bind_local_register_program(empty, "main")


def test_local_binding_transfers_reused_input_as_output() -> None:
    from flagquantum.core.ir.program.registers import RegisterSide

    bit = quantum_bit_type()
    builder = ComponentProgramBuilder(
        Signature(
            (
                Register("old", bit, side=RegisterSide.LEFT),
                Register("new", bit, side=RegisterSide.RIGHT),
            )
        )
    )
    old = builder.inputs["old"]
    assert isinstance(old, RegisterBinding)
    builder.free(old)
    new = builder.allocate(dtype=bit)
    bound = _bind_local_register_program(builder.finalize({"new": new}), "main")
    assert bound.outputs == ((0,),)
    assert bound.circuit.n_wires == 1
    assert [(event.kind, event.wires) for event in bound.events] == [
        ("free", (0,)),
        ("allocate", (0,)),
    ]


def test_local_empty_component_retains_scalar_carrier() -> None:
    builder = ComponentProgramBuilder(Signature(()))
    bound = _bind_local_register_program(builder.finalize({}), "main")
    assert bound.circuit.n_wires == 1
    assert bound.circuit.instructions == ()
    assert bound.outputs == ()
    assert bound.events == ()

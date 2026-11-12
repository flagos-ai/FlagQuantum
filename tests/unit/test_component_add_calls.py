"""Executable Add call signatures and the pinned zero-input failure boundary."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from itertools import product

import numpy as np
import pytest
import torch

from flagquantum.compiler.registers.arithmetic.addition import add_adjoint_ir, add_ir
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_expansion import expand_component_calls
from flagquantum.compiler.registers.component_graph import copy_component_program
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program import FunctionRecord, verify_program_record
from flagquantum.core.ir.program.register_bindings import RegisterBinding
from flagquantum.core.ir.program.registers import Register, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.measurement_checks import forced_branch
from tests.unit.register_binding_fixtures import allocated_circuit
from tests.unit.test_component_call_boundaries import original
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def executable_add(adjoint: bool = False) -> tuple[FunctionRecord, ...]:
    """Expose exact a/b word ports around the existing measured Add body."""
    emitter = add_adjoint_ir if adjoint else add_ir
    gates = emitter((0, 1), (2, 3, 4), (5, 6), (0, 1))
    private = allocated_circuit(CircuitIR(7, gates), 5)
    definitions = tuple(
        replace(function, name="private_add") if function.name == "main" else function
        for function in private.functions
    )
    registers = (
        Register("a", RegisterType("uint", 2)),
        Register("b", RegisterType("uint", 3)),
    )
    builder = ComponentProgramBuilder(Signature(registers), definitions, name="add")
    bits: list[RegisterBinding] = []
    for register in registers:
        word = builder.inputs[register.name]
        assert isinstance(word, RegisterBinding)
        bits.extend(builder.split(word))
    packed = builder.join(bits)
    result = builder.call_dict("private_add", {"data": packed})["data"]
    assert isinstance(result, RegisterBinding)
    returned = builder.split(result)
    outputs = {
        "a": builder.join(returned[:2], dtype=registers[0].dtype),
        "b": builder.join(returned[2:], dtype=registers[1].dtype),
    }
    module = builder.finalize(outputs)
    assert verify_program_record(module) == ()
    if not adjoint:
        return module.functions
    return tuple(
        replace(
            function,
            name=function.name + "_adjoint",
            blocks=tuple(
                replace(
                    block,
                    operations=tuple(
                        (
                            replace(op, callee=op.callee + "_adjoint")
                            if op.callee is not None
                            else op
                        )
                        for op in block.operations
                    ),
                )
                for block in function.blocks
            ),
        )
        for function in module.functions
    )


@pytest.mark.parametrize("rebuild", [False, True])
@pytest.mark.parametrize("expand", [False, True])
def test_two_actual_add_calls_keep_exact_signature_gates_all_branches_and_resources(
    rebuild: bool,
    expand: bool,
) -> None:
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic.addition import Add
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    registers = (
        Register("a", RegisterType("uint", 2)),
        Register("b", RegisterType("uint", 3)),
        Register("spectator", RegisterType("uint", 1)),
    )
    signature = Signature(registers)
    native = ComponentProgramBuilder(signature, executable_add())
    source = original(signature)
    add = Add(QUInt(2), QUInt(3))
    outputs = {name: native.inputs[name] for name in ("a", "b")}
    expected_outputs = {name: source.inputs[name] for name in ("a", "b")}
    for _ in range(2):
        outputs = native.call_dict("add", outputs)
        expected_outputs = source.builder.add_d(add, **expected_outputs)
        assert tuple(outputs) == tuple(expected_outputs) == ("a", "b")
    module = native.finalize({**outputs, "spectator": native.inputs["spectator"]})
    if rebuild:
        module = copy_component_program(module)
    if expand:
        module, _ = expand_component_calls(module)
    composite = source.builder.finalize(
        **expected_outputs, spectator=source.inputs["spectator"]
    )
    assert verify_program_record(module) == ()
    assert [len(op.operands) for op in native._operations] == [2, 2]
    assert [len(op.results) for op in native._operations] == [2, 2]
    bound = bind_register_program(module, ((0, 1), (2, 3, 4), (5,)), clean=(6, 7))
    assert bound.outputs == ((0, 1), (2, 3, 4), (5,))
    measurements = tuple(
        gate for gate in bound.circuit.instructions if gate.name == "measure"
    )
    assert len(measurements) == 4
    expected_gates = tuple(
        gate
        for start in (0, 2)
        for gate in reference_arithmetic_gates(
            add,
            {"a": (0, 1), "b": (2, 3, 4)},
            # The original generator reverses qalloc before assigning carry
            # significance. Bind its reversed allocation to native carry roles.
            (7, 6),
            {
                gate.wires[0]: gate.metadata["classical_bit"]
                for gate in measurements[start : start + 2]
            },
            cirq_style=True,
        )
    )
    assert_source(bound.circuit.instructions, expected_gates)
    assert Counter(gate.name for gate in bound.circuit.instructions) == {
        "cx": 40,
        "h": 12,
        "t": 8,
        "tdg": 8,
        "s": 4,
        "measure": 4,
        "cz": 4,
        "reset": 4,
    }
    model = get_cost_value(composite, QECGatesCost())
    assert model.total_t_count() == 16 and model.measurement == 4
    assert get_cost_value(composite, QubitCount()) == 8
    initial = torch.zeros(256, dtype=torch.complex128)
    phases = torch.tensor([(1j) ** (index % 4) / 8 for index in range(64)])
    initial[::4] = phases
    expected = torch.zeros_like(initial)
    for prefix in range(64):
        a, b, spectator = prefix >> 4, (prefix >> 1) & 7, prefix & 1
        target = (a << 4) | (((b + 2 * a) & 7) << 1) | spectator
        expected[target << 2] = initial[prefix << 2]
    for history in product((0, 1), repeat=4):
        result = forced_branch(initial, bound.circuit.instructions, history)
        torch.testing.assert_close(result, expected / 4, rtol=0, atol=1e-12)
        assert torch.count_nonzero(result.reshape(64, 4)[:, 1:]).item() == 0


def test_zero_left_add_constructs_in_source_but_fails_original_decomposition() -> None:
    import cirq
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic.addition import Add

    registers = (
        Register("a", RegisterType("uint", 0)),
        Register("b", RegisterType("uint", 2)),
    )
    source = original(Signature(registers))
    add = Add(QUInt(0), QUInt(2))
    values = source.inputs
    for _ in range(2):
        values = source.builder.add_d(add, **values)
        assert tuple(values) == ("a", "b")
    source.builder.finalize(**values)
    context = cirq.DecompositionContext(cirq.SimpleQubitManager())
    with pytest.raises(IndexError, match="index 0"):
        list(
            add.decompose_from_registers(
                context=context,
                a=np.array([], dtype=object),
                b=np.array(cirq.LineQubit.range(2)),
            )
        )
    with pytest.raises(ValueError, match="nonempty input"):
        add_ir((), (0, 1), (2,), (0,))

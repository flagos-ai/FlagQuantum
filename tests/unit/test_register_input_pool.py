"""Bind explicitly owned input lifetimes and reuse their physical wires."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from itertools import product

import numpy as np
import pytest
import torch

from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import Operation, ProgramRecord, ValueRef
from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.measurement_checks import forced_branch
from tests.unit.register_binding_fixtures import allocated_circuit
from tests.unit.test_register_program_lowering import packed_circuit

pytestmark = pytest.mark.unit


def release_then_allocate(width: int, dirty: bool, free_dirty: bool) -> ProgramRecord:
    builder = ComponentProgramBuilder(
        Signature(
            [
                Register(
                    "input", RegisterType("opaque", width), side=RegisterSide.LEFT
                ),
                Register("out", RegisterType("opaque", width), side=RegisterSide.RIGHT),
            ]
        )
    )
    reversed_input = builder.join(builder.split(builder.inputs["input"])[::-1])
    builder.free(reversed_input, dirty=free_dirty)
    output = builder.allocate(width, dirty=dirty)
    return builder.finalize({"out": output})


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("width", [1, 3])
@pytest.mark.parametrize("dirty,free_dirty", list(product((False, True), repeat=2)))
def test_preowned_input_matches_original_manager_reuse(
    width: int, dirty: bool, free_dirty: bool
) -> None:
    from tests.unit.test_register_input_pool_contract import (
        test_source_preowned_input_returns_to_manager_before_output_allocation,
    )

    test_source_preowned_input_returns_to_manager_before_output_allocation(
        width, dirty, free_dirty
    )
    inputs = (4, 2, 7)[:width]
    bound = bind_register_program(
        release_then_allocate(width, dirty, free_dirty),
        (inputs,),
        input_ownership=tuple((wire, dirty) for wire in inputs),
        allow_workspace_outputs=True,
    )
    assert bound.outputs == (inputs[::-1],)
    assert bound.circuit.instructions == ()
    assert [(event.kind, event.wires, event.dirty) for event in bound.events] == [
        ("free", inputs[::-1], free_dirty),
        ("allocate", inputs[::-1], dirty),
    ]
    assert not Counter(gate.name for gate in bound.circuit.instructions)
    with pytest.raises(ValueError, match="ownership transfer"):
        bind_register_program(
            release_then_allocate(width, dirty, free_dirty),
            (inputs,),
            input_ownership=tuple((wire, dirty) for wire in inputs),
        )


@pytest.mark.parametrize("dirty", [False, True])
def test_live_returned_adopted_input_needs_no_workspace_output_policy(
    dirty: bool,
) -> None:
    builder = ComponentProgramBuilder(
        Signature([Register("input", RegisterType("opaque", 2))])
    )
    record = builder.finalize(builder.inputs)
    bound = bind_register_program(
        record, ((3, 1),), input_ownership=((1, dirty), (3, dirty))
    )
    assert bound.outputs == ((3, 1),) and bound.events == ()
    assert bound.circuit.instructions == ()


@pytest.mark.parametrize(
    "ownership",
    [
        [(1, False)],
        ((1, False), (1, True)),
        ((1, 0),),
        ((1, np.bool_(False)),),
        ((1, False, 3),),
        ((2, False),),
        ((True, False),),
        ((1, False), [3, False]),
    ],
)
def test_invalid_input_ownership_is_rejected_before_binding(ownership: object) -> None:
    record = release_then_allocate(1, False, False)
    with pytest.raises(ValueError, match="input ownership"):
        bind_register_program(
            record, ((1,),), input_ownership=ownership, allow_workspace_outputs=True
        )
    bound = bind_register_program(
        record, ((1,),), input_ownership=((1, False),), allow_workspace_outputs=True
    )
    assert bound.outputs == ((1,),)


def test_partial_ownership_cannot_free_a_word_or_release_any_prefix() -> None:
    record = release_then_allocate(3, False, False)
    with pytest.raises(ValueError, match="caller input wires"):
        bind_register_program(
            record,
            ((3, 1, 2),),
            input_ownership=((3, False), (1, False)),
            allow_workspace_outputs=True,
        )
    result = bind_register_program(
        record,
        ((3, 1, 2),),
        input_ownership=((3, False), (1, False), (2, False)),
        allow_workspace_outputs=True,
    )
    assert result.events[0].wires == (2, 1, 3)
    assert result.events[1].wires == (2, 1, 3)


def test_mixed_adopted_and_allocated_words_return_to_origin_pools() -> None:
    builder = ComponentProgramBuilder(
        Signature(
            [
                Register("input", RegisterType("opaque", 2), side=RegisterSide.LEFT),
                Register("out", RegisterType("opaque", 2), side=RegisterSide.RIGHT),
            ]
        )
    )
    work = builder.allocate(1, dirty=True)
    input_bits = builder.split(builder.inputs["input"])
    mixed = builder.join([input_bits[0], *builder.split(work)])
    builder.free(mixed, dirty=False)
    builder.free(builder.join(input_bits[1:]), dirty=True)
    out = builder.allocate(2)
    record = builder.finalize({"out": out})
    bound = bind_register_program(
        record,
        ((2, 4),),
        clean=(7,),
        borrowed=(8,),
        input_ownership=((2, False), (4, True)),
        allow_workspace_outputs=True,
    )
    assert [(event.kind, event.wires) for event in bound.events] == [
        ("free", (4,)),
        ("allocate", (4,)),
        ("free", (2, 4)),
        ("allocate", (2, 7)),
    ]
    assert bound.events[-1].kind == "allocate"
    assert bound.outputs == ((2, 7),)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("dirty,free_dirty", list(product((False, True), repeat=2)))
def test_released_input_restores_complex_joint_state_and_exact_resource_counts(
    dirty: bool, free_dirty: bool
) -> None:
    import cirq
    from qualtran import BloqBuilder, QAny, Side
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature
    from qualtran.bloqs.basic_gates import CNOT
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    import flagquantum as fq
    from examples.algorithms.binary_field_arithmetic import FieldOperation

    gates = CircuitIR(2, (Instruction("cx", (0, 1)),) * 2)
    native_module = packed_circuit(gates, (1, 1))
    definitions = tuple(
        replace(function, name="cnot_pair") if function.name == "main" else function
        for function in native_module.functions
    )
    builder = ComponentProgramBuilder(
        Signature(
            [
                Register("input", RegisterType("opaque", 2), side=RegisterSide.LEFT),
                Register("out", RegisterType("opaque", 2), side=RegisterSide.RIGHT),
            ]
        ),
        definitions,
    )
    bits = builder.split(builder.inputs["input"])
    first, second = builder.call_tuple(
        "cnot_pair", {"word0": bits[0], "word1": bits[1]}
    )
    builder.free(builder.join([second, first]), dirty=free_dirty)
    out = builder.allocate(2, dirty=dirty)
    bound = bind_register_program(
        builder.finalize({"out": out}),
        ((0, 1),),
        input_ownership=((0, dirty), (1, dirty)),
        allow_workspace_outputs=True,
    )
    assert bound.outputs == ((1, 0),)
    assert [(event.kind, event.wires) for event in bound.events] == [
        ("free", (1, 0)),
        ("allocate", (1, 0)),
    ]
    assert Counter(gate.name for gate in bound.circuit.instructions) == {"cx": 2}

    manager = cirq.GreedyQubitManager("adopted", size=2, maximize_reuse=False)
    source_inputs = tuple(manager.qborrow(2) if dirty else manager.qalloc(2))
    source, arguments = BloqBuilder.from_signature(
        SourceSignature(
            [
                SourceRegister("input", QAny(2), side=Side.LEFT),
                SourceRegister("out", QAny(2), side=Side.RIGHT),
            ]
        )
    )
    source_bits = source.split(arguments["input"])
    first_source, second_source = source.add(
        CNOT(), ctrl=source_bits[0], target=source_bits[1]
    )
    first_source, second_source = source.add(
        CNOT(), ctrl=first_source, target=second_source
    )
    source.free(source.join([second_source, first_source]), dirty=free_dirty)
    source_out = source.allocate(dtype=QAny(2), dirty=dirty)
    composite = source.finalize(out=source_out)
    reference, source_outputs = composite.to_cirq_circuit_and_quregs(
        qubit_manager=manager, input=np.array(source_inputs)
    )
    assert tuple(source_outputs["out"]) == source_inputs[::-1]
    assert tuple(reference.all_operations()) == (cirq.CNOT(*source_inputs),) * 2
    model = get_cost_value(composite, QECGatesCost())
    assert model.clifford == 2 and model.total_t_count() == 0 and model.measurement == 0
    assert get_cost_value(composite, QubitCount()) == 2

    state = torch.complex(
        torch.arange(1, 9, dtype=torch.float64),
        torch.arange(8, dtype=torch.float64).remainder(5) - 2,
    )
    if not dirty:
        state[2:] = 0
    state /= state.norm()
    actual = (
        fq.Circuit(3, inputs=state, dtype=state.dtype)
        .append(FieldOperation(3, bound.circuit.instructions))
        .state()
        .flatten()
    )
    expected = (
        cirq.Simulator(dtype=np.complex128)
        .simulate(
            reference,
            initial_state=state.numpy(),
            qubit_order=(*source_inputs, cirq.NamedQubit("spectator")),
        )
        .final_state_vector
    )
    torch.testing.assert_close(actual, torch.as_tensor(expected), rtol=0, atol=2e-12)
    if not dirty:
        assert torch.count_nonzero(actual.reshape(4, 2)[1:]) == 0
    else:
        torch.testing.assert_close(actual, state, rtol=0, atol=2e-12)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("dirty", [False, True])
def test_adopted_input_release_after_measured_and_matches_both_source_branches(
    dirty: bool,
) -> None:
    from qualtran import BloqBuilder
    from qualtran import Signature as SourceSignature
    from qualtran.bloqs.mcmt.and_bloq import And
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    from flagquantum.compiler.registers.primitives.and_compute import (
        and_compute_ir,
        and_uncompute_ir,
    )
    from tests.unit.test_register_constant_addition import reference_arithmetic_gates
    from tests.unit.test_register_subtraction import assert_source

    gates = (*and_compute_ir(0, 1, 2), *and_uncompute_ir(0, 1, 2, 0))
    record = allocated_circuit(CircuitIR(3, gates), 2)
    main, leaf = record.functions
    body = main.blocks[0]
    assert body.return_values is not None
    output = ValueRef("entry", 10002, "register")
    dtype = RegisterType("opaque", 2)
    body = replace(
        body,
        operations=(
            *body.operations,
            Operation("free_register", body.return_values, workspace_dirty=dirty),
            Operation(
                "allocate_register",
                results=(output,),
                register_type=dtype,
                workspace_dirty=dirty,
            ),
        ),
        return_values=(output,),
    )
    main = replace(
        main,
        blocks=(body,),
        registers=(
            Register("data", dtype, side=RegisterSide.LEFT),
            Register("out", dtype, side=RegisterSide.RIGHT),
        ),
    )
    bound = bind_register_program(
        replace(record, functions=(main, leaf)),
        ((0, 1),),
        clean=(2,),
        input_ownership=((0, dirty), (1, dirty)),
        allow_workspace_outputs=True,
    )
    assert bound.outputs == ((0, 1),)
    assert Counter(gate.name for gate in bound.circuit.instructions) == Counter(
        gate.name for gate in gates
    )
    assert [(event.kind, event.wires) for event in bound.events] == [
        ("allocate", (2,)),
        ("free", (2,)),
        ("free", (0, 1)),
        ("allocate", (0, 1)),
    ]

    source, inputs = BloqBuilder.from_signature(SourceSignature.build(a=1, b=1))
    controls, target = source.add(And(), ctrl=np.array([inputs["a"], inputs["b"]]))
    controls = source.add(And().adjoint(), ctrl=controls, target=target)
    source_component = source.finalize(a=controls[0], b=controls[1])
    reference = reference_arithmetic_gates(
        source_component, {"a": (0,), "b": (1,)}, (2,), {2: 0}
    )
    assert_source(bound.circuit.instructions, reference)
    model = get_cost_value(source_component, QECGatesCost())
    assert model.total_t_count() == 4 and model.measurement == 1
    assert get_cost_value(source_component, QubitCount()) == 3

    state = torch.zeros(16, dtype=torch.complex128)
    for controls in range(4 if dirty else 1):
        for spectator in range(2):
            state[(controls << 2) | spectator] = complex(controls + 1, spectator - 2)
    state /= state.norm()
    for outcome in (0, 1):
        actual = forced_branch(state, bound.circuit.instructions, (outcome,))
        torch.testing.assert_close(
            actual, forced_branch(state, reference, (outcome,)), rtol=0, atol=2e-12
        )
        torch.testing.assert_close(actual, state / 2**0.5, rtol=0, atol=2e-12)
        assert torch.count_nonzero(actual.reshape(4, 2, 2)[:, 1, :]) == 0

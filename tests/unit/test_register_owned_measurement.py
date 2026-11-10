"""Execute destructive register measurement with source instrument evidence."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from itertools import product

import numpy as np
import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.register_binding import (
    BoundRegisterProgram,
    bind_register_program,
)
from flagquantum.compiler.registers.register_programs import lower_register_program
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.register_types import (
    classical_bit_type,
    quantum_bit_type,
)
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_program_classical_registers import identity
from tests.unit.test_program_owned_measurement import measured_function

pytestmark = pytest.mark.unit


def measured_array() -> ProgramRecord:
    cast = identity("cast", (Register("reg", classical_bit_type()),))
    builder = ComponentProgramBuilder(
        Signature(
            [
                Register("q", quantum_bit_type(), (2,), RegisterSide.LEFT),
                Register("spectator", quantum_bit_type()),
                Register("c", classical_bit_type(), (2,), RegisterSide.RIGHT),
            ]
        ),
        [measured_function("measurement"), cast],
    )
    outputs = []
    for q in np.asarray(builder.inputs["q"]).flat:
        c = builder.call_dict("measurement", {"q": q})["c"]
        for _ in range(2):
            c = builder.call_dict("cast", {"reg": c})["reg"]
        outputs.append(c)
    work = builder.allocate(dtype=quantum_bit_type())
    builder.free(work)
    return builder.finalize(
        {"spectator": builder.inputs["spectator"], "c": np.array(outputs[::-1])}
    )


def test_measurement_slots_survive_nested_calls_and_reordered_array_outputs() -> None:
    bound = bind_register_program(measured_array(), ((0,), (1,), (2,)), clean=(3,))
    assert bound.outputs == ((2,), (), ())
    slots = {
        gate.wires[0]: gate.metadata["classical_bit"]
        for gate in bound.circuit.instructions
    }
    assert bound.classical_outputs == ((), (slots[1],), (slots[0],))
    assert Counter(gate.name for gate in bound.circuit.instructions) == {"measure": 2}
    assert [(event.kind, event.wires) for event in bound.events] == [
        ("allocate", (3,)),
        ("free", (3,)),
    ]
    assert len(set(slots.values())) == 2
    assert all(gate.metadata["is_dynamic"] for gate in bound.circuit.instructions)


@pytest.mark.ftqc_reference
def test_all_complex_branches_match_original_tensor_instrument_and_work_zero() -> None:
    from qualtran import BloqBuilder, CBit, QBit, Side
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature
    from qualtran.bloqs.basic_gates import MeasureZ
    from qualtran.bloqs.bookkeeping.cast import Cast
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    original, inputs = BloqBuilder.from_signature(
        SourceSignature(
            [
                SourceRegister("q", QBit(), shape=(2,), side=Side.LEFT),
                SourceRegister("c", CBit(), shape=(2,), side=Side.RIGHT),
            ]
        )
    )
    returned = []
    for q in inputs["q"]:
        c = original.add(MeasureZ(), q=q)
        for _ in range(2):
            c = original.add(Cast(CBit(), CBit()), reg=c)
        returned.append(c)
    source = original.finalize(c=np.array(returned[::-1]))
    assert Counter(type(b.bloq).__name__ for b, _, _ in source.iter_bloqsoqs()) == {
        "MeasureZ": 2,
        "Cast": 4,
    }
    for left, right in product((0, 1), repeat=2):
        (result,) = source.call_classically(q=np.array([left, right]))
        np.testing.assert_array_equal(result, [right, left])
    model = get_cost_value(source, QECGatesCost())
    assert model.measurement == 2
    assert model.total_t_count() == 0

    bound = bind_register_program(measured_array(), ((0,), (1,), (2,)), clean=(3,))
    circuit = Circuit(4).append(FieldOperation(4, bound.circuit.instructions))
    gates = circuit.to_ir().instructions
    assert gates == bound.circuit.instructions
    emitted = Counter(gate.name for gate in gates)
    assert emitted == {"measure": 2}
    generator = torch.Generator().manual_seed(1084)
    state = torch.randn(8, dtype=torch.complex128, generator=generator)
    state /= state.norm()
    with_work = torch.zeros(16, dtype=torch.complex128)
    with_work[::2] = state
    copy, _ = MeasureZ().my_tensors({"q": "input"}, {"c": "output"})
    total_probability = 0.0
    for history in product((0, 1), repeat=2):
        outcomes = {gate.wires[0]: b for gate, b in zip(gates, history, strict=True)}
        branch = forced_branch(with_work, gates, history).reshape(2, 2, 2, 2)
        expected = np.einsum(
            "qa,rb,qrs->abs",
            copy.data[:, :, outcomes[0]],
            copy.data[:, :, outcomes[1]],
            state.numpy().reshape(2, 2, 2),
        )
        np.testing.assert_allclose(
            branch[:, :, :, 0].numpy(), expected, rtol=0, atol=2e-15
        )
        assert torch.count_nonzero(branch[:, :, :, 1]) == 0
        total_probability += float(branch.abs().square().sum())
        # Outcome one retains a projected one on the consumed physical wire.
        if outcomes[0]:
            assert torch.count_nonzero(branch[0]) == 0
            assert torch.count_nonzero(branch[1]) > 0
    assert total_probability == pytest.approx(1)


@pytest.mark.parametrize("dirty", [False, True])
@pytest.mark.parametrize("transfer", [False, True])
def test_adopted_workspace_cannot_be_destructively_measured(
    dirty: bool, transfer: bool
) -> None:
    record = ProgramRecord((measured_function(),))
    with pytest.raises(ValueError, match="retirement ownership contract"):
        bind_register_program(
            record,
            ((0,),),
            input_ownership=((0, dirty),),
            allow_workspace_outputs=transfer,
        )
    # A failed binding has no effect on a later independent binding.
    assert bind_register_program(record, ((0,),)).classical_outputs == ((0,),)


@pytest.mark.parametrize("dirty", [False, True])
def test_allocated_workspace_cannot_be_destructively_measured(dirty: bool) -> None:
    builder = ComponentProgramBuilder(
        Signature([Register("c", classical_bit_type(), side=RegisterSide.RIGHT)]),
        [measured_function("measurement")],
    )
    work = builder.allocate(dtype=quantum_bit_type(), dirty=dirty)
    result = builder.call_dict("measurement", {"q": work})
    record = builder.finalize(result)
    with pytest.raises(ValueError, match="retirement ownership contract"):
        bind_register_program(
            record,
            (),
            clean=() if dirty else (0,),
            borrowed=(0,) if dirty else (),
            allow_workspace_outputs=True,
        )


def test_classical_entry_remains_uninitialized_and_quantum_constructor_is_compatible() -> (
    None
):
    cast = identity("main", (Register("c", classical_bit_type()),))
    with pytest.raises(ValueError, match="unsupported executable"):
        bind_register_program(ProgramRecord((cast,)), ((),))
    with pytest.raises(ValueError, match="requires physical register binding"):
        lower_register_program(ProgramRecord((measured_function(),)))
    pure = identity("main", (Register("q", quantum_bit_type()),))
    bound = bind_register_program(ProgramRecord((pure,)), ((0,),))
    assert bound == BoundRegisterProgram(CircuitIR(1, ()), ((0,),), ())
    assert bound.classical_outputs == ()


def test_owned_measurement_slots_do_not_collide_with_existing_feedback_slots() -> None:
    from tests.unit.test_register_program_lowering import packed_circuit

    # The existing non-destructive measurement retains its quantum successor.
    packed = packed_circuit(
        CircuitIR(
            1,
            (
                Instruction(
                    "measure", (0,), metadata={"is_dynamic": True, "classical_bit": 0}
                ),
            ),
        ),
        (1,),
    )
    leaf = replace(
        packed.functions[0],
        name="feedback",
        registers=(Register("reg", quantum_bit_type()),),
    )
    assert leaf.registers is not None
    signature = measured_function().registers
    assert signature is not None
    builder = ComponentProgramBuilder(
        Signature(signature),
        [leaf, *packed.functions[1:], measured_function("destructive")],
    )
    quantum = builder.call_dict(
        "feedback", {leaf.registers[0].name: builder.inputs["q"]}
    )
    classical = builder.call_dict("destructive", {"q": next(iter(quantum.values()))})
    bound = bind_register_program(builder.finalize(classical), ((0,),))
    assert [gate.metadata["classical_bit"] for gate in bound.circuit.instructions] == [
        0,
        1,
    ]
    assert bound.classical_outputs == ((1,),)

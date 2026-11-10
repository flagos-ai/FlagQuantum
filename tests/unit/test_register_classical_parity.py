"""Measurement-produced parity, owned forwarding and bool feedback execution."""

from __future__ import annotations

from collections import Counter
from itertools import product

import numpy as np
import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_classical import parity_word_function
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.compiler.registers.register_programs import lower_register_program
from flagquantum.core.classical_dataflow import parse_classical_xor
from flagquantum.core.ir.program import (
    BlockRecord,
    ConditionRef,
    FunctionRecord,
    Operation,
    ProgramRecord,
    ValueRef,
    verify_program_record,
)
from flagquantum.core.ir.program.register_types import (
    classical_bit_type,
    quantum_bit_type,
)
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_program_owned_measurement import measured_function

pytestmark = pytest.mark.unit


def feedback_function(polarity: bool) -> FunctionRecord:
    c, left, right, successor = (ValueRef("entry", i, "register") for i in range(4))
    snapshot = ValueRef("entry", 4, "bool")
    qleft, qright, outleft, outright = (
        ValueRef("entry", i, "qubit") for i in range(5, 9)
    )
    rleft, rright = (ValueRef("entry", i, "register") for i in (9, 10))
    condition = ((ConditionRef(snapshot, polarity),),)
    return FunctionRecord(
        "feedback",
        (
            BlockRecord(
                "entry",
                (c, left, right),
                operations=(
                    Operation("read_classical_register", (c,), (successor, snapshot)),
                    Operation("split_register", (left,), (qleft,)),
                    Operation("split_register", (right,), (qright,)),
                    Operation("z", (qleft,), (outleft,), condition_clauses=condition),
                    Operation("x", (qright,), (outright,), condition_clauses=condition),
                    Operation(
                        "join_register",
                        (outleft,),
                        (rleft,),
                        register_type=quantum_bit_type(),
                    ),
                    Operation(
                        "join_register",
                        (outright,),
                        (rright,),
                        register_type=quantum_bit_type(),
                    ),
                ),
                return_values=(successor, rleft, rright),
            ),
        ),
        registers=(
            Register("c", classical_bit_type()),
            Register("targets", quantum_bit_type(), (2,)),
        ),
    )


def parity_feedback_program(count: int, polarity: bool) -> ProgramRecord:
    builder = ComponentProgramBuilder(
        Signature(
            [
                Register("q", quantum_bit_type(), (count,), RegisterSide.LEFT),
                Register("targets", quantum_bit_type(), (2,)),
                Register("spectator", quantum_bit_type()),
                Register("c", classical_bit_type(), (count,), RegisterSide.RIGHT),
                Register("parities", classical_bit_type(), (2,), RegisterSide.RIGHT),
            ]
        ),
        [
            measured_function("measurement"),
            parity_word_function("parity", count),
            parity_word_function("nested", 1),
            feedback_function(polarity),
        ],
    )
    measured = np.array(
        [
            builder.call_dict("measurement", {"q": q})["c"]
            for q in np.asarray(builder.inputs["q"]).flat
        ],
        dtype=object,
    )
    first = builder.call_dict("parity", {"x": measured})
    nested = builder.call_dict(
        "nested", {"x": np.array([first["parity"]], dtype=object)}
    )
    feedback = builder.call_dict(
        "feedback", {"c": nested["parity"], "targets": builder.inputs["targets"]}
    )
    work = builder.allocate(dtype=quantum_bit_type())
    builder.free(work)
    return builder.finalize(
        {
            "c": first["x"],
            "parities": np.array(
                [np.asarray(nested["x"])[0], feedback["c"]], dtype=object
            ),
            "targets": feedback["targets"],
            "spectator": builder.inputs["spectator"],
        }
    )


@pytest.mark.parametrize("count", [0, 1, 2, 3])
def test_nested_parity_has_fresh_slots_and_preserves_owned_inputs(count: int) -> None:
    record = parity_feedback_program(count, True)
    assert verify_program_record(record) == ()
    bound = bind_register_program(
        record, tuple((i,) for i in range(count + 3)), clean=(count + 3,)
    )
    gates = bound.circuit.instructions
    assert Counter(g.name for g in gates) == Counter(
        {
            "measure": count,
            "classical_xor": 2,
            "z": 1,
            "x": 1,
        }
    )
    slots = {
        g.wires[0]: int(g.metadata["classical_bit"])
        for g in gates
        if g.name == "measure"
    }
    assignments = tuple(
        parse_classical_xor(g) for g in gates if g.name == "classical_xor"
    )
    assert assignments[0].sources == tuple(slots[i] for i in range(count))
    assert assignments[1].sources == (assignments[0].target,)
    assert sorted([*slots.values(), *(a.target for a in assignments)]) == list(
        range(count + 2)
    )
    assert bound.outputs == ((count,), (count + 1,), (count + 2,), *((),) * (count + 2))
    assert bound.classical_outputs == (
        (),
        (),
        (),
        *((slots[i],) for i in range(count)),
        *((a.target,) for a in assignments),
    )
    assert [e.kind for e in bound.events] == ["allocate", "free"]
    for g in gates:
        if g.name in {"x", "z"}:
            assert g.metadata["conditions"] == ((assignments[-1].target, 1),)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("count", [0, 1, 2, 3])
@pytest.mark.parametrize("polarity", [False, True])
def test_all_branches_match_original_measurement_tensor_and_parity(
    count: int, polarity: bool
) -> None:
    from qualtran import CBit, CtrlSpec
    from qualtran.bloqs.basic_gates import MeasureZ, XGate, ZGate
    from qualtran.bloqs.gf_arithmetic.gf2_multiplication import Parity
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    bound = bind_register_program(
        parity_feedback_program(count, polarity),
        tuple((i,) for i in range(count + 3)),
        clean=(count + 3,),
    )
    gates = (
        Circuit(count + 4)
        .append(FieldOperation(count + 4, bound.circuit.instructions))
        .to_ir()
        .instructions
    )
    assert gates == bound.circuit.instructions
    generator = torch.Generator().manual_seed(1084 + count)
    state = torch.randn(1 << (count + 3), dtype=torch.complex128, generator=generator)
    state /= state.norm()
    with_work = torch.zeros(1 << (count + 4), dtype=torch.complex128)
    with_work[::2] = state
    copy, _ = MeasureZ().my_tensors({"q": "input"}, {"c": "output"})
    measurements = [g for g in gates if g.name == "measure"]
    total = 0.0
    for history in product((0, 1), repeat=count):
        outcomes = {g.wires[0]: b for g, b in zip(measurements, history, strict=True)}
        original = state.numpy().reshape((2,) * (count + 3)).copy()
        for wire in range(count):
            instrument = copy.data[:, :, outcomes[wire]]
            original = np.moveaxis(
                np.tensordot(instrument.T, original, axes=([1], [wire])), 0, wire
            )
        values = np.array([outcomes[i] for i in range(count)], dtype=np.uint8)
        returned, parity = Parity(count).call_classically(x=values)
        np.testing.assert_array_equal(returned, values)
        _, nested = Parity(1).call_classically(x=np.array([parity]))
        controls = CtrlSpec(qdtypes=[CBit()], cvs=[int(polarity)])
        for gate, wire in ((ZGate(), count), (XGate(), count + 1)):
            tensor = gate.controlled(controls)._tensor_data()
            matrix = tensor[int(nested), :, int(nested), :]
            original = np.moveaxis(
                np.tensordot(matrix, original, axes=([1], [wire])), 0, wire
            )
        branch = forced_branch(with_work, gates, history).reshape((2,) * (count + 4))
        np.testing.assert_allclose(branch[..., 0].numpy(), original, rtol=0, atol=3e-15)
        assert torch.count_nonzero(branch[..., 1]) == 0
        total += float(branch.abs().square().sum())
    assert total == pytest.approx(1)
    # This is the measurement submodel. Original Parity has no QEC model.
    measurement_model = get_cost_value(MeasureZ(), QECGatesCost())
    assert measurement_model.measurement * count == len(measurements)
    assert measurement_model.total_t_count() == 0


@pytest.mark.parametrize("strategy", ["trajectory", "batched"])
@pytest.mark.parametrize("polarity", [False, True])
def test_existing_append_and_runtime_execute_owned_parity_feedback(
    strategy: str, polarity: bool
) -> None:
    count = 2
    bound = bind_register_program(
        parity_feedback_program(count, polarity),
        tuple((i,) for i in range(5)),
        clean=(5,),
    )
    state = torch.zeros(64, dtype=torch.complex128)
    generator = torch.Generator().manual_seed(1084)
    state[::2] = torch.randn(32, dtype=torch.complex128, generator=generator)
    state /= state.norm()
    circuit = DynamicCircuit(6, inputs=state, dtype=torch.complex128)
    circuit.append(FieldOperation(6, bound.circuit.instructions))
    result = run_dynamic(circuit, shots=32, seed=1084, strategy=strategy)
    measurements = [g for g in bound.circuit.instructions if g.name == "measure"]
    assignments = [
        parse_classical_xor(g)
        for g in bound.circuit.instructions
        if g.name == "classical_xor"
    ]
    for bits, actual in zip(
        result.classical_bits, result.final_states.reshape(32, 64), strict=True
    ):
        history = tuple(
            int(bits[int(g.metadata["classical_bit"])]) for g in measurements
        )
        expected = forced_branch(state, bound.circuit.instructions, history)
        expected /= expected.norm()
        torch.testing.assert_close(actual, expected, rtol=0, atol=3e-15)
        assert torch.count_nonzero(actual[1::2]) == 0
        for assignment in assignments:
            assert bits[assignment.target] == assignment.evaluate(
                tuple(int(b) for b in bits)
            )


@pytest.mark.parametrize("count", [-1, True])
def test_parity_helper_rejects_invalid_executable_counts(count: int) -> None:
    with pytest.raises(ValueError, match="nonnegative count"):
        parity_word_function("parity", count)


def test_parity_and_snapshot_remain_outside_quantum_ssa_lowering() -> None:
    for record in (
        ProgramRecord((parity_word_function("main", 0),)),
        ProgramRecord((feedback_function(True),)),
    ):
        with pytest.raises(ValueError, match="requires physical register binding"):
            lower_register_program(record)

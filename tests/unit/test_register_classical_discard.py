"""Measurement, parity and discard preserve shared feedback snapshots."""

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
from flagquantum.compiler.registers.component_classical import (
    discard_word_function,
    parity_word_function,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.compiler.registers.register_programs import lower_register_program
from flagquantum.core.ir.program import Operation, ProgramRecord, verify_program_record
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
from tests.unit.test_register_classical_parity import feedback_function

pytestmark = pytest.mark.unit


def discard_feedback_program(count: int, polarity: bool) -> ProgramRecord:
    feedback = feedback_function(polarity)
    block = feedback.blocks[0]
    read = block.operations[0]
    assert block.return_values is not None and feedback.registers is not None
    feedback = replace(
        feedback,
        blocks=(
            replace(
                block,
                operations=(
                    read,
                    Operation("discard_classical_register", (read.results[0],)),
                    *block.operations[1:],
                ),
                return_values=block.return_values[1:],
            ),
        ),
        registers=(
            Register("c", classical_bit_type(), side=RegisterSide.LEFT),
            *feedback.registers[1:],
        ),
    )
    builder = ComponentProgramBuilder(
        Signature(
            [
                Register("q", quantum_bit_type(), (count,), RegisterSide.LEFT),
                Register("targets", quantum_bit_type(), (2,)),
                Register("spectator", quantum_bit_type()),
            ]
        ),
        [
            measured_function("measurement"),
            parity_word_function("parity", count),
            discard_word_function("discard"),
            feedback,
        ],
    )
    measured = np.array(
        [
            builder.call_dict("measurement", {"q": q})["c"]
            for q in np.asarray(builder.inputs["q"]).flat
        ],
        dtype=object,
    )
    result = builder.call_dict("parity", {"x": measured})
    for bit in np.asarray(result["x"]).flat:
        assert builder.call_dict("discard", {"c": bit}) == {}
    targets = builder.call_dict(
        "feedback", {"c": result["parity"], "targets": builder.inputs["targets"]}
    )["targets"]
    work = builder.allocate(dtype=quantum_bit_type())
    builder.free(work)
    return builder.finalize(
        {"targets": targets, "spectator": builder.inputs["spectator"]}
    )


@pytest.mark.parametrize("count", [0, 1, 2, 3])
def test_discard_emits_no_gate_slot_or_quantum_retirement(count: int) -> None:
    record = discard_feedback_program(count, True)
    assert verify_program_record(record) == ()
    bound = bind_register_program(
        record, tuple((i,) for i in range(count + 3)), clean=(count + 3,)
    )
    assert Counter(g.name for g in bound.circuit.instructions) == Counter(
        {"measure": count, "classical_xor": 1, "z": 1, "x": 1}
    )
    assert bound.outputs == ((count,), (count + 1,), (count + 2,))
    assert bound.classical_outputs == ()
    assert [e.kind for e in bound.events] == ["allocate", "free"]
    for gate in bound.circuit.instructions:
        if gate.name in {"z", "x"}:
            assert gate.metadata["conditions"] == ((count, 1),)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("count", [0, 1, 2, 3])
@pytest.mark.parametrize("polarity", [False, True])
def test_discard_all_complex_branches_match_original_tensors(
    count: int, polarity: bool
) -> None:
    from qualtran import CBit, CtrlSpec
    from qualtran.bloqs.basic_gates import MeasureZ, XGate, ZGate
    from qualtran.bloqs.basic_gates.discard import Discard
    from qualtran.bloqs.gf_arithmetic.gf2_multiplication import Parity
    from qualtran.resource_counting import QECGatesCost, get_cost_value
    from qualtran.simulation.tensor import DiscardInd

    bound = bind_register_program(
        discard_feedback_program(count, polarity),
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
    state = torch.randn(
        1 << (count + 3),
        dtype=torch.complex128,
        generator=torch.Generator().manual_seed(108429 + count),
    )
    state /= state.norm()
    with_work = torch.zeros(1 << (count + 4), dtype=torch.complex128)
    with_work[::2] = state
    copy, _ = MeasureZ().my_tensors({"q": "input"}, {"c": "output"})
    total = 0.0
    expected_density = np.zeros((8, 8), dtype=np.complex128)
    actual_density = np.zeros((8, 8), dtype=np.complex128)
    for history in product((0, 1), repeat=count):
        original = state.numpy().reshape((2,) * (count + 3)).copy()
        for wire, outcome in enumerate(history):
            original = np.moveaxis(
                np.tensordot(copy.data[:, :, outcome].T, original, axes=([1], [wire])),
                0,
                wire,
            )
        _, parity = Parity(count).call_classically(x=np.array(history, dtype=np.uint8))
        for bit in (*history, int(parity)):
            assert Discard().on_classical_vals(c=bit) == {}
        assert Discard().my_tensors({"c": "c"}, {}) == [DiscardInd(("c", 0))]
        for op, wire in ((ZGate(), count), (XGate(), count + 1)):
            matrix = op.controlled(
                CtrlSpec(qdtypes=[CBit()], cvs=[int(polarity)])
            )._tensor_data()
            matrix = matrix[int(parity), :, int(parity), :]
            original = np.moveaxis(
                np.tensordot(matrix, original, axes=([1], [wire])), 0, wire
            )
        branch = forced_branch(with_work, gates, history).reshape((2,) * (count + 4))
        np.testing.assert_allclose(branch[..., 0].numpy(), original, rtol=0, atol=3e-15)
        assert torch.count_nonzero(branch[..., 1]) == 0
        expected = original.reshape(-1, 8)
        actual = branch[..., 0].numpy().reshape(-1, 8)
        expected_density += np.einsum("ki,kj->ij", expected, expected.conj())
        actual_density += np.einsum("ki,kj->ij", actual, actual.conj())
        total += float(branch.abs().square().sum())
    assert total == pytest.approx(1)
    np.testing.assert_allclose(actual_density, expected_density, rtol=0, atol=3e-15)
    model = get_cost_value(Discard(), QECGatesCost())
    assert model.total_t_count() == model.measurement == model.clifford == 0


@pytest.mark.parametrize("strategy", ["trajectory", "batched"])
@pytest.mark.parametrize("polarity", [False, True])
def test_append_runtime_can_read_snapshot_after_owner_discard(
    strategy: str, polarity: bool
) -> None:
    bound = bind_register_program(
        discard_feedback_program(2, polarity), tuple((i,) for i in range(5)), clean=(5,)
    )
    state = torch.zeros(64, dtype=torch.complex128)
    state[::2] = torch.randn(
        32, dtype=torch.complex128, generator=torch.Generator().manual_seed(108429)
    )
    state /= state.norm()
    circuit = DynamicCircuit(6, inputs=state, dtype=torch.complex128)
    circuit.append(FieldOperation(6, bound.circuit.instructions))
    result = run_dynamic(circuit, shots=32, seed=1084, strategy=strategy)
    for bits, actual in zip(
        result.classical_bits, result.final_states.reshape(32, 64), strict=True
    ):
        history = tuple(int(bits[i]) for i in range(2))
        expected = forced_branch(state, bound.circuit.instructions, history)
        expected /= expected.norm()
        torch.testing.assert_close(actual, expected, rtol=0, atol=3e-15)
        assert torch.count_nonzero(actual[1::2]) == 0
        assert int(bits[2]) == sum(history) % 2


def test_discard_requires_physical_binding_in_quantum_only_lowering() -> None:
    with pytest.raises(ValueError, match="requires physical register binding"):
        lower_register_program(ProgramRecord((discard_word_function("main"),)))

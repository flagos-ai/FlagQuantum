"""Classical flank graphs preserve source sequencing and complex overlaps."""

from collections.abc import Mapping
from dataclasses import replace

import numpy as np
import pytest
import torch

from flagquantum.compiler.registers.component_classical_vectors import (
    flank_component_with_classical_vectors,
)
from flagquantum.compiler.registers.component_graph import iter_component_calls
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import (
    ProgramRecord,
    verify_program_record,
)
from flagquantum.core.ir.program.registers import RegisterType
from tests.unit.component_vector_checks import contract, tracked_flank
from tests.unit.test_component_bit_values import identity_record
from tests.unit.test_component_zero_initialization import phase_fixture

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("width", [1, 2])
@pytest.mark.parametrize("shape", [(), (2,), (1, 2)])
def test_shapes_call_order_and_complete_complex_overlap(
    monkeypatch: pytest.MonkeyPatch, width: int, shape: tuple[int, ...]
) -> None:
    from qualtran.bloqs.basic_gates import IntEffect, IntState
    from qualtran.simulation.xcheck_classical_quimb import flank_with_classical_vectors

    record, source = phase_fixture(width, shape)
    inputs: dict[str, object] = {
        "q": np.array([1, 0]),
        "idle": np.zeros(shape, dtype=int) if shape else 0,
    }
    outputs: dict[str, object] = {
        "q": np.array([0, 1]),
        "idle": np.zeros(shape, dtype=int) if shape else 0,
    }
    native, vectors = tracked_flank(monkeypatch, record, inputs, outputs)
    original = flank_with_classical_vectors(source, inputs, outputs)
    assert verify_program_record(native) == ()
    assert all(
        a is b
        for a, b in zip(
            record.functions,
            native.functions[1 : 1 + len(record.functions)],
            strict=True,
        )
    )
    calls = [
        native.functions[0].blocks[0].operations[i].callee
        for i, _, _ in iter_component_calls(native, "classical_vectors")
    ]
    kinds = [
        "state" if vectors[n][1] else "effect" if n in vectors else "child"
        for n in calls
        if n in vectors
    ]
    source_kinds = [
        (
            "state"
            if isinstance(b.bloq, IntState)
            else "effect" if isinstance(b.bloq, IntEffect) else "child"
        )
        for b, _, _ in original.iter_bloqsoqs()
    ]
    assert source_kinds == [
        *("state" for _ in range(2 + int(np.prod(shape)))),
        "child",
        *("effect" for _ in range(2 + int(np.prod(shape)))),
    ]
    assert kinds == [k for k in source_kinds if k != "child"]
    theta = torch.tensor(0.17, dtype=torch.float64, requires_grad=True)
    matrix = torch.as_tensor(
        source.tensor_contract(), dtype=torch.complex128
    ) * torch.exp(1j * theta)
    actual = contract(native, vectors, matrix)
    expected = torch.as_tensor(
        original.tensor_contract(), dtype=torch.complex128
    ) * torch.exp(1j * theta)
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=0)
    (gradient,) = torch.autograd.grad(actual.real, (theta,))
    torch.testing.assert_close(gradient, (1j * actual).real)


@pytest.mark.parametrize("shape", [(), (2,), (1, 2), (0,)])
def test_automatic_outputs_evaluates_once_and_explicit_outputs_bypass(
    monkeypatch: pytest.MonkeyPatch, shape: tuple[int, ...]
) -> None:
    record = identity_record(shape)
    values: dict[str, object] = {"x": np.ones(shape, dtype=int) if shape else True}
    seen = []

    def evaluate(inputs: Mapping[str, object]) -> Mapping[str, object]:
        seen.append(inputs)
        return inputs

    native, vectors = tracked_flank(monkeypatch, record, values, evaluate=evaluate)
    assert len(seen) == 1 and seen[0] is values
    matrix = torch.eye(1 << int(np.prod(shape)), dtype=torch.complex128)
    torch.testing.assert_close(
        contract(native, vectors, matrix), torch.tensor(1, dtype=torch.complex128)
    )
    flank_component_with_classical_vectors(
        record, values, values, evaluate=lambda _: pytest.fail("explicit outputs")
    )
    bound = bind_register_program(
        native,
        (),
        entry="classical_vectors",
        clean=tuple(range(max(1, int(np.prod(shape))))),
    )
    assert bound.outputs == ()


@pytest.mark.parametrize("stage", ["inputs", "evaluate", "outputs"])
def test_original_mapping_and_failure_order(
    monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    record = identity_record(())
    marker = RuntimeError("evaluator failure")

    def evaluate(values: Mapping[str, object]) -> Mapping[str, object]:
        if stage == "inputs":
            pytest.fail("input must fail before evaluation")
        if stage == "evaluate":
            raise marker
        return {}

    with pytest.raises(KeyError if stage != "evaluate" else RuntimeError) as error:
        flank_component_with_classical_vectors(
            record, {} if stage == "inputs" else {"x": 0}, evaluate=evaluate
        )
    if stage == "evaluate":
        assert error.value is marker


def test_native_entry_refusals_and_name_collision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = identity_record(())
    with pytest.raises(ValueError, match="evaluator"):
        flank_component_with_classical_vectors(record, {"x": 0})
    with pytest.raises(ValueError, match="unknown"):
        flank_component_with_classical_vectors(record, {}, entry="missing")
    with pytest.raises(ValueError, match="conflicts"):
        flank_component_with_classical_vectors(record, {}, name="main")
    with pytest.raises(ValueError, match="classical ports"):
        flank_component_with_classical_vectors(identity_record((), classical=True), {})
    for value in [0.5, -1, 2]:
        with pytest.raises(ValueError):
            flank_component_with_classical_vectors(record, {"x": value}, {"x": 0})
    renamed = replace(
        record,
        functions=(
            *record.functions,
            replace(record.functions[1], name="__fq_vector_0_reserved"),
        ),
    )
    native = flank_component_with_classical_vectors(renamed, {"x": 1}, {"x": 1})
    assert any(f.name.startswith("__fq_vector_1_") for f in native.functions)


def test_measured_add_all_inputs_histories_work_and_resource_accounts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from collections import Counter
    from itertools import product

    from qualtran.resource_counting import QECGatesCost, get_cost_value
    from qualtran.simulation.xcheck_classical_quimb import flank_with_classical_vectors

    from tests.unit.measurement_checks import forced_branch
    from tests.unit.test_component_hierarchy_adjoint import wrapped_add

    record, source = wrapped_add(2)
    base = bind_register_program(record, ((0, 1), (2, 3, 4), (5,)), clean=(6, 7))
    baseline_counts = Counter(g.name for g in base.circuit.instructions)
    initial = torch.zeros(256, dtype=torch.complex128)
    initial[0] = torch.tensor(1j, dtype=torch.complex128)
    for a, b, spectator in product(range(4), range(8), range(2)):
        inputs = {"a": a, "b": b, "spectator": spectator}
        outputs = source.on_classical_vals(**inputs)
        native, _ = tracked_flank(
            monkeypatch,
            record,
            inputs,
            evaluate=lambda values: source.on_classical_vals(**values),
        )
        bound = bind_register_program(
            native, (), entry="classical_vectors", clean=tuple(range(8))
        )
        x_count = sum(int(v).bit_count() for v in (*inputs.values(), *outputs.values()))
        counts = Counter(g.name for g in bound.circuit.instructions)
        assert counts == baseline_counts + Counter({"x": x_count})
        for history in product((0, 1), repeat=4):
            actual = forced_branch(initial, bound.circuit.instructions, history)
            torch.testing.assert_close(actual, initial / 4, atol=2e-12, rtol=0)
        assert bound.outputs == ()
    original = flank_with_classical_vectors(source, inputs)
    before = get_cost_value(source, QECGatesCost()).asdict()
    after = get_cost_value(original, QECGatesCost()).asdict()
    assert after.get("clifford", 0) - before.get("clifford", 0) == 6
    assert {k: v for k, v in after.items() if k != "clifford"} == {
        k: v for k, v in before.items() if k != "clifford"
    }
    np.testing.assert_allclose(original.tensor_contract(), 1, atol=2e-12)
    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic

    circuit = DynamicCircuit(8, dtype=torch.complex128)
    circuit.append(FieldOperation(8, bound.circuit.instructions))
    execution = run_dynamic(circuit, shots=8, seed=37, strategy="trajectory")
    expected_zero = torch.zeros(256, dtype=torch.complex128)
    expected_zero[0] = 1
    torch.testing.assert_close(
        execution.final_states.reshape(8, -1),
        expected_zero.expand(8, -1),
        atol=2e-12,
        rtol=0,
    )


def test_automatic_outputs_compose_existing_native_classical_evaluator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from flagquantum.simulation.component_classical_state import ComponentClassicalState

    record = identity_record(())
    inputs = {"x": 1}

    def evaluate(values: Mapping[str, object]) -> Mapping[str, object]:
        return ComponentClassicalState(
            record, (0,), values, lambda function, values: values
        ).simulate()

    native, vectors = tracked_flank(monkeypatch, record, inputs, evaluate=evaluate)
    torch.testing.assert_close(
        contract(native, vectors, torch.eye(2, dtype=torch.complex128)),
        torch.tensor(1, dtype=torch.complex128),
    )


@pytest.mark.parametrize("state", [False, True])
def test_directional_child_and_explicit_empty_outputs(
    monkeypatch: pytest.MonkeyPatch, state: bool
) -> None:
    from qualtran.bloqs.basic_gates import IntEffect, IntState
    from qualtran.simulation.xcheck_classical_quimb import flank_with_classical_vectors

    from flagquantum.compiler.registers.component_basis_vectors import (
        integer_basis_function,
    )

    record = ProgramRecord(
        (integer_basis_function("main", RegisterType("uint", 2), 2, state=state),)
    )
    source = (IntState if state else IntEffect)(2, 2)
    inputs = {} if state else {"val": 2}
    outputs = {"val": 2} if state else {}
    native, vectors = tracked_flank(
        monkeypatch,
        record,
        inputs,
        outputs,
        evaluate=lambda _: pytest.fail("explicit empty output must bypass evaluator"),
    )
    actual = contract(
        native,
        vectors,
        torch.as_tensor(source.tensor_contract(), dtype=torch.complex128),
    )
    np.testing.assert_allclose(
        actual.numpy(),
        flank_with_classical_vectors(source, inputs, outputs).tensor_contract(),
        atol=2e-12,
    )


def test_invalid_and_untyped_graphs_are_refused_before_evaluator() -> None:
    from flagquantum.core.ir import CircuitIR, Instruction
    from flagquantum.core.ir.program import circuit_to_program

    invalid = replace(identity_record(()), functions=())
    with pytest.raises(ValueError):
        flank_component_with_classical_vectors(
            invalid, {}, evaluate=lambda _: pytest.fail("invalid graph")
        )
    conversion = circuit_to_program(CircuitIR(1, (Instruction("x", (0,)),)))
    assert conversion.record is not None
    untyped = conversion.record
    with pytest.raises(ValueError, match="typed logical"):
        flank_component_with_classical_vectors(
            untyped, {}, evaluate=lambda _: pytest.fail("untyped graph")
        )


@pytest.mark.parametrize("output", [None, 0])
def test_original_xgate_inference_and_mismatched_tensor_effect(
    monkeypatch: pytest.MonkeyPatch, output: int | None
) -> None:
    from qualtran.bloqs.basic_gates import XGate
    from qualtran.simulation.xcheck_classical_quimb import flank_with_classical_vectors

    from tests.unit.test_component_expansion import _hierarchy

    record, _ = _hierarchy(1, 1)
    record = replace(
        record,
        functions=tuple(
            replace(
                f,
                blocks=tuple(
                    replace(
                        b,
                        operations=tuple(
                            replace(op, opcode="x") if op.opcode == "h" else op
                            for op in b.operations
                        ),
                    )
                    for b in f.blocks
                ),
            )
            for f in record.functions
        ),
    )
    outputs = None if output is None else {"q": output}
    native, vectors = tracked_flank(
        monkeypatch, record, {"q": 0}, outputs, evaluate=lambda values: {"q": 1}
    )
    actual = contract(
        native, vectors, torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
    )
    original = flank_with_classical_vectors(XGate(), {"q": 0}, outputs)
    np.testing.assert_allclose(actual.numpy(), original.tensor_contract(), atol=2e-12)
    assert actual.item() == (1 if output is None else 0)

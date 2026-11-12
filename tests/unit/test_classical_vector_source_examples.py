"""Original classical/tensor cross-check examples run through real native bodies."""

from collections import Counter
from dataclasses import replace
from functools import cache
from itertools import product
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from flagquantum.compiler.registers.arithmetic.addition import add_adjoint_ir
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.tensor_network.classical_action_tensor import (
    classical_action_tensor,
)
from tests.unit.classical_vector_fixtures import (
    KINDS,
    evaluate,
    example,
    original_inputs,
)
from tests.unit.component_vector_checks import contract, tracked_flank
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@cache
def native_matrix(kind: str) -> torch.Tensor:
    record, _, _ = example(kind)
    signature = Signature(record.functions[0].registers or ())
    data = classical_action_tensor(
        signature, lambda **v: tuple(evaluate(kind, v).values()), operation_name=kind
    )
    return torch.as_tensor(data, dtype=torch.complex128)


@pytest.mark.parametrize("kind", KINDS)
def test_unmodified_original_test_with_native_flank_replacement(
    kind: str, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    request.config.addinivalue_line("markers", "notebook: upstream notebook execution")
    import qualtran.simulation.xcheck_classical_quimb_test as original

    record, source, _ = example(kind)
    calls = []

    def replacement(bloq: Any, in_vals: Any, out_vals: Any = None) -> Any:
        assert bloq == source
        native, vectors = tracked_flank(
            monkeypatch,
            record,
            in_vals,
            out_vals,
            evaluate=lambda values: evaluate(kind, values),
        )
        result = contract(native, vectors, native_matrix(kind))
        calls.append(result)
        return SimpleNamespace(tensor_contract=lambda: result.detach().numpy())

    monkeypatch.setattr(original, "flank_with_classical_vectors", replacement)
    getattr(original, "test_" + kind)()
    assert len(calls) == (2 if kind == "xgate" else 1)
    assert calls[0].item() == 1
    if kind == "xgate":
        assert calls[1].item() == 0


@pytest.mark.parametrize("kind", KINDS)
def test_complete_native_matrix_against_original_tensor(kind: str) -> None:
    _, source, _ = example(kind)
    np.testing.assert_allclose(
        native_matrix(kind).numpy(), source.tensor_contract(), atol=2e-12, rtol=0
    )
    if kind == "add":
        assert native_matrix(kind).shape == (1024, 1024)


def input_words(record: ProgramRecord) -> tuple[tuple[int, ...], ...]:
    words = []
    position = 0
    for register in record.functions[0].registers or ():
        for _ in register.all_idxs():
            width = int(register.bitsize)
            words.append(tuple(range(position, position + width)))
            position += width
    return tuple(words)


@pytest.mark.parametrize("kind", KINDS)
def test_exact_gates_complex_entangled_input_all_branches_and_work(kind: str) -> None:
    import cirq

    record, source, emitted = example(kind)
    words = input_words(record)
    data_width = sum(map(len, words))
    work = emitted.n_wires - data_width
    bound = bind_register_program(
        record, words, clean=tuple(range(data_width, emitted.n_wires))
    )
    assert bound.outputs == words
    slots = {
        g.wires[0]: g.metadata["classical_bit"]
        for g in emitted.instructions
        if g.name == "measure"
    }
    mapping = {
        g.metadata["classical_bit"]: slots[g.wires[0]]
        for g in bound.circuit.instructions
        if g.name == "measure"
    }
    normalized = []
    for gate in bound.circuit.instructions:
        metadata = dict(gate.metadata)
        if gate.name == "measure":
            metadata["classical_bit"] = mapping[metadata["classical_bit"]]
        if "conditions" in metadata:
            metadata["conditions"] = tuple(
                (mapping[slot], value) for slot, value in metadata["conditions"]
            )
        normalized.append(replace(gate, metadata=metadata))
    assert_source(tuple(normalized), emitted.instructions)
    if kind == "add":
        expected = reference_arithmetic_gates(
            source,
            {"a": tuple(range(5)), "b": tuple(range(5, 10))},
            tuple(range(10, 14)),
            dict(zip(range(10, 14), range(4), strict=True)),
        )
        assert_source(emitted.instructions, expected)
    else:
        quregs = {}
        position = 0
        for register in source.signature:
            count = int(register.total_bits())
            quregs[register.name] = np.array(
                cirq.LineQubit.range(position, position + count)
            ).reshape((*register.shape, int(register.bitsize)))
            position += count
        operation, _ = source.as_cirq_op(cirq.SimpleQubitManager(), **quregs)
        assert tuple(q.x for q in operation.qubits) == emitted.instructions[0].wires
        assert (
            operation.gate
            == {"xgate": cirq.X, "cnot": cirq.CNOT, "toffoli": cirq.TOFFOLI}[kind]
        )
    size = 1 << (data_width + 1)
    values = torch.arange(size, dtype=torch.float64)
    amplitudes = torch.complex(values + 1, 3 * values - 7)
    amplitudes /= amplitudes.norm()
    indices = ((torch.arange(size) >> 1) << (work + 1)) | (torch.arange(size) & 1)
    initial = torch.zeros(1 << (emitted.n_wires + 1), dtype=torch.complex128)
    initial[indices] = amplitudes
    expected_state = torch.zeros_like(initial)
    expected_state[indices] = (
        native_matrix(kind) @ amplitudes.reshape(-1, 2)
    ).flatten()
    first = None
    for history in product((0, 1), repeat=work):
        result = forced_branch(initial, bound.circuit.instructions, history)
        torch.testing.assert_close(
            result, expected_state / (2 ** (work / 2)), atol=2e-12, rtol=0
        )
        if first is None:
            first = result
    if kind == "add":
        inverse = add_adjoint_ir(
            tuple(range(5)), tuple(range(5, 10)), tuple(range(10, 14)), tuple(range(4))
        )
        assert first is not None
        for history in product((0, 1), repeat=4):
            restored = forced_branch(first, inverse, history)
            torch.testing.assert_close(restored, initial / 16, atol=2e-12, rtol=0)


@pytest.mark.parametrize("kind", KINDS)
def test_original_closed_inputs_append_and_separate_resource_accounts(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value
    from qualtran.simulation.xcheck_classical_quimb import flank_with_classical_vectors

    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic

    record, source, emitted = example(kind)
    inputs = original_inputs(kind)
    outputs = evaluate(kind, inputs)
    native, vectors = tracked_flank(
        monkeypatch, record, inputs, evaluate=lambda values: evaluate(kind, values)
    )
    torch.testing.assert_close(
        contract(native, vectors, native_matrix(kind)),
        torch.tensor(1, dtype=torch.complex128),
    )
    closed = bind_register_program(
        native, (), entry="classical_vectors", clean=tuple(range(emitted.n_wires))
    )
    assert closed.outputs == ()
    counts = Counter(g.name for g in closed.circuit.instructions)
    x_count = sum(
        int(v).bit_count()
        for collection in (inputs, outputs)
        for value in collection.values()
        for v in np.asarray(value).flat
    )
    assert counts == Counter(g.name for g in emitted.instructions) + Counter(
        {"x": x_count}
    )
    initial = torch.zeros(1 << emitted.n_wires, dtype=torch.complex128)
    initial[0] = 1
    measurements = counts["measure"]
    for history in product((0, 1), repeat=measurements):
        torch.testing.assert_close(
            forced_branch(initial, closed.circuit.instructions, history),
            initial / (2 ** (measurements / 2)),
            atol=2e-12,
            rtol=0,
        )
    circuit = DynamicCircuit(emitted.n_wires, dtype=torch.complex128)
    circuit.append(FieldOperation(emitted.n_wires, closed.circuit.instructions))
    execution = run_dynamic(circuit, shots=4, seed=37, strategy="trajectory")
    torch.testing.assert_close(
        execution.final_states.reshape(4, -1), initial.expand(4, -1), atol=2e-12, rtol=0
    )
    original = flank_with_classical_vectors(source, inputs)
    source_cost = get_cost_value(source, QECGatesCost()).asdict()
    closed_cost = get_cost_value(original, QECGatesCost()).asdict()
    assert closed_cost.get("clifford", 0) - source_cost.get("clifford", 0) == len(
        vectors
    )
    assert {k: v for k, v in source_cost.items() if k != "clifford"} == {
        k: v for k, v in closed_cost.items() if k != "clifford"
    }
    assert get_cost_value(original, QubitCount()) == emitted.n_wires

"""Executable cross-checks retain physical leaves and factorized addition."""

from dataclasses import replace
from itertools import product

import numpy as np
import pytest
import torch

from examples.algorithms import classical_tensor_crosscheck as example
from flagquantum.compiler.registers.component_classical_vectors import (
    flank_component_with_classical_vectors,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from tests.unit.measurement_checks import forced_branch

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def cases() -> list[tuple[str, int, dict[str, object]]]:
    result: list[tuple[str, int, dict[str, object]]] = []
    result.extend(("xgate", 1, {"q": q}) for q in range(2))
    result.extend(
        ("cnot", 1, {"ctrl": c, "target": t}) for c, t in product(range(2), repeat=2)
    )
    result.extend(
        ("toffoli", 1, {"ctrl": [a, b], "target": t})
        for a, b, t in product(range(2), repeat=3)
    )
    for width in (1, 2, 3):
        result.extend(
            ("add", width, {"a": a, "b": b})
            for a, b in product(range(1 << width), repeat=2)
        )
    return result


@pytest.mark.parametrize("kind,width,inputs", cases())
def test_complete_small_basis_overlap_and_wrong_output(
    kind: str, width: int, inputs: dict[str, object]
) -> None:
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic import Add
    from qualtran.bloqs.basic_gates import CNOT, Toffoli, XGate

    source = {
        "xgate": XGate(),
        "cnot": CNOT(),
        "toffoli": Toffoli(),
        "add": Add(QUInt(width)),
    }[kind]
    source_inputs = {
        k: np.asarray(v) if isinstance(v, list) else v for k, v in inputs.items()
    }
    expected = source.call_classically(**source_inputs)
    outputs = example.crosscheck_values(kind, inputs, width)
    for actual, original in zip(outputs.values(), expected, strict=True):
        np.testing.assert_array_equal(actual, original)
    actual = example.contract_classical_crosscheck(kind, inputs, width=width)
    torch.testing.assert_close(actual, torch.ones_like(actual), atol=2e-12, rtol=0)
    wrong = dict(outputs)
    key = {"xgate": "q", "cnot": "target", "toffoli": "target", "add": "b"}[kind]
    value = wrong[key]
    assert isinstance(value, int)
    wrong[key] = value ^ 1
    rejected = example.contract_classical_crosscheck(kind, inputs, wrong, width=width)
    torch.testing.assert_close(rejected, torch.zeros_like(rejected), atol=2e-12, rtol=0)


@pytest.mark.parametrize(
    "kind,width,inputs",
    [
        ("xgate", 1, {"q": 0}),
        ("cnot", 1, {"ctrl": 1, "target": 0}),
        ("toffoli", 1, {"ctrl": [1, 1], "target": 0}),
        *(("add", w, {"a": 1, "b": 1}) for w in (1, 2, 3, 5)),
    ],
)
def test_real_closed_graph_all_measurement_histories_and_append(
    kind: str, width: int, inputs: dict[str, object]
) -> None:
    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic

    record = example.crosscheck_program(kind, width)
    closed = flank_component_with_classical_vectors(
        record, inputs, evaluate=lambda v: example.crosscheck_values(kind, v, width)
    )
    size = {"xgate": 1, "cnot": 2, "toffoli": 3, "add": 3 * width - 1}[kind]
    bound = bind_register_program(
        closed, (), entry="classical_vectors", clean=tuple(range(size))
    )
    assert bound.outputs == ()
    count = sum(g.name == "measure" for g in bound.circuit.instructions)
    initial = torch.zeros(1 << size, dtype=torch.complex128)
    initial[0] = 1
    for history in product((0, 1), repeat=count):
        actual = forced_branch(initial, bound.circuit.instructions, history)
        torch.testing.assert_close(
            actual, initial / (2 ** (count / 2)), atol=2e-12, rtol=0
        )
    circuit = DynamicCircuit(size, dtype=torch.complex128)
    circuit.append(FieldOperation(size, bound.circuit.instructions))
    result = run_dynamic(circuit, shots=4, seed=37, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(4, -1), initial.expand(4, -1), atol=2e-12, rtol=0
    )


def test_add_never_uses_whole_word_unitary(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("whole-word unitary is forbidden")

    monkeypatch.setattr(example, "get_unitary", refuse)
    result = example.contract_classical_crosscheck("add", {"a": 2, "b": 3})
    assert result.item() == 1


def test_explicit_output_bypasses_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("inference is forbidden")

    monkeypatch.setattr(example, "crosscheck_values", refuse)
    assert (
        example.contract_classical_crosscheck("xgate", {"q": 0}, {"q": 1}).item() == 1
    )


def test_refusal_boundaries_and_numpy_integer() -> None:
    for action in (
        lambda: example.crosscheck_program("unknown"),
        lambda: example.crosscheck_values("unknown", {}),
        lambda: example.crosscheck_providers(
            example.crosscheck_program("xgate"), "unknown", torch.empty(())
        ),
        lambda: example.crosscheck_values("xgate", {"q": 0.5}),
    ):
        with pytest.raises(ValueError):
            action()
    record = example.crosscheck_program("xgate")
    with pytest.raises(ValueError, match="typed ports"):
        example.crosscheck_providers(
            replace(record, functions=(replace(record.functions[0], registers=None),)),
            "xgate",
            torch.empty(()),
        )
    assert example.crosscheck_values("xgate", {"q": np.int64(0)}) == {"q": 1}


def test_executable_example(capsys: pytest.CaptureFixture[str]) -> None:
    example.main()
    assert "add: (1+0j)" in capsys.readouterr().out


@pytest.mark.parametrize(
    "kind,width",
    [("xgate", 1), ("cnot", 1), ("toffoli", 1), *(("add", w) for w in (1, 2, 3))],
)
def test_complete_provider_matrix_matches_unmodified_source(
    kind: str, width: int
) -> None:
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic import Add
    from qualtran.bloqs.basic_gates import CNOT, Toffoli, XGate

    from flagquantum.compiler.registers.component_graph import iter_component_calls
    from flagquantum.compiler.registers.component_tensor_expansion import (
        expand_component_root_for_tensor,
    )
    from flagquantum.simulation.tensor_network.component_dense import (
        contract_component_tensors,
    )

    record = example.crosscheck_program(kind, width)
    like = torch.empty((), dtype=torch.complex128)
    providers = example.crosscheck_providers(record, kind, like)
    selected, _ = expand_component_root_for_tensor(
        record, providers.has_custom_tensor, full_flatten=False
    )
    order = tuple(i for i, _, _ in iter_component_calls(selected, "tensor_root"))
    actual = contract_component_tensors(
        selected, order, providers.supply, like, "tensor_root"
    )
    source = {
        "xgate": XGate(),
        "cnot": CNOT(),
        "toffoli": Toffoli(),
        "add": Add(QUInt(width)),
    }[kind]
    np.testing.assert_allclose(
        actual.numpy(), source.tensor_contract(), atol=2e-12, rtol=0
    )

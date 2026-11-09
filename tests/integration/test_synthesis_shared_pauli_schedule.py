"""Replace shared product-formula scheduling without modifying its consumers."""

from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.algorithms import Hamiltonian, pauli_term, trotter_circuit
from flagquantum.compiler import _pauli_schedule
from tests.integration.test_synthesis_qmod_suzuki_trotter import _reference
from tests.unit.test_trotter import _coefficient_value_and_gradient

pytestmark = pytest.mark.integration
SOURCE = (
    Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod/suzuki_trotter.qmod"
)


def alternative_steps(
    term_count: int, time: float, order: int, repetitions: int
) -> Iterator[tuple[int, float]]:
    """Use a small materialized tuple for the test replacement implementation."""
    step = time / repetitions
    indices = tuple(range(term_count))
    if order == 2:
        indices += indices[::-1]
        step /= 2
    return iter((index, step) for index in indices * repetitions)


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("repetitions", [1, 3])
@pytest.mark.parametrize("time", [0.37, -0.29])
def test_both_consumers_keep_full_complex_states_and_identical_instruction_sequences(
    monkeypatch: pytest.MonkeyPatch, order: int, repetitions: int, time: float
) -> None:
    hamiltonian = Hamiltonian(
        [pauli_term(1.5, "ZXX", (0, 1, 2)), pauli_term(0.5, "ZXY", (0, 1, 2))]
    )
    source = SOURCE.read_text().replace("x, 1, 1", f"x, {order}, {repetitions}")

    def consumers() -> tuple[fq.Circuit, synth.Program]:
        return (
            trotter_circuit(
                hamiltonian,
                time,
                order=order,
                steps=repetitions,
                dtype=torch.complex128,
            ),
            synth.synthesize(
                source, source_format="qmod", classical_inputs={"x": time}
            ),
        )

    original_circuit, original_program = consumers()
    calls = []

    def replacement(
        count: int, duration: float, product_order: int, steps: int
    ) -> Iterator[tuple[int, float]]:
        calls.append((count, duration, product_order, steps))
        return alternative_steps(count, duration, product_order, steps)

    monkeypatch.setattr(_pauli_schedule, "ordered_pauli_steps", replacement)
    circuit, program = consumers()
    assert len(calls) >= 2
    assert all(call == (2, time, order, repetitions) for call in calls)
    assert circuit.to_ir().content_hash == original_circuit.to_ir().content_hash
    assert program.circuit.content_hash == original_program.circuit.content_hash
    expected = fq.run(_reference(time, order, repetitions)).statevector()
    for value in (original_circuit, original_program, circuit, program):
        torch.testing.assert_close(
            fq.run(value).statevector(), expected, atol=1e-12, rtol=0
        )


@pytest.mark.parametrize("order", [1, 2])
def test_replacement_keeps_live_tensor_coefficient_gradients(
    monkeypatch: pytest.MonkeyPatch, order: int
) -> None:
    expected = _coefficient_value_and_gradient(0.6, order=order, steps=3)
    monkeypatch.setattr(_pauli_schedule, "ordered_pauli_steps", alternative_steps)
    actual = _coefficient_value_and_gradient(0.6, order=order, steps=3)
    assert actual == pytest.approx(expected, abs=1e-12, rel=0)

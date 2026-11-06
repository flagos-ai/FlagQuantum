"""Exercise final outputs and count actual draws on a dynamic program."""

from __future__ import annotations

import math
from collections import Counter
from unittest.mock import patch

import pytest
import torch

import flagquantum as fq
from flagquantum.dynamic import DynamicCircuit
from flagquantum.observables import OutputRequest
from flagquantum.runtime.dynamic import execution
from flagquantum.runtime.dynamic.result import DynamicExecutionResult

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("strategy", ["trajectory", "batched"])
def test_final_outputs_reuse_one_readout(
    strategy: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    circuit = DynamicCircuit(3)
    circuit.h(0)
    circuit.measure(0, classical_bit=0)
    circuit.conditional("x", 1, classical_bit=0)
    circuit.x(0)
    circuit.h(2)  # This unmeasured qubit keeps the terminal draw stochastic.
    shots = 24
    options = fq.ExecutionOptions(seed=7)
    original_run = execution.run_dynamic
    original_draw = torch.multinomial

    def run(
        program: DynamicCircuit, *, shots: int, seed: int | None, max_batched_bytes: int
    ) -> DynamicExecutionResult:
        return original_run(
            program,
            shots=shots,
            seed=seed,
            max_batched_bytes=max_batched_bytes,
            strategy=strategy,
        )

    monkeypatch.setattr(execution, "run_dynamic", run)

    def execute(
        outputs: OutputRequest | tuple[OutputRequest, ...] | None,
    ) -> tuple[fq.ExecutionResult, list[tuple[int, ...]]]:
        with (
            patch.object(execution, "run_dynamic", wraps=run) as execute_spy,
            patch.object(torch, "multinomial", wraps=original_draw) as draw_spy,
        ):
            result = fq.run(circuit, outputs=outputs, options=options, shots=shots)
        assert execute_spy.call_count == 1
        return result, [tuple(call.args[0].shape) for call in draw_spy.call_args_list]

    baseline, baseline_draws = execute(None)
    # The mid-circuit measurement has two outcomes. The terminal draw has eight.
    terminal_draws = [shape for shape in baseline_draws if shape[-1] == 8]
    assert len(terminal_draws) == (shots if strategy == "trajectory" else 1)
    for outputs, expected_kinds in (
        (fq.samples(name="final"), ("sample",)),
        (fq.counts(), ("counts",)),
        ((fq.samples(name="final"), fq.counts()), ("sample", "counts")),
    ):
        actual, draws = execute(outputs)
        assert draws == baseline_draws
        assert actual.samples is not None
        assert baseline.samples is not None
        assert torch.equal(actual.samples, baseline.samples)
        mid = actual.measurement("mid_circuit_measurement").value
        baseline_mid = baseline.measurement("mid_circuit_measurement").value
        assert isinstance(mid, torch.Tensor)
        assert isinstance(baseline_mid, torch.Tensor)
        assert torch.equal(mid, baseline_mid)
        assert torch.equal(actual.samples[:, 0], 1 - mid[:, 0])
        assert torch.equal(actual.samples[:, 1], mid[:, 0])
        kinds = tuple(item.kind for item in actual.measurements)
        assert kinds == (*expected_kinds, "mid_circuit_measurement")
        if "sample" in kinds:
            assert actual.measurement("final").value is actual.samples
        if "counts" in kinds:
            expected = Counter(
                "".join(map(str, row)) for row in actual.samples.tolist()
            )
            assert actual.counts[0] == dict(expected)
            assert sum(actual.counts[0].values()) == shots


@pytest.mark.parametrize(
    ("feedback", "outcome", "terminal_weights"),
    [
        ("x", 0, (1.0, 0.0, 0.0, 0.0)),
        ("x", 1, (0.0, 0.0, 0.0, 1.0)),
        ("h", 0, (1.0, 0.0, 0.0, 0.0)),
        ("h", 1, (0.0, 0.0, 0.5, 0.5)),
    ],
)
def test_mixed_feedback_preserves_born_weights_and_final_outputs(
    feedback: str, outcome: int, terminal_weights: tuple[float, ...]
) -> None:
    circuit = (
        DynamicCircuit(2, dtype=torch.complex128, device="cpu")
        .h(0)
        .t(0)
        .h(0)
        .measure(0, classical_bit=0)
        .conditional(feedback, 1, classical_bit=0)
    )
    born = torch.tensor(
        [(2 + math.sqrt(2)) / 4, (2 - math.sqrt(2)) / 4], dtype=torch.float64
    )
    terminal = torch.tensor(terminal_weights, dtype=torch.float64)
    draws: list[tuple[int, ...]] = []
    original_draw = torch.multinomial

    def draw(
        probabilities: torch.Tensor,
        num_samples: int,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        assert num_samples == 1
        draws.append(tuple(probabilities.shape))
        if len(draws) == 1:
            torch.testing.assert_close(probabilities, born, atol=1e-12, rtol=0)
            assert probabilities[outcome] > 0
            return torch.tensor([outcome], dtype=torch.int64)
        torch.testing.assert_close(probabilities, terminal, atol=1e-12, rtol=0)
        return original_draw(probabilities, num_samples, generator=generator)

    with patch.object(torch, "multinomial", side_effect=draw):
        result = fq.run(
            circuit,
            outputs=(fq.samples(name="final"), fq.counts()),
            options=fq.ExecutionOptions(
                seed=7,
                device="cpu",
                allow_approximate=False,
                allow_backend_fallback=False,
            ),
            shots=1,
        )

    assert draws == [(2,), (4,)]
    mid = result.measurement("mid_circuit_measurement")
    final = result.measurement("final")
    assert mid.shots == final.shots == 1
    assert isinstance(mid.value, torch.Tensor) and mid.value.shape == (1, 1)
    assert isinstance(final.value, torch.Tensor) and final.value.shape == (1, 2)
    assert mid.value.tolist() == [[outcome]]
    control_bit, target_bit = final.value[0].tolist()
    assert control_bit == outcome
    assert target_bit in (0, 1)
    assert terminal_weights[2 * control_bit + target_bit] > 0
    assert result.samples is final.value
    assert result.counts[0] == {f"{control_bit}{target_bit}": 1}

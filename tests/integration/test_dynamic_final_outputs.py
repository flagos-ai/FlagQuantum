"""Exercise final outputs and count actual draws on a dynamic program."""

from __future__ import annotations

from collections import Counter
from typing import Any
from unittest.mock import patch

import pytest
import torch

import flagquantum as fq
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic import execution

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

    def run(*args: Any, **kwargs: Any) -> Any:
        return original_run(*args, strategy=strategy, **kwargs)

    monkeypatch.setattr(execution, "run_dynamic", run)

    def execute(outputs: Any) -> tuple[Any, list[tuple[int, ...]]]:
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
    for outputs in (
        fq.samples(name="final"),
        fq.counts(),
        (fq.samples(name="final"), fq.counts()),
    ):
        actual, draws = execute(outputs)
        assert draws == baseline_draws
        assert torch.equal(actual.samples, baseline.samples)
        mid = actual.measurement("mid_circuit_measurement").value
        assert torch.equal(mid, baseline.measurement("mid_circuit_measurement").value)
        assert torch.equal(actual.samples[:, 0], 1 - mid[:, 0])
        assert torch.equal(actual.samples[:, 1], mid[:, 0])
        kinds = tuple(item.kind for item in actual.measurements)
        if "sample" in kinds:
            assert kinds.count("sample") == 1
            assert torch.equal(actual.measurement("final").value, actual.samples)
        if "counts" in kinds:
            expected = Counter(
                "".join(map(str, row)) for row in actual.samples.tolist()
            )
            assert actual.counts[0] == dict(expected)
            assert sum(actual.counts[0].values()) == shots

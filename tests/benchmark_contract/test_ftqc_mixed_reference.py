"""Protect the approved point envelope and phase-sensitive observer semantics."""

import cmath
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum.benchmarking import ftqc_mixed_reference as mixed
from flagquantum.errors import CapabilityError

pytestmark = pytest.mark.benchmark_contract


@pytest.mark.parametrize("mode", ["statevector", "mps", "tensor_network"])
def test_single_point_uses_exact_route_and_independent_reference(mode: str) -> None:
    report = mixed.run_benchmark(n=4, t=2, topology="ring", mode=mode)
    assert report["status"] == "completed", report
    assert report["authored_t_count"] == 2
    assert report["performance_targets_enabled"] is False
    assert report["scalability_claim_allowed"] is False
    assert report["output_probability_bytes"] == 128
    digest = report["ir_sha256"]
    assert isinstance(digest, str) and len(digest) == 64
    phases = []
    for key in ("construction_seconds", "planning_seconds", "execution_seconds"):
        value = report[key]
        assert isinstance(value, float)
        phases.append(value)
    assert report["timed_phase_sum_seconds"] == pytest.approx(sum(phases))


@pytest.mark.parametrize("n", [4, 8])
@pytest.mark.parametrize("t", [0, 2, 4, 8])
@pytest.mark.parametrize("topology", ["chain", "ring"])
def test_authored_matrix_and_reference_normalization(
    n: int, t: int, topology: str
) -> None:
    instructions = mixed._circuit(n, t, topology).to_ir().instructions
    assert sum(item.name == "t" for item in instructions) == t
    expected = mixed._reference(n, t, topology)
    assert float(expected.abs().square().sum()) == pytest.approx(1.0, abs=1e-12)


def test_relative_phase_corruption_is_visible_when_probabilities_agree() -> None:
    expected = torch.tensor([1, 1j], dtype=torch.complex128) / math.sqrt(2)
    state = expected.conj()[None, :]
    result = fq.ExecutionResult(
        state=state,
        measurements=(
            fq.MeasurementResult(
                kind="probabilities", qubits=(0,), value=state.abs().square()
            ),
        ),
    )
    errors = mixed._errors(result, expected)
    assert errors["probability_max_absolute_error"] == 0
    assert errors["normalized_density_max_absolute_error"] > 0.9
    identity = torch.eye(2, dtype=torch.complex128)
    phase = torch.diag(torch.tensor([1, 1j], dtype=torch.complex128))
    b = (cmath.exp(1j * math.pi / 4) - 1) / (1j - 1)
    torch.testing.assert_close(
        (1 - b) * identity + b * phase,
        torch.diag(
            torch.tensor([1, cmath.exp(1j * math.pi / 4)], dtype=torch.complex128)
        ),
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("n", True),
        ("n", 16),
        ("t", 1),
        ("t", False),
        ("topology", "grid"),
        ("mode", "auto"),
    ],
)
def test_refuses_unapproved_inputs_before_execution(
    field: str, value: int | str
) -> None:
    with pytest.raises(ValueError):
        if field == "n":
            assert isinstance(value, int)
            mixed.run_benchmark(n=value)
        elif field == "t":
            assert isinstance(value, int)
            mixed.run_benchmark(t=value)
        elif field == "topology":
            assert isinstance(value, str)
            mixed.run_benchmark(topology=value)
        else:
            assert isinstance(value, str)
            mixed.run_benchmark(mode=value)


def test_capability_refusal_is_retained_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def refuse(*args: object, **kwargs: object) -> fq.ExecutionPlan:
        calls.append(1)
        raise CapabilityError("exact request unavailable")

    monkeypatch.setattr(fq, "plan", refuse)
    report = mixed.run_benchmark()
    assert report["status"] == "refused"
    assert report["diagnostic"] == "exact request unavailable"
    assert len(calls) == 1


def test_runtime_failure_is_propagated_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fail(*args: object, **kwargs: object) -> fq.ExecutionResult:
        calls.append(1)
        raise RuntimeError("execution failed")

    monkeypatch.setattr(fq, "run", fail)
    with pytest.raises(RuntimeError, match="execution failed"):
        mixed.run_benchmark()
    assert len(calls) == 1


def test_multi_process_is_outside_point_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(ValueError, match="one local process"):
        mixed.run_benchmark()


def test_analytic_bell_t_anchor_matches_current_public_execution() -> None:
    phase = cmath.exp(1j * math.pi / 4)
    expected = torch.tensor(
        [1 + phase, 1 - phase, 1 - phase, 1 + phase], dtype=torch.complex128
    ) / (2 * math.sqrt(2))
    result = fq.run(
        fq.Circuit(2).h(0).cx(0, 1).t(0).h(0).h(1),
        outputs=fq.probabilities([0, 1]),
        options=fq.ExecutionOptions(
            mode="statevector",
            device="cpu",
            precision="complex128",
            allow_approximate=False,
            allow_backend_fallback=False,
        ),
    )
    assert all(error <= 1e-12 for error in mixed._errors(result, expected).values())


def test_phase_error_is_written_before_nonzero_cli_exit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    original = fq.run

    def corrupt(*args: object, **kwargs: object) -> fq.ExecutionResult:
        result = original(*args, **kwargs)
        state = result.to_statevector().clone()
        index = int(torch.argmax(state.abs()))
        state.reshape(-1)[index] *= 1j
        return replace(result, state=state)

    monkeypatch.setattr(fq, "run", corrupt)
    output = tmp_path / "failure.json"
    monkeypatch.setattr(
        sys, "argv", ["mixed", "--t", "2", "--json-output", str(output)]
    )
    assert mixed.main() == 1
    report = json.loads(output.read_text())
    assert report["status"] == "failed"
    assert report["errors"]["normalized_density_max_absolute_error"] > 1e-12
    assert report["errors"]["probability_max_absolute_error"] < 1e-12


def test_report_refuses_to_certify_truncated_mps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = fq.run

    def truncated(*args: object, **kwargs: object) -> fq.ExecutionResult:
        result = original(*args, **kwargs)
        changed = replace(result, runtime={**result.runtime, "truncation_error": 0.125})
        object.__setattr__(changed, "_native_output", result.native())
        return changed

    monkeypatch.setattr(fq, "run", truncated)
    report = mixed.run_benchmark(mode="mps")
    assert report["status"] == "failed"
    diagnostic = report["diagnostic"]
    assert isinstance(diagnostic, str) and "truncation" in diagnostic

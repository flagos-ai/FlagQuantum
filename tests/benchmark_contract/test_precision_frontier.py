"""Independent complete-circuit, gradient and frontier acceptance checks."""

from __future__ import annotations

from decimal import Decimal, localcontext

import pytest
import torch

np = pytest.importorskip("numpy")

from benchmarks.precision_frontier import (  # noqa: E402
    cpu_output,
    evaluate,
    frontier,
    run_case,
)
from benchmarks.precision_workloads import (  # noqa: E402
    Case,
    output_hash,
    reference,
    state_quality,
)
from flagquantum.simulation.numerics.double_single import (  # noqa: E402
    DoubleSingleTensor,
)

pytestmark = [pytest.mark.benchmark_contract, pytest.mark.integration]


@pytest.mark.parametrize("family", ["noncommuting", "cancellation_echo"])
@pytest.mark.parametrize("qubits,depth", [(2, 2), (4, 8)])
def test_independent_reference_tangent_matches_finite_difference(
    family: str, qubits: int, depth: int
) -> None:
    case = Case(qubits, depth, family, 11)
    state, tangent, _, gradient = reference(case)
    epsilon = 1e-5
    upper = reference(case, case.theta + epsilon)
    lower = reference(case, case.theta - epsilon)
    assert np.max(np.abs(tangent - (upper[0] - lower[0]) / (2 * epsilon))) < 2e-9
    assert gradient == pytest.approx(
        (upper[2]["objective"] - lower[2]["objective"]) / (2 * epsilon), abs=2e-9
    )
    assert np.vdot(state, state).real == pytest.approx(1, abs=2e-13)


@pytest.mark.parametrize(
    "mode", ["fp32", "double_single_p4", "fp64", "selective_p2", "double_single_p3"]
)
def test_existing_precision_routes_match_independent_noncommuting_circuit(
    mode: str,
) -> None:
    case = Case(2, 2, "noncommuting", 29)
    result = evaluate(case, mode, "cpu", preflight=True)
    state, value, gradient = cpu_output(result)
    expected, _, observables, derivative = reference(case)
    tolerance = 2e-6 if mode in ("fp32", "selective_p2") else 1e-11
    assert np.max(np.abs(state - expected)) < tolerance
    assert value == pytest.approx(observables["objective"], abs=tolerance)
    assert gradient[0] == pytest.approx(derivative, abs=tolerance)
    assert result["summary"].get("state_host_fallback", False) is False


def test_cancellation_and_slow_cases_are_retained_with_real_samples() -> None:
    case = Case(2, 2, "cancellation_echo", 11)
    result = run_case(case, "fp32", "cpu", repeats=2, warmup=1)
    assert result["status"] == "completed"
    assert result["quality"]["gradient_max_absolute_error"] > 0.1
    assert len(result["timing"]["samples_seconds"]) == 2
    assert result["timing"]["statistic"] == "median"
    assert result["timing"]["median_seconds"] == pytest.approx(
        sum(result["timing"]["samples_seconds"]) / 2
    )
    assert len(result["output_word_hashes_sha256"]) == 2
    assert frontier([result], [1e-3])[0]["feasible_modes"] == []
    assert result["synthesis"]["coefficient_conversion_max_error"] == 0
    assert result["synthesis"]["parameter_conversion_absolute_error"] > 0
    assert result["memory"]["frontier_peak_memory_bytes"] > 0


def test_equal_scalar_errors_do_not_imply_output_identity() -> None:
    expected = np.array([1, 0], dtype=np.complex128)
    positive = np.array([1, 1e-5], dtype=np.complex128)
    negative = np.array([1, -1e-5], dtype=np.complex128)
    assert state_quality(positive, expected) == state_quality(negative, expected)
    assert output_hash([torch.from_numpy(positive)]) != output_hash(
        [torch.from_numpy(negative)]
    )
    assert output_hash([torch.tensor([0.0])]) != output_hash([torch.tensor([-0.0])])


def test_small_compensated_arithmetic_matches_decimal_reference() -> None:
    a, b = 1 + 2**-24, 1 - 2**-24
    left = DoubleSingleTensor.from_float64(torch.tensor(a, dtype=torch.float64))
    right = DoubleSingleTensor.from_float64(torch.tensor(b, dtype=torch.float64))
    product = float(left.multiply(right).to_float64())
    with localcontext() as context:
        context.prec = 100
        exact = Decimal.from_float(a) * Decimal.from_float(b)
    assert Decimal.from_float(product) == exact
    total = (
        DoubleSingleTensor.from_float64(torch.tensor(1e8, dtype=torch.float64))
        .add(DoubleSingleTensor.from_float64(torch.tensor(1.0, dtype=torch.float64)))
        .subtract(
            DoubleSingleTensor.from_float64(torch.tensor(1e8, dtype=torch.float64))
        )
    )
    assert float(total.to_float64()) == float(
        Decimal("1e8") + Decimal(1) - Decimal("1e8")
    )


def test_frontier_uses_observed_peak_memory_and_keeps_tradeoffs() -> None:
    def point(mode: str, error: float, seconds: float, peak: int | None) -> dict:
        return {
            "status": "completed",
            "case": {"qubits": 2, "depth": 2, "family": "noncommuting", "seed": 11},
            "mode": mode,
            "device": "cpu",
            "quality": {"accuracy_error": error},
            "timing": {"median_seconds": seconds},
            "memory": {"frontier_peak_memory_bytes": peak, "state_payload_bytes": 1},
        }

    rows = frontier(
        [
            point("fast", 1e-5, 1, 100),
            point("compact", 1e-9, 2, 50),
            point("dominated", 1e-9, 3, 100),
            point("unknown_peak", 1e-9, 0.5, None),
        ],
        [1e-4, 1e-8],
    )
    assert set(rows[0]["pareto_modes"]) == {"fast", "compact"}
    assert rows[1]["pareto_modes"] == ["compact"]
    assert rows[0]["cost_axes"][-1] == "observed_peak_memory_bytes"


def test_mps_fp64_refuses_cpu_substitution() -> None:
    from benchmarks.precision_frontier import UnsupportedRouteError

    with pytest.raises(UnsupportedRouteError, match="CPU substitution is prohibited"):
        evaluate(Case(2, 2, "noncommuting", 11), "fp64", "mps", preflight=False)


def test_unsupported_stage_inventory_is_retained_without_worker(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    from benchmarks import precision_frontier as producer

    monkeypatch.setattr(producer, "source_identity", lambda: {"code_version": "test"})
    monkeypatch.setattr(producer, "environment", lambda device: {"device": device})
    monkeypatch.setattr(
        producer,
        "probe_stages",
        lambda device: [
            {
                "stage": name,
                "status": "unsupported",
                "error": "explicit unavailable test stage",
            }
            for name in ("P1", "P4", "native_fp64")
        ],
    )

    def forbidden_worker(*args, **kwargs):
        raise AssertionError(
            "an unavailable stage must not launch or substitute a worker"
        )

    monkeypatch.setattr(producer, "run_worker", forbidden_worker)
    output = tmp_path / "campaign"
    assert producer.campaign(output, "mps", smoke=True, controls=False) == 0
    summary = json.loads((output / "summary.json").read_text())
    assert len(summary["cases"]) == 3
    assert all(
        row["status"] == "unsupported" and not row["worker_started"]
        for row in summary["cases"]
    )
    assert json.loads((output / "frontier.json").read_text())["frontier"] == []
    summary["cases"].pop()
    (output / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="inventory"):
        producer.render_report(output)

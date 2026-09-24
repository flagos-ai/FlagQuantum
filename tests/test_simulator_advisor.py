from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from flagquantum.ecosystem.simulators import (
    SimulatorAdvisorEvidenceError,
    recommend,
)

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[1]
REPORT = (
    ROOT
    / "benchmarks"
    / "results"
    / "comparison"
    / "simulator_comparison_cpu_arm64_20260923.json"
)
ALL_ENGINES = {
    "flagquantum_native",
    "qiskit_aer",
    "cirq_simulator",
    "pennylane_lightning_qubit",
}


def _report() -> dict[str, object]:
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _environment(report: dict[str, object]) -> dict[str, object]:
    value = report["comparison_identity"]
    assert isinstance(value, dict)
    return value


def test_advisor_import_does_not_import_optional_simulators() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from flagquantum.ecosystem.simulators import recommend; "
            "assert callable(recommend); "
            "assert not {'qiskit', 'qiskit_aer', 'cirq', 'pennylane'} & sys.modules.keys()",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_bundled_advisor_evidence_tracks_the_measured_report() -> None:
    raw = REPORT.read_bytes()
    report = _report()

    for measured_row in report["rows"]:
        decision = recommend(
            n_wires=measured_row["workload"]["n_wires"],
            environment=_environment(report),
            available_engines=ALL_ENGINES,
        )

        assert decision.evidence_source == REPORT.relative_to(ROOT).as_posix()
        assert decision.evidence_sha256 == hashlib.sha256(raw).hexdigest()
        by_engine = {candidate.engine: candidate for candidate in decision.candidates}
        for engine, measured in measured_row["engines"].items():
            candidate = by_engine[engine]
            assert candidate.label == measured["label"]
            assert candidate.version == measured["version"]
            assert candidate.median_seconds == measured["median_seconds"]
            assert (
                candidate.relative_median_absolute_deviation
                == measured["relative_median_absolute_deviation"]
            )
            assert (
                candidate.relative_to_flagquantum
                == measured_row["ratios_to_flagquantum"][engine]
            )


def test_advisor_prefers_native_when_external_gain_is_inside_tie_margin() -> None:
    report = _report()

    decision = recommend(
        n_wires=10,
        environment=_environment(report),
        available_engines=ALL_ENGINES,
    )

    assert decision.status == "recommended"
    assert decision.recommended_engine == "flagquantum_native"
    assert decision.reason == "native_within_tie_margin"
    assert decision.candidates[0].engine == "pennylane_lightning_qubit"
    cirq = next(item for item in decision.candidates if item.engine == "cirq_simulator")
    assert cirq.eligible is False
    assert cirq.exclusion_reasons == ("measurement_not_stable",)


@pytest.mark.parametrize("n_wires", (14, 18, 22, 24))
def test_advisor_recommends_the_fastest_stable_measured_engine(n_wires: int) -> None:
    report = _report()

    decision = recommend(
        n_wires=n_wires,
        environment=_environment(report),
        available_engines=ALL_ENGINES,
    )

    assert decision.status == "recommended"
    assert decision.recommended_engine == "flagquantum_native"
    assert decision.reason == "fastest_stable_measured_engine"
    assert decision.candidates[0].engine == "flagquantum_native"


def test_advisor_filters_unavailable_engines_without_importing_them() -> None:
    report = _report()

    decision = recommend(
        n_wires=22,
        environment=_environment(report),
        available_engines={"qiskit_aer"},
    )

    assert decision.recommended_engine == "qiskit_aer"
    assert decision.candidates[0].engine == "qiskit_aer"
    native = next(
        item for item in decision.candidates if item.engine == "flagquantum_native"
    )
    assert native.exclusion_reasons == ("engine_not_available",)


def test_advisor_filters_an_installed_version_that_does_not_match_evidence() -> None:
    report = _report()

    decision = recommend(
        n_wires=22,
        environment=_environment(report),
        available_engines={
            "flagquantum_native": "0.2.0",
            "qiskit_aer": "999.0",
        },
    )

    qiskit = next(item for item in decision.candidates if item.engine == "qiskit_aer")
    assert qiskit.available is True
    assert qiskit.eligible is False
    assert qiskit.installed_version == "999.0"
    assert qiskit.exclusion_reasons == ("version_mismatch",)


def test_advisor_fails_closed_for_unmeasured_workload_or_environment() -> None:
    report = _report()
    environment = _environment(report)

    workload_miss = recommend(
        n_wires=20,
        environment=environment,
        available_engines=ALL_ENGINES,
    )
    environment_miss = recommend(
        n_wires=22,
        environment={**environment, "machine": "x86_64"},
        available_engines=ALL_ENGINES,
    )

    assert workload_miss.status == "insufficient_evidence"
    assert workload_miss.reason == "workload_not_measured"
    assert workload_miss.workload_matches is False
    assert environment_miss.status == "insufficient_evidence"
    assert environment_miss.reason == "environment_not_measured"
    assert environment_miss.environment_matches is False


def test_advisor_can_select_a_clear_external_winner_from_future_evidence() -> None:
    report = _report()
    changed = deepcopy(report)
    row = next(item for item in changed["rows"] if item["workload"]["n_wires"] == 22)
    row["engines"]["qiskit_aer"]["median_seconds"] = 0.1
    row["ratios_to_flagquantum"]["qiskit_aer"] = (
        0.1 / row["engines"]["flagquantum_native"]["median_seconds"]
    )

    decision = recommend(
        n_wires=22,
        evidence=changed,
        environment=_environment(report),
        available_engines=ALL_ENGINES,
    )

    assert decision.recommended_engine == "qiskit_aer"
    assert decision.reason == "fastest_stable_measured_engine"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("passed", False, "evidence passed"),
        ("release_gate_allowed", True, "release_gate_allowed"),
    ),
)
def test_advisor_rejects_unsafe_report_claims(
    field: str, value: object, message: str
) -> None:
    report = _report()
    report[field] = value

    with pytest.raises(SimulatorAdvisorEvidenceError, match=message):
        recommend(n_wires=22, evidence=report)


def test_advisor_rejects_hidden_fallback_evidence() -> None:
    report = _report()
    identity = report["comparison_identity"]
    assert isinstance(identity, dict)
    identity["hidden_fallback_allowed"] = True

    with pytest.raises(SimulatorAdvisorEvidenceError, match="hidden backend fallback"):
        recommend(n_wires=22, evidence=report)


def test_advisor_rejects_an_inconsistent_timing_ratio() -> None:
    report = _report()
    row = next(item for item in report["rows"] if item["workload"]["n_wires"] == 22)
    row["ratios_to_flagquantum"]["qiskit_aer"] = 0.01

    with pytest.raises(SimulatorAdvisorEvidenceError, match="inconsistent"):
        recommend(
            n_wires=22,
            evidence=report,
            environment=_environment(report),
            available_engines=ALL_ENGINES,
        )


def test_advisor_rejects_duplicate_json_fields(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text(
        '{"schema":"flagquantum.simulator_comparison_report.v1","schema":"other"}',
        encoding="utf-8",
    )

    with pytest.raises(SimulatorAdvisorEvidenceError, match="duplicate field"):
        recommend(n_wires=22, evidence=path)


def test_recommendation_is_json_compatible() -> None:
    report = _report()
    decision = recommend(
        n_wires=22,
        environment=_environment(report),
        available_engines=ALL_ENGINES,
    )

    restored = json.loads(json.dumps(decision.to_dict()))
    assert restored["recommended_engine"] == "flagquantum_native"

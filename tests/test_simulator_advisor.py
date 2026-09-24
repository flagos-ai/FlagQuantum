from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum.benchmarking.simulator_compare import build_workload
from flagquantum.ecosystem.simulators import (
    SimulatorAdvisorEvidenceError,
    _calibration,
    recommend,
)
from flagquantum.ecosystem.simulators._evidence import bundled_report

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[1]
REPORT = (
    ROOT
    / "benchmarks"
    / "results"
    / "comparison"
    / "simulator_comparison_cpu_arm64_20260923.json"
)
WORKLOAD_MANIFEST = ROOT / "benchmarks" / "manifests" / "simulator_workload_ir_v1.json"
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


def test_workload_manifest_binds_builder_ir_to_existing_timing_rows() -> None:
    report = _report()
    raw = WORKLOAD_MANIFEST.read_bytes()
    manifest = json.loads(raw)

    assert manifest["schema"] == "flagquantum.simulator_workload_ir_manifest.v1"
    assert manifest["benchmark_report"] == {
        "path": REPORT.relative_to(ROOT).as_posix(),
        "sha256": hashlib.sha256(REPORT.read_bytes()).hexdigest(),
    }
    assert hashlib.sha256(raw).hexdigest() == (
        "0e8d5ec9797b43c65e1c668abc90e7190e93510ef6633f0c9c744dfa7a0147d0"
    )

    measured_by_fingerprint = {
        row["workload_fingerprint"]: row for row in report["rows"]
    }
    for entry in manifest["entries"]:
        workload = entry["workload"]
        circuit = build_workload(n_wires=workload["n_wires"], layers=workload["layers"])
        measured = measured_by_fingerprint[entry["workload_fingerprint"]]

        assert entry["ir_content_hash"] == circuit.to_ir().content_hash
        assert entry["workload"] == measured["workload"]


def test_advisor_recommends_only_for_an_exact_manifest_circuit() -> None:
    report = _report()
    circuit = build_workload(n_wires=22, layers=2)

    decision = recommend(
        circuit,
        environment=_environment(report),
        available_engines=ALL_ENGINES,
    )

    assert decision.status == "recommended"
    assert decision.recommended_engine == "flagquantum_native"
    assert decision.circuit_matches is True
    assert decision.circuit_ir_hash == circuit.to_ir().content_hash
    assert (
        decision.workload_manifest_source
        == WORKLOAD_MANIFEST.relative_to(ROOT).as_posix()
    )
    assert (
        decision.workload_manifest_sha256
        == hashlib.sha256(WORKLOAD_MANIFEST.read_bytes()).hexdigest()
    )


def test_advisor_rejects_same_width_and_gate_count_with_different_ir() -> None:
    report = _report()
    measured = build_workload(n_wires=22, layers=2).to_ir()
    first = measured.instructions[0]
    changed_first = replace(
        first,
        params={**first.params, "theta": float(first.params["theta"]) + 0.001},
    )
    changed = replace(
        measured,
        instructions=(changed_first, *measured.instructions[1:]),
    )

    assert changed.n_wires == measured.n_wires
    assert len(changed.instructions) == len(measured.instructions)
    assert changed.content_hash != measured.content_hash

    decision = recommend(
        changed,
        environment=_environment(report),
        available_engines=ALL_ENGINES,
    )

    assert decision.status == "insufficient_evidence"
    assert decision.recommended_engine is None
    assert decision.reason == "circuit_not_measured"
    assert decision.circuit_matches is False
    assert decision.workload_matches is False
    assert all(
        "circuit_not_measured" in candidate.exclusion_reasons
        for candidate in decision.candidates
    )


def test_advisor_live_calibrates_an_unmeasured_circuit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    circuit = fq.Circuit(2, dtype=torch.complex128).h(0).cx(0, 1)

    def fake_runner(engine: str):
        delay = 0.002 if engine == "flagquantum_native" else 0.0002

        def execute(ir):
            time.sleep(delay)
            state = torch.zeros(ir.shape, dtype=torch.complex128)
            state[..., 0] = 2**-0.5
            state[..., -1] = 2**-0.5
            return state

        return execute

    monkeypatch.setattr(_calibration, "_runner", fake_runner)
    decision = recommend(
        circuit,
        available_engines={
            "flagquantum_native": "0.2.0",
            "qiskit_aer": "0.17.2",
        },
        calibration_budget_seconds=0.2,
        calibration_warmup=0,
        calibration_repeats=3,
        calibration_use_cache=False,
    )

    assert decision.status == "recommended"
    assert decision.recommended_engine == "qiskit_aer"
    assert decision.reason == "fastest_live_calibrated_engine"
    assert decision.evidence_level == "live_calibration"
    assert decision.circuit_matches is True
    assert decision.confidence == pytest.approx(0.9)
    assert decision.calibration_cache_hit is False
    assert decision.calibration_elapsed_seconds is not None
    assert decision.calibration_elapsed_seconds > 0.0
    assert all(
        candidate.median_seconds is not None
        for candidate in decision.candidates
        if candidate.eligible
    )


def test_advisor_reuses_an_exact_process_local_calibration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    circuit = fq.Circuit(3, dtype=torch.complex128).h(0).cx(0, 2)
    calls = 0

    def fake_runner(engine: str):
        del engine

        def execute(ir):
            nonlocal calls
            calls += 1
            state = torch.zeros(ir.shape, dtype=torch.complex128)
            state[..., 0] = 2**-0.5
            state[..., 5] = 2**-0.5
            return state

        return execute

    monkeypatch.setattr(_calibration, "_runner", fake_runner)
    kwargs = {
        "available_engines": {"flagquantum_native": "0.2.0"},
        "calibration_budget_seconds": 0.2,
        "calibration_warmup": 0,
        "calibration_repeats": 2,
    }

    first = recommend(circuit, **kwargs)
    calls_after_first = calls
    second = recommend(circuit, **kwargs)

    assert first.calibration_cache_hit is False
    assert second.calibration_cache_hit is True
    assert calls == calls_after_first
    assert second.evidence_sha256 == first.evidence_sha256


def test_live_calibration_excludes_a_faster_incorrect_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    circuit = fq.Circuit(2, dtype=torch.complex128).h(0).cx(0, 1)

    def fake_runner(engine: str):
        def execute(ir):
            time.sleep(0.001 if engine == "flagquantum_native" else 0.0001)
            state = torch.zeros(ir.shape, dtype=torch.complex128)
            if engine == "flagquantum_native":
                state[..., 0] = 2**-0.5
                state[..., -1] = 2**-0.5
            else:
                state[..., 0] = 1.0
            return state

        return execute

    monkeypatch.setattr(_calibration, "_runner", fake_runner)
    decision = recommend(
        circuit,
        available_engines={
            "flagquantum_native": "0.2.0",
            "qiskit_aer": "0.17.2",
        },
        calibration_budget_seconds=0.2,
        calibration_warmup=0,
        calibration_repeats=3,
        calibration_use_cache=False,
    )

    qiskit = next(item for item in decision.candidates if item.engine == "qiskit_aer")
    assert decision.recommended_engine == "flagquantum_native"
    assert qiskit.eligible is False
    assert "correctness_not_passed" in qiskit.exclusion_reasons


def test_exact_evidence_does_not_execute_live_calibration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _report()

    def unexpected_runner(engine: str):
        raise AssertionError(f"unexpected calibration for {engine}")

    monkeypatch.setattr(_calibration, "_runner", unexpected_runner)
    decision = recommend(
        build_workload(n_wires=10, layers=2),
        environment=_environment(report),
        available_engines=ALL_ENGINES,
        calibration_budget_seconds=0.2,
    )

    assert decision.evidence_level == "exact"
    assert decision.confidence == 1.0


def test_profile_query_cannot_request_live_calibration() -> None:
    with pytest.raises(ValueError, match="requires a program"):
        recommend(n_wires=22, calibration_budget_seconds=1.0)


def test_advisor_rejects_malformed_ir_hash_evidence() -> None:
    report = bundled_report()
    report["rows"][0]["ir_content_hash"] = "not-a-sha256"

    with pytest.raises(SimulatorAdvisorEvidenceError, match="ir_content_hash"):
        recommend(build_workload(n_wires=10, layers=2), evidence=report)


def test_advisor_rejects_ir_hash_bound_to_inconsistent_metadata() -> None:
    report = bundled_report()
    report["rows"][0]["workload"]["gate_count"] = 81

    with pytest.raises(SimulatorAdvisorEvidenceError, match="inconsistent"):
        recommend(build_workload(n_wires=10, layers=2), evidence=report)


def test_advisor_rejects_mixed_circuit_and_profile_selectors() -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        recommend(build_workload(n_wires=10, layers=2), n_wires=10)


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

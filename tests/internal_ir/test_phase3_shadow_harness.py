from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import pytest

import flagquantum as fq
from flagquantum._compiler.runtime_abi import ExecutionState
from flagquantum._compiler.shadow_contracts import (
    ShadowEvidence,
    ShadowKillReason,
    ShadowMismatch,
    ShadowObservation,
    ShadowPolicy,
    ShadowResultKind,
    ShadowSkipReason,
)
from flagquantum._compiler.shadow_harness import (
    ExplicitShadowHarness,
    ShadowKillSwitch,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests/fixtures/internal_ir/phase3_batch_f_shadow.json"
LEGACY_IDENTITY = "a" * 64
CANDIDATE_IDENTITY = "b" * 64


class _StepClock:
    def __init__(self, elapsed: int = 100) -> None:
        self._value = 0
        self._elapsed = elapsed

    def __call__(self) -> int:
        value = self._value
        self._value += self._elapsed
        return value


def _policy(**changes: object) -> ShadowPolicy:
    values = {
        "enabled": True,
        "max_comparisons": 100,
        "max_input_bytes": 1024,
        "max_evidence_bytes": 4096,
        "max_candidate_time_ns": 1_000_000_000,
        "max_mismatches": 50,
    }
    values.update(changes)
    return ShadowPolicy(**values)


def _observation(
    payload: bytes = b"same",
    *,
    status: ExecutionState = ExecutionState.SUCCEEDED,
    kind: ShadowResultKind = ShadowResultKind.STATE,
    shape: tuple[int, ...] = (2,),
    identity: str = LEGACY_IDENTITY,
) -> ShadowObservation:
    return ShadowObservation(status, kind, shape, payload, identity)


def _fixture_records() -> list[dict[str, str]]:
    return json.loads(FIXTURES.read_text(encoding="utf-8"))["cases"]


def _candidate_for(name: str):
    if name == "status":
        return lambda _: _observation(status=ExecutionState.FAILED)
    if name == "type":
        return lambda _: _observation(kind=ShadowResultKind.PROBABILITIES)
    if name == "shape":
        return lambda _: _observation(shape=(1, 2))
    if name == "value":
        return lambda _: _observation(payload=b"different")
    if name == "identity":
        return lambda _: _observation(identity=CANDIDATE_IDENTITY)
    if name == "candidate_failure":
        return lambda _: (_ for _ in ()).throw(RuntimeError("secret=do-not-leak"))
    return lambda _: _observation()


def _run_fixture(record: dict[str, str]):
    harness = ExplicitShadowHarness(
        _policy(),
        ShadowKillSwitch(),
        clock_ns=_StepClock(),
    )
    return harness.compare(
        b"private-input",
        lambda _: _observation(),
        _candidate_for(record["name"]),
    )


def test_legacy_result_is_authoritative_for_every_comparison_class() -> None:
    for record in _fixture_records():
        authoritative = _observation()
        harness = ExplicitShadowHarness(
            _policy(), ShadowKillSwitch(), clock_ns=_StepClock()
        )
        outcome = harness.compare(
            b"private-input",
            lambda _: authoritative,
            _candidate_for(record["name"]),
        )

        assert outcome.authoritative is authoritative
        assert outcome.candidate_executed is True
        assert outcome.classification is ShadowMismatch(record["classification"])
        assert outcome.evidence is not None
        assert not hasattr(outcome, "candidate")


def test_mismatch_taxonomy_is_closed_and_legacy_runs_before_candidate() -> None:
    assert {item.value for item in ShadowMismatch} == {
        "match",
        "status_mismatch",
        "type_mismatch",
        "shape_mismatch",
        "value_mismatch",
        "identity_mismatch",
        "candidate_failure",
        "limit_breach",
    }
    order: list[str] = []

    def legacy(_: bytes) -> ShadowObservation:
        order.append("legacy")
        return _observation()

    def candidate(_: bytes) -> ShadowObservation:
        order.append("candidate")
        return _observation()

    harness = ExplicitShadowHarness(
        _policy(), ShadowKillSwitch(), clock_ns=_StepClock()
    )
    harness.compare(b"input", legacy, candidate)

    assert order == ["legacy", "candidate"]


def test_fixture_taxonomy_and_privacy_evidence_have_stable_identities() -> None:
    for record in _fixture_records():
        outcome = _run_fixture(record)
        assert outcome.classification is ShadowMismatch(record["classification"])
        assert (
            outcome.evidence.evidence_identity == record["expected_evidence_identity"]
        )
        assert {item.name for item in fields(ShadowEvidence)}.isdisjoint(
            {"input_payload", "legacy_payload", "candidate_payload", "exception"}
        )
        assert "private-input" not in repr(outcome.evidence)
        assert "do-not-leak" not in repr(outcome.evidence)


def test_fixture_evidence_identities_are_stable_across_python_hash_seeds() -> None:
    script = f"""
import json
import runpy
namespace = runpy.run_path({str(Path(__file__))!r})
records = namespace['_fixture_records']()
identities = [
    namespace['_run_fixture'](record).evidence.evidence_identity
    for record in records
]
print(json.dumps(identities))
"""

    def identities(seed: str) -> list[str]:
        environment = dict(os.environ)
        environment["PYTHONHASHSEED"] = seed
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(completed.stdout)

    expected = [record["expected_evidence_identity"] for record in _fixture_records()]
    assert identities("1") == identities("8675309") == expected


def test_disabled_input_and_comparison_limits_skip_candidate_after_legacy() -> None:
    calls = {"legacy": 0, "candidate": 0}

    def legacy(_: bytes) -> ShadowObservation:
        calls["legacy"] += 1
        return _observation()

    def candidate(_: bytes) -> ShadowObservation:
        calls["candidate"] += 1
        return _observation()

    disabled = ExplicitShadowHarness(_policy(enabled=False), ShadowKillSwitch())
    assert disabled.compare(b"x", legacy, candidate).skip_reason is (
        ShadowSkipReason.POLICY_DISABLED
    )
    limited_input = ExplicitShadowHarness(
        _policy(max_input_bytes=1), ShadowKillSwitch()
    )
    assert limited_input.compare(b"xx", legacy, candidate).skip_reason is (
        ShadowSkipReason.INPUT_LIMIT
    )
    limited_count = ExplicitShadowHarness(
        _policy(max_comparisons=1, max_mismatches=1), ShadowKillSwitch()
    )
    assert limited_count.compare(b"x", legacy, candidate).evidence is not None
    assert limited_count.compare(b"x", legacy, candidate).skip_reason is (
        ShadowSkipReason.COMPARISON_LIMIT
    )
    assert calls == {"legacy": 4, "candidate": 1}


def test_overhead_and_evidence_breakers_trip_without_replacing_legacy() -> None:
    authoritative = _observation(payload=b"legacy")
    overhead_switch = ShadowKillSwitch()
    overhead = ExplicitShadowHarness(
        _policy(max_candidate_time_ns=10),
        overhead_switch,
        clock_ns=_StepClock(11),
    )
    first = overhead.compare(b"x", lambda _: authoritative, lambda _: _observation())
    assert first.authoritative is authoritative
    assert first.classification is ShadowMismatch.LIMIT_BREACH
    assert first.evidence.candidate_time_ns == 11
    assert overhead_switch.reason is ShadowKillReason.OVERHEAD_LIMIT
    second = overhead.compare(b"x", lambda _: authoritative, lambda _: _observation())
    assert second.authoritative is authoritative
    assert second.skip_reason is ShadowSkipReason.KILL_SWITCHED

    evidence_switch = ShadowKillSwitch()
    evidence_limited = ExplicitShadowHarness(
        _policy(max_evidence_bytes=1),
        evidence_switch,
        clock_ns=_StepClock(),
    )
    outcome = evidence_limited.compare(
        b"x", lambda _: authoritative, lambda _: _observation()
    )
    assert outcome.authoritative is authoritative
    assert outcome.candidate_executed is True
    assert outcome.skip_reason is ShadowSkipReason.EVIDENCE_LIMIT
    assert evidence_switch.reason is ShadowKillReason.EVIDENCE_LIMIT


def test_operator_and_mismatch_kill_switches_are_one_way() -> None:
    switch = ShadowKillSwitch()
    assert switch.trip(ShadowKillReason.OPERATOR) is True
    assert switch.trip(ShadowKillReason.OVERHEAD_LIMIT) is False
    assert switch.reason is ShadowKillReason.OPERATOR
    harness = ExplicitShadowHarness(_policy(), switch)
    candidate_called = {"value": False}
    outcome = harness.compare(
        b"x",
        lambda _: _observation(),
        lambda _: candidate_called.update(value=True),
    )
    assert outcome.skip_reason is ShadowSkipReason.KILL_SWITCHED
    assert candidate_called["value"] is False

    mismatch_switch = ShadowKillSwitch()
    mismatch_harness = ExplicitShadowHarness(
        _policy(max_mismatches=1),
        mismatch_switch,
        clock_ns=_StepClock(),
    )
    mismatch_harness.compare(
        b"x", lambda _: _observation(), lambda _: _observation(payload=b"different")
    )
    assert mismatch_switch.reason is ShadowKillReason.MISMATCH_LIMIT


def test_comparison_reservation_is_thread_safe_and_bounded() -> None:
    limit = 16
    lock = threading.Lock()
    calls = {"legacy": 0, "candidate": 0}

    def legacy(_: bytes) -> ShadowObservation:
        with lock:
            calls["legacy"] += 1
        return _observation()

    def candidate(_: bytes) -> ShadowObservation:
        with lock:
            calls["candidate"] += 1
        return _observation()

    harness = ExplicitShadowHarness(
        _policy(max_comparisons=limit, max_mismatches=limit),
        ShadowKillSwitch(),
    )
    with ThreadPoolExecutor(max_workers=8) as executor:
        outcomes = list(
            executor.map(lambda _: harness.compare(b"x", legacy, candidate), range(64))
        )
    assert calls == {"legacy": 64, "candidate": limit}
    assert harness.comparison_count == limit
    assert sum(item.candidate_executed for item in outcomes) == limit


def test_policy_and_observations_are_immutable_and_validate_inputs() -> None:
    policy = _policy()
    observation = _observation()
    with pytest.raises(FrozenInstanceError):
        policy.enabled = False
    with pytest.raises(FrozenInstanceError):
        observation.payload = b"changed"
    with pytest.raises(ValueError):
        _policy(max_mismatches=101)
    with pytest.raises(ValueError):
        ShadowObservation(
            ExecutionState.RUNNING,
            ShadowResultKind.STATE,
            (),
            b"",
            LEGACY_IDENTITY,
        )


def test_harness_has_no_implicit_activation_or_export_path() -> None:
    import flagquantum._compiler.shadow_harness as module

    source = inspect.getsource(module)
    assert "os.environ" not in source
    assert "getenv(" not in source
    for name in ("ExplicitShadowHarness", "ShadowPolicy", "ShadowKillSwitch"):
        assert not hasattr(fq, name)

"""Explicit, private shadow harness with legacy-authoritative safety semantics."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

from .shadow_contracts import (
    ShadowKillReason,
    ShadowMismatch,
    ShadowObservation,
    ShadowOutcome,
    ShadowPolicy,
    ShadowSkipReason,
    classify_observations,
    create_shadow_evidence,
)

ShadowRunner = Callable[[bytes], ShadowObservation]


class ShadowKillSwitch:
    """Thread-safe one-way circuit breaker; a tripped switch cannot be reset."""

    def __init__(self) -> None:
        self._reason: ShadowKillReason | None = None
        self._lock = threading.Lock()

    @property
    def tripped(self) -> bool:
        with self._lock:
            return self._reason is not None

    @property
    def reason(self) -> ShadowKillReason | None:
        with self._lock:
            return self._reason

    def trip(self, reason: ShadowKillReason) -> bool:
        if not isinstance(reason, ShadowKillReason):
            raise ValueError("shadow kill reason must use the closed enum")
        with self._lock:
            if self._reason is None:
                self._reason = reason
                return True
            return False


class ExplicitShadowHarness:
    """Run a candidate only under an explicit policy; always return legacy output."""

    def __init__(
        self,
        policy: ShadowPolicy,
        kill_switch: ShadowKillSwitch,
        *,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        if not isinstance(policy, ShadowPolicy):
            raise ValueError("shadow harness requires an immutable ShadowPolicy")
        if not isinstance(kill_switch, ShadowKillSwitch):
            raise ValueError("shadow harness requires a ShadowKillSwitch")
        if not callable(clock_ns):
            raise ValueError("shadow harness clock must be callable")
        self._policy = policy
        self._kill_switch = kill_switch
        self._clock_ns = clock_ns
        self._comparisons = 0
        self._mismatches = 0
        self._lock = threading.Lock()

    @property
    def policy(self) -> ShadowPolicy:
        return self._policy

    @property
    def kill_switch(self) -> ShadowKillSwitch:
        return self._kill_switch

    @property
    def comparison_count(self) -> int:
        with self._lock:
            return self._comparisons

    @property
    def mismatch_count(self) -> int:
        with self._lock:
            return self._mismatches

    def _reserve_comparison(self) -> bool:
        with self._lock:
            if self._comparisons >= self._policy.max_comparisons:
                return False
            self._comparisons += 1
            return True

    def _record_mismatch(self) -> None:
        with self._lock:
            self._mismatches += 1
            should_trip = self._mismatches >= self._policy.max_mismatches
        if should_trip:
            self._kill_switch.trip(ShadowKillReason.MISMATCH_LIMIT)

    def compare(
        self,
        input_payload: bytes,
        legacy_runner: ShadowRunner,
        candidate_runner: ShadowRunner,
    ) -> ShadowOutcome:
        if not isinstance(input_payload, bytes):
            raise ValueError("shadow input must be immutable bytes")
        if not callable(legacy_runner) or not callable(candidate_runner):
            raise ValueError("shadow runners must be callable")

        authoritative = legacy_runner(input_payload)
        if not isinstance(authoritative, ShadowObservation):
            raise ValueError("legacy runner must return ShadowObservation")
        if not self._policy.enabled:
            return ShadowOutcome(
                authoritative,
                False,
                skip_reason=ShadowSkipReason.POLICY_DISABLED,
            )
        if self._kill_switch.tripped:
            return ShadowOutcome(
                authoritative,
                False,
                skip_reason=ShadowSkipReason.KILL_SWITCHED,
            )
        if len(input_payload) > self._policy.max_input_bytes:
            return ShadowOutcome(
                authoritative,
                False,
                skip_reason=ShadowSkipReason.INPUT_LIMIT,
            )
        if not self._reserve_comparison():
            return ShadowOutcome(
                authoritative,
                False,
                skip_reason=ShadowSkipReason.COMPARISON_LIMIT,
            )
        if self._kill_switch.tripped:
            return ShadowOutcome(
                authoritative,
                False,
                skip_reason=ShadowSkipReason.KILL_SWITCHED,
            )

        started = self._clock_ns()
        candidate: ShadowObservation | None = None
        try:
            observed = candidate_runner(input_payload)
            if not isinstance(observed, ShadowObservation):
                raise TypeError("candidate result has the wrong type")
            candidate = observed
        except Exception:  # noqa: BLE001 - evidence never records exception content.
            classification = ShadowMismatch.CANDIDATE_FAILURE
        else:
            classification = classify_observations(authoritative, candidate)
        elapsed = max(0, int(self._clock_ns()) - int(started))
        exceeded = elapsed > self._policy.max_candidate_time_ns
        bounded_elapsed = min(elapsed, self._policy.max_candidate_time_ns + 1)
        if exceeded:
            classification = ShadowMismatch.LIMIT_BREACH
            self._kill_switch.trip(ShadowKillReason.OVERHEAD_LIMIT)
        evidence = create_shadow_evidence(
            classification,
            input_payload,
            authoritative,
            candidate,
            candidate_time_ns=bounded_elapsed,
            candidate_time_exceeded=exceeded,
        )
        if evidence.encoded_size > self._policy.max_evidence_bytes:
            self._kill_switch.trip(ShadowKillReason.EVIDENCE_LIMIT)
            return ShadowOutcome(
                authoritative,
                True,
                skip_reason=ShadowSkipReason.EVIDENCE_LIMIT,
            )
        if classification is not ShadowMismatch.MATCH:
            self._record_mismatch()
        return ShadowOutcome(
            authoritative,
            True,
            classification,
            evidence,
        )


__all__ = ["ExplicitShadowHarness", "ShadowKillSwitch", "ShadowRunner"]

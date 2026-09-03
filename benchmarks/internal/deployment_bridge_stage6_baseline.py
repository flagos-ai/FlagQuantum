"""Measure the private offline Deployment Bridge Stage 6 scripted connector."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import time
import tracemalloc
from collections.abc import Callable, Sequence
from typing import Any

from flagquantum._compiler.provider_sandbox_connector import (
    CredentialReference,
    SandboxConnectorError,
    SandboxConnectorOperation,
    SandboxConnectorPolicy,
    SandboxConnectorRequest,
    SandboxLifecycleState,
    SandboxTargetAttestation,
    SandboxTerminalState,
    ScriptedSandboxResponse,
    ScriptedSandboxTransport,
    run_scripted_sandbox_connector,
)


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    rank = (len(ordered) - 1) * float(fraction)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _identity(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _build_connector_inputs():
    operation = SandboxConnectorOperation
    state = SandboxLifecycleState
    target = _identity("anonymous-sandbox-target")
    responses = (
        ScriptedSandboxResponse(
            operation.PREFLIGHT,
            state.ADMITTED,
            target,
            10,
            10,
            SandboxConnectorError.NONE,
        ),
        ScriptedSandboxResponse(
            operation.SUBMIT_ONCE,
            state.ACCEPTED,
            target,
            10,
            10,
            SandboxConnectorError.NONE,
        ),
        ScriptedSandboxResponse(
            operation.READ_STATUS,
            state.RUNNING,
            target,
            10,
            10,
            SandboxConnectorError.NONE,
        ),
        ScriptedSandboxResponse(
            operation.READ_RESULT,
            state.SUCCEEDED,
            target,
            10,
            10,
            SandboxConnectorError.NONE,
        ),
    )
    provider = _identity("anonymous-provider-namespace")
    request = SandboxConnectorRequest(
        tuple(item.operation for item in responses),
        _identity("anonymous-request"),
        _identity("anonymous-program"),
        _identity("anonymous-artifact"),
        _identity("anonymous-idempotency"),
        target,
        _identity("anonymous-capability"),
        10,
        True,
        True,
        True,
        True,
        True,
    )
    attestation = SandboxTargetAttestation(
        provider,
        target,
        request.capability_identity,
        _identity("anonymous-attestation"),
        True,
        True,
        100,
    )
    credential = CredentialReference(
        _identity("anonymous-credential-reference"),
        provider,
        _identity("anonymous-sandbox-scope"),
        100,
    )
    transport = ScriptedSandboxTransport(responses)
    policy = SandboxConnectorPolicy(5, 5, 65536, 65536, 100, 100, 100)
    return request, attestation, credential, transport, policy


def _time(
    operation: Callable[[], str], iterations: int
) -> tuple[list[float], list[str]]:
    timings = []
    identities = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(iterations):
            started = time.perf_counter_ns()
            identities.append(operation())
            timings.append((time.perf_counter_ns() - started) / 1_000_000.0)
    finally:
        if gc_was_enabled:
            gc.enable()
    return timings, identities


def measure_case(
    connector_count: int, *, iterations: int, warmup: int
) -> dict[str, Any]:
    request, attestation, credential, transport, policy = _build_connector_inputs()

    def evaluate_batch() -> str:
        evidence_identity = ""
        for _ in range(connector_count):
            report = run_scripted_sandbox_connector(
                request,
                attestation,
                credential,
                transport,
                policy,
            )
            assert report.terminal_state is SandboxTerminalState.SUCCEEDED
            evidence_identity = report.evidence_identity
        return evidence_identity

    for _ in range(warmup):
        evaluate_batch()
    gc.collect()
    timings, identities = _time(evaluate_batch, iterations)
    gc.collect()
    tracemalloc.start()
    try:
        evaluate_batch()
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return {
        "connector_count": connector_count,
        "scripted_connector_p95_ms": round(percentile(timings, 0.95), 6),
        "scripted_connector_peak_host_memory_bytes": int(peak_bytes),
        "deterministic_evidence_identity": len(set(identities)) == 1,
    }


def evaluate(*, iterations: int, warmup: int) -> dict[str, Any]:
    cases = [
        measure_case(count, iterations=iterations, warmup=warmup)
        for count in (10, 100, 1000, 10000)
    ]
    return {
        "schema_version": "1.0",
        "status": "measurement_only",
        "claim_scope": "Deployment Bridge Stage 6 private CPU baseline; not a budget or SLA",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "method": {
            "iterations": iterations,
            "warmup": warmup,
            "clock": "time.perf_counter_ns",
            "timing_gc": "disabled after explicit collection",
            "peak_memory": "tracemalloc",
            "pipeline": "anonymous offline successful scripted connector",
        },
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()
    if args.iterations < 5 or args.warmup < 1:
        raise ValueError("iterations must be >= 5 and warmup must be >= 1")
    print(
        json.dumps(evaluate(iterations=args.iterations, warmup=args.warmup), indent=2)
    )


if __name__ == "__main__":
    main()

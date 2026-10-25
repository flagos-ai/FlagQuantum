"""Observe one approved Clifford+T point; never run a campaign implicitly."""

from __future__ import annotations

import argparse
import cmath
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

import flagquantum as fq
from flagquantum.errors import CapabilityError

from .contract import runtime_metadata, write_json_atomic


def _circuit(n: int, t: int, topology: str) -> fq.Circuit:
    circuit = fq.Circuit(n)
    for wire in range(n):
        circuit.h(wire)
    for layer in range(4):
        circuit.s((layer + 1) % n)
        for wire in range(n - 1):
            circuit.cx(wire, wire + 1)
        if topology == "ring":
            circuit.cx(n - 1, 0)
        if layer < t:
            circuit.t(layer % n)
        if layer + 4 < t:
            circuit.t((layer + 1) % n)
    for wire in range(n):
        circuit.h(wire)
    return circuit


def _reference(n: int, t: int, topology: str) -> torch.Tensor:
    # Scalar basis amplitudes: no production gates, compiler or state engine.
    state = [0j] * (1 << n)
    state[0] = 1 + 0j

    def hadamard(wire: int) -> None:
        mask = 1 << (n - 1 - wire)
        for index in range(1 << n):
            if index & mask == 0:
                a, b = state[index], state[index | mask]
                state[index] = (a + b) / math.sqrt(2)
                state[index | mask] = (a - b) / math.sqrt(2)

    for wire in range(n):
        hadamard(wire)
    for layer in range(4):
        for index in range(1 << n):
            if index & (1 << (n - 1 - ((layer + 1) % n))):
                state[index] *= 1j
        # Permute basis indices, independently of the Circuit builder.
        for control in range(n if topology == "ring" else n - 1):
            target = (control + 1) % n
            permuted = [0j] * (1 << n)
            for index, amplitude in enumerate(state):
                output = index
                if index & (1 << (n - 1 - control)):
                    output ^= 1 << (n - 1 - target)
                permuted[output] = amplitude
            state = permuted
        for ordinal in (layer, layer + 4):
            if ordinal < t:
                wire = (layer + (ordinal >= 4)) % n
                for index in range(1 << n):
                    if index & (1 << (n - 1 - wire)):
                        state[index] *= cmath.exp(1j * math.pi / 4)
    for wire in range(n):
        hadamard(wire)
    return torch.tensor(state, dtype=torch.complex128)


def _peak_rss_bytes() -> int | None:
    if sys.platform == "win32":
        return None
    import resource

    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _errors(result: fq.ExecutionResult, expected: torch.Tensor) -> dict[str, float]:
    values = result.probabilities
    if (
        values is None
        or values.shape != (1, expected.numel())
        or values.dtype != torch.float64
        or values.device.type != "cpu"
        or not bool(torch.isfinite(values).all())
    ):
        raise ValueError("mixed reference requires finite CPU float64 probabilities")
    state = result.to_statevector().reshape(-1)
    if (
        state.shape != expected.shape
        or state.dtype != torch.complex128
        or state.device.type != "cpu"
        or not bool(torch.isfinite(state).all())
    ):
        raise ValueError("mixed reference requires a finite CPU complex128 state")
    norm = float(torch.linalg.vector_norm(state))
    if norm == 0:
        raise ValueError("mixed reference cannot normalize a zero state")
    normalized = state / norm
    reference = expected / torch.linalg.vector_norm(expected)
    density = normalized[:, None] * normalized.conj()[None, :]
    target = reference[:, None] * reference.conj()[None, :]
    return {
        "probability_max_absolute_error": float(
            (values[0] - expected.abs().square()).abs().max()
        ),
        "probability_normalization_error": abs(float(values.sum()) - 1),
        "state_normalization_error": abs(norm * norm - 1),
        "normalized_density_max_absolute_error": float((density - target).abs().max()),
    }


def run_benchmark(
    *, n: int = 4, t: int = 0, topology: str = "chain", mode: str = "statevector"
) -> dict[str, object]:
    """Observe exactly one construction/plan/run; repeats belong to the controller.

    Examples:
        Run ``flagquantum-benchmark run ftqc_mixed_reference --n 4 --t 2``.
    """
    if type(n) is not int or n not in (4, 8):
        raise ValueError("n must be 4 or 8")
    if type(t) is not int or t not in (0, 2, 4, 8):
        raise ValueError("t must be 0, 2, 4 or 8")
    if topology not in ("chain", "ring"):
        raise ValueError("topology must be chain or ring")
    if mode not in ("statevector", "mps", "tensor_network"):
        raise ValueError("mode must be statevector, mps or tensor_network")
    if os.environ.get("WORLD_SIZE", "1") != "1":
        raise ValueError("mixed reference requires one local process")
    torch.set_num_threads(1)
    report: dict[str, object] = {
        **runtime_metadata(
            runner="ftqc_mixed_reference",
            schema="flagquantum.ftqc_mixed_reference.point.v1",
        ),
        "torch_version": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "flagquantum_source_version": fq.__version__,
        "n": n,
        "authored_t_count": t,
        "topology": topology,
        "requested_mode": mode,
        "device": "cpu",
        "precision": "complex128",
        "distribution_semantics": "single_device_fast_path",
        "local_world_size": 1,
        "node_count": 1,
        "rank_ownership": {"0": "whole_workload"},
        "communication_bytes": 0,
        "scalability_claim_allowed": False,
        "release_gate_allowed": False,
        "performance_targets_enabled": False,
        "non_release_evidence": True,
        "blockers": ["bounded static CPU study; no adaptive or capacity qualification"],
        "timing_boundaries": "Initial imports, reference, validation, native state conversion and report IO excluded; lazy imports in timed APIs included. One construct/plan/run, no inner warmup or retry. External watchdog owns process-group RSS and wall budget.",
        "atol": 1e-12,
        "rtol": 0,
    }
    started = time.perf_counter()
    circuit = _circuit(n, t, topology)
    constructed = time.perf_counter()
    options = fq.ExecutionOptions(
        mode=mode,
        device="cpu",
        precision="complex128",
        allow_approximate=False,
        allow_backend_fallback=False,
    )
    try:
        plan = fq.plan(
            circuit, outputs=fq.probabilities(list(range(n))), options=options
        )
        planned = time.perf_counter()
        report["plan"] = plan.to_dict()
        report["ir_sha256"] = hashlib.sha256(
            json.dumps(circuit.to_ir().to_dict(), sort_keys=True).encode()
        ).hexdigest()
        report["optimized_instruction_count"] = plan.summary()["n_instructions"]
        report["optimized_t_count"] = sum(
            instruction["opcode"] == "t"
            for instruction in plan.to_dict()["program"]["instructions"]
        )
        report["construction_seconds"] = constructed - started
        report["planning_seconds"] = planned - constructed
        # Plan serialization above is deliberately outside execution timing.
        execution_start = time.perf_counter()
        result = fq.run(plan)
        executed = time.perf_counter()
    except CapabilityError as error:
        report.update(
            status="refused", diagnostic=str(error), exception_type=type(error).__name__
        )
        return report
    report["execution_seconds"] = executed - execution_start
    report["timed_phase_sum_seconds"] = (
        (constructed - started) + (planned - constructed) + (executed - execution_start)
    )
    report.update(
        _execution_evidence(plan, result, n=n, t=t, topology=topology, mode=mode)
    )
    return report


def _execution_evidence(
    plan: fq.ExecutionPlan,
    result: fq.ExecutionResult,
    *,
    n: int,
    t: int,
    topology: str,
    mode: str,
) -> dict[str, object]:
    report: dict[str, object] = {}
    report["result_summary_before_dense_validation"] = result.summary()
    decision = plan.to_dict()["decision"]
    if (
        not isinstance(decision, dict)
        or decision["mode"] != mode
        or decision["allow_approximate"] is not False
        or decision["allow_backend_fallback"] is not False
        or result.runtime.get("mode") != mode
    ):
        report.update(
            status="failed", diagnostic="plan changed the exact requested route"
        )
        return report
    if mode == "mps":
        config = getattr(result.native(), "config", None)
        cutoff, maximum = getattr(config, "cutoff", None), getattr(
            config, "max_bond", None
        )
        report["mps_configuration"] = {"cutoff": cutoff, "max_bond": maximum}
        if (
            config is None
            or cutoff != 0.0
            or maximum is not None
            or result.runtime.get("truncation_error") != 0.0
        ):
            report.update(
                status="failed",
                diagnostic="MPS exactness evidence is missing or truncation is enabled",
            )
            return report
    if mode == "tensor_network":
        native = result.native()
        report["tensor_network_configuration"] = {
            "contraction_strategy": getattr(native, "contraction_strategy", None),
            "max_intermediate_size": getattr(native, "max_intermediate_size", None),
            "max_intermediate_bytes": getattr(native, "max_intermediate_bytes", None),
        }
    errors = _errors(result, _reference(n, t, topology))
    report["errors"] = errors
    report["status"] = (
        "completed" if all(value <= 1e-12 for value in errors.values()) else "failed"
    )
    report["producer_peak_rss_bytes_after_validation"] = _peak_rss_bytes()
    report["output_probability_bytes"] = (1 << n) * 8
    report["dense_validation_density_bytes_each"] = (1 << (2 * n)) * 16
    return report


def main() -> int:
    """Write one point, preserving numerical failure before returning nonzero."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, choices=(4, 8), default=4)
    parser.add_argument("--t", type=int, choices=(0, 2, 4, 8), default=0)
    parser.add_argument("--topology", choices=("chain", "ring"), default="chain")
    parser.add_argument(
        "--mode",
        choices=("statevector", "mps", "tensor_network"),
        default="statevector",
    )
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    report = run_benchmark(n=args.n, t=args.t, topology=args.topology, mode=args.mode)
    if args.json_output is not None:
        write_json_atomic(args.json_output, report)
    print(json.dumps(report, sort_keys=True))
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())

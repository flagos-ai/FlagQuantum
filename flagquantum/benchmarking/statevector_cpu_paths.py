#!/usr/bin/env python3
"""Path-decomposed CPU statevector performance characterization.

``statevector_local`` reports one aggregate number for a mixed circuit. That is
enough to detect a broad regression, but it cannot attribute a change to the
statevector path that actually moved, because the CPU fast paths differ per gate
family: single-qubit rotations run through fused matrix regions, CX ladders run
through permutations, diagonal gates have their own elementwise kernel, and
joint marginals are recovered from Pauli contractions rather than from a state
reduction.

This runner measures those paths separately on CPU, records the runtime
statistics the engine already publishes for each case, and fails closed when a
case stops agreeing with its reference.

Reference scope. For the statevector cases the reference is the same-IR
sequential dense-matrix application, which itself calls ``_apply_matrix``. It
therefore detects fusion, routing, permutation, and composition errors, but not
a defect inside ``_apply_matrix``. For the marginal case the reference
re-simulates the program and reduces the probabilities of that state over the
complement of the requested wires, so the reported ratio spans the whole program
on both sides and pins the reduction semantics rather than the statevector
kernel; statevector kernels are pinned by the other cases. Independent
small-width unitary references live in the unit test suite, not here.

Why the marginal case prepares its state with gates. ``Circuit.state()`` and
``Circuit.initial_state()`` honour the initial state a circuit carries in
``circuit_param["inputs"]``, but local ``fq.run`` executes only the circuit IR,
whose metadata does not carry that tensor; it therefore starts from |0...0>.
The two entry points disagree whenever ``inputs`` is set, so a marginal case
built on ``inputs`` would measure a different state than the reference it is
compared against. The four kernel cases stay on ``inputs`` because they read the
state through ``state()``/``initial_state()`` and are self-consistent; the
marginal case prepares its state with gates so that both entry points agree.

Scope of the evidence. This payload describes one host under local CPU
execution. It never promotes a distributed scalability claim and never licenses
a release gate.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

import flagquantum as fq
from flagquantum.benchmarking.contract import runtime_metadata, write_json_atomic
from flagquantum.benchmarking.statevector_local import sequential_reference

SCHEMA = "flagquantum.statevector.cpu_paths.v1"
DEFAULT_SEED = 4417
DEFAULT_N_WIRES = 20
DEFAULT_LAYERS = 8
DEFAULT_MARGINAL_WIRES = 8
STATEVECTOR_TOLERANCE = 2e-5
MARGINAL_TOLERANCE = 1e-6
# Gate on the standard error of the point estimate, not on the spread of the
# samples. The spread is a property of the host and does not shrink with more
# iterations, so a spread gate would reject a correctly working harness on a
# busy machine and accept it on an idle one. The standard error does shrink,
# and it is what decides whether a claimed ratio is separable from noise.
MAX_RELATIVE_STANDARD_ERROR = 0.25
MINIMUM_ITERATIONS = 2


@dataclass(frozen=True)
class PathCase:
    """One measurable CPU statevector path."""

    name: str
    gate_family: str
    reference: str
    executes_public_program: bool = False


CASES: tuple[PathCase, ...] = (
    PathCase(
        name="rotation_chain",
        gate_family="non-diagonal single-qubit rotations",
        reference="same_ir_sequential_dense_matrix_application",
    ),
    PathCase(
        name="diagonal_chain",
        gate_family="diagonal single-qubit rotations",
        reference="same_ir_sequential_dense_matrix_application",
    ),
    PathCase(
        name="cx_chain",
        gate_family="nearest-neighbour CX ladder",
        reference="same_ir_sequential_dense_matrix_application",
    ),
    PathCase(
        name="mixed_chain",
        gate_family="interleaved rotations and CX",
        reference="same_ir_sequential_dense_matrix_application",
    ),
    PathCase(
        name="marginal_probabilities",
        gate_family="joint marginal probabilities over k wires",
        reference=(
            "resimulated_state_reduced_over_the_complement_of_the_requested_wires"
        ),
        executes_public_program=True,
    ),
)

_CASE_BY_NAME = {case.name: case for case in CASES}

# The CPU kernel switches this runner's numbers depend on. Both are read afresh
# on every dispatch, so the payload records what was in force rather than what
# the default is; ``source`` distinguishes "the caller set this" from "the code
# default applied", because for a switch whose default is off those two produce
# very different states.
_CPU_KERNEL_SWITCHES: tuple[tuple[str, bool], ...] = (
    ("FQ_CPU_CX_SEQUENCE_GATHER", True),
    ("FQ_CPU_SINGLE_WIRE_ELEMENTWISE", False),
)


def cpu_kernel_switch_state() -> dict[str, dict[str, Any]]:
    """Return the resolved state of each CPU kernel switch for this process."""
    state: dict[str, dict[str, Any]] = {}
    for name, default in _CPU_KERNEL_SWITCHES:
        raw = os.environ.get(name)
        value = (raw if raw is not None else ("1" if default else "0")).strip().lower()
        if default:
            effective = value not in {"0", "false", "off", "no"}
        else:
            effective = value in {"1", "true", "on", "yes"}
        state[name] = {
            "effective": effective,
            "source": "environment" if raw is not None else "code_default",
            "raw": raw,
        }
    return state


def _angles(*, n_wires: int, layers: int) -> torch.Tensor:
    return torch.linspace(
        -0.37,
        0.41,
        steps=layers * n_wires * 3,
        dtype=torch.float32,
    ).reshape(layers, n_wires, 3)


def _normalized_initial_state(
    *, n_wires: int, batch_size: int, seed: int
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    initial = torch.randn(
        batch_size,
        2**n_wires,
        dtype=torch.complex64,
        generator=generator,
    )
    return initial / torch.linalg.vector_norm(initial, dim=-1, keepdim=True)


def build_rotation_chain(
    *, n_wires: int, layers: int, batch_size: int, seed: int
) -> fq.Circuit:
    """Non-diagonal single-qubit rotations, one per wire per layer."""
    values = _angles(n_wires=n_wires, layers=layers)
    circuit = fq.Circuit(
        n_wires,
        bsz=batch_size,
        device="cpu",
        inputs=_normalized_initial_state(
            n_wires=n_wires, batch_size=batch_size, seed=seed
        ),
    )
    for layer in range(layers):
        for wire in range(n_wires):
            if layer % 2 == 0:
                circuit.ry(wire, values[layer, wire, 0])
            else:
                circuit.rx(wire, values[layer, wire, 1])
    return circuit


def build_diagonal_chain(
    *, n_wires: int, layers: int, batch_size: int, seed: int
) -> fq.Circuit:
    """Diagonal single-qubit rotations, one per wire per layer."""
    values = _angles(n_wires=n_wires, layers=layers)
    circuit = fq.Circuit(
        n_wires,
        bsz=batch_size,
        device="cpu",
        inputs=_normalized_initial_state(
            n_wires=n_wires, batch_size=batch_size, seed=seed
        ),
    )
    for layer in range(layers):
        for wire in range(n_wires):
            circuit.rz(wire, values[layer, wire, 2])
    return circuit


def build_cx_chain(
    *, n_wires: int, layers: int, batch_size: int, seed: int
) -> fq.Circuit:
    """A nearest-neighbour CX ladder repeated once per layer."""
    circuit = fq.Circuit(
        n_wires,
        bsz=batch_size,
        device="cpu",
        inputs=_normalized_initial_state(
            n_wires=n_wires, batch_size=batch_size, seed=seed
        ),
    )
    for _ in range(layers):
        for wire in range(n_wires - 1):
            circuit.cx(wire, wire + 1)
    return circuit


def build_mixed_chain(
    *, n_wires: int, layers: int, batch_size: int, seed: int
) -> fq.Circuit:
    """Rotations interleaved with a CX ladder, so fusion regions stay fragmented."""
    values = _angles(n_wires=n_wires, layers=layers)
    circuit = fq.Circuit(
        n_wires,
        bsz=batch_size,
        device="cpu",
        inputs=_normalized_initial_state(
            n_wires=n_wires, batch_size=batch_size, seed=seed
        ),
    )
    for layer in range(layers):
        for wire in range(n_wires):
            circuit.ry(wire, values[layer, wire, 0])
        for wire in range(n_wires - 1):
            circuit.cx(wire, wire + 1)
    return circuit


def build_marginal_circuit(
    *, n_wires: int, layers: int, batch_size: int, seed: int
) -> fq.Circuit:
    """An entangled state whose joint marginals are recovered over k wires.

    The state is prepared with gates rather than an ``inputs`` tensor so that
    ``fq.run`` and ``circuit.state()`` agree; see the module docstring.
    """
    values = _angles(n_wires=n_wires, layers=layers)
    circuit = fq.Circuit(n_wires, bsz=batch_size, device="cpu")
    for layer in range(layers):
        for wire in range(n_wires):
            circuit.h(wire)
        for wire in range(n_wires - 1):
            circuit.cx(wire, wire + 1)
        circuit.rz(layer % n_wires, values[layer, layer % n_wires, 2])
    return circuit


_BUILDERS: dict[str, Callable[..., "fq.Circuit"]] = {
    "rotation_chain": build_rotation_chain,
    "diagonal_chain": build_diagonal_chain,
    "cx_chain": build_cx_chain,
    "mixed_chain": build_mixed_chain,
    "marginal_probabilities": build_marginal_circuit,
}


def _percentile(samples: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile, so no numpy dependency is introduced."""
    ordered = sorted(samples)
    if len(ordered) == 1:
        return ordered[0]
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def _timing(samples: tuple[float, ...]) -> dict[str, Any]:
    mean = statistics.fmean(samples)
    deviation = statistics.pstdev(samples) if len(samples) > 1 else 0.0
    standard_error = deviation / math.sqrt(len(samples)) if len(samples) > 1 else 0.0
    relative_standard_error = standard_error / mean if mean else 0.0
    return {
        "samples_seconds": samples,
        "sample_count": len(samples),
        "median_seconds": statistics.median(samples),
        "mean_seconds": mean,
        "min_seconds": min(samples),
        "max_seconds": max(samples),
        "p95_seconds": _percentile(samples, 0.95),
        "coefficient_of_variation": deviation / mean if mean else 0.0,
        "relative_standard_error": relative_standard_error,
        "resolvable_ratio": 1.0 + 2.0 * relative_standard_error,
    }


def _measure(
    function: Callable[[], torch.Tensor], *, warmup: int, iterations: int
) -> tuple[tuple[float, ...], torch.Tensor]:
    output = function()
    for _ in range(warmup):
        output = function()
    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        output = function()
        samples.append(time.perf_counter() - started)
    return tuple(samples), output


def _measure_pair(
    candidate: Callable[[], torch.Tensor],
    reference: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iterations: int,
) -> tuple[tuple[float, ...], tuple[float, ...], torch.Tensor, torch.Tensor]:
    """Time the candidate and its reference adjacent inside every iteration.

    Interleaving the two measurements gives a pairwise ratio: each pair sees the
    same contention and the same clock state, so a slow iteration moves both
    sides. On the reference host this did *not* reduce the observed spread --
    measured coefficients of variation for the paired ratio were as large as
    those of the candidate alone (0.02 to 0.99 at 6 wires, 0.24 to 0.44 at 20
    wires), because the jitter here is per-call rather than drift over the run.
    The interleaving is kept because the pairwise ratio remains the more
    comparable quantity, and the spread it fails to cancel is handled by gating
    on the standard error instead; see ``_timing``.
    """
    candidate_output = candidate()
    reference_output = reference()
    for _ in range(warmup):
        candidate_output = candidate()
        reference_output = reference()
    candidate_samples = []
    reference_samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        candidate_output = candidate()
        candidate_samples.append(time.perf_counter() - started)
        started = time.perf_counter()
        reference_output = reference()
        reference_samples.append(time.perf_counter() - started)
    return (
        tuple(candidate_samples),
        tuple(reference_samples),
        candidate_output,
        reference_output,
    )


def _speedup_samples(
    candidate_samples: Sequence[float], reference_samples: Sequence[float]
) -> tuple[float, ...]:
    """Per-iteration reference/candidate ratios, so drift cancels pairwise."""
    return tuple(
        reference / candidate if candidate else math.inf
        for candidate, reference in zip(
            candidate_samples, reference_samples, strict=True
        )
    )


def _direct_marginal_probabilities(
    state: torch.Tensor, wires: tuple[int, ...]
) -> torch.Tensor:
    """Reduce probabilities over the complement of ``wires``, big-endian per wire.

    ``wires`` must be sorted; the surviving axes then come out in wire order,
    which is the order the measurement path reports.
    """
    n_wires = int(round(math.log2(state.shape[-1])))
    probabilities = torch.abs(state) ** 2
    unselected = tuple(index + 1 for index in range(n_wires) if index not in set(wires))
    if not unselected:
        return probabilities
    expanded = probabilities.reshape(probabilities.shape[0], *([2] * n_wires))
    return expanded.sum(dim=unselected).reshape(probabilities.shape[0], -1)


def _case_arguments(
    *,
    n_wires: int,
    layers: int,
    batch_size: int,
    seed: int,
    warmup: int,
    iterations: int,
) -> dict[str, int]:
    return {
        "n_wires": n_wires,
        "layers": layers,
        "batch_size": batch_size,
        "seed": seed,
        "warmup": warmup,
        "iterations": iterations,
    }


def _stability(speedup: dict[str, Any], samples: Sequence[float]) -> dict[str, Any]:
    """Gate on whether the paired ratio is separable from host noise.

    The gate is a statement about the measurement, not about the library. A
    small ``resolvable_ratio`` at a given sample count is a false negative caused
    by a busy host; it is reported rather than repaired, because raising the
    iteration count to fit the gate on one machine would silently change what
    every other run measured.
    """
    return {
        "max_relative_standard_error": MAX_RELATIVE_STANDARD_ERROR,
        "minimum_sample_count": MINIMUM_ITERATIONS,
        "gated_statistic": "speedup_vs_reference",
        "observed_relative_standard_error": speedup["relative_standard_error"],
        "resolvable_ratio": speedup["resolvable_ratio"],
        "passed": (
            speedup["relative_standard_error"] <= MAX_RELATIVE_STANDARD_ERROR
            and len(samples) >= MINIMUM_ITERATIONS
        ),
    }


def _paired_measurement(
    candidate: Callable[[], torch.Tensor],
    reference: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iterations: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], torch.Tensor, torch.Tensor]:
    """Return candidate timing, reference timing, speedup timing, and outputs."""
    candidate_samples, reference_samples, candidate_output, reference_output = (
        _measure_pair(candidate, reference, warmup=warmup, iterations=iterations)
    )
    return (
        _timing(candidate_samples),
        _timing(reference_samples),
        _timing(_speedup_samples(candidate_samples, reference_samples)),
        candidate_output,
        reference_output,
    )


def _run_statevector_case(
    case: PathCase,
    *,
    n_wires: int,
    layers: int,
    batch_size: int,
    seed: int,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    circuit = _BUILDERS[case.name](
        n_wires=n_wires, layers=layers, batch_size=batch_size, seed=seed
    )
    reference = sequential_reference(circuit)

    def execute() -> torch.Tensor:
        return circuit.state(refresh=True)

    def execute_reference() -> torch.Tensor:
        return sequential_reference(circuit)

    deterministic = bool(torch.equal(execute(), execute()))
    timing, reference_timing, speedup, output, _ = _paired_measurement(
        execute, execute_reference, warmup=warmup, iterations=iterations
    )
    max_error = float(torch.max(torch.abs(output - reference)).item())
    return {
        "case": case.name,
        "gate_family": case.gate_family,
        "reference": case.reference,
        "gate_count": len(circuit),
        "workload": {
            "n_wires": n_wires,
            "batch_size": batch_size,
            "layers": layers,
            "dtype": "complex64",
        },
        **timing,
        "reference_timing": reference_timing,
        "speedup_vs_reference": speedup["median_seconds"],
        "speedup_timing": speedup,
        "correctness": {
            "passed": max_error <= STATEVECTOR_TOLERANCE,
            "max_abs_error": max_error,
            "absolute_tolerance": STATEVECTOR_TOLERANCE,
        },
        "deterministic_across_repeats": deterministic,
        "runtime_statistics": dict(circuit._last_statevector_runtime),
        "stability_gate": _stability(speedup, timing["samples_seconds"]),
    }


def _run_marginal_case(
    case: PathCase,
    *,
    n_wires: int,
    layers: int,
    batch_size: int,
    seed: int,
    warmup: int,
    iterations: int,
    marginal_wires: int,
) -> dict[str, Any]:
    """Measure the shipped marginal request against a direct state reduction.

    Both sides run the whole program, so the ratio spans both the entry point and
    the reduction. The reference must re-simulate: ``circuit.state()`` is cached,
    and reading that cache made the reference a measurement of the reduction
    alone (0.64 ms at 20 wires) against a candidate that re-ran the program,
    which understated the candidate by the whole cost of the simulation.

    The reduction itself is nearly free once the state exists; the shipped path
    reaches the same marginal through ``2**k - 1`` parity expectations, which is
    where its time goes and why ``max_marginal_wires`` has a default at all.
    """
    wires = tuple(range(marginal_wires))
    circuit = _BUILDERS[case.name](
        n_wires=n_wires, layers=layers, batch_size=batch_size, seed=seed
    )
    options = fq.ExecutionOptions(mode="statevector")
    request = fq.probabilities(wires)
    reference = _direct_marginal_probabilities(circuit.state(), wires)

    def execute() -> torch.Tensor:
        return fq.run(circuit, options=options, outputs=request).measurements[0].value

    def execute_reference() -> torch.Tensor:
        return _direct_marginal_probabilities(circuit.state(refresh=True), wires)

    deterministic = bool(torch.equal(execute(), execute()))
    timing, reference_timing, speedup, output, _ = _paired_measurement(
        execute, execute_reference, warmup=warmup, iterations=iterations
    )
    max_error = float(torch.max(torch.abs(output - reference)).item())
    return {
        "case": case.name,
        "gate_family": case.gate_family,
        "reference": case.reference,
        "gate_count": len(circuit),
        "workload": {
            "n_wires": n_wires,
            "batch_size": batch_size,
            "layers": layers,
            "dtype": "complex64",
            "marginal_wires": marginal_wires,
        },
        **timing,
        "reference_timing": reference_timing,
        "speedup_vs_reference": speedup["median_seconds"],
        "speedup_timing": speedup,
        "correctness": {
            "passed": max_error <= MARGINAL_TOLERANCE,
            "max_abs_error": max_error,
            "absolute_tolerance": MARGINAL_TOLERANCE,
        },
        "deterministic_across_repeats": deterministic,
        "runtime_statistics": dict(circuit._last_statevector_runtime),
        "stability_gate": _stability(speedup, timing["samples_seconds"]),
    }


def run_benchmark(
    *,
    n_wires: int = DEFAULT_N_WIRES,
    layers: int = DEFAULT_LAYERS,
    batch_size: int = 1,
    seed: int = DEFAULT_SEED,
    warmup: int = 2,
    iterations: int = 5,
    cases: Sequence[str] | None = None,
    marginal_wires: int = DEFAULT_MARGINAL_WIRES,
    threads: int | None = None,
) -> dict[str, Any]:
    """Measure every requested CPU path and return the evidence payload."""
    if n_wires < 3:
        raise ValueError("n_wires >= 3 required")
    if layers < 1:
        raise ValueError("layers >= 1 required")
    if batch_size < 1:
        raise ValueError("batch_size >= 1 required")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if iterations < MINIMUM_ITERATIONS:
        raise ValueError(f"iterations must be >= {MINIMUM_ITERATIONS}")
    if marginal_wires < 1 or marginal_wires > n_wires:
        raise ValueError("marginal_wires must be between 1 and n_wires")
    if threads is not None and threads < 1:
        raise ValueError("threads must be >= 1")
    requested = tuple(cases) if cases else tuple(case.name for case in CASES)
    unknown = sorted(name for name in requested if name not in _CASE_BY_NAME)
    if unknown:
        raise ValueError(f"unknown cases: {unknown}")
    if threads is not None:
        torch.set_num_threads(threads)

    arguments = _case_arguments(
        n_wires=n_wires,
        layers=layers,
        batch_size=batch_size,
        seed=seed,
        warmup=warmup,
        iterations=iterations,
    )
    results: list[dict[str, Any]] = []
    for name in requested:
        case = _CASE_BY_NAME[name]
        if case.executes_public_program:
            results.append(
                _run_marginal_case(case, marginal_wires=marginal_wires, **arguments)
            )
        else:
            results.append(_run_statevector_case(case, **arguments))

    return {
        **runtime_metadata(
            runner="statevector_cpu_paths",
            schema=SCHEMA,
            device="cpu",
            device_name=platform.processor() or "cpu",
            torch=torch.__version__,
            cpu_count=os.cpu_count(),
            torch_threads=torch.get_num_threads(),
            torch_interop_threads=torch.get_num_interop_threads(),
            omp_num_threads=os.environ.get("OMP_NUM_THREADS"),
            seed=seed,
        ),
        "schema_version": SCHEMA,
        "benchmark": "cpu_statevector_paths",
        "artifact_class": "measured_local_run",
        "claim_evidence_type": "local_performance",
        "distribution_semantics": "single_device_fast_path",
        "scalability_claim_allowed": False,
        "release_gate_allowed": False,
        "workload": {
            "n_wires": n_wires,
            "layers": layers,
            "batch_size": batch_size,
            "warmup": warmup,
            "iterations": iterations,
            "marginal_wires": marginal_wires,
            "cases": list(requested),
        },
        "execution_flags": cpu_kernel_switch_state(),
        "cases": results,
        "all_cases_correct": all(item["correctness"]["passed"] for item in results),
        "all_cases_stable": all(item["stability_gate"]["passed"] for item in results),
        "all_cases_deterministic": all(
            item["deterministic_across_repeats"] for item in results
        ),
        "evidence_limits": (
            "Single-host local CPU characterization. Every number here is "
            "internal: the candidate is the shipped FlagQuantum path on this "
            "host and the reference is another route in the same repository or a "
            "reduction computed from the same state. No cross-framework "
            "comparison was run, so none of these ratios is a claim about any "
            "other library. Repeated 20-wire runs of this harness on the "
            "reference host produced coefficients of variation between 0.24 and "
            "0.44 for a candidate measured on its own, which is wider than some "
            "effects being tracked; each case therefore times its candidate and "
            "its reference adjacent within one iteration and gates stability on "
            "that paired ratio. Read the paired ratio, not a ratio of separately "
            "recorded medians. No distributed scalability claim and no release "
            "gate is licensed. The CPU kernel switches recorded under "
            "``execution_flags`` change which kernel runs, so two payloads are "
            "comparable only when that block agrees; the default state of "
            "``FQ_CPU_SINGLE_WIRE_ELEMENTWISE`` is off and an unset variable "
            "therefore means the pre-existing path, not the new one."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-wires", type=int, default=DEFAULT_N_WIRES)
    parser.add_argument("--layers", type=int, default=DEFAULT_LAYERS)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--marginal-wires", type=int, default=DEFAULT_MARGINAL_WIRES)
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=tuple(case.name for case in CASES),
        help="restrict the run to these paths",
    )
    parser.add_argument(
        "--threads",
        type=int,
        help="set the torch intra-op thread count and record the effective value",
    )
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    payload = run_benchmark(
        n_wires=args.n_wires,
        layers=args.layers,
        batch_size=args.batch_size,
        seed=args.seed,
        warmup=args.warmup,
        iterations=args.iterations,
        cases=args.cases,
        marginal_wires=args.marginal_wires,
        threads=args.threads,
    )
    if args.json_output is not None:
        write_json_atomic(args.json_output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["all_cases_correct"] else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

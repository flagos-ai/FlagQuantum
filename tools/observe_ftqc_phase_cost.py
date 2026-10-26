"""One explicitly admitted research observation; never invoked by CI tests.

Run this file with the admitted installed interpreter, outside the checkout.
No registry, public mode, or production dispatch imports this private tool.
"""

from __future__ import annotations

import argparse
import cmath
import hashlib
import itertools
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    import torch

    import flagquantum as fq

Word = tuple[tuple[str, tuple[int, ...]], ...]
CASES = tuple(f"n{n}t{t}" for n, t in itertools.product((2, 3), (0, 1, 2, 4))) + (
    "hth",
    "bell",
)
ROUTES = ("statevector", "mps", "tensor_network", "candidate")
ATOL = 1e-12


def description(name: str) -> tuple[int, Word]:
    if name not in CASES:
        raise ValueError("outside the approved ten descriptions")
    if name == "hth":
        return 1, (("h", (0,)), ("t", (0,)), ("h", (0,)))
    if name == "bell":
        return 2, (("h", (0,)), ("cx", (0, 1)), ("t", (0,)), ("h", (0,)), ("h", (1,)))
    n, t = int(name[1]), int(name[3])
    word: list[tuple[str, tuple[int, ...]]] = [("h", (q,)) for q in range(n)]
    for layer in range(4):
        word.append(("s", ((layer + 1) % n,)))
        word.extend(("cx", (q, q + 1)) for q in range(n - 1))
        if layer < t:
            word.append(("t", (layer % n,)))
    word.extend(("h", (q,)) for q in range(n))
    return n, tuple(word)


def expansion(word: Word) -> tuple[tuple[complex, Word], ...]:
    b = (cmath.exp(1j * math.pi / 4) - 1) / (1j - 1)
    branches: list[tuple[complex, Word]] = []
    for choices in itertools.product((0, 1), repeat=sum(g == "t" for g, _ in word)):
        selected = iter(choices)
        coefficient = 1 + 0j
        branch: list[tuple[str, tuple[int, ...]]] = []
        for gate, wires in word:
            if gate != "t":
                branch.append((gate, wires))
            elif next(selected):
                coefficient *= b
                branch.append(("s", wires))
            else:
                coefficient *= 1 - b
        branches.append((coefficient, tuple(branch)))
    return tuple(branches)


def reference(name: str) -> torch.Tensor:
    import torch

    from flagquantum.benchmarking.ftqc_mixed_reference import _reference

    n, _ = description(name)
    if name not in ("hth", "bell"):
        return _reference(n, int(name[3]), "chain")
    phase = cmath.exp(1j * math.pi / 4)
    if name == "hth":
        return torch.tensor([1 + phase, 1 - phase], dtype=torch.complex128) / 2
    return torch.tensor(
        [1 + phase, 1 - phase, 1 - phase, 1 + phase], dtype=torch.complex128
    ) / (2 * math.sqrt(2))


def candidate(n: int, word: Word) -> tuple[torch.Tensor, dict[str, Any]]:
    importing = time.perf_counter()
    import torch

    from flagquantum.simulation.stabilizer._phase_overlap import (
        _CliffordState,
        _overlap,
    )

    started = time.perf_counter()
    terms = expansion(word)
    states = [_CliffordState.from_word(n, branch) for _, branch in terms]
    bases = [
        _CliffordState.from_word(
            n, tuple(("x", (q,)) for q in range(n) if index & (1 << (n - 1 - q)))
        )
        for index in range(1 << n)
    ]
    coefficients = torch.tensor([c for c, _ in terms], dtype=torch.complex128)
    built = time.perf_counter()
    gram = torch.tensor(
        [[_overlap(bra, ket) for ket in states] for bra in states],
        dtype=torch.complex128,
    )
    grams_done = time.perf_counter()
    amplitudes = torch.tensor(
        [[_overlap(bra, ket) for ket in states] for bra in bases],
        dtype=torch.complex128,
    )
    overlaps_done = time.perf_counter()
    norm = complex(torch.vdot(coefficients, gram @ coefficients))
    if not math.isfinite(abs(norm)) or abs(norm - 1) > ATOL:
        raise ValueError(f"coherent Gram norm {norm!r} failed before reconstruction")
    state = amplitudes @ coefficients
    values = state.abs().square()
    reconstructed = time.perf_counter()
    return values, {
        "candidate_import_seconds": started - importing,
        "expansion_construction_seconds": built - started,
        "gram_seconds": grams_done - built,
        "basis_overlap_seconds": overlaps_done - grams_done,
        "reconstruction_seconds": reconstructed - overlaps_done,
        "numerical_seconds": reconstructed - started,
        "state_constructions": len(states) + len(bases),
        "overlap_queries": len(states) ** 2 + len(states) * len(bases),
        "public_executions": 0,
        "retained_state_counter_scope": "from_word outputs only; overlap scratch is included in query timings",
        "branch_words": [branch for _, branch in terms],
        "coefficients_real_imag": [[c.real, c.imag] for c, _ in terms],
        "coherent_norm": [norm.real, norm.imag],
        # Validation uses independently computed scalar states, never another query.
        "captured_amplitudes": amplitudes,
        "captured_gram": gram,
        "coefficients": coefficients,
    }


def scalar_branches(n: int, word: Word) -> torch.Tensor:
    """Independent scalar Clifford oracle, without production gates or projectors."""
    import torch

    vectors: list[list[complex]] = []
    for _, branch in expansion(word):
        state = [0j] * (1 << n)
        state[0] = 1 + 0j
        for gate, wires in branch:
            mask = 1 << (n - 1 - wires[0])
            previous = state[:]
            if gate == "h":
                state = [
                    (previous[i & ~mask] + (-1 if i & mask else 1) * previous[i | mask])
                    / math.sqrt(2)
                    for i in range(1 << n)
                ]
            elif gate == "s":
                state = [a * (1j if i & mask else 1) for i, a in enumerate(previous)]
            elif gate == "cx":
                target = 1 << (n - 1 - wires[1])
                state = [previous[i ^ target if i & mask else i] for i in range(1 << n)]
            else:
                raise ValueError("unexpected independent Clifford instruction")
        vectors.append(state)
    return torch.tensor(vectors, dtype=torch.complex128).T


def atomic(path: Path, data: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError("attempt evidence cannot be overwritten")
    temporary = path.with_suffix(".tmp")
    with temporary.open("x") as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def public_values(
    circuit: fq.Circuit, n: int, route: str
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    import flagquantum as fq
    from flagquantum.errors import CapabilityError

    report: dict[str, Any] = {}
    planning = time.perf_counter()
    try:
        plan = fq.plan(
            circuit,
            outputs=fq.probabilities(list(range(n))),
            options=fq.ExecutionOptions(
                mode=cast(Literal["statevector", "mps", "tensor_network"], route),
                device="cpu",
                precision="complex128",
                allow_approximate=False,
                allow_backend_fallback=False,
            ),
        )
    except CapabilityError as error:
        report.update(
            status="refused",
            diagnostic=str(error),
            refusal_stage="planning",
            public_executions=0,
            state_constructions=0,
            overlap_queries=0,
        )
        return None, report
    planned = time.perf_counter()
    report["plan"] = plan.to_dict()
    report["planning_seconds"] = planned - planning
    execution = time.perf_counter()
    report["plan_serialization_seconds"] = execution - planned
    result = fq.run(plan)
    executed = time.perf_counter()
    report["execution_seconds"] = executed - execution
    report.update(
        public_executions=1,
        state_constructions=0,
        overlap_queries=0,
        result_summary=result.summary(),
    )
    decision = plan.to_dict()["decision"]
    if (
        decision["mode"] != route
        or decision["allow_approximate"] is not False
        or decision["allow_backend_fallback"] is not False
        or result.runtime.get("mode") != route
    ):
        raise ValueError("public route or exactness changed")
    report["compiled_t"] = sum(
        i["opcode"] == "t" for i in plan.to_dict()["program"]["instructions"]
    )
    if route == "mps":
        config = getattr(result.native(), "config", None)
        report["mps_configuration"] = {
            "cutoff": getattr(config, "cutoff", None),
            "max_bond": getattr(config, "max_bond", None),
        }
        if (
            config is None
            or config.cutoff != 0
            or config.max_bond is not None
            or result.runtime.get("truncation_error") != 0
        ):
            raise ValueError("missing exact MPS evidence")
    if route == "tensor_network":
        native = result.native()
        report["contraction_configuration"] = {
            key: getattr(native, key, None)
            for key in (
                "contraction_strategy",
                "max_intermediate_size",
                "max_intermediate_bytes",
            )
        }
    if result.probabilities is None:
        raise ValueError("public output is absent")
    values = result.probabilities.reshape(-1)
    report["numerical_seconds"] = (
        report["planning_seconds"] + report["execution_seconds"]
    )
    report["post_execution_evidence_seconds"] = time.perf_counter() - executed
    return values, report


def observe(name: str, route: str, provenance: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    import resource

    import torch

    import flagquantum as fq

    if route not in ROUTES or os.environ.get("WORLD_SIZE", "1") != "1":
        raise ValueError("require one admitted local CPU route")
    if "site-packages" not in str(Path(fq.__file__).resolve()):
        raise ValueError("worker must use installed numerical code outside checkout")
    package = Path(fq.__file__).resolve().parent.parent
    for member, checksum in provenance["package_members_sha256"].items():
        if hashlib.sha256((package / member).read_bytes()).hexdigest() != checksum:
            raise ValueError(f"admitted installed package changed: {member}")
    if torch.__version__ != provenance["torch_version"]:
        raise ValueError("installed runtime version changed")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    # This is a one-shot worker; inter-op ownership ends with its process.
    torch.set_num_interop_threads(1)
    try:
        setup = time.perf_counter()
        n, word = description(name)
        report: dict[str, Any] = {
            "case": name,
            "route": route,
            "status": "running",
            "provenance": provenance,
            "word": word,
            "word_sha256": hashlib.sha256(json.dumps(word).encode()).hexdigest(),
            "n": n,
            "literal_t": sum(g == "t" for g, _ in word),
            "compiled_t": None,
            "compiled_t_scope": "candidate has no compiled plan; public routes replace this with plan count",
            "torch_version": torch.__version__,
            "python_version": sys.version,
            "worker_cwd": str(Path.cwd()),
            "numerical_package": str(Path(fq.__file__).resolve()),
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "module_setup_seconds": setup - started,
            "bootstrap_scope": "script setup/imports before observe excluded here; controller outer wall includes all",
            "dtype": "complex128/float64",
            "atol": ATOL,
            "rtol": 0,
            "performance_targets_enabled": False,
            "release_gate_allowed": False,
            "non_release_evidence": True,
            "blockers": [
                "finite one-to-three-qubit research costs only; no method selection or capacity qualification"
            ],
            "scalability_claim_allowed": False,
            "distribution_semantics": "single_device_fast_path",
            "world_size": 1,
            "local_world_size": 1,
            "node_count": 1,
            "rank_ownership": {"0": "whole_workload"},
            "communication_bytes": 0,
        }
        circuit = fq.Circuit(n)
        for gate, wires in word:
            circuit.gate(gate, wires)
        report["ir_sha256"] = hashlib.sha256(
            json.dumps(circuit.to_ir().to_dict(), sort_keys=True).encode()
        ).hexdigest()
        constructed = time.perf_counter()
        report["description_ir_seconds"] = constructed - setup
        if route == "candidate":
            values, details = candidate(n, word)
            report.update(details)
        else:
            public, details = public_values(circuit, n, route)
            report.update(details)
            if public is None:
                return report
            values = public
        validating = time.perf_counter()
        expected = reference(name)
        if (
            values.shape != (1 << n,)
            or values.dtype != torch.float64
            or values.device.type != "cpu"
            or not bool(torch.isfinite(values).all())
            or bool((values < 0).any())
        ):
            raise ValueError("require finite ordered CPU float64 probabilities")
        errors = {
            "probability_error": float((values - expected.abs().square()).abs().max()),
            "normalization_error": abs(float(values.sum()) - 1),
        }
        if route == "candidate":
            independent = scalar_branches(n, word)
            amplitudes = report.pop("captured_amplitudes")
            gram = report.pop("captured_gram")
            report.pop("coefficients")
            report["complex_capture"] = {
                "amplitudes_real": amplitudes.real.tolist(),
                "amplitudes_imag": amplitudes.imag.tolist(),
                "gram_real": gram.real.tolist(),
                "gram_imag": gram.imag.tolist(),
            }
            errors["complex_amplitude_error"] = float(
                (amplitudes - independent).abs().max()
            )
            errors["complex_gram_error"] = float(
                (gram - independent.conj().T @ independent).abs().max()
            )
        if not all(math.isfinite(e) and 0 <= e <= ATOL for e in errors.values()):
            raise ValueError(f"independent numerical comparison failed: {errors}")
        report.update(
            status="completed",
            errors=errors,
            probabilities=values.tolist(),
            validation_seconds=time.perf_counter() - validating,
            expected_probabilities=expected.abs().square().tolist(),
            worker_elapsed_seconds=time.perf_counter() - started,
            worker_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024),
        )
        return report
    finally:
        torch.set_num_threads(previous_threads)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--route", choices=ROUTES, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    try:
        data = observe(args.case, args.route, json.loads(args.provenance.read_text()))
    except Exception as error:  # Preserve one process-boundary failure, never retry.
        data = {
            "case": args.case,
            "route": args.route,
            "status": "failed",
            "diagnostic": f"{type(error).__name__}: {error}",
        }
    data["worker_outer_seconds_before_io"] = time.perf_counter() - started
    writing = time.perf_counter()
    atomic(args.output, data)
    elapsed = time.perf_counter() - writing
    atomic(
        args.output.with_name("report-io.json"),
        {
            "producer_write_seconds": elapsed,
            "scope": "producer JSON serialization/fsync/replace; sidecar overhead included in controller wall",
            "producer_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        },
    )
    return 0 if data["status"] in ("completed", "refused") else 1


if __name__ == "__main__":
    raise SystemExit(main())

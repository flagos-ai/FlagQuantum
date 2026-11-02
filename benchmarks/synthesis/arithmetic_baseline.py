"""AO-00: static circuit costs and explicitly bounded independent verification.

Run ``python -m benchmarks.synthesis.arithmetic_baseline --output /tmp/ao00.json``.
This is a local compiler baseline, not a performance/release/scalability claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, TypedDict

import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler._construction.target import (
    PHASE_BASIS,
    Target,
    compile_target,
)
from flagquantum.compiler.coupling import CouplingMap
from flagquantum.compiler.resource_estimation import estimate_resources
from flagquantum.core.ir import CircuitIR, Instruction

from .expressions import Case, corpus, exchanged_energy, helper_energy, square_helper
from .oracle import expected_amplitudes, output_width
from .verification import Verification, basis_checks, coherent_check

SCHEMA = "flagquantum.synthesis.arithmetic_baseline.v1"
BASIS = PHASE_BASIS
MAX_QUBITS = 256
MAX_OPERATIONS = 100_000
# Planning ceiling only; the runner never executes these large state buffers.
PLANNING_BYTES = 16 << MAX_QUBITS
MAX_EXECUTION_QUBITS = 20
ROOT = Path(__file__).resolve().parents[2]


class Environment(TypedDict):
    python: str
    torch: str
    platform: str
    torch_threads: int


class Configuration(TypedDict):
    seed: int
    max_qubits: int
    max_operations: int
    max_execution_qubits: int
    large_case_timeout_seconds: int
    worker_threshold_input_bits: int
    state_execution_limit_bytes: int
    basis_input_bits_exhaustive_max: int
    objective: str
    timing_scope: str
    cost_scope: str
    claims: str


class Baseline(TypedDict):
    schema: str
    source_revision: str
    source_hashes: dict[str, str]
    environment: Environment
    configuration: Configuration
    summary: dict[str, int]
    rows: list[dict[str, object]]
    sentinel_targets: list[dict[str, object]]
    spelling_variants: list[dict[str, object]]


@dataclass(frozen=True)
class Costs:
    allocated_qubits: int
    used_qubits: int
    workspace_qubits: int
    operations: int
    operation_counts: dict[str, int]
    depth: int
    two_qubit_operations: int
    planned_complex128_state_bytes: int
    circuit_hash: str


def costs(circuit: CircuitIR, workspace: int) -> Costs:
    """Read all counts from the authoritative estimator, not a second scheduler."""
    resource = estimate_resources(circuit)
    return Costs(
        resource.n_qubits,
        resource.used_qubits,
        workspace,
        resource.n_operations,
        dict(sorted(resource.operation_counts.items())),
        resource.depth,
        sum(len(gate.wires) == 2 for gate in circuit.instructions),
        16 << resource.n_qubits,
        circuit.content_hash,
    )


def qmod_source(case: Case) -> str:
    names = case.expression.arguments
    ports = ",".join(
        f"output {name}:qnum<{width}>"
        for name, width in zip(names, case.widths, strict=True)
    )
    preparation = "".join(
        f"allocate({name});hadamard_transform({name});" for name in names
    )
    return f"qfunc main({ports},output result:qnum) {{{preparation}result={case.expression.text};}}"


def synthesize_case(case: Case, *, phased: bool = False) -> synth.Program:
    inputs = {
        name: synth.UInt(width)
        for name, width in zip(case.expression.arguments, case.widths, strict=True)
    }
    if not phased:
        return synth.synthesize(
            case.expression.function,
            inputs=inputs,
            input_state=synth.InputState.uniform(),
            memory_limit_bytes=PLANNING_BYTES,
        )
    gates: list[Instruction] = []
    mapping: dict[str, tuple[int, ...]] = {}
    position = 0
    for name, width in zip(case.expression.arguments, case.widths, strict=True):
        mapping[name] = tuple(range(position, position + width))
        for wire in mapping[name]:
            gates.extend(
                (
                    Instruction("h", (wire,)),
                    Instruction("phase", (wire,), params={"theta": (wire + 1) * 0.173}),
                )
            )
        position += width
    return synth.synthesize(
        case.expression.function,
        inputs=inputs,
        preparation=CircuitIR(position, tuple(gates), dtype="complex128"),
        preparation_map=mapping,
        memory_limit_bytes=PLANNING_BYTES,
    )


@dataclass(frozen=True)
class Row:
    case: str
    expression: str
    widths: tuple[int, ...]
    output_width: int
    status: Literal["compiled", "refused", "timeout"]
    resources: Costs | None
    strategy: str | None
    verification: Verification | None
    synthesis_and_planning_seconds: float | None
    diagnostic_codes: tuple[str, ...]
    candidate_seconds: tuple[tuple[str, float], ...] = ()
    selected_target_seconds: float | None = None
    worker_wall_seconds: float | None = None


def measure(case: Case) -> Row:
    started = time.perf_counter()
    try:
        program = synthesize_case(case)
    except synth.SynthesisError as error:
        return Row(
            case.identifier,
            case.expression.text,
            case.widths,
            output_width(case),
            "refused",
            None,
            None,
            None,
            time.perf_counter() - started,
            tuple(d.code for d in error.diagnostics),
        )
    elapsed = time.perf_counter() - started
    if len(program.registers["result"]) != output_width(case):
        raise AssertionError(
            f"complete-domain output width mismatch: {case.identifier}"
        )
    count, exhaustive = basis_checks(case, program)
    coherent = sum(case.widths) <= 6 and program.circuit.n_wires <= MAX_EXECUTION_QUBITS
    errors: list[float] = []
    leakages: list[float] = []
    if coherent:
        for phased in (False, True):
            amplitude_error, leakage = coherent_check(
                case,
                synthesize_case(case, phased=True) if phased else program,
                phased=phased,
            )
            errors.append(amplitude_error)
            leakages.append(leakage)
    selected = program.selection.selected
    assert selected is not None and selected.arithmetic is not None
    verified = Verification(
        count,
        exhaustive,
        True,
        2 if coherent else 0,
        max(errors) if errors else None,
        max(leakages) if leakages else None,
        (
            "exact gate interpretation; bounded full-state verification"
            if coherent
            else "exact gate interpretation; no quantum state executed"
        ),
    )
    return Row(
        case.identifier,
        case.expression.text,
        case.widths,
        output_width(case),
        "compiled",
        costs(program.circuit, selected.arithmetic.peak_workspace),
        selected.strategy,
        verified,
        elapsed,
        (),
        tuple(
            (row.strategy, row.construction_seconds) for row in program.selection.rows
        ),
        selected.target.elapsed_seconds if selected.target else None,
    )


def manual_reference() -> CircuitIR:
    """Independent two-bit derivation: b^d, c^bd, a(1-b)^bcd, ab."""
    gates = [Instruction("h", (wire,)) for wire in range(4)]
    gates.extend(
        Instruction(name, wires)
        for name, wires in (
            ("cx", (1, 7)),
            ("cx", (3, 7)),
            ("cx", (2, 6)),
            ("ccx", (1, 3, 6)),
            ("x", (1,)),
            ("ccx", (0, 1, 5)),
            ("x", (1,)),
            ("ccx", (1, 2, 8)),
            ("ccx", (8, 3, 5)),
            ("ccx", (1, 2, 8)),
            ("ccx", (0, 1, 4)),
        )
    )
    return CircuitIR(9, tuple(gates), dtype="complex128")


@dataclass(frozen=True)
class TargetRow:
    implementation: str
    profile: str
    resources: Costs


def workspace_of(program: synth.Program) -> int:
    selected = program.selection.selected
    assert selected is not None and selected.arithmetic is not None
    return selected.arithmetic.peak_workspace


def sentinel_targets() -> list[TargetRow]:
    """Fixed source -> target conversion; this does not rerun candidate selection."""
    case = next(case for case in corpus() if case.identifier == "energy-2x2")
    native = synthesize_case(case)
    migrated = synth.synthesize(
        qmod_source(case), source_format="qmod", memory_limit_bytes=PLANNING_BYTES
    )
    # Independent whole-state oracle covers both Qmod and the hand-derived circuit.
    coherent_check(case, migrated, phased=False)
    reference = manual_reference()
    state = fq.run(reference).statevector().reshape(-1)
    expected = expected_amplitudes(
        case, {"x": (0, 1), "y": (2, 3), "result": (4, 5, 6, 7)}, 9
    )
    for index, amplitude in expected.items():
        state[index] -= amplitude
    if float(state.abs().max()) > 1e-12:
        raise AssertionError(
            "independent hand-derived Circuit disagrees with mathematics"
        )
    rows: list[TargetRow] = []
    for name, source, workspace in (
        ("python", native.circuit, workspace_of(native)),
        ("qmod", migrated.circuit, workspace_of(migrated)),
        ("manual", reference, reference.n_wires - 8),
    ):
        for profile in ("logical", "exact_basis_all_to_all", "exact_basis_line"):
            target = Target(
                basis=None if profile == "logical" else BASIS,
                couplings=(
                    (CouplingMap.line(source.n_wires),)
                    if profile == "exact_basis_line"
                    else ()
                ),
            )
            converted = compile_target(source, target, MAX_OPERATIONS)
            rows.append(
                TargetRow(name, profile, costs(converted.executable, workspace))
            )
    return rows


def spelling_variants() -> list[TargetRow]:
    """Record present costs, including differences between equivalent spellings."""
    case = next(case for case in corpus() if case.identifier == "energy-2x2")
    rows = []
    for name, function, helpers in (
        ("helper", helper_energy, (square_helper,)),
        ("exchanged", exchanged_energy, ()),
    ):
        program = synth.synthesize(
            function,
            inputs={"x": synth.UInt(2), "y": synth.UInt(2)},
            functions=helpers,
            input_state=synth.InputState.uniform(),
            memory_limit_bytes=PLANNING_BYTES,
        )
        basis_checks(case, program)
        coherent_check(case, program, phased=False)
        rows.append(
            TargetRow(name, "logical", costs(program.circuit, workspace_of(program)))
        )
    return rows


def revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def bounded_measure(case: Case, timeout_seconds: float = 30.0) -> dict[str, object]:
    """Keep large-domain compilation from hanging the complete benchmark.

    Child startup is outside the inner synthesis timer. A timeout has no cost
    result or correctness claim and is distinct from a compiler refusal.
    """
    if sum(case.widths) <= 12:
        return asdict(measure(case))
    with tempfile.TemporaryDirectory(prefix="fq-ao00-") as folder:
        output = Path(folder) / "case.json"
        started = time.perf_counter()
        try:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "benchmarks.synthesis.arithmetic_baseline",
                    "--case",
                    case.identifier,
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                check=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return asdict(
                Row(
                    case.identifier,
                    case.expression.text,
                    case.widths,
                    output_width(case),
                    "timeout",
                    None,
                    None,
                    None,
                    None,
                    ("benchmark.wall_timeout",),
                    worker_wall_seconds=time.perf_counter() - started,
                )
            )
        payload: object = json.loads(output.read_text())
        if not isinstance(payload, dict):
            raise ValueError("case worker returned a non-object")
        return payload


def run(output: Path) -> None:
    torch.set_num_threads(1)
    rows = []
    for case in corpus():
        row = bounded_measure(case)
        rows.append(row)
        print(case.identifier, row["status"], flush=True)
    source_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path(__file__).parent.glob("*.py"))
    }
    payload: Baseline = {
        "schema": SCHEMA,
        "source_revision": revision(),
        "source_hashes": source_hashes,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "torch_threads": torch.get_num_threads(),
        },
        "configuration": {
            "seed": 20261102,
            "max_qubits": MAX_QUBITS,
            "max_operations": MAX_OPERATIONS,
            "max_execution_qubits": MAX_EXECUTION_QUBITS,
            "large_case_timeout_seconds": 30,
            "worker_threshold_input_bits": 12,
            "state_execution_limit_bytes": 16 << MAX_EXECUTION_QUBITS,
            "basis_input_bits_exhaustive_max": 6,
            "objective": "width",
            "timing_scope": "one synthesize call including capture, selection, target compilation and planning; excludes verification and process startup",
            "cost_scope": "static instruction counts including input preparation and all cleanup; state bytes are not process peak memory",
            "claims": "local CPU correctness and static resources only; no speed, optimality, hardware, release or scalability claim",
        },
        "summary": dict(Counter(str(row["status"]) for row in rows)),
        "rows": rows,
        "sentinel_targets": [asdict(row) for row in sentinel_targets()],
        "spelling_variants": [asdict(row) for row in spelling_variants()],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compare", type=Path, nargs=2, metavar=("BEFORE", "AFTER"))
    parser.add_argument("--case", choices=[case.identifier for case in corpus()])
    args = parser.parse_args()
    if args.compare is not None:
        from .results import compare

        changes = compare(*args.compare)
        print(json.dumps({"changed_resources_or_status": changes}))
        if changes:
            raise SystemExit(1)
        return
    if args.output is None:
        parser.error("--output is required unless --compare is used")
    if args.case is not None:
        torch.set_num_threads(1)
        case = next(case for case in corpus() if case.identifier == args.case)
        args.output.write_text(json.dumps(asdict(measure(case)), sort_keys=True) + "\n")
    else:
        run(args.output)


if __name__ == "__main__":
    main()

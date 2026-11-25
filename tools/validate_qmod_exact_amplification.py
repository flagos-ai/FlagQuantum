"""Independent Circuit and analytic checks, with the inconsistent original retained."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler.resource_estimation import ResourceEstimate

FIXTURES = Path(__file__).resolve().parents[1] / "tests/fixtures/classiq_qmod"
DTYPE = torch.complex128


def preparation_data(case: str) -> tuple[int, tuple[float, ...], tuple[int, ...]]:
    if case == "exact_amplitude_amplification":
        return 1, (0.93, 0.07), (1,)
    if case != "partial_exponential_state_preparation":
        raise ValueError("unknown original")
    weights = tuple(math.exp(0.5 * i) for i in range(32))
    total = math.fsum(weights)
    return 5, tuple(w / total for w in weights), tuple(range(21, 32))


def actual_amplitude(case: str) -> float:
    _, probabilities, marked = preparation_data(case)
    return math.sqrt(math.fsum(probabilities[w] for w in marked))


def load_original(case: str) -> str:
    manifest = json.loads((FIXTURES / "amplification_sources.json").read_text())
    row = next(r for r in manifest["fixtures"] if r["file"] == case + ".qmod")
    filename = row["file"]
    if not isinstance(filename, str):
        raise TypeError("fixture filename must be a string")
    raw = (FIXTURES / filename).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == row["sha256"]
    assert (
        hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == row["git_blob_sha1"]
    )
    assert (
        hashlib.sha256((FIXTURES / manifest["license_file"]).read_bytes()).hexdigest()
        == manifest["license_sha256"]
    )
    return raw.decode()


def corrected_source() -> str:
    # Diagnostic copy only: never replace the original fixture or corpus source.
    return load_original("partial_exponential_state_preparation").replace(
        "0.996625424765961",
        format(actual_amplitude("partial_exponential_state_preparation"), ".17g"),
    )


def reference(case: str, amplitude: float) -> fq.Circuit:
    width, probabilities, marked = preparation_data(case)
    logical = width + 1
    c = fq.Circuit(logical + max(0, logical - 3), dtype=DTYPE)
    a_angles = []
    for bit in range(width):
        mass = math.fsum(p for word, p in enumerate(probabilities) if word & (1 << bit))
        a_angles.append(2 * math.atan2(math.sqrt(mass), math.sqrt(max(0, 1 - mass))))
    rounds = max(0, math.ceil(math.pi / (4 * math.asin(amplitude)) - 0.5 - 1e-14))
    beta = 2 * math.acos(min(1, math.sin(math.pi / (4 * rounds + 2)) / amplitude))

    def prepare(inverse: bool = False) -> None:
        for bit, angle in enumerate(a_angles):
            c.ry(width - 1 - bit, theta=-angle if inverse else angle)
        c.ry(width, theta=-beta if inverse else beta)

    def flip(word: int) -> None:
        zeros = [i for i in range(logical) if not word & (1 << (logical - 1 - i))]
        for i in zeros:
            c.x(i)
        if logical == 2:
            c.cz(0, 1)
        elif logical == 3:
            c.h(2).ccx(0, 1, 2).h(2)
        else:
            c.h(logical - 1).ccx(0, 1, logical)
            for i in range(2, logical - 2):
                c.ccx(logical + i - 2, i, logical + i - 1)
            c.ccx(2 * logical - 4, logical - 2, logical - 1)
            for i in reversed(range(2, logical - 2)):
                c.ccx(logical + i - 2, i, logical + i - 1)
            c.ccx(0, 1, logical).h(logical - 1)
        for i in reversed(zeros):
            c.x(i)

    prepare()
    for _ in range(rounds):
        for word in marked:
            flip(word << 1)
        prepare(True)
        flip(0)
        c.x(0).phase(0, theta=math.pi).x(0).phase(0, theta=math.pi)
        prepare()
    if width == 5:
        permutation = torch.zeros((32, 32), dtype=DTYPE)
        for word in range(32):
            permutation[(word - 18) % 32, word] = 1
        c.any(*range(5), matrix=permutation)
    return c


def validate(case: str) -> dict[str, object]:
    raw = load_original(case)
    corrected = case == "partial_exponential_state_preparation"
    source = corrected_source() if corrected else raw
    program = synth.synthesize(
        source, source_format="qmod", memory_limit_bytes=64 * 1024**2
    )
    resources = program.report["resources"]
    if not isinstance(resources, ResourceEstimate):
        raise TypeError("missing resources")
    width, probabilities, marked = preparation_data(case)
    state = fq.run(program).statevector().reshape(1 << width, -1)
    ref = (
        fq.run(reference(case, actual_amplitude(case)))
        .statevector()
        .reshape(1 << width, -1)
    )
    expected = torch.zeros(1 << width, dtype=DTYPE)
    good_mass = math.fsum(probabilities[w] for w in marked)
    for word in marked:
        expected[(word - 18) % 32 if corrected else word] = math.sqrt(
            probabilities[word] / good_mass
        )
    error = float((state[:, 0] - ref[:, 0]).abs().max())
    analytic_error = float((state[:, 0] - expected).abs().max())
    dirty = float(state[:, 1:].abs().square().sum())
    reference_dirty = float(ref[:, 1:].abs().square().sum())
    norm = float(state.abs().square().sum())
    assert (
        max(error, analytic_error) < 1e-12
        and max(dirty, reference_dirty) < 1e-24
        and abs(norm - 1) < 1e-9
    )
    report: dict[str, object] = {
        "case": case,
        "original_accepted": not corrected,
        "diagnostic_copy": corrected,
        "original_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "validation_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "actual_amplitude": actual_amplitude(case),
        "qubits": resources.n_qubits,
        "operations": resources.n_operations,
        "depth": resources.depth,
        "workspace": resources.n_qubits - width,
        "max_abs_state_error": error,
        "analytic_max_abs_error": analytic_error,
        "workspace_nonzero_probability": dirty,
        "reference_workspace_nonzero_probability": reference_dirty,
        "state_norm": norm,
        "validation": "passed",
    }
    if corrected:
        try:
            synth.synthesize(raw, source_format="qmod")
        except ValueError as e:
            assert "amplification.amplitude" in str(e)
            report["original_refusal"] = str(e)
        else:
            raise AssertionError("inconsistent original was silently accepted")
        unsafe = (
            fq.run(reference(case, 0.996625424765961)).statevector().reshape(32, -1)
        )
        report["unsafe_reference_workspace_nonzero_probability"] = float(
            unsafe[:, 1:].abs().square().sum()
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=[
            "exact_amplitude_amplification",
            "partial_exponential_state_preparation",
        ],
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(validate(args.case), indent=2) + "\n")


if __name__ == "__main__":
    main()

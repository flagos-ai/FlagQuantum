"""Independent gate-level and analytic references for unchanged callback originals."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler.resource_estimation import ResourceEstimate

FIXTURES = Path(__file__).resolve().parents[1] / "tests/fixtures/classiq_qmod"


def reference(case: str) -> fq.Circuit:
    width = 3 if case == "deutsch_jozsa" else 4
    c = fq.Circuit(width, dtype=torch.complex128)

    def hadamards() -> None:
        for bit in range(width):
            c.h(bit)

    hadamards()
    if case == "deutsch_jozsa":
        # max(x xor 4, x and 3) >= 4 iff the three-bit x has MSB zero.
        c.x(0).z(0).x(0)
    elif case == "hidden_shift":
        # Source array index zero is numeric bit zero, physical qubit three.
        c.x(3).cz(3, 2).x(3).cz(1, 0)
        hadamards()
        c.cz(3, 2).cz(1, 0)
    else:
        raise ValueError("unknown callback original")
    hadamards()
    return c


def validate(case: str) -> dict[str, object]:
    manifest = json.loads((FIXTURES / "quantum_callback_sources.json").read_text())
    row = next(r for r in manifest["fixtures"] if r["file"] == case + ".qmod")
    source = FIXTURES / row["file"]
    raw = source.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == row["sha256"]
    assert (
        hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == row["git_blob_sha1"]
    )
    assert (
        hashlib.sha256((FIXTURES / manifest["license_file"]).read_bytes()).hexdigest()
        == manifest["license_sha256"]
    )
    program = synth.synthesize(source, memory_limit_bytes=64 * 1024**2)
    resources = program.report["resources"]
    if not isinstance(resources, ResourceEstimate):
        raise TypeError("missing resources")
    state = fq.run(program).statevector().reshape(-1)
    expected = fq.run(reference(case)).statevector().reshape(-1)
    width = 3 if case == "deutsch_jozsa" else 4
    workspace = resources.n_qubits - width
    data = state.reshape(1 << width, 1 << workspace)
    error = float((data[:, 0] - expected).abs().max())
    dirty = float(data[:, 1:].abs().square().sum())
    analytic = torch.zeros_like(expected)
    analytic[4 if width == 3 else 1] = -1 if width == 3 else 1
    analytic_error = float((data[:, 0] - analytic).abs().max())
    assert max(error, analytic_error) < 1e-12 and dirty < 1e-24
    return {
        "case": case,
        "source_blob": row["git_blob_sha1"],
        "qubits": resources.n_qubits,
        "operations": resources.n_operations,
        "depth": resources.depth,
        "workspace": workspace,
        "max_abs_state_error": error,
        "analytic_max_abs_error": analytic_error,
        "workspace_nonzero_probability": dirty,
        "validation": "passed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=["hidden_shift", "deutsch_jozsa"], required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(validate(args.case), indent=2) + "\n")


if __name__ == "__main__":
    main()

"""Unchanged BPDE full-state checks against independent Strang matrices and Circuit."""

from __future__ import annotations

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
# Source branch zero/one: identity, Z, X. Recorded independently of parsing.
H0 = (-0.530427259238194, 0.536032051970536, 0.196711389481122)
H1 = (-0.531672967442228, 0.534084446291781, 0.196869810923244)


def branch_matrix(
    coefficients: tuple[float, float, float], include_identity: bool = True
) -> torch.Tensor:
    z = torch.diag(torch.tensor([1.0, -1.0], dtype=DTYPE))
    x = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=DTYPE)
    scalar, cz, cx = coefficients
    half_z = torch.matrix_exp(-1j * 0.25 * cz * z)
    full_x = torch.matrix_exp(-1j * 0.5 * cx * x)
    step = half_z @ full_x @ half_z
    if include_identity:
        step = step * complex(math.cos(-0.5 * scalar), math.sin(-0.5 * scalar))
    return step.matrix_power(8)


def reference(delta: float, include_identity: bool = True) -> fq.Circuit:
    u0 = branch_matrix(H0, include_identity)
    u1 = branch_matrix(H1, include_identity)
    matrix = torch.zeros((4, 4), dtype=DTYPE)
    matrix[:2, :2] = u0
    matrix[2:, 2:] = u1
    return (
        fq.Circuit(2, dtype=DTYPE)
        .x(1)
        .h(0)
        .any(0, 1, matrix=matrix)
        .phase(0, theta=4 * delta)
        .h(0)
    )


def validate(delta: float) -> dict[str, object]:
    manifest = json.loads((FIXTURES / "controlled_evolution_sources.json").read_text())
    row = manifest["fixtures"][0]
    raw = (FIXTURES / "bpde.qmod").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == row["sha256"]
    assert (
        hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == row["git_blob_sha1"]
    )
    assert (
        hashlib.sha256((FIXTURES / manifest["license_file"]).read_bytes()).hexdigest()
        == manifest["license_sha256"]
    )
    program = synth.synthesize(
        FIXTURES / "bpde.qmod",
        classical_inputs={"delta_e": delta},
        memory_limit_bytes=64 * 1024**2,
    )
    resources = program.report["resources"]
    if not isinstance(resources, ResourceEstimate):
        raise TypeError("missing resources")
    actual = fq.run(program).statevector().reshape(4, -1)
    expected = fq.run(reference(delta)).statevector().reshape(-1)
    v0 = branch_matrix(H0)[:, 1]
    v1 = branch_matrix(H1)[:, 1] * complex(math.cos(4 * delta), math.sin(4 * delta))
    algebra = torch.cat(((v0 + v1) / 2, (v0 - v1) / 2))
    error = float((actual[:, 0] - expected).abs().max())
    algebra_error = float((actual[:, 0] - algebra).abs().max())
    dirty = float(actual[:, 1:].abs().square().sum())
    no_identity = fq.run(reference(delta, False)).statevector().reshape(2, 2)
    observable_defect = float(
        (
            actual[:, 0].reshape(2, 2).abs().square().sum(1)
            - no_identity.abs().square().sum(1)
        )
        .abs()
        .max()
    )
    assert (
        max(error, algebra_error) < 1e-12 and dirty < 1e-24 and observable_defect > 1e-4
    )
    return {
        "case": "bpde",
        "delta_e": delta,
        "source_blob": row["git_blob_sha1"],
        "qubits": resources.n_qubits,
        "operations": resources.n_operations,
        "depth": resources.depth,
        "workspace": program.report["workspace_qubits"],
        "max_abs_state_error": error,
        "algebra_max_abs_state_error": algebra_error,
        "workspace_nonzero_probability": dirty,
        "identity_omission_readout_probability_defect": observable_defect,
        "dropped_state_is_semantic_data": True,
        "time": 4.0,
        "order": 2,
        "repetitions": 8,
        "validation": "passed",
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delta", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(validate(args.delta), indent=2) + "\n")


if __name__ == "__main__":
    main()

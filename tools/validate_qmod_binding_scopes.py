"""Unchanged quantum-autoencoder full-state checks against an independent Circuit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import TypedDict

import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler.resource_estimation import ResourceEstimate

FIXTURES = Path(__file__).resolve().parents[1] / "tests/fixtures/classiq_qmod"
WEIGHTS = (
    (0.11, -0.23, 0.37, -0.41, 0.53, -0.67, 0.71, -0.83, 0.97, -1.09),
    (-0.17, 0.29, -0.47, 0.61, -0.73, 0.89, -1.01, 1.13, -1.27, 1.39),
)
DATA = ((0.19, -0.31, 0.43, -0.59), (-0.21, 0.34, -0.55, 0.68))


class Validation(TypedDict):
    case: int
    source_blob: str
    weights: tuple[float, ...]
    input_data: tuple[float, ...]
    qubits: int
    operations: int
    depth: int
    workspace: int
    max_abs_state_error: float
    trash_max_probability_error: float
    workspace_nonzero_probability: float
    dropped_coded_is_semantic_data: bool
    validation: str


def reference(case: int) -> fq.Circuit:
    c = fq.Circuit(4, dtype=torch.complex128)
    # Source concatenation is low-to-high: coded[0:2], then trash[0:2].
    for index, value in enumerate(DATA[case]):
        c.ry(3 - index, theta=math.pi * value)
    for layer in range(2):
        for index in range(4):
            c.ry(3 - index, theta=WEIGHTS[case][4 * layer + index])
        for index in range(3):
            c.cx(3 - index, 2 - index)
    for index in range(2):
        c.ry(1 - index, theta=WEIGHTS[case][8 + index])
    return c


def validate(case: int) -> Validation:
    manifest = json.loads((FIXTURES / "binding_scope_sources.json").read_text())
    row = manifest["fixtures"][0]
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
    program = synth.synthesize(
        source,
        classical_inputs={"w": WEIGHTS[case], "input_data": DATA[case]},
        memory_limit_bytes=64 * 1024**2,
    )
    resources = program.report["resources"]
    if not isinstance(resources, ResourceEstimate):
        raise TypeError("missing resource estimate")
    workspace = program.report["workspace_qubits"]
    if not isinstance(workspace, int):
        raise TypeError("missing workspace count")
    result = fq.run(program)
    state = result.statevector().reshape(16, -1)
    expected = fq.run(reference(case)).statevector().reshape(-1)
    error = float((state[:, 0] - expected).abs().max())
    dirty = float(state[:, 1:].abs().square().sum())
    marginal = expected.reshape(4, 4).abs().square().sum(1)
    observed = dict(program.probabilities(result, registers=("trash",)))
    probability_error = max(
        abs(observed.get((i,), 0) - float(marginal[i])) for i in range(4)
    )
    assert max(error, probability_error) < 1e-12 and dirty < 1e-24
    return {
        "case": case,
        "source_blob": row["git_blob_sha1"],
        "weights": WEIGHTS[case],
        "input_data": DATA[case],
        "qubits": resources.n_qubits,
        "operations": resources.n_operations,
        "depth": resources.depth,
        "workspace": workspace,
        "max_abs_state_error": error,
        "trash_max_probability_error": probability_error,
        "workspace_nonzero_probability": dirty,
        "dropped_coded_is_semantic_data": True,
        "validation": "passed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=int, choices=(0, 1), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(validate(args.case), indent=2) + "\n")


if __name__ == "__main__":
    main()

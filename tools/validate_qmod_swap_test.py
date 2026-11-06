"""Preserve the swap-test original refusal and validate a labelled diagnostic copy."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler.resource_estimation import ResourceEstimate
from flagquantum.ecosystem import InteropConversionError

FIXTURES = Path(__file__).resolve().parents[1] / "tests/fixtures/classiq_qmod"
DTYPE = torch.complex128
FIRST = (0.41, -0.17, 0.30, 0.04, 0.55, -0.34, -0.33, 0.54)
SECOND = (-0.30, 0.53, 0.38, 0.06, -0.29, -0.19, 0.67, 0.14)


def diagnostic_source() -> str:
    source = (FIXTURES / "swap_test.qmod").read_text()
    for name, values in (("psi1_amplitudes", FIRST), ("psi2_amplitudes", SECOND)):
        vector = torch.tensor(values, dtype=torch.float64)
        normalized = vector / vector.norm()
        replacement = ", ".join(format(float(v), ".17f") for v in normalized)
        source, count = re.subn(
            rf"({name}: real\[\] = )\[[^\]]*\]", rf"\g<1>[{replacement}]", source
        )
        assert count == 1
    return source


def preparation(values: torch.Tensor) -> torch.Tensor:
    """Independent Householder unitary whose first column is the normalized vector."""
    identity = torch.eye(len(values), dtype=DTYPE)
    delta = identity[:, 0] - values
    return identity.sub(
        torch.outer(delta, delta.conj()).mul(2).div(delta.norm().square())
    )


def reference(first: torch.Tensor, second: torch.Tensor) -> fq.Circuit:
    c = fq.Circuit(7, dtype=DTYPE)
    c.any(1, 2, 3, matrix=preparation(first))
    c.any(4, 5, 6, matrix=preparation(second))
    # Direct permutation is independent of the production three-X Fredkin recipe.
    permutation = torch.zeros((8, 8), dtype=DTYPE)
    for control in (0, 1):
        for a in (0, 1):
            for b in (0, 1):
                source = 4 * control + 2 * a + b
                target = 4 * control + (2 * b + a if control else 2 * a + b)
                permutation[target, source] = 1
    c.h(0)
    for offset in range(3):
        c.any(0, 1 + offset, 4 + offset, matrix=permutation)
    return c.h(0)


def validate() -> dict[str, object]:
    manifest = json.loads((FIXTURES / "swap_test_sources.json").read_text())
    row = manifest["fixtures"][0]
    raw = (FIXTURES / "swap_test.qmod").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == row["sha256"]
    assert (
        hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == row["git_blob_sha1"]
    )
    assert (
        hashlib.sha256((FIXTURES / manifest["license_file"]).read_bytes()).hexdigest()
        == manifest["license_sha256"]
    )
    try:
        synth.synthesize(FIXTURES / "swap_test.qmod")
    except InteropConversionError as import_error:
        assert "normalized" in str(import_error)
        refusal = str(import_error)
    else:
        raise AssertionError("the unchanged non-normalized original must refuse")
    source = diagnostic_source()
    program = synth.synthesize(
        source, source_format="qmod", source_name="normalized-swap-test-diagnostic.qmod"
    )
    resource = program.report["resources"]
    if not isinstance(resource, ResourceEstimate):
        raise TypeError("missing resource report")
    a, b = (torch.tensor(values, dtype=DTYPE) for values in (FIRST, SECOND))
    original_norms = [float(v.abs().square().sum()) for v in (a, b)]
    a, b = a / a.norm(), b / b.norm()
    state = fq.run(program).statevector().reshape(128, -1)
    expected = fq.run(reference(a, b)).statevector().reshape(-1)
    ab, ba = torch.kron(a, b), torch.kron(b, a)
    algebra = torch.cat(((ab + ba) / 2, (ab - ba) / 2))
    error = float((state[:, 0] - expected).abs().max())
    analytic_error = float((state[:, 0] - algebra).abs().max())
    dirty = float(state[:, 1:].abs().square().sum())
    probability = float(state[:64].abs().square().sum())
    overlap_probability = (1 + float(torch.vdot(a, b).abs().square())) / 2
    assert max(error, analytic_error, abs(probability - overlap_probability)) < 1e-12
    assert dirty < 1e-24
    return {
        "case": "swap_test",
        "original_accepted": False,
        "diagnostic_copy": True,
        "original_refusal": refusal,
        "original_squared_norms": original_norms,
        "source_blob": row["git_blob_sha1"],
        "validation_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "qubits": resource.n_qubits,
        "operations": resource.n_operations,
        "depth": resource.depth,
        "workspace": program.report["workspace_qubits"],
        "max_abs_state_error": error,
        "analytic_max_abs_error": analytic_error,
        "workspace_nonzero_probability": dirty,
        "readout_zero_probability": probability,
        "analytic_readout_zero_probability": overlap_probability,
        "dropped_states_are_semantic_data": True,
        "validation": "passed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(validate(), indent=2) + "\n")


if __name__ == "__main__":
    main()

"""Validate original qperm calls with independent Circuits and clean workspace."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

import flagquantum as fq
from flagquantum.compiler.resource_estimation import ResourceEstimate
from flagquantum.services.synthesis import _synthesize

FIXTURES = Path(__file__).resolve().parents[1] / "tests/fixtures/classiq_qmod"
BLOBS = {
    "discrete_quantum_walk": "27185f41232ce8872e17110a0b46c53c43825ad0",
    "molecular_energy_curve": "d294a3e7acbb8eb7825577473196f8981cd6812e",
    "molecule_eigensolver": "d294a3e7acbb8eb7825577473196f8981cd6812e",
}


@dataclass(frozen=True)
class Validation:
    case: str
    source_blob: str
    parameter: float | None
    qubits: int
    operations: int
    depth: int
    workspace: int
    max_abs_state_error: float
    max_workspace_probability: float
    state_norm: float
    named_registers: dict[str, tuple[int, ...]]
    validation: str = "passed"


def validate(case: str, parameter: float | None = None) -> Validation:
    source = FIXTURES / f"{case}.qmod"
    raw = source.read_bytes()
    blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
    if blob != BLOBS[case]:
        raise ValueError("original source blob mismatch")
    walk = case == "discrete_quantum_walk"
    if not walk and (
        parameter is None or not math.isfinite(parameter) or not parameter
    ):
        raise ValueError("molecular examples require one finite nonzero time")
    if walk and parameter is not None:
        raise ValueError("the original walk has no classical parameter")
    inputs: dict[str, list[float]] | None = None
    if not walk:
        assert parameter is not None
        inputs = {"params": [parameter]}
    program = _synthesize(
        source, classical_inputs=inputs, memory_limit_bytes=64 * 1024**2
    )
    resources = program.report["resources"]
    if not isinstance(resources, ResourceEstimate):
        raise TypeError("expected actual Compiler resource estimates")
    workspace = program.report["workspace_qubits"]
    if type(workspace) is not int or workspace < 0:
        raise TypeError("expected the actual workspace count")
    reference = fq.Circuit(resources.n_qubits, dtype=torch.complex128)
    if walk:
        # From x=0, the two coin branches end at 1 and 1023 respectively.
        reference.h(10).x(9)
        for wire in range(9):
            reference.cx(10, wire)
    else:
        assert parameter is not None
        # H = -Y/2, so exp(-itH) = RY(-t), after basis-state |1> preparation.
        reference.x(0).ry(0, theta=-parameter)
    result = fq.run(program)
    actual = result.statevector().reshape(-1)
    expected = fq.run(reference).statevector().reshape(-1)
    error = float((actual - expected).abs().max())
    norm = float(actual.abs().square().sum())
    dirty = actual.reshape(-1, 1 << workspace)[:, 1:]
    leakage = float(dirty.abs().square().sum())
    if (
        not all(math.isfinite(value) for value in (error, norm, leakage))
        or error > 1e-12
        or abs(norm - 1) > 1e-12
        or leakage > 1e-24
    ):
        raise AssertionError(
            f"state/norm/workspace mismatch: {error}, {norm}, {leakage}"
        )
    if walk:
        distribution = program.probabilities(result, registers=("x", "coin"))
        if any(
            abs(distribution.get(key, 0) - 0.5) > 1e-12 for key in ((1, 0), (1023, 1))
        ):
            raise AssertionError("walk integer readout mismatch")
    return Validation(
        case,
        blob,
        parameter,
        resources.n_qubits,
        resources.n_operations,
        resources.depth,
        workspace,
        error,
        leakage,
        norm,
        dict(program.registers),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=tuple(BLOBS), required=True)
    parser.add_argument("--parameter", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = asdict(validate(args.case, args.parameter))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()

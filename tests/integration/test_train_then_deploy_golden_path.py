from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

#: The exported program's instruction order. A deployment binds the trained
#: angles into the circuit it exports, so the QASM a provider would receive is
#: part of what this capability delivers rather than a rendering detail.
_QASM_LINES = (
    "OPENQASM 2.0;",
    'include "qelib1.inc";',
    "qreg q[2];",
    "creg c[2];",
    "cx q[0], q[1];",
    "measure q[0] -> c[0];",
    "measure q[1] -> c[1];",
)


def _run() -> str:
    root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [sys.executable, "-m", "examples.train_parameterized_circuit_then_deploy"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def _value(stdout: str, label: str) -> float:
    line = next(line for line in stdout.splitlines() if line.startswith(label))
    return float(line.split(":", 1)[1].strip())


def test_train_then_deploy_user_example_runs_end_to_end() -> None:
    stdout = _run()

    assert "optimized_parameters:" in stdout
    assert "deployment_counts:" in stdout
    assert "THEORETICAL EXACT DIAGONALIZATION:" in stdout
    assert "Ground state energy: -1.2" in stdout


def test_the_trained_circuit_reaches_the_exact_ground_state() -> None:
    """The example is scored against a diagonalization, not against itself.

    A two-qubit Hamiltonian can be diagonalized exactly, so the run reports
    the full spectrum beside the energy it trained to. That is what makes the
    training result evidence rather than a loss curve: the target is computed
    independently of the optimizer that approached it.
    """

    stdout = _run()

    assert "Full spectrum: [-1.2, -0.8, 0.8, 1.2]" in stdout

    training = _value(stdout, "training_energy")
    assert training == pytest.approx(-1.2, abs=1e-4), training

    gap_line = next(
        line for line in stdout.splitlines() if "Gap to ground state" in line
    )
    gap = abs(float(gap_line.split(":")[1].strip()))
    assert gap < 1e-6, gap_line


def test_the_deployed_package_agrees_with_the_trained_circuit() -> None:
    """A deployment that changed the answer would be the failure to catch.

    The package binds the trained parameters and is executed through the
    deployment path, so its sampled estimate has to land on the energy the
    training reached. The counts come from a finite number of shots, so the
    estimate is compared to a tolerance while the agreement is the claim.
    """

    stdout = _run()

    training = _value(stdout, "training_energy")
    deployed = _value(stdout, "deployment_energy_estimate")
    assert deployed == pytest.approx(training, abs=1e-4), (training, deployed)

    # The sampled distribution is concentrated on one basis state here, and
    # the shot total is what the estimate was formed from.
    counts_line = next(
        line for line in stdout.splitlines() if line.startswith("deployment_counts:")
    )
    shots = sum(int(value) for value in re.findall(r":\s*(\d+)", counts_line))
    assert shots == 1024, counts_line


def test_the_exported_program_carries_the_trained_angles() -> None:
    """An exported program with untrained angles would deploy a different model."""

    stdout = _run()

    for line in _QASM_LINES:
        assert line in stdout, line

    # Each rotation in the export must carry one of the optimized parameters,
    # so the angles are read from both places and compared rather than pinned.
    parameters_line = next(
        line for line in stdout.splitlines() if line.startswith("optimized_parameters:")
    )
    parameters = [float(value) for value in re.findall(r"-?\d+\.\d+", parameters_line)]
    exported = [
        float(match) for match in re.findall(r"r[xyz]\((-?\d+\.\d+)\) q\[\d\];", stdout)
    ]
    assert len(exported) == len(parameters), (exported, parameters)
    assert exported == pytest.approx(parameters, abs=1e-6)

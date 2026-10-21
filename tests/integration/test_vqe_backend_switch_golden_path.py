from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

#: The three representations the example trains the same VQE circuit on. The
#: point of the example is that the circuit and the observable do not change
#: when the representation does, so the three energies have to agree.
_MODES = ("statevector", "mps", "tensor_network")


def _run(tmp_path: Path) -> dict[str, object]:
    """Run the example, keeping its report file out of the working tree.

    The example writes its report to ``vqe_backend_switch.json`` in the
    current directory by default, so the destination is passed explicitly:
    a test that left the default in place would drop an untracked file in
    the repository root on every run.
    """

    root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.vqe_switch_sv_mps_tn",
            "--output",
            str(tmp_path / "vqe_backend_switch.json"),
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_vqe_backend_switch_user_example_runs_end_to_end(tmp_path: Path) -> None:
    report = _run(tmp_path)

    assert set(report) == {"final_energy", "precision_aligned"}
    energies = report["final_energy"]
    assert isinstance(energies, dict)
    assert tuple(energies) == _MODES


def test_the_three_representations_reach_the_same_energy(tmp_path: Path) -> None:
    """Switching the representation must not change the program's meaning.

    This is the architectural invariant the example exists to show: backend
    selection may change execution, but not the result contract. The energies
    are compared against each other rather than against a pinned value,
    because the value is a property of the ansatz and the optimizer schedule
    while the agreement is the property under test.
    """

    report = _run(tmp_path)
    energies = report["final_energy"]
    assert isinstance(energies, dict)

    values = [float(energies[mode]) for mode in _MODES]
    spread = max(values) - min(values)

    # The three paths run different arithmetic -- a dense amplitude store, a
    # truncated matrix product state, and a contracted network -- so they
    # agree to a tolerance rather than bitwise.
    assert spread < 1e-4, f"energies disagree by {spread}: {energies}"

    # The example asserts the alignment itself, and that verdict is what a
    # reader acts on.
    assert report["precision_aligned"] is True

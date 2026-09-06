import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples" / "single_machine_quantum_ai"


def _run_example(name: str, *args: str) -> str:
    completed = subprocess.run(
        [sys.executable, str(EXAMPLES / name), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
        timeout=60,
    )
    return completed.stdout


def test_hybrid_ai_quick_start_smoke():
    completed = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "quick_start.py"), "--steps", "2"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
        timeout=60,
    )
    assert "FlagQuantum quick start complete" in completed.stdout
    assert "qubits     : 4" in completed.stdout
    assert "final loss :" in completed.stdout


def test_local_fast_path_example_smoke():
    out = _run_example("00_local_fast_path_check.py")
    assert "Local Fast Path Preflight" in out
    assert "status : passed" in out
    assert "Backend Speed Comparison" in out


def test_vqe_statevector_example_smoke():
    out = _run_example("01_vqe_statevector.py", "--steps", "2", "--n-qubits", "3")
    assert "single_machine_vqe_statevector" in out


def test_quantum_classifier_example_smoke():
    out = _run_example("02_quantum_classifier.py", "--steps", "2")
    assert "single_machine_quantum_classifier" in out


def test_mps_training_example_smoke():
    out = _run_example(
        "03_mps_training.py", "--steps", "2", "--n-qubits", "4", "--max-bond", "8"
    )
    assert "single_machine_mps_training" in out
    assert "reference" in out


def test_mps_training_scale_report_smoke():
    pytest.importorskip("jax")
    out = _run_example(
        "03_mps_training.py",
        "--scale-report",
        "3,4",
        "--layers",
        "1",
        "--max-bond",
        "4",
        "--scale-iters",
        "1",
        "--scale-warmup",
        "0",
    )
    assert "MPS JAX Scale Report" in out
    assert "first_loss_grad_s" in out
    assert "steady_loss_grad_s" in out


def test_jax_kernel_torch_layer_example_smoke():
    out = _run_example(
        "04_jax_kernel_torch_layer.py", "--steps", "1", "--bench-iters", "1"
    )
    assert (
        "single_machine_jax_kernel_torch_layer" in out or "'status': 'skipped'" in out
    )


def test_1000q_structured_mps_example_smoke():
    pytest.importorskip("jax")
    out = _run_example(
        "05_mps_1000q_dimer_training.py",
        "--n-qubits",
        "20",
        "--steps",
        "2",
        "--bench-iters",
        "1",
        "--bench-warmup",
        "0",
    )
    assert "single_machine_1000q_dimer_mps_training" in out
    assert "FlagQuantum structured JAX MPS" in out

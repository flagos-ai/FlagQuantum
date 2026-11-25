from __future__ import annotations

import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from flagquantum.errors import ExecutionError

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.integration
@pytest.mark.parametrize("backend", ["pytorch", "jax"])
def test_maintained_hybrid_acceptance_source_script(backend: str) -> None:
    if backend == "jax":
        pytest.importorskip("jax")
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "examples/hybrid_model_acceptance.py"),
            "--model",
            "classifier",
            "--backend",
            backend,
            "--steps",
            "2",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["model"] == "HybridQuantumClassifier"
    assert payload["accuracy_is_performance_claim"] is False
    assert set(payload) >= {"correctness", "accuracy", "performance", "deployment"}
    assert payload["policy"]["execution_options"]["backend"] == backend
    assert payload["deployment"]["runtime_backend"] == backend
    assert payload["deployment"]["fallback_used"] is False
    assert payload["correctness"]["finite"] is True


@pytest.mark.integration
def test_classifier_example_matches_torch_training_on_jax() -> None:
    pytest.importorskip("jax")
    classifier_run = runpy.run_path(str(ROOT / "examples/hybrid_model_acceptance.py"))[
        "classifier_run"
    ]
    native = classifier_run(5, "pytorch")
    accelerated = classifier_run(5, "jax")
    torch.testing.assert_close(
        torch.tensor(accelerated["accuracy"]["losses"]),
        torch.tensor(native["accuracy"]["losses"]),
        atol=2e-5,
        rtol=2e-5,
    )
    assert (
        accelerated["accuracy"]["classification_accuracy"]
        == native["accuracy"]["classification_accuracy"]
    )


@pytest.mark.integration
def test_classifier_example_refuses_missing_requested_jax(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    classifier_run = runpy.run_path(str(ROOT / "examples/hybrid_model_acceptance.py"))[
        "classifier_run"
    ]
    monkeypatch.setitem(sys.modules, "jax", None)
    with pytest.raises(ExecutionError, match="JAX module execution failed") as error:
        classifier_run(1, "jax")
    assert isinstance(error.value.__cause__, ModuleNotFoundError)

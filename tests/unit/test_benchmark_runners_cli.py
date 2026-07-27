import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.runners.registry import names, register, resolve

ROOT = Path(__file__).parents[2]


def test_runner_registry_is_lazy_and_sorted():
    assert names() == (
        "environment_probe",
        "statevector_strong_scaling",
        "statevector_training_scaling",
        "statevector_weak_scaling",
    )
    assert callable(resolve("environment_probe"))


def test_runner_registry_rejects_unknown_name():
    with pytest.raises(KeyError, match="unknown benchmark runner"):
        resolve("does_not_exist")


def test_runner_registry_rejects_duplicate_registration():
    with pytest.raises(ValueError, match="already registered"):
        register("environment_probe", "benchmarks.runners.environment_probe")


def test_runner_registry_enforces_name_convention():
    with pytest.raises(ValueError, match="lowercase snake_case"):
        register("BadRunner", "benchmarks.runners.environment_probe")


def test_runner_cli_writes_contract_payload(tmp_path):
    output = tmp_path / "probe.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.runners",
            "environment_probe",
            "--json-output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(output.read_text())
    assert payload["runner"] == "environment_probe"
    assert payload["schema"].endswith(".v1")


def test_runner_package_lists_available_runner():
    result = subprocess.run(
        [sys.executable, "-m", "benchmarks.runners"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "environment_probe" in result.stdout

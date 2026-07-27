import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.runners.registry import describe, names, register, resolve

ROOT = Path(__file__).parents[2]


def test_runner_registry_is_lazy_and_sorted():
    assert names() == (
        "environment_probe",
        "mps_training",
        "statevector_local",
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
    assert "statevector_local" in result.stdout
    assert "mps_training" in result.stdout


def test_runner_info_is_discoverable_without_importing_implementation():
    spec = describe("statevector_local")
    assert spec.category == "statevector"
    assert "CPU or one GPU" in spec.hardware
    result = subprocess.run(
        [sys.executable, "-m", "benchmarks.runners", "info", "statevector_local"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "flagquantum-benchmark run statevector_local" in result.stdout


def test_explicit_run_subcommand_writes_contract_payload(tmp_path):
    output = tmp_path / "probe.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.runners",
            "run",
            "environment_probe",
            "--json-output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(output.read_text())["runner"] == "environment_probe"


def test_pyproject_installs_benchmark_command_and_runner_namespace():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert (
        'flagquantum-benchmark = "benchmarks.runners.__main__:main"' in text
    )
    assert '"benchmarks*"' in text


def test_benchmark_root_stays_free_of_internal_assets():
    benchmark_root = ROOT / "benchmarks"
    assert not list(benchmark_root.glob("*.json"))
    assert not list(benchmark_root.glob("*.sh"))
    assert not list(benchmark_root.glob("issue*.py"))
    assert not list(benchmark_root.glob("aggregate_issue*.py"))
    assert not list(benchmark_root.glob("plot_*.py"))
    assert not [
        path
        for path in benchmark_root.rglob("*")
        if "issue" in path.name.lower() and "__pycache__" not in path.parts
    ]

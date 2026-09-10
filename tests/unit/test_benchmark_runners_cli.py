import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from flagquantum.benchmarking import registry
from flagquantum.benchmarking.registry import describe, names, register, resolve

ROOT = Path(__file__).parents[2]


def test_resolve_rejects_noncallable_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        registry, "import_module", lambda name: SimpleNamespace(main=42)
    )
    with pytest.raises(TypeError, match="entrypoint must be callable"):
        resolve("environment_probe")


@pytest.mark.parametrize("exit_code", [None, "0", 0.5])
def test_runner_rejects_invalid_exit_code(
    monkeypatch: pytest.MonkeyPatch, exit_code: object
) -> None:
    monkeypatch.setattr(
        registry, "import_module", lambda name: SimpleNamespace(main=lambda: exit_code)
    )
    with pytest.raises(TypeError, match="integer exit code"):
        resolve("environment_probe")()


def test_resolve_is_lazy_and_preserves_nonzero_exit_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def main() -> int:
        calls.append("run")
        return 3

    monkeypatch.setattr(
        registry, "import_module", lambda name: SimpleNamespace(main=main)
    )
    runner = resolve("environment_probe")
    assert calls == []
    assert runner() == 3
    assert calls == ["run"]


@pytest.mark.parametrize(
    "module",
    (
        "environment_probe",
        "statevector_strong_scaling",
        "statevector_weak_scaling",
        "statevector_training_scaling",
        "statevector_weak_scaling_report",
        "statevector_training_scaling_report",
    ),
)
def test_benchmark_scripts_run_without_package_context(module: str) -> None:
    script = ROOT / "flagquantum" / "benchmarking" / f"{module}.py"
    result = subprocess.run(
        [sys.executable, "-S", str(script), "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert "usage:" in result.stdout


def test_runner_registry_is_lazy_and_sorted() -> None:
    assert names() == (
        "environment_probe",
        "statevector_local",
        "statevector_strong_scaling",
        "statevector_training_scaling",
        "statevector_weak_scaling",
    )
    assert callable(resolve("environment_probe"))


def test_runner_registry_rejects_unknown_name() -> None:
    with pytest.raises(KeyError, match="unknown benchmark runner"):
        resolve("does_not_exist")


def test_runner_registry_rejects_duplicate_registration() -> None:
    with pytest.raises(ValueError, match="already registered"):
        register("environment_probe", "flagquantum.benchmarking.environment_probe")


def test_runner_registry_enforces_name_convention() -> None:
    with pytest.raises(ValueError, match="lowercase snake_case"):
        register("BadRunner", "flagquantum.benchmarking.environment_probe")


def test_runner_cli_writes_contract_payload(tmp_path: Path) -> None:
    output = tmp_path / "probe.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "flagquantum.benchmarking",
            "environment_probe",
            "--json-output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    payload = json.loads(output.read_text())
    assert payload["runner"] == "environment_probe"
    assert payload["schema"].endswith(".v1")


def test_runner_package_lists_available_runner() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "flagquantum.benchmarking"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert "environment_probe" in result.stdout
    assert "statevector_local" in result.stdout


def test_runner_info_is_discoverable_without_importing_implementation() -> None:
    spec = describe("statevector_local")
    assert spec.category == "statevector"
    assert "CPU or one GPU" in spec.hardware
    result = subprocess.run(
        [sys.executable, "-m", "flagquantum.benchmarking", "info", "statevector_local"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert "flagquantum-benchmark run statevector_local" in result.stdout


def test_explicit_run_subcommand_writes_contract_payload(tmp_path: Path) -> None:
    output = tmp_path / "probe.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "flagquantum.benchmarking",
            "run",
            "environment_probe",
            "--json-output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert json.loads(output.read_text())["runner"] == "environment_probe"


def test_pyproject_installs_benchmark_command_and_runner_namespace() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'flagquantum-benchmark = "flagquantum.benchmarking.__main__:main"' in text
    assert 'include = ["flagquantum*"]' in text


def test_benchmark_root_stays_free_of_internal_assets() -> None:
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

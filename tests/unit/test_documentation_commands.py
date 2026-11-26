"""Complete frozen documentation script entry points and real CLI execution."""

import csv
import hashlib
import importlib
import os
import runpy
import subprocess
import sys
import types
from itertools import product
from pathlib import Path

import pytest

from tests.unit.notebook_data import installed_jupyter_data
from tests.unit.test_development_tools import REFERENCE
from tools.ftqc_source_inputs import FTQC_SOURCE_INVENTORY

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def original_command(
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
    modules: dict[str, dict[str, object]],
    arguments: list[str],
) -> None:
    """Execute an unchanged source script with only its imported owners replaced."""
    package = types.ModuleType("qualtran_dev_tools")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, package.__name__, package)
    for name, attributes in modules.items():
        module = types.ModuleType("qualtran_dev_tools." + name)
        vars(module).update(attributes)
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(sys, "argv", [filename, *arguments])
    runpy.run_path(str(REFERENCE / "dev_tools" / filename), run_name="__main__")


def test_frozen_command_blobs() -> None:
    inventory = FTQC_SOURCE_INVENTORY
    selected = {"Q-0042", "Q-0043", "Q-0045", "Q-0050"}
    found = set()
    for row in csv.DictReader(inventory.open(encoding="utf-8-sig")):
        if row["id"] in selected:
            data = (REFERENCE / row["path"]).read_bytes()
            assert (
                hashlib.sha1(
                    b"blob " + str(len(data)).encode() + b"\0" + data
                ).hexdigest()
                == row["git_blob"]
            )
            found.add(row["id"])
    assert found == selected


def test_execution_arguments_match_original(monkeypatch: pytest.MonkeyPatch) -> None:
    native = importlib.import_module("tools.development.notebook_execution")
    source_calls: list[dict[str, object]] = []
    native_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        native,
        "execute_and_export_notebooks",
        lambda **kwargs: native_calls.append(kwargs),
    )
    choices: list[list[str]] = [[]]
    for outputs, markdown, freshness in product((False, True), repeat=3):
        choices.append(
            [
                "--output-nbs" if outputs else "--no-output-nbs",
                "--output-md" if markdown else "--no-output-md",
                "--only-out-of-date" if freshness else "--no-only-out-of-date",
                "--n-workers",
                "2",
            ]
        )
    choices.extend([["--n-workers", value] for value in ("0", "-1", "1")])
    for arguments in choices:
        original_command(
            monkeypatch,
            "execute-notebooks.py",
            {
                "notebook_execution": {
                    "execute_and_export_notebooks": lambda **kwargs: source_calls.append(
                        kwargs
                    )
                },
            },
            arguments,
        )
        native.main(arguments)
        assert native_calls[-1] == source_calls[-1]
    assert native_calls[0] == dict(
        output_nbs=True, output_md=False, only_out_of_date=True, n_workers=None
    )
    for arguments in (["--n-workers", "bad"], ["--n-workers"], ["--unknown"]):
        before = len(native_calls), len(source_calls)
        for entry in (
            lambda arguments=arguments: native.main(arguments),
            lambda arguments=arguments: original_command(
                monkeypatch,
                "execute-notebooks.py",
                {
                    "notebook_execution": {
                        "execute_and_export_notebooks": lambda **kwargs: source_calls.append(
                            kwargs
                        )
                    },
                },
                arguments,
            ),
        ):
            with pytest.raises(SystemExit) as error:
                entry()
            assert error.value.code == 2
        assert before == (len(native_calls), len(source_calls))


def test_cleaning_arguments_root_and_exit_count(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    native = importlib.import_module("tools.development.notebook_cleaning")
    calls: list[tuple[Path, bool]] = []

    def clean(root: Path, do_clean: bool) -> int:
        calls.append((root, do_clean))
        return 3

    monkeypatch.setattr(native, "get_git_root", lambda: tmp_path)
    monkeypatch.setattr(native, "clean_notebooks", clean)
    for arguments, apply in (([], False), (["--apply"], True), (["--no-apply"], False)):
        with pytest.raises(SystemExit) as error:
            original_command(
                monkeypatch,
                "clean-notebooks.py",
                {
                    "git_tools": {"get_git_root": lambda: tmp_path},
                    "clean_notebooks": {"clean_notebooks": clean},
                },
                arguments,
            )
        assert error.value.code == 3
        assert native.main(arguments) == 3
        assert calls[-2:] == [
            (tmp_path / "qualtran", apply),
            (tmp_path / "flagquantum", apply),
        ]


def test_reference_roots_defaults_and_failure_propagation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    static = importlib.import_module("tools.development.reference_docs")
    templates = importlib.import_module("tools.development.reference_templates")
    calls: list[tuple[object, ...]] = []

    def record(*args: object) -> None:
        calls.append(args)

    monkeypatch.setattr(static, "get_git_root", lambda: tmp_path)
    monkeypatch.setattr(static, "make_reference_docs", record)
    original_command(
        monkeypatch,
        "build-reference-docs-2.py",
        {
            "git_tools": {"get_git_root": lambda: tmp_path},
            "make_reference_docs": {"make_reference_docs": record},
        },
        [],
    )
    static.main()
    assert calls == [(tmp_path,), (tmp_path,)]
    monkeypatch.setattr(templates, "build_reference_docs", record)
    original_command(
        monkeypatch,
        "build-reference-docs.py",
        {
            "reference_docs": {"build_reference_docs": record},
        },
        [],
    )
    templates.main()
    assert calls[-2:] == [(), ()]

    def fail(*args: object, **kwargs: object) -> None:
        raise ValueError("original owner failure")

    execution = importlib.import_module("tools.development.notebook_execution")
    monkeypatch.setattr(execution, "execute_and_export_notebooks", fail)
    for entry in (
        lambda: execution.main([]),
        lambda: original_command(
            monkeypatch,
            "execute-notebooks.py",
            {
                "notebook_execution": {"execute_and_export_notebooks": fail},
            },
            [],
        ),
    ):
        with pytest.raises(ValueError, match="original owner failure"):
            entry()


def test_real_module_cleanup_and_empty_execution(tmp_path: Path) -> None:
    nbformat = importlib.import_module("nbformat")
    subprocess.run(
        ["git", "init"], cwd=tmp_path, capture_output=True, check=True, timeout=15
    )
    package = tmp_path / "flagquantum"
    package.mkdir()
    (tmp_path / "examples/notebooks").mkdir(parents=True)
    path = package / "case.ipynb"
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("print(7)")])
    notebook.cells[0].execution_count = 1
    notebook.cells[0].outputs = [
        nbformat.v4.new_output("stream", name="stdout", text="7\n")
    ]
    nbformat.write(notebook, path)
    subprocess.run(
        ["git", "add", "flagquantum/case.ipynb"],
        cwd=tmp_path,
        capture_output=True,
        check=True,
        timeout=15,
    )
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))

    def command(module: str, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "tools.development." + module, *arguments],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
        )

    before = path.read_bytes()
    result = command("notebook_cleaning")
    assert result.returncode == 1, result.stdout + result.stderr
    assert path.read_bytes() == before
    result = command("notebook_cleaning", "--apply")
    assert result.returncode == 1, result.stdout + result.stderr
    assert nbformat.read(path, as_version=4).cells[0].outputs == []
    assert command("notebook_cleaning").returncode == 0
    subprocess.run(
        ["git", "rm", "--cached", "-f", "flagquantum/case.ipynb"],
        cwd=tmp_path,
        capture_output=True,
        check=True,
        timeout=15,
    )
    result = command("notebook_execution", "--n-workers", "1", "--no-output-nbs")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Found 0 notebooks." in result.stdout
    assert "Slowest 10 notebooks:" in result.stdout
    for module in ("notebook_cleaning", "notebook_execution"):
        result = command(module, "--help")
        assert result.returncode == 0 and "usage:" in result.stdout


def test_actual_execution_command_complex_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Reuse the accepted notebook owner's actual kernel environment, including its
    # controlled interpreter and environment. The CLI adds no execution machinery.
    from tests.unit.test_notebook_execution import configure_kernel_environment

    monkeypatch.setenv("JUPYTER_PATH", str(installed_jupyter_data()))
    kernel_root = configure_kernel_environment(tmp_path, monkeypatch)
    nbformat = importlib.import_module("nbformat")
    subprocess.run(
        ["git", "init"], cwd=tmp_path, capture_output=True, check=True, timeout=15
    )
    package = tmp_path / "flagquantum"
    package.mkdir()
    (tmp_path / "examples/notebooks").mkdir(parents=True)
    path = package / "case.ipynb"
    code = (
        "import torch\nimport flagquantum as fq\n"
        "state = torch.tensor([1, 1j], dtype=torch.complex128) / 2**0.5\n"
        "circuit = fq.Circuit(1, inputs=state, dtype=torch.complex128)\n"
        "from flagquantum.core.ir import Instruction\n"
        "from examples.algorithms.binary_field_arithmetic import FieldOperation\n"
        "op = FieldOperation(1, (Instruction('x', (0,)),))\ncircuit.append(op)\n"
        "torch.testing.assert_close(circuit.state().flatten(), state.flip(0))\n"
        "print('complex CLI state passed')"
    )
    nbformat.write(
        nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(code)]), path
    )
    subprocess.run(
        ["git", "add", "flagquantum/case.ipynb"],
        cwd=tmp_path,
        capture_output=True,
        check=True,
        timeout=15,
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.development.notebook_execution",
            "--n-workers",
            "1",
            "--output-md",
            "--no-only-out-of-date",
        ],
        cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)),
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    exported = tmp_path / "docs/case.ipynb"
    assert (
        nbformat.read(exported, as_version=4).cells[0].outputs[0].text
        == "complex CLI state passed\n"
    )
    assert (tmp_path / "docs/case.md").exists()
    assert not list((kernel_root / "runtime").glob("kernel-*.json"))

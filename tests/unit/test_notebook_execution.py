"""Complete locked notebook execution workflow and shared native kernel owner."""

import asyncio
import csv
import hashlib
import json
import os
import subprocess
import sys
import types
from contextlib import contextmanager
from copy import deepcopy
from importlib import import_module
from pathlib import Path
from typing import Any, ClassVar

import pytest
from nbformat import NotebookNode

from flagquantum.testing.notebook_execution import execute_notebook_node
from tests.unit.notebook_data import installed_jupyter_data
from tests.unit.test_development_tools import REFERENCE, load_source
from tools.development import notebook_execution as native
from tools.ftqc_source_inputs import FTQC_SOURCE_INVENTORY

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
nbformat = import_module("nbformat")


@pytest.fixture(autouse=True)
def jupyter_templates(monkeypatch: pytest.MonkeyPatch) -> None:
    location = installed_jupyter_data()
    assert (location / "nbconvert" / "templates" / "markdown" / "conf.json").exists()
    monkeypatch.setenv("JUPYTER_PATH", str(location))


@pytest.fixture
def original(monkeypatch: pytest.MonkeyPatch) -> Any:
    package = types.ModuleType("_development_reference")
    package.__path__ = [str(REFERENCE / "dev_tools" / "qualtran_dev_tools")]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    source = load_source("notebook_execution")
    monkeypatch.setitem(
        sys.modules, "_development_reference.notebook_execution", source
    )
    return source


def notebook() -> NotebookNode:
    result = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell("Check the `Bloq` documentation."),
            nbformat.v4.new_code_cell("print(7)"),
        ]
    )
    assert isinstance(result, NotebookNode)
    return result


def test_original_blobs_and_unchanged_source_tests(
    original: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remaining = {"Q-0078", "Q-0079"}
    with (FTQC_SOURCE_INVENTORY).open(encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if row["id"] not in remaining:
                continue
            raw = (REFERENCE / row["path"]).read_bytes()
            assert (
                hashlib.sha1(
                    b"blob " + str(len(raw)).encode() + b"\0" + raw
                ).hexdigest()
                == row["git_blob"]
            )
            remaining.remove(row["id"])
    assert not remaining
    provider = types.ModuleType("_development_reference.notebook_execution")
    setattr(provider, "is_out_of_date", native.is_out_of_date)
    setattr(
        provider, "linkify", lambda nb: native.linkify(nb, original._LINK_REPLACEMENTS)
    )
    monkeypatch.setitem(sys.modules, provider.__name__, provider)
    tests = load_source("notebook_execution_test")
    tests.test_is_out_of_date(tmp_path)
    tests.test_linkify()


@pytest.mark.parametrize("output_md", [False, True])
@pytest.mark.parametrize("output_nbs", [False, True])
def test_source_paths_mtime_and_link_replacements(
    original: Any, tmp_path: Path, output_md: bool, output_nbs: bool
) -> None:
    input_root = tmp_path / "source"
    relative = Path("nested/case.ipynb")
    path = input_root / relative
    path.parent.mkdir(parents=True)
    path.write_text("source")
    expected = original._NBInOutPaths.from_nb_rel_path(
        relative, tmp_path, input_root, output_md, output_nbs
    )
    actual = native.NotebookExportPaths.from_nb_rel_path(
        relative, tmp_path, input_root, output_md, output_nbs
    )
    assert (actual.nb_in, actual.md_out, actual.nb_out) == (
        expected.nb_in,
        expected.md_out,
        expected.nb_out,
    )
    for method in ("md_needs_reexport", "nb_needs_reexport", "needs_reexport"):
        assert getattr(actual, method)() == getattr(expected, method)()
    for output in (actual.md_out, actual.nb_out):
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("output")
            os.utime(output, (1, 1))
    assert actual.needs_reexport() == expected.needs_reexport()
    for output in (actual.md_out, actual.nb_out):
        if output is not None:
            stamp = path.stat().st_mtime + 10
            os.utime(output, (stamp, stamp))
    assert not actual.needs_reexport() and not expected.needs_reexport()
    left = notebook()
    right = deepcopy(left)
    for _ in range(2):
        original.linkify(left)
        native.linkify(right, original._LINK_REPLACEMENTS)
        assert left == right
    for _, target in native._make_link_replacements():
        destination = target.split("](", 1)[1].rstrip(")")
        assert (Path(__file__).parents[2] / "docs" / destination.lstrip("/")).exists()
    path.unlink()
    with pytest.raises(AssertionError):
        native.is_out_of_date(path, tmp_path / "missing")


class FakeExecutor:
    events: ClassVar[list[str]] = []
    fail = False

    def __init__(self, **kwargs: Any) -> None:
        self.settings = kwargs
        self.km = None

    def create_kernel_manager(self) -> object:
        self.events.append("precreate")
        return object()

    async def async_start_new_kernel(self) -> None:
        self.events.append("prestart")

    async def async_start_new_kernel_client(self) -> None:
        self.events.append("preclient")

    @contextmanager
    def setup_kernel(self, **kwargs: Any) -> Any:
        self.events.append("actual-start")
        try:
            yield
        finally:
            self.events.append("actual-stop")

    def preprocess(self, value: NotebookNode) -> tuple[NotebookNode, dict[str, object]]:
        with self.setup_kernel():
            if self.fail:
                raise RuntimeError("cell failed")
            value.cells[1].execution_count = 1
            value.cells[1].outputs = [
                nbformat.v4.new_output("stream", name="stdout", text="7\n")
            ]
        return value, {}


@pytest.mark.parametrize(
    "output_md,output_nbs,fail",
    [
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (True, True, False),
        (True, True, True),
    ],
)
def test_complete_original_execution_exports_and_error_outcomes(
    original: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    output_md: bool,
    output_nbs: bool,
    fail: bool,
) -> None:
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "case.ipynb"
    nbformat.write(notebook(), path)
    before = path.read_bytes()
    monkeypatch.setattr(original, "ExecutePreprocessor", FakeExecutor)
    owner = import_module("flagquantum.testing.notebook_execution")
    monkeypatch.setattr(
        owner,
        "import_module",
        lambda name: (
            types.SimpleNamespace(ExecutePreprocessor=FakeExecutor)
            if name == "nbconvert.preprocessors"
            else import_module(name)
        ),
    )
    monkeypatch.setattr(FakeExecutor, "events", [])
    monkeypatch.setattr(FakeExecutor, "fail", fail)
    expected = original._NBInOutPaths(
        path,
        tmp_path / "source.md" if output_md else None,
        tmp_path / "source.ipynb" if output_nbs else None,
    )
    actual = native.NotebookExportPaths(
        path,
        tmp_path / "native.md" if output_md else None,
        tmp_path / "native.ipynb" if output_nbs else None,
    )
    expected_error = original.execute_and_export_notebook(expected)
    expected_events = list(FakeExecutor.events)
    FakeExecutor.events.clear()
    actual_error = native.execute_and_export_notebook(
        actual, replacements=original._LINK_REPLACEMENTS
    )
    assert type(actual_error) is type(expected_error)
    assert path.read_bytes() == before
    assert expected_events[:3] == ["precreate", "prestart", "preclient"]
    assert FakeExecutor.events == ["actual-start", "actual-stop"]
    for expected_out, actual_out in (
        (expected.nb_out, actual.nb_out),
        (expected.md_out, actual.md_out),
    ):
        if actual_out is not None:
            if fail:
                assert not actual_out.exists()
            else:
                assert actual_out.read_bytes() == expected_out.read_bytes()


def test_export_failure_preserves_old_outputs_and_reclaims_temporaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "case.ipynb"
    nbformat.write(notebook(), path)
    paths = native.NotebookExportPaths(
        path, tmp_path / "case.md", tmp_path / "output.ipynb"
    )
    assert paths.nb_out is not None and paths.md_out is not None
    paths.nb_out.write_text("old-notebook")
    paths.md_out.write_text("old-markdown")
    owner = import_module("flagquantum.testing.notebook_execution")
    monkeypatch.setattr(
        owner,
        "import_module",
        lambda name: (
            types.SimpleNamespace(ExecutePreprocessor=FakeExecutor)
            if name == "nbconvert.preprocessors"
            else import_module(name)
        ),
    )

    def fail_export(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("renderer failed")

    monkeypatch.setattr(import_module("nbconvert"), "export", fail_export)
    with pytest.raises(RuntimeError, match="renderer failed"):
        native.execute_and_export_notebook(paths, startup_lock=tmp_path / "lock")
    assert (
        paths.nb_out.read_text() == "old-notebook"
        and paths.md_out.read_text() == "old-markdown"
    )
    before = set(tmp_path.iterdir())

    def fail_replace(*args: Any, **kwargs: Any) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        native._write_export(paths.md_out, "replacement")
    assert (
        set(tmp_path.iterdir()) == before and paths.md_out.read_text() == "old-markdown"
    )


def configure_kernel_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    base = tmp_path / "jupyter"
    folder = base / "kernels" / "python3"
    folder.mkdir(parents=True)
    (folder / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [
                    sys.executable,
                    "-m",
                    "ipykernel_launcher",
                    "-f",
                    "{connection_file}",
                ],
                "display_name": "G2 locked Python",
                "language": "python",
                "env": {
                    "PYTHONPATH": os.pathsep.join(sys.path),
                    "OMP_NUM_THREADS": "1",
                },
            }
        )
    )
    monkeypatch.setenv(
        "JUPYTER_PATH", str(base) + os.pathsep + os.environ.get("JUPYTER_PATH", "")
    )
    monkeypatch.setenv("JUPYTER_RUNTIME_DIR", str(base / "runtime"))
    monkeypatch.setenv("IPYTHONDIR", str(base / "ipython"))
    monkeypatch.setenv("JUPYTER_CONFIG_DIR", str(base / "config"))
    return base


@pytest.fixture
def kernel_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    return configure_kernel_environment(tmp_path, monkeypatch)


def test_real_source_and_native_execution_with_kernel_cleanup(
    original: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kernel_environment: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "case.ipynb"
    nbformat.write(notebook(), path)
    preprocessor = import_module("nbconvert.preprocessors").ExecutePreprocessor
    create = preprocessor.create_kernel_manager
    client_start = preprocessor.async_start_new_kernel_client
    managers: list[Any] = []
    clients: list[Any] = []

    def create_manager(self: Any) -> Any:
        manager = create(self)
        managers.append(manager)
        return manager

    async def create_client(self: Any) -> Any:
        client = await client_start(self)
        clients.append(client)
        return client

    monkeypatch.setattr(preprocessor, "create_kernel_manager", create_manager)
    monkeypatch.setattr(preprocessor, "async_start_new_kernel_client", create_client)
    expected = original._NBInOutPaths(
        path, tmp_path / "source.md", tmp_path / "source.ipynb"
    )
    actual = native.NotebookExportPaths(
        path, tmp_path / "native.md", tmp_path / "native.ipynb"
    )
    try:
        assert original.execute_and_export_notebook(expected) is None
        assert len(managers) == 2
        assert sum(manager.has_kernel for manager in managers) == 1
        for manager in managers:
            if manager.has_kernel:
                asyncio.run(manager.shutdown_kernel(now=True))
        for client in clients:
            client.stop_channels()
        managers.clear()
        clients.clear()
        assert (
            native.execute_and_export_notebook(
                actual, replacements=original._LINK_REPLACEMENTS
            )
            is None
        )
        assert len(managers) == 1 and not managers[0].has_kernel
        assert nbformat.read(actual.nb_out, as_version=4) == nbformat.read(
            expected.nb_out, as_version=4
        )
        assert actual.md_out is not None
        assert actual.md_out.read_bytes() == expected.md_out.read_bytes()
    finally:
        for manager in managers:
            if manager.has_kernel:
                asyncio.run(manager.shutdown_kernel(now=True))
        for client in clients:
            client.stop_channels()


def test_real_shared_owner_preserves_native_complex_state_and_append(
    kernel_environment: Path,
) -> None:
    code = """import torch
import flagquantum as fq
from flagquantum.core.ir import Instruction
from examples.algorithms.binary_field_arithmetic import FieldOperation
circuit = fq.Circuit(1)
circuit.append(FieldOperation(1, (Instruction('h', (0,)), Instruction('s', (0,)))))
expected = torch.tensor([1, 1j], dtype=torch.complex64) / (2 ** 0.5)
torch.testing.assert_close(circuit.state().flatten(), expected)
assert len(circuit.to_ir().instructions) == 2
print('complex-state-verified')"""
    value = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(code)])
    result, resources = execute_notebook_node(
        value, timeout=30, record_timing=False, startup_lock=kernel_environment / "lock"
    )
    assert result.cells[0].outputs[0].text == "complex-state-verified\n" and isinstance(
        resources, dict
    )


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("bad", [False, True])
def test_complete_source_serial_pool_summaries_and_failures(
    original: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    workers: int,
    bad: bool,
) -> None:
    import multiprocessing

    context = multiprocessing.get_context("fork")

    def paths(sourceroot: Path) -> list[tuple[Path, Path]]:
        name = "first" if sourceroot.name in {"qualtran", "native_a"} else "second"
        return [(Path(name + ".ipynb"), tmp_path / "inputs")]

    def export(value: Any, **kwargs: Any) -> Exception | None:
        return RuntimeError("failed") if bad and value.nb_in.stem == "second" else None

    monkeypatch.setattr(original, "get_git_root", lambda: tmp_path)
    monkeypatch.setattr(original, "get_nb_rel_paths", paths)
    monkeypatch.setattr(native, "get_nb_rel_paths", paths)
    monkeypatch.setattr(original, "execute_and_export_notebook", export)
    monkeypatch.setattr(native, "execute_and_export_notebook", export)
    monkeypatch.setattr(
        original, "multiprocessing", types.SimpleNamespace(Pool=context.Pool)
    )
    monkeypatch.setattr(original.random, "shuffle", lambda values: None)
    for run in (
        lambda: original.execute_and_export_notebooks(
            output_nbs=True, output_md=True, only_out_of_date=False, n_workers=workers
        ),
        lambda: native.execute_and_export_notebooks(
            output_nbs=True,
            output_md=True,
            only_out_of_date=False,
            n_workers=workers,
            reporoot=tmp_path,
            sourceroots=[tmp_path / "native_a", tmp_path / "native_b"],
            start_method="fork",
        ),
    ):
        if bad:
            with pytest.raises(SystemExit) as error:
                run()
            assert error.value.code == 1
        else:
            run()
        output = capsys.readouterr().out
        assert "Found 2 notebooks." in output
        assert ("Errors in notebooks:" in output) == bad
        assert ("Slowest 10 notebooks:" in output) != bad
        if bad:
            assert str(tmp_path / "inputs" / "second.ipynb") in output


def test_unchanged_stale_closure_and_empty_parallel_boundary(
    original: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "input"
    root.mkdir()
    source = root / "case.ipynb"
    source.write_text("source")
    source_run = original._NotebookRunClosure(tmp_path, False, False, True)
    native_run = native._NotebookRunClosure(tmp_path, False, False, True)
    expected, actual = source_run(Path("case.ipynb"), root), native_run(
        Path("case.ipynb"), root
    )
    assert (
        (actual.nb_in, actual.err, actual.duration_s)
        == (expected.nb_in, expected.err, expected.duration_s)
        == (source, None, 0.0)
    )
    monkeypatch.setattr(original, "get_git_root", lambda: tmp_path)
    monkeypatch.setattr(original, "get_nb_rel_paths", lambda sourceroot: [])
    monkeypatch.setattr(native, "get_nb_rel_paths", lambda sourceroot: [])

    class EmptyPool:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> "EmptyPool":
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def starmap(self, *args: Any) -> list[object]:
            return []

    monkeypatch.setattr(
        original, "multiprocessing", types.SimpleNamespace(Pool=EmptyPool)
    )
    with pytest.raises(AssertionError):
        original.execute_and_export_notebooks(
            output_nbs=True, output_md=True, n_workers=2
        )
    native.execute_and_export_notebooks(
        output_nbs=True,
        output_md=True,
        n_workers=2,
        reporoot=tmp_path,
        sourceroots=[root],
    )


def test_tracked_notebook_discovery_matches_original(
    original: Any, tmp_path: Path
) -> None:
    subprocess.run(
        ["git", "init"], cwd=tmp_path, capture_output=True, check=True, timeout=15
    )
    path = tmp_path / "nested" / "case.ipynb"
    path.parent.mkdir()
    nbformat.write(notebook(), path)
    nbformat.write(notebook(), tmp_path / "untracked.ipynb")
    subprocess.run(
        ["git", "add", "nested/case.ipynb"],
        cwd=tmp_path,
        capture_output=True,
        check=True,
        timeout=15,
    )
    assert (
        native.get_nb_rel_paths(tmp_path)
        == original.get_nb_rel_paths(tmp_path)
        == [(Path("nested/case.ipynb"), tmp_path)]
    )


@pytest.mark.parametrize("bad", [False, True])
def test_real_spawn_batch_exports_and_aggregates_failures(
    tmp_path: Path, kernel_environment: Path, bad: bool
) -> None:
    subprocess.run(
        ["git", "init"], cwd=tmp_path, capture_output=True, check=True, timeout=15
    )
    code = "raise ValueError('bounded notebook failure')" if bad else "print(11)"
    nbformat.write(notebook(), tmp_path / "first.ipynb")
    nbformat.write(
        nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(code)]),
        tmp_path / "second.ipynb",
    )
    subprocess.run(
        ["git", "add", "first.ipynb", "second.ipynb"],
        cwd=tmp_path,
        capture_output=True,
        check=True,
        timeout=15,
    )
    script = tmp_path / "run_export.py"
    script.write_text(
        "from pathlib import Path\n"
        "from tools.development.notebook_execution import execute_and_export_notebooks\n"
        "if __name__ == '__main__':\n"
        "    execute_and_export_notebooks(output_nbs=True, output_md=True, "
        "only_out_of_date=False, n_workers=2, reporoot=Path(__file__).parent, "
        "sourceroots=[Path(__file__).parent], start_method='spawn')\n"
    )
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == int(bad), result.stdout + result.stderr
    assert "Found 2 notebooks." in result.stdout
    assert ("Errors in notebooks:" in result.stdout) == bad
    first = nbformat.read(tmp_path / "docs" / "first.ipynb", as_version=4)
    assert first.cells[1].outputs[0].text == "7\n"
    assert (tmp_path / "docs" / "first.md").exists()
    assert (tmp_path / "docs" / "second.ipynb").exists() != bad
    assert (tmp_path / "docs" / "second.md").exists() != bad
    assert not list((kernel_environment / "runtime").glob("kernel-*.json"))


def test_startup_failure_scope_is_explicit(
    original: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "case.ipynb"
    nbformat.write(notebook(), path)

    class StartupFailure(FakeExecutor):
        async def async_start_new_kernel(self) -> None:
            raise RuntimeError("kernel startup failed")

        @contextmanager
        def setup_kernel(self, **kwargs: Any) -> Any:
            raise RuntimeError("kernel startup failed")
            yield

    monkeypatch.setattr(original, "ExecutePreprocessor", StartupFailure)
    owner = import_module("flagquantum.testing.notebook_execution")
    monkeypatch.setattr(
        owner,
        "import_module",
        lambda name: (
            types.SimpleNamespace(ExecutePreprocessor=StartupFailure)
            if name == "nbconvert.preprocessors"
            else import_module(name)
        ),
    )
    with pytest.raises(RuntimeError, match="kernel startup failed"):
        original.execute_and_export_notebook(
            original._NBInOutPaths(path, tmp_path / "original.md", None)
        )
    error = native.execute_and_export_notebook(
        native.NotebookExportPaths(path, tmp_path / "native.md", None)
    )
    assert isinstance(error, RuntimeError) and str(error) == "kernel startup failed"
    assert not (tmp_path / "original.md").exists()
    assert not (tmp_path / "native.md").exists()

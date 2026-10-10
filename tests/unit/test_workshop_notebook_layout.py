from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = (
    "cloud/04_jiuding_jobs.ipynb",
    "cloud/05_quafu_hardware.ipynb",
    "cloud/06_compare_results.ipynb",
    "cloud/13_trained_hardware.ipynb",
    "cloud/16_cloud_sweep.ipynb",
    "simulation/18_one_problem_three_simulators.ipynb",
    "training/07_hybrid_model.ipynb",
)


def _notebook(relative_path: str) -> dict[str, object]:
    path = ROOT / "workshops" / "flagos2026" / "notebooks" / relative_path
    return json.loads(path.read_text(encoding="utf-8"))


def _locator_source(relative_path: str) -> str:
    notebook = _notebook(relative_path)
    matches = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "search_roots" in "".join(cell["source"])
    ]
    assert len(matches) == 1
    return matches[0]


def _code_source(relative_path: str) -> str:
    notebook = _notebook(relative_path)
    return "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


def test_source_notebooks_are_clean_and_contain_valid_python() -> None:
    notebook_root = ROOT / "workshops" / "flagos2026" / "notebooks"
    paths = sorted(notebook_root.rglob("*.ipynb"))
    assert len(paths) == 18
    for path in paths:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            ast.parse(source, filename=str(path))
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            assert "pip install flagquantum" not in source.lower()
            assert "export QUAFU_" not in source


@pytest.mark.parametrize(
    "relative_path",
    ("cloud/05_quafu_hardware.ipynb", "cloud/13_trained_hardware.ipynb"),
)
def test_quafu_hardware_labs_use_the_current_task_api(relative_path: str) -> None:
    source = _code_source(relative_path)

    assert 'os.getenv("QUAFU_API_KEY")' in source
    assert "compiler=None" in source
    assert "QUAFU_API_TOKEN" not in source
    assert 'compiler="qsteed"' not in source


@pytest.mark.parametrize(
    "relative_path",
    ("cloud/04_jiuding_jobs.ipynb", "cloud/16_cloud_sweep.ipynb"),
)
def test_jiuding_labs_use_the_current_submit_contract(relative_path: str) -> None:
    source = _code_source(relative_path)

    assert 'target = "jiuding:cpu"' in source
    assert "target=target" in source
    assert "gpus=" not in source


def test_jiuding_helper_uses_target_instead_of_removed_gpus_argument() -> None:
    source = (
        ROOT / "workshops" / "flagos2026" / "scripts" / "jiuding_job.py"
    ).read_text(encoding="utf-8")

    assert '"--target"' in source
    assert 'parser.error("submit requires --target")' in source
    assert "target=args.target" in source
    assert "gpus=" not in source


def test_first_lab_reports_configuration_without_exposing_values() -> None:
    source = _code_source("basics/01_first_circuit.ipynb")

    assert 'bool(os.getenv("QUAFU_API_KEY"))' in source
    assert 'bool(os.getenv("QUAFU_TASK_SERVER_URL"))' in source


@pytest.mark.parametrize("relative_path", NOTEBOOKS)
@pytest.mark.parametrize("full_checkout", (False, True))
def test_workshop_locator_supports_both_deployment_layouts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative_path: str,
    full_checkout: bool,
) -> None:
    if full_checkout:
        repository = tmp_path / "repository"
        workshop = repository / "workshops" / "flagos2026"
        (repository / "pyproject.toml").parent.mkdir(parents=True)
        (repository / "pyproject.toml").touch()
        expected_root = repository
    else:
        workshop = tmp_path / "flagos2026"
        expected_root = workshop
    notebook_dir = workshop / "notebooks" / Path(relative_path).parent
    notebook_dir.mkdir(parents=True)
    monkeypatch.chdir(notebook_dir)

    namespace: dict[str, object] = {}
    exec(_locator_source(relative_path), namespace)

    assert namespace["WORKSHOP"] == workshop.resolve()
    assert namespace["ROOT"] == expected_root.resolve()
    assert namespace["OUTPUTS"] == workshop.resolve() / "outputs"
    assert (workshop / "outputs").is_dir()


def test_workshop_locator_reports_a_clear_error_when_not_deployed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.chdir(empty)

    with pytest.raises(RuntimeError, match="Cannot locate flagos2026/notebooks"):
        exec(_locator_source(NOTEBOOKS[0]), {})

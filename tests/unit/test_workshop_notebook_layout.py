from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = (
    "05_jiuding_jobs.ipynb",
    "06_quafu_hardware.ipynb",
    "../reference_notebooks/cloud/06_compare_results.ipynb",
    "../reference_notebooks/cloud/13_trained_hardware.ipynb",
    "../reference_notebooks/cloud/16_cloud_sweep.ipynb",
    "04_one_problem_three_simulators.ipynb",
    "03_hybrid_model.ipynb",
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
    paths = sorted(notebook_root.glob("*.ipynb"))
    assert len(paths) == 6
    paths += sorted((notebook_root.parent / "reference_notebooks").rglob("*.ipynb"))
    assert len(paths) == 18
    for path in paths:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            ast.parse(
                source.replace(
                    "%matplotlib inline",
                    'get_ipython().run_line_magic("matplotlib", "inline")',
                ),
                filename=str(path),
            )
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            assert "pip install flagquantum" not in source.lower()
            assert "export QUAFU_" not in source


@pytest.mark.parametrize(
    "relative_path",
    (
        "06_quafu_hardware.ipynb",
        "../reference_notebooks/cloud/13_trained_hardware.ipynb",
    ),
)
def test_quafu_hardware_labs_use_the_current_task_api(relative_path: str) -> None:
    source = _code_source(relative_path)

    assert 'os.getenv("QUAFU_API_KEY")' in source
    assert "compiler=None" in source
    assert "QUAFU_API_TOKEN" not in source
    assert 'compiler="qsteed"' not in source


@pytest.mark.parametrize(
    "relative_path",
    ("05_jiuding_jobs.ipynb", "../reference_notebooks/cloud/16_cloud_sweep.ipynb"),
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
    source = _code_source("01_first_circuit.ipynb")

    assert 'bool(os.getenv("QUAFU_API_KEY"))' in source
    assert 'bool(os.getenv("QUAFU_TASK_SERVER_URL"))' in source


@pytest.mark.parametrize(
    "relative_path",
    ("01_first_circuit.ipynb", "../notebooks_cn/01_first_circuit.ipynb"),
)
def test_first_lab_asymmetric_order_example_runs_without_mutating_lesson(
    relative_path: str,
) -> None:
    import flagquantum as fq

    cells = _notebook(relative_path)["cells"]
    examples = [
        "".join(cell["source"])
        for cell in cells
        if cell["cell_type"] == "code" and "order_check =" in "".join(cell["source"])
    ]
    assert len(examples) == 1
    original = fq.Circuit(2)
    namespace = {
        "fq": fq,
        "labels": ["00", "01", "10", "11"],
        "circuit": original,
    }
    # This core-only test checks numerical semantics; real PNG rendering is
    # exercised separately by the workshop's Jupyter validation script.
    tree = ast.parse(examples[0])
    tree.body = [
        node
        for node in tree.body
        if not (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "draw"
        )
        and not (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and isinstance(node.value.func.value, ast.Name)
            and node.value.func.value.id == "plt"
        )
    ]
    exec(compile(tree, relative_path, "exec"), namespace)
    assert namespace["order_probabilities"].tolist() == [0.0, 0.0, 1.0, 0.0]
    assert namespace["circuit"] is original
    assert fq.run(original).to_statevector().reshape(-1).abs().square().tolist() == [
        1.0,
        0.0,
        0.0,
        0.0,
    ]


@pytest.mark.parametrize("relative_path", NOTEBOOKS)
@pytest.mark.parametrize("full_checkout", (False, True))
def test_workshop_locator_supports_both_deployment_layouts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative_path: str,
    full_checkout: bool,
) -> None:
    # This fixture models an offline checkout, not an authenticated platform pod.
    monkeypatch.delenv("JIUDING_AK", raising=False)
    path_exists = Path.exists
    monkeypatch.setattr(
        Path,
        "exists",
        lambda path: False if path == Path("/etc/accesskey") else path_exists(path),
    )
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
    (workshop / "notebooks").mkdir(parents=True, exist_ok=True)
    notebook_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(notebook_dir)
    monkeypatch.syspath_prepend(str(ROOT / "workshops/flagos2026/scripts"))

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

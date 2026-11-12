"""Classified examples retain source identity, honest coverage and executable cases."""

import ast
import json
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest

from tools.run_qmod_catalog import run_case
from tools.synthesis_catalog import (
    CATALOG,
    ROOT,
    case_readme,
    example_path,
    load_catalog,
    render,
    verify_catalog,
)

pytestmark = pytest.mark.unit


def test_native_user_library_contains_no_importer_sources_or_calls() -> None:
    library = ROOT / "examples/synthesis"
    assert not list(library.rglob("*.qmod"))
    for path in library.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith("flagquantum.")
            ):
                assert not any(
                    part.startswith("_") for part in node.module.split(".")
                ), path
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert not re.search(
                    r"\bqfunc\s+\w+\s*\([^)]*\)\s*\{", node.value
                ), path
            if isinstance(node, ast.Call):
                assert not any(
                    keyword.arg == "source_format"
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value == "qmod"
                    for keyword in node.keywords
                ), path


def test_expanded_inventory_separates_quantum_and_workflow_acceptance() -> None:
    data = json.loads((CATALOG.parent / "native_library_inventory.json").read_text())
    assert set(data["categories"]) == {
        "algorithms",
        "applications",
        "community",
        "functions",
        "tutorials",
    }
    groups = data["groups"]
    assert len(groups) == 417 == len({group["id"] for group in groups})
    sources = [source for group in groups for source in group["sources"]]
    assert len(sources) == 577 == len({source["path"] for source in sources})
    assert Counter(Path(source["path"]).suffix for source in sources) == {
        ".qmod": 216,
        ".ipynb": 213,
        ".py": 148,
    }
    for group in groups:
        assert group["category"] == Path(group["id"]).parts[0]
        assert group["workflow_status"] in {"pending", "validated"}
        if group["workflow_status"] == "validated":
            proof = group["workflow_validation"]
            assert group["native_program"]
            assert proof["reviewed_code_cells"]
            assert (ROOT / proof["acceptance_test"]).is_file()
            notebook = next(
                source for source in group["sources"] if source["kind"] == "notebook"
            )
            assert proof["git_blob_sha1"] == notebook["git_blob_sha1"]
            assert proof["upstream_commit"] == data["upstream_commit"]
            provenance = ROOT / group["native_program"]["path"]
            assert (
                json.loads(provenance.with_name("provenance.json").read_text()) == proof
            )
        if group["native_program"]:
            assert (ROOT / group["native_program"]["path"]).is_file()


def test_complete_catalog_preserves_sources_and_generated_navigation() -> None:
    catalog = load_catalog()
    verify_catalog(catalog)
    for path, text in render(catalog).items():
        assert path.read_text(encoding="utf-8") == text
    assert {case["category"] for case in catalog["cases"]} == {
        "tutorials",
        "algorithms",
        "applications",
        "community",
        "functions",
    }
    assert CATALOG.is_file()


@pytest.mark.parametrize("replacement", ["qpe", "missing_qpe"])
def test_inventory_rejects_missing_or_case_mismatched_native_program(
    monkeypatch: pytest.MonkeyPatch, replacement: str
) -> None:
    inventory = CATALOG.parent / "native_library_inventory.json"
    original_read = Path.read_text

    def read_text(
        path: Path, encoding: str | None = None, errors: str | None = None
    ) -> str:
        text = original_read(path, encoding=encoding, errors=errors)
        if path == inventory:
            data = json.loads(text)
            group = next(
                group
                for group in data["groups"]
                if group["id"]
                == "tutorials/technology_demonstrations/QPE/qpe_for_unitary_matrix/qpe_for_unitary_matrix"
            )
            group["native_program"]["path"] = group["native_program"]["path"].replace(
                "/QPE/", f"/{replacement}/"
            )
            return json.dumps(data)
        return text

    monkeypatch.setattr(Path, "read_text", read_text)
    with pytest.raises(ValueError, match="missing or case-mismatched native program"):
        render(load_catalog())


@pytest.mark.parametrize(
    "source",
    [
        "def reference_circuit() -> None:\n    return None\n",
        'if __name__ == "__main__":\n    pass\n',
    ],
)
def test_catalog_rejects_reference_only_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    catalog = load_catalog()
    comparison = catalog["cases"][0]["native_comparison"]
    assert comparison is not None
    reference_only = tmp_path / "circuit.py"
    reference_only.write_text(source)

    def comparison_path(path: str) -> Path:
        return reference_only if path == comparison else example_path(path)

    monkeypatch.setattr("tools.synthesis_catalog.example_path", comparison_path)
    with pytest.raises(ValueError, match="executable comparison"):
        verify_catalog(catalog)


@pytest.mark.parametrize(
    "change", ["missing", "duplicate", "source_digest", "bindings", "revision"]
)
def test_catalog_rejects_incomplete_or_mislabelled_examples(change: str) -> None:
    catalog = deepcopy(load_catalog())
    if change == "missing":
        catalog["cases"].pop()
    elif change == "duplicate":
        catalog["cases"][-1] = catalog["cases"][0]
    elif change == "source_digest":
        catalog["cases"][0]["original"]["sha256"] = "0" * 64
    elif change == "bindings":
        catalog["cases"][0]["classical_inputs"] = {"invented": 0.4}
    else:
        catalog["cases"][0]["approved"]["revised"] = True
    with pytest.raises(ValueError):
        verify_catalog(catalog)


def test_native_coverage_requires_the_exact_original_source() -> None:
    catalog = deepcopy(load_catalog())
    native = next(case for case in catalog["cases"] if case["native_example"])
    other = next(case for case in catalog["cases"] if case["id"] != native["id"])
    other["native_example"] = native["native_example"]
    other["native_comparison"] = native["native_comparison"]
    with pytest.raises(ValueError, match="not tied to this original"):
        verify_catalog(catalog)


def test_example_assets_cannot_escape_the_catalog_directory() -> None:
    with pytest.raises(ValueError, match="outside the synthesis examples"):
        example_path("examples/synthesis/../../../source.qmod")


def test_original_and_approved_swap_inputs_have_distinct_results() -> None:
    catalog = load_catalog()
    case = next(
        case
        for case in catalog["cases"]
        if case["id"] == "algorithms/quantum_primitives/swap_test/swap_test.qmod"
    )
    original = run_case(case, catalog, revision="original")
    approved = run_case(case, catalog, revision="approved")
    assert original["stage"] == "frontend_refused"
    assert approved["stage"] == "execution_smoke_passed"
    assert original["source_sha256"] != approved["source_sha256"]
    assert approved["operations"] == case["recorded_approved"]["operations"]


def test_original_simple_case_uses_its_recorded_resources() -> None:
    catalog = load_catalog()
    case = next(
        case
        for case in catalog["cases"]
        if case["id"].endswith("bernstein_vazirani.qmod")
    )
    observed = run_case(case, catalog, revision="original")
    assert observed["stage"] == "execution_smoke_passed"
    for field in ("qubits", "operations", "depth"):
        assert observed[field] == case["recorded_approved"][field]


def test_companion_preferences_keep_the_source_basename() -> None:
    catalog = load_catalog()
    case = next(case for case in catalog["cases"] if case["companion_files"])
    case["companion_files"][0]["path"] = catalog["license"]["path"]
    case["companion_files"][0]["path"] += ".synthesis_options.json"
    with pytest.raises(ValueError, match="companion preferences"):
        verify_catalog(catalog)


def test_autocallable_companion_precision_is_consumed_after_relocation() -> None:
    catalog = load_catalog()
    case = next(case for case in catalog["cases"] if case["companion_files"])
    observed = run_case(case, catalog, revision="original")
    assert observed["stage"] == "execution_smoke_passed"
    for field in ("qubits", "operations", "depth"):
        assert observed[field] == case["recorded_original"][field]


@pytest.mark.parametrize("validated", [False, True])
def test_workflow_guide_does_not_determine_notebook_acceptance(validated: bool) -> None:
    catalog = load_catalog()
    case = next(
        case
        for case in catalog["cases"]
        if case["id"].endswith("/facility_location.qmod")
    )
    assert (example_path(case["directory"]) / "WORKFLOW.md").is_file()
    text = case_readme(case, catalog, notebook_validated=validated)
    assert ("complete local Notebook workflow are validated" in text) == validated
    assert ("quantum-program guide" in text) == (not validated)


def test_rendered_workflow_claims_follow_inventory_acceptance() -> None:
    catalog = load_catalog()
    inventory = json.loads(
        (CATALOG.parent / "native_library_inventory.json").read_text()
    )
    validated = {
        (group["id"], group["native_program"]["path"])
        for group in inventory["groups"]
        if group["kind"] == "notebook_workflow"
        and group["workflow_status"] == "validated"
        and group["native_program"]
    }
    assert len(validated) == 24
    rendered = render(catalog)
    assert "**24/213**" in rendered[ROOT / "examples/synthesis/CATALOG.md"]
    for case in catalog["cases"]:
        if case["native_example"]:
            text = rendered[example_path(case["directory"]) / "README.md"]
            assert ("complete local Notebook workflow are validated" in text) == (
                (case["id"][:-5], case["native_example"]) in validated
            ), case["id"]

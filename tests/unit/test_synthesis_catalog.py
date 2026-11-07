"""Classified examples retain source identity, honest coverage and executable cases."""

from copy import deepcopy

import pytest

from examples.synthesis.run_qmod import run_case
from tools.synthesis_catalog import (
    CATALOG,
    example_path,
    load_catalog,
    render,
    verify_catalog,
)

pytestmark = pytest.mark.unit


def test_complete_catalog_preserves_sources_and_generated_navigation() -> None:
    catalog = load_catalog()
    verify_catalog(catalog)
    for path, text in render(catalog).items():
        assert path.read_text(encoding="utf-8") == text
    assert {case["category"] for case in catalog["cases"]} == {
        "tutorials",
        "algorithms",
        "applications",
    }
    assert CATALOG.is_file()


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
    missing = next(case for case in catalog["cases"] if not case["native_example"])
    missing["native_example"] = native["native_example"]
    missing["native_comparison"] = native["native_comparison"]
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

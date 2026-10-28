"""A failing check should say whether the pull request caused it.

CI here runs only on pull requests, and each run reports every failure in the
tree — including ones that were already failing on `main`. A reader cannot tell
the two apart, so a contributor has to re-run the same tests against a pristine
base by hand to find out, which is what this tool automates.

The properties pinned here are the ones that make the answer trustworthy: a
failure that could not be attributed is reported as unattributable rather than
assumed pre-existing, and a run that cannot verify which tree it measured is
refused rather than compared.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from tools.partition_failures_against_base import (
    UnattributableComparisonError,
    failure_ids,
    partition,
    require_base_package,
)

pytestmark = pytest.mark.unit


def _report(path: Path, *cases: tuple[str, str, str]) -> Path:
    """Write a junit report with ``(classname, name, outcome)`` cases."""

    suite = ET.Element("testsuite", name="pytest")
    for classname, name, outcome in cases:
        case = ET.SubElement(suite, "testcase", classname=classname, name=name)
        if outcome in {"failure", "error"}:
            ET.SubElement(case, outcome, message="boom")
        elif outcome == "skipped":
            ET.SubElement(case, "skipped")
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)
    return path


def test_a_failure_and_an_error_are_both_failures(tmp_path: Path) -> None:
    """A test that errors in teardown failed just as much as one that asserted.

    pytest reports the two under different element names, and a tool that read
    only `<failure>` would call an erroring test healthy.
    """

    report = _report(
        tmp_path / "head.xml",
        ("tests.unit.test_a", "test_broken", "failure"),
        ("tests.unit.test_a", "test_errors", "error"),
        ("tests.unit.test_a", "test_fine", "passed"),
        ("tests.unit.test_a", "test_skipped", "skipped"),
    )

    assert failure_ids(report) == frozenset(
        {
            "tests/unit/test_a.py::test_broken",
            "tests/unit/test_a.py::test_errors",
        }
    )


def test_a_parameterised_id_keeps_its_parameter_suffix(tmp_path: Path) -> None:
    """Two parameters of one test are two checks, and must stay separable."""

    report = _report(
        tmp_path / "head.xml",
        ("tests.unit.test_a", "test_x[dtype0]", "failure"),
        ("tests.unit.test_a", "test_x[dtype1]", "passed"),
    )

    assert failure_ids(report) == frozenset({"tests/unit/test_a.py::test_x[dtype0]"})


def test_the_partition_names_who_owns_each_failure() -> None:
    head = frozenset({"a::1", "b::1", "c::1"})
    base = frozenset({"b::1", "c::1", "d::1"})

    result = partition(head, base)

    assert result["introduced_by_this_change"] == ["a::1"]
    assert result["pre_existing_on_base"] == ["b::1", "c::1"]
    assert result["fixed_by_this_change"] == ["d::1"]


def test_a_failure_absent_from_the_base_run_is_attributed_to_the_change() -> None:
    """A new test cannot have been failing before it existed.

    The base run simply has no record of a node that the head added, so treating
    "not in the base failures" as "pre-existing" would invert the answer for
    exactly the case a contributor most needs.
    """

    result = partition(frozenset({"new_module.py::test_new"}), frozenset())

    assert result["introduced_by_this_change"] == ["new_module.py::test_new"]
    assert result["pre_existing_on_base"] == []


def test_the_partition_is_reported_in_a_stable_order() -> None:
    """Two runs over the same failures must produce the same report."""

    result = partition(frozenset({"z::1", "a::1", "m::1"}), frozenset())

    assert result["introduced_by_this_change"] == ["a::1", "m::1", "z::1"]


def test_a_base_package_imported_from_elsewhere_is_refused(
    tmp_path: Path,
) -> None:
    """Measuring the wrong tree would answer a question nobody asked.

    A virtualenv that already has the package installed shadows an extracted
    tree unless the search path is arranged, and every comparison then describes
    the installed version rather than the base revision.
    """

    tree = tmp_path / "base"
    (tree / "flagquantum").mkdir(parents=True)
    (tree / "flagquantum" / "__init__.py").write_text("", encoding="utf-8")

    with pytest.raises(UnattributableComparisonError, match="outside"):
        require_base_package(
            tree, resolve=lambda: "/some/venv/site-packages/flagquantum/__init__.py"
        )


def test_a_base_package_resolved_inside_the_tree_is_accepted(tmp_path: Path) -> None:
    tree = tmp_path / "base"
    (tree / "flagquantum").mkdir(parents=True)
    (tree / "flagquantum" / "__init__.py").write_text("", encoding="utf-8")
    resolved = str(tree / "flagquantum" / "__init__.py")

    require_base_package(tree, resolve=lambda: resolved)


def test_a_missing_base_tree_is_refused_rather_than_compared(tmp_path: Path) -> None:
    with pytest.raises(UnattributableComparisonError, match="does not contain"):
        require_base_package(tmp_path / "absent", resolve=lambda: "/anywhere")

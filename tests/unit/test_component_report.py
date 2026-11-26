"""Complete source report-card behavior and the native shared-check consumer."""

import sys
import time
import types
from collections.abc import Callable
from functools import partial
from typing import Any

import pandas as pd
import pytest

from flagquantum.testing.component_examples import ComponentExample
from flagquantum.testing.example_checks import (
    CHECKS,
    CheckKind,
    CheckResult,
    CheckStatus,
    RegisterExample,
    check_example,
)
from tests.unit.test_library_call_graph import load_reference
from tests.unit.test_reference_testing_file import lower_fixture, parallel_example
from tools.development import component_report as native
from tools.ftqc_source_inputs import FTQC_SOURCE_INVENTORY

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture
def original(monkeypatch: pytest.MonkeyPatch) -> Any:
    from tests.unit.test_development_tools import REFERENCE

    package = types.ModuleType("_library_reference")
    package.__path__ = [str(REFERENCE / "dev_tools" / "qualtran_dev_tools")]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    return load_reference("bloq_report_card")


def source_example(name: str, component_type: type) -> Any:
    return types.SimpleNamespace(name=name, bloq_cls=component_type)


def source_statuses(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rename(columns={"bloq_cls": "component_cls"}).map(
        lambda value: CheckStatus[value.name] if hasattr(value, "name") else value
    )


def test_complete_source_status_palette_and_missing_records(original: Any) -> None:
    from qualtran.testing import BloqCheckResult

    for status in CheckStatus:
        source = BloqCheckResult[status.name]
        assert native.color_status(status) == original.color_status(source)
        assert native.format_status(status) == original.format_status(source)
    classes = [int, str, float, int]
    examples: list[ComponentExample[object]] = [
        ComponentExample(int, "a", int),
        ComponentExample(int, "b", int),
    ]
    source_examples = [
        source_example(item.name, item.component_type) for item in examples
    ]
    assert (
        native.classes_without_examples(classes, examples)
        == (original.bloq_classes_with_no_examples(classes, source_examples))
        == {str, float}
    )
    for cls in classes:
        assert (
            native.record_for_class_without_examples(cls)
            == source_statuses(
                pd.DataFrame([original.record_for_class_with_no_examples(cls)])
            )
            .iloc[0]
            .to_dict()
        )


def install_checks(
    original: Any, monkeypatch: pytest.MonkeyPatch, calls: list[tuple[str, str]]
) -> Callable[[ComponentExample[object], CheckKind], CheckResult]:
    from qualtran.testing import BloqCheckResult

    names = (
        "check_bloq_example_make",
        "check_bloq_example_decompose",
        "check_equivalent_bloq_example_counts",
        "check_bloq_example_serializes",
        "check_bloq_example_qtyping",
    )
    statuses = tuple(CheckStatus)

    def check(example: ComponentExample[object], kind: CheckKind) -> CheckResult:
        calls.append((example.name, kind))
        index = (int(example.name[-1]) + CHECKS.index(kind)) % len(statuses)
        return CheckResult(statuses[index])

    def reference_check(kind: CheckKind, example: Any) -> tuple[Any, str]:
        result = check(ComponentExample(int, example.name, example.bloq_cls), kind)
        return BloqCheckResult[result.status.name], ""

    for name, kind in zip(names, CHECKS, strict=True):
        monkeypatch.setattr(original, name, partial(reference_check, kind))
    return check


def test_complete_source_report_summary_and_styling(
    original: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, str]] = []
    check = install_checks(original, monkeypatch, calls)
    classes = [int, str, float]
    examples: list[ComponentExample[object]] = [
        ComponentExample(int, f"sample{i}", int) for i in range(6)
    ]
    source_examples = [
        source_example(item.name, item.component_type) for item in examples
    ]
    expected = original.get_bloq_report_card(classes, source_examples, "built")
    expected_calls = list(calls)
    calls.clear()
    actual = native.get_report(
        classes, examples, "built", check=check, is_component=lambda cls: True
    )
    assert (
        calls
        == expected_calls
        == [(example.name, kind) for example in examples for kind in CHECKS]
    )
    pd.testing.assert_frame_equal(actual, source_statuses(expected))
    pd.testing.assert_frame_equal(
        native.summarize_results(actual), original.summarize_results(expected)
    )
    # Normalize only the display identity column and generated table UUID.
    expected_html = (
        original.show_bloq_report_card(expected).set_uuid("report").to_html()
    )
    actual_html = native.show_report(actual).set_uuid("report").to_html()
    assert actual_html == expected_html.replace("bloq_cls", "component_cls")


def test_default_discovery_exclusions_and_explicit_override(
    original: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    names = [
        "qubitization_qpe_hubbard_model_small",
        "qubitization_qpe_hubbard_model_large",
        "included0",
    ]
    examples: list[ComponentExample[object]] = [
        ComponentExample(int, name, int) for name in names
    ]
    source_examples = [
        source_example(item.name, item.component_type) for item in examples
    ]
    monkeypatch.setattr(original, "get_bloq_classes", lambda: [int, str])
    monkeypatch.setattr(original, "get_bloq_examples", lambda: source_examples)
    monkeypatch.setattr(native, "get_component_classes", lambda **kwargs: [int, str])
    monkeypatch.setattr(native, "get_component_examples", lambda: examples)
    from qualtran.testing import BloqCheckResult

    for name in (
        "check_bloq_example_make",
        "check_bloq_example_decompose",
        "check_equivalent_bloq_example_counts",
        "check_bloq_example_serializes",
        "check_bloq_example_qtyping",
    ):
        monkeypatch.setattr(original, name, lambda example: (BloqCheckResult.PASS, ""))

    def check(example: ComponentExample[object], kind: CheckKind) -> CheckResult:
        return CheckResult(CheckStatus.PASS)

    expected = original.get_bloq_report_card()
    actual = native.get_report(check=check, is_component=lambda cls: True)
    pd.testing.assert_frame_equal(actual, source_statuses(expected))
    assert set(actual["name"]) == {"-", "included0"}
    expected = original.get_bloq_report_card(bexamples=source_examples)
    actual = native.get_report(
        examples=examples, check=check, is_component=lambda cls: True
    )
    pd.testing.assert_frame_equal(actual, source_statuses(expected))
    assert len(actual) == 4


def test_source_slow_warning_failures_and_iterable_boundaries(
    original: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from qualtran.testing import BloqCheckResult

    for name in (
        "check_bloq_example_make",
        "check_bloq_example_decompose",
        "check_equivalent_bloq_example_counts",
        "check_bloq_example_serializes",
        "check_bloq_example_qtyping",
    ):
        monkeypatch.setattr(original, name, lambda example: (BloqCheckResult.PASS, ""))
    example: ComponentExample[object] = ComponentExample(int, "slow", int)

    def check(example: ComponentExample[object], kind: CheckKind) -> CheckResult:
        return CheckResult(CheckStatus.PASS)

    times = iter([0.0, 2.0, 0.0, 2.0])
    monkeypatch.setattr(time, "perf_counter", lambda: next(times))
    with pytest.warns(UserWarning, match="slow took 2.0 to check"):
        original.record_for_bloq_example(source_example("slow", int))
    with pytest.warns(UserWarning, match="slow took 2.0 to check"):
        native.record_for_example(example, check)
    monkeypatch.undo()
    empty_reports: tuple[Callable[[], object], ...] = (
        lambda: original.get_bloq_report_card([], []),
        lambda: native.get_report([], [], check=check, is_component=lambda cls: True),
    )
    for report in empty_reports:
        with pytest.raises(KeyError, match="package"):
            report()
    # Original generator exhaustion is retained as a documented input boundary.
    expected = original.get_bloq_report_card(
        [int, str], iter([source_example("a", int)])
    )
    actual = native.get_report(
        [int, str], iter([example]), check=check, is_component=lambda cls: True
    )
    pd.testing.assert_frame_equal(actual, source_statuses(expected))

    def broken(example: ComponentExample[object], kind: CheckKind) -> CheckResult:
        raise RuntimeError("checker failed")

    with pytest.raises(RuntimeError, match="checker failed"):
        native.record_for_example(example, broken)


def test_real_native_register_checks_and_existing_test_import_identity() -> None:
    from flagquantum.core.ir.program import ProgramRecord
    from tests.unit import register_example_checks as adapter

    assert adapter.check_example is check_example
    assert adapter.CheckStatus is CheckStatus
    example: ComponentExample[object] = ComponentExample(
        parallel_example, "parallel", ProgramRecord
    )
    owned = RegisterExample(parallel_example, lower_fixture, lambda: {"x": 3})
    report = native.get_report(
        [ProgramRecord],
        [example],
        check=lambda example, kind: check_example(owned, kind),
        is_component=lambda cls: cls is ProgramRecord,
    )
    assert report.iloc[0][native.CHECK_COLUMNS].tolist() == [CheckStatus.PASS] * 5
    assert native.summarize_results(report)["pass"].tolist() == [1] * 5


def test_source_report_blob() -> None:
    import csv
    import hashlib

    from tests.unit.test_development_tools import REFERENCE

    with (FTQC_SOURCE_INVENTORY).open(encoding="utf-8-sig") as stream:
        row = next(row for row in csv.DictReader(stream) if row["id"] == "Q-0058")
    raw = (REFERENCE / row["path"]).read_bytes()
    assert (
        hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        == row["git_blob"]
    )

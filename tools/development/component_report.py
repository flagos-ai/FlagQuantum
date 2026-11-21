# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 dev_tools/qualtran_dev_tools/bloq_report_card.py.

"""Tabulate example checks without owning decomposition or execution policy."""

import time
import warnings
from collections.abc import Callable, Iterable

import pandas as pd
from pandas.io.formats.style import Styler

from flagquantum.testing.component_examples import ComponentExample
from flagquantum.testing.example_checks import CheckKind, CheckResult, CheckStatus
from tools.development.component_discovery import (
    get_component_classes,
    get_component_examples,
)

ID_COLUMNS = ["package", "component_cls", "name"]
CHECK_COLUMNS = ["make", "decomp", "counts", "serialize", "qtyping"]
_CHECK_KINDS: tuple[CheckKind, ...] = (
    "make",
    "decompose",
    "counts",
    "serialize",
    "qtyping",
)


def _get_package(component_cls: type) -> str:
    return ".".join(component_cls.__module__.split(".")[:-1])


def color_status(value: CheckStatus) -> str:
    """Return the original report's CSS for each check outcome."""
    if value is CheckStatus.PASS:
        return "background-color:lightgreen"
    if value is CheckStatus.MISSING:
        return "background-color:lightyellow"
    if value is CheckStatus.NA:
        return "background-color:lightgrey"
    if value is CheckStatus.UNVERIFIED:
        return "background-color:lightblue"
    return "background-color:red"


def format_status(value: CheckStatus) -> str:
    """Render a check outcome as its lowercase status name."""
    return value.name.lower()


def classes_without_examples(
    classes: Iterable[type], examples: Iterable[ComponentExample[object]]
) -> set[type]:
    """Find classes absent from the supplied example metadata."""
    missing = set(classes)
    for example in examples:
        missing.discard(example.component_type)
    return missing


def record_for_class_without_examples(component_cls: type) -> dict[str, object]:
    """Make a report row with all checks marked missing."""
    return {
        "component_cls": component_cls.__name__,
        "package": _get_package(component_cls),
        "name": "-",
    } | dict.fromkeys(CHECK_COLUMNS, CheckStatus.MISSING)


def record_for_example(
    example: ComponentExample[object],
    check: Callable[[ComponentExample[object], CheckKind], CheckResult],
) -> dict[str, object]:
    """Run the five domain-supplied checks in the original order.

    The checker owns conversion to an existing RegisterExample or another
    domain fixture. This tool never imports a test module or guesses a lowerer.
    """
    start = time.perf_counter()
    record: dict[str, object] = {
        "component_cls": example.component_type.__name__,
        "package": _get_package(example.component_type),
        "name": example.name,
    }
    for column, kind in zip(CHECK_COLUMNS, _CHECK_KINDS, strict=True):
        record[column] = check(example, kind).status
    duration = time.perf_counter() - start
    if duration > 1.0:
        warnings.warn(f"{example.name} took {duration} to check.")
    return record


def show_report(report: pd.DataFrame) -> Styler:
    """Style the five check columns using the original status palette."""
    return report.style.map(color_status, CHECK_COLUMNS).format(
        format_status, CHECK_COLUMNS
    )


def get_report(
    classes: Iterable[type] | None = None,
    examples: Iterable[ComponentExample[object]] | None = None,
    package_prefix: str = "flagquantum.algorithms.",
    *,
    check: Callable[[ComponentExample[object], CheckKind], CheckResult],
    is_component: Callable[[type], bool],
) -> pd.DataFrame:
    """Discover or consume examples and return the sorted status table.

    Supply explicit examples to override the original default slow-example
    exclusions. Like the original, examples are traversed twice. Use a reusable
    iterable to retain example rows. An entirely empty input raises KeyError.
    """
    if classes is None:
        classes = get_component_classes(is_component=is_component)
    if examples is None:
        examples = get_component_examples()
        skips = {
            "qubitization_qpe_hubbard_model_small",
            "qubitization_qpe_hubbard_model_large",
        }
        examples = [example for example in examples if example.name not in skips]
    missing = classes_without_examples(classes, examples)
    records = [record_for_class_without_examples(cls) for cls in missing]
    records.extend(record_for_example(example, check) for example in examples)
    report = pd.DataFrame(records)
    report["package"] = report["package"].str.removeprefix(package_prefix)
    return (
        report.sort_values(by=ID_COLUMNS).loc[:, ID_COLUMNS + CHECK_COLUMNS].reindex()
    )


def summarize_results(report: pd.DataFrame) -> pd.DataFrame:
    """Count each status per check in the original first-observed order."""
    summary = (
        pd.DataFrame(
            [report[column].value_counts().rename(column) for column in CHECK_COLUMNS]
        )
        .fillna(0)
        .astype(int)
    )
    summary.columns = [value.name.lower() for value in summary.columns]
    return summary

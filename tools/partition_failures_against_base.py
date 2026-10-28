#!/usr/bin/env python3
"""Say which failing tests a change introduced and which were already failing.

CI here runs on pull requests only, and a run reports every failure in the tree.
A test that `main` already fails therefore appears in every pull request's checks
with nothing to distinguish it, so a contributor has to re-run the same identifiers
against a pristine base by hand before they can tell whether they broke anything.
This tool is that comparison, with the safeguards that make its answer usable.

Three properties matter more than the plumbing.

It compares identifiers, not counts. Two failures with the same total can be
disjoint sets, and the whole question is which identifiers are whose.

It fails closed. A base run that could not be produced, a base revision that
cannot be extracted, and a base interpreter that imports the package from
somewhere other than the extracted tree all report the comparison as
unattributable instead of returning "nothing introduced". A virtualenv that
already has the package installed shadows an extracted tree unless the search
path is arranged, and a comparison against the installed version rather than the
base revision answers a different question than the one asked.

It does not suppress anything. The tool reports a partition; it does not tell CI
to ignore a failure, and nothing here changes what a check requires.

Examples
--------
Partition a run's failures against the merge base::

    python -m pytest -m "smoke or unit" -q --junitxml=/tmp/head.xml
    python tools/partition_failures_against_base.py \\
        --head-report /tmp/head.xml --base origin/main

Exit status is 0 when nothing was introduced, 1 when something was, and 2 when
the comparison could not be made.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

#: Element names pytest uses for a test that did not pass. Both are failures for
#: this purpose: an error in a fixture or teardown is a broken check, not a pass.
FAILURE_ELEMENTS = ("failure", "error")


class UnattributableComparisonError(RuntimeError):
    """The comparison cannot be made, so no partition is claimed."""


def _node_id(classname: str, name: str) -> str:
    """Return the pytest node identifier a junit case came from.

    pytest writes the module as a dotted path and the test as its parameterised
    name, which is enough to rebuild the identifier a reader would type.
    """

    module = classname.replace(".", "/") + ".py"
    return f"{module}::{name}"


def failure_ids(report: Path) -> frozenset[str]:
    """Return every test identifier that did not pass in one junit report."""

    tree = ET.parse(report)
    failed: set[str] = set()
    for case in tree.iter("testcase"):
        classname = case.get("classname") or ""
        name = case.get("name") or ""
        if not classname or not name:
            continue
        if any(case.find(element) is not None for element in FAILURE_ELEMENTS):
            failed.add(_node_id(classname, name))
    return frozenset(failed)


def partition(head: Iterable[str], base: Iterable[str]) -> dict[str, list[str]]:
    """Split two failure sets into introduced, pre-existing and fixed.

    A failure the base run does not mention belongs to this change: a test that
    did not exist there cannot have been failing there, and reading "absent" as
    "pre-existing" would invert the answer for exactly the case a contributor
    most needs.
    """

    head_set = set(head)
    base_set = set(base)
    return {
        "introduced_by_this_change": sorted(head_set - base_set),
        "pre_existing_on_base": sorted(head_set & base_set),
        "fixed_by_this_change": sorted(base_set - head_set),
    }


def require_base_package(
    tree: Path,
    *,
    resolve: Callable[[], str],
) -> None:
    """Refuse to compare unless the base tree is the package being imported.

    ``resolve`` returns the path of the imported package. It is injected so the
    guard can be exercised without a subprocess, and so the caller decides which
    interpreter asks.
    """

    package = tree / "flagquantum"
    if not (package / "__init__.py").is_file():
        raise UnattributableComparisonError(
            f"base tree {tree} does not contain flagquantum/__init__.py, so the "
            "base revision was not extracted"
        )
    resolved = Path(resolve()).resolve()
    if package.resolve() not in resolved.parents:
        raise UnattributableComparisonError(
            "the base interpreter imports flagquantum from outside the extracted "
            f"tree: {resolved} is not under {package.resolve()}. A comparison "
            "against an installed copy answers a different question than the one "
            "asked, so this run is not attributed."
        )


def _run(
    arguments: Sequence[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(arguments),
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def extract_base(revision: str, destination: Path) -> Path:
    """Write the tree at ``revision`` into ``destination`` and return it."""

    archive = subprocess.run(
        ["git", "archive", "--format=tar", revision],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    if archive.returncode != 0:
        raise UnattributableComparisonError(
            f"cannot extract base revision {revision!r}: "
            f"{archive.stderr.decode('utf-8', 'replace').strip()}"
        )
    # Written to a file rather than streamed: `tarfile.open` takes a path
    # without a custom reader, and one fewer class here is one fewer thing to
    # be wrong about the archive it is reading.
    bundle = destination.parent / "base.tar"
    bundle.write_bytes(archive.stdout)
    with tarfile.open(bundle) as table:
        table.extractall(destination, filter="data")
    bundle.unlink()
    return destination


def _base_environment(tree: Path) -> dict[str, str]:
    """Return an environment that resolves the package from ``tree`` first."""

    environment = dict(os.environ)
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(tree) if not existing else os.pathsep.join((str(tree), existing))
    )
    return environment


def _resolve_from_base(python: str, tree: Path, *, timeout: float) -> str:
    probe = _run(
        [
            python,
            "-c",
            "import flagquantum, sys; sys.stdout.write(flagquantum.__file__)",
        ],
        cwd=tree,
        env=_base_environment(tree),
        timeout=timeout,
    )
    if probe.returncode != 0:
        raise UnattributableComparisonError(
            "the base interpreter cannot import flagquantum: " f"{probe.stderr.strip()}"
        )
    return probe.stdout.strip()


def run_base_failures(
    revision: str,
    node_ids: Sequence[str],
    *,
    python: str = sys.executable,
    timeout: float = 1800.0,
) -> frozenset[str]:
    """Return which of ``node_ids`` fail on ``revision``."""

    if not node_ids:
        return frozenset()
    with tempfile.TemporaryDirectory(prefix="fq-base-") as directory:
        tree = extract_base(revision, Path(directory))
        require_base_package(
            tree,
            resolve=lambda: _resolve_from_base(
                python, tree, timeout=min(timeout, 120.0)
            ),
        )
        report = Path(directory) / "base.xml"
        completed = _run(
            [
                python,
                "-m",
                "pytest",
                *node_ids,
                "-q",
                "-p",
                "no:cacheprovider",
                f"--junitxml={report}",
            ],
            cwd=tree,
            env=_base_environment(tree),
            timeout=timeout,
        )
        if not report.is_file():
            raise UnattributableComparisonError(
                "the base run produced no report, so nothing can be attributed: "
                f"{completed.stdout.strip()[-400:]}"
            )
        return failure_ids(report)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--head-report",
        type=Path,
        required=True,
        help="junit report of the run being attributed",
    )
    parser.add_argument(
        "--base",
        default="origin/main",
        help="revision to compare against, as a ref this repository holds",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)

    head = failure_ids(arguments.head_report)
    print(f"failures in this run: {len(head)}")
    if not head:
        payload: dict[str, Any] = {
            "base": arguments.base,
            "head_failures": [],
            "introduced_by_this_change": [],
            "pre_existing_on_base": [],
            "fixed_by_this_change": [],
        }
        _write(arguments.output, payload)
        return 0

    try:
        base = run_base_failures(
            arguments.base,
            sorted(head),
            python=arguments.python,
            timeout=arguments.timeout,
        )
    except (UnattributableComparisonError, subprocess.TimeoutExpired) as error:
        print(f"unattributable: {error}", file=sys.stderr)
        print(
            "No partition is claimed. Re-run the failing identifiers against the "
            "base revision by hand, or fix the environment so the base tree is "
            "the package under test.",
            file=sys.stderr,
        )
        return 2

    result = partition(head, base)
    payload = {
        "base": arguments.base,
        "head_failures": sorted(head),
        "base_failures": sorted(base),
        **result,
    }
    _write(arguments.output, payload)
    print(f"introduced by this change: {len(result['introduced_by_this_change'])}")
    for node in result["introduced_by_this_change"]:
        print(f"  {node}")
    print(f"pre-existing on {arguments.base}: {len(result['pre_existing_on_base'])}")
    for node in result["pre_existing_on_base"]:
        print(f"  {node}")
    if result["fixed_by_this_change"]:
        print(f"fixed by this change: {len(result['fixed_by_this_change'])}")
    return 1 if result["introduced_by_this_change"] else 0


def _write(output: Path | None, payload: dict[str, Any]) -> None:
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

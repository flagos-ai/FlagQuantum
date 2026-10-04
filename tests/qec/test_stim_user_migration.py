"""Run the Stim migration guide the way a reader would.

`docs/guides/STIM_USER_MIGRATION.md` is the route a Stim user takes into this
package, and every command in it is a claim about what happens. This module
executes every Python fence in the guide, in one shared namespace and in the
order the guide prints them, and compares each print against the transcript
quoted under it.

**A quoted value has to carry the ``--`` separator to count as a transcript.**
The guide also carries ordinary explanatory comments, and a print followed by
a sentence about it is not a promise about its output; requiring the separator
keeps this check from turning prose into an assertion.

The last fence compares the in-tree matcher against the PyMatching cross-check
and is therefore the one part of the guide that needs the ``pymatching`` extra.
It is its own test so that a caller without the extra still runs the other
seven fences, which are the migration itself.
"""

from __future__ import annotations

import ast
import contextlib
import io
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

stim = pytest.importorskip("stim")

ROOT = Path(__file__).resolve().parents[2]
GUIDE = ROOT / "docs/guides/STIM_USER_MIGRATION.md"
EXAMPLE = "examples.qec.stim_user_migration"
TIMEOUT_SECONDS = 600

PYTHON_FENCE = re.compile(r"^```python\n(.*?)^```", re.MULTILINE | re.DOTALL)
QUOTE_SEPARATOR = re.compile(r"(?:^|\s)--\s")

#: The fence that decodes through the PyMatching cross-check, identified by the
#: name it asks the registry for rather than by its position, so inserting a
#: fence above it does not silently move the skip onto a different one.
CROSS_CHECK_MARKER = 'get_decoder("pymatching"'


def _fences() -> list[str]:
    """Return every Python fence in the guide, in guide order."""

    fences = [match.group(1) for match in PYTHON_FENCE.finditer(_guide())]
    assert fences, "the migration guide carries no Python fence"
    return fences


def _guide() -> str:
    return GUIDE.read_text(encoding="utf-8")


def _quoted_runs(source: str) -> list[tuple[str, int]]:
    """Pair every print in one fence with the transcript quoted under it.

    Returns:
        One ``(expected, line)`` pair per print, in source order, where
        ``expected`` is the text before the ``--`` separator and ``line`` is
        the 1-based line of the print, so a mismatch names its own site.
    """

    lines = source.splitlines()
    runs: list[tuple[str, int]] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if not (isinstance(function, ast.Name) and function.id == "print"):
            continue
        assert node.end_lineno is not None
        following = node.end_lineno
        if following >= len(lines):
            continue
        quoted = lines[following].strip()
        if not quoted.startswith("#"):
            continue
        body = quoted[1:].strip()
        if not QUOTE_SEPARATOR.search(body):
            continue
        runs.append((QUOTE_SEPARATOR.split(body, maxsplit=1)[0].strip(), node.lineno))
    return sorted(runs, key=lambda run: run[1])


def _execute(fences: list[str]) -> list[list[str]]:
    """Execute fences in one shared namespace, one printed-lines list each."""

    namespace: dict = {"__name__": "__documentation_example__"}
    printed: list[list[str]] = []
    for index, fence in enumerate(fences):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            exec(compile(fence, f"{GUIDE.name}#fence-{index}", "exec"), namespace)
        printed.append(buffer.getvalue().splitlines())
    return printed


def _assert_matches(fences: list[str]) -> list[list[str]]:
    """Assert every quoted transcript in ``fences``, returning the real output."""

    printed = _execute(fences)
    expected = [run for fence in fences for run in _quoted_runs(fence)]
    flat = [line for fence in printed for line in fence]
    assert len(flat) == len(expected), (
        f"the guide quotes {len(expected)} values and its fences printed "
        f"{len(flat)}: {flat}"
    )
    for (value, line), actual in zip(expected, flat, strict=True):
        assert actual == value, f"{GUIDE.name}:{line} quotes {value!r}, got {actual!r}"
    return printed


def test_every_quoted_fence_runs_and_prints_what_the_guide_says():
    fences = _fences()
    migration = [fence for fence in fences if CROSS_CHECK_MARKER not in fence]
    assert (
        len(migration) == len(fences) - 1
    ), "the guide has more than one cross-check fence"
    _assert_matches(migration)


def test_the_cross_check_fence_needs_the_pymatching_extra():
    pytest.importorskip("pymatching")

    fences = _fences()
    cross_check = [
        index for index, fence in enumerate(fences) if CROSS_CHECK_MARKER in fence
    ]
    assert len(cross_check) == 1
    printed = _assert_matches(fences)[cross_check[0]]
    # The guide's load-bearing claim is that the rate does not separate the two
    # readings while the per-shot agreement does, so the last line is asserted
    # against its own numbers rather than only against the transcript.
    assert printed[-1].split() == ["1.0", "0.983"]


def _example_values() -> dict[str, str]:
    """Run the QEC golden path and read its labelled report lines."""

    completed = subprocess.run(
        [sys.executable, "-m", EXAMPLE],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=TIMEOUT_SECONDS,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    values: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        label, separator, value = line.strip().partition(":")
        if separator and line.startswith("  "):
            values[label.strip()] = value.strip()
    return values


def test_the_qec_golden_path_runs_and_reports_its_premises():
    values = _example_values()
    assert values["signatures unique"] == "False"
    assert values["after merge_duplicate"] == "103 mechanisms"
    # The script samples as many shots as its default asks for, and the detector
    # column count is the model's rather than the sampler's, so this pair pins the
    # default and the reading at once.
    assert values["sampled detection events"] == "(4000, 42)"
    assert values["registered decoders"] == "minimum_weight_matching, pymatching"
    assert values["default reading"] == "219 mechanisms, 113 hyperedges"
    assert values["graphlike reading"] == "78 mechanisms, unique True"
    assert values["matcher on default"].startswith("CapabilityError")
    # The verdict the script exists to print: the faithful reading agrees with
    # Stim's sampler to within shot noise and the graphlike one does not.
    assert float(values["default reading misses by"].split()[0]) <= 3.0
    assert float(values["graphlike reading misses by"].split()[0]) >= 20.0
    if "pymatching cross-check" in values:
        assert values["pymatching cross-check"].startswith("unavailable")
    else:
        assert values["matcher == pymatching"] == "1.0"

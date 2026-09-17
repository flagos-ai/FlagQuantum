"""Every test file must carry a seeded marker the tier commands can select.

`docs/development/TESTING.md` and `AGENTS.md` define verification as marker
selection (`python -m pytest -m "smoke or unit" -q` and friends) and warn that
"an empty selection is not verification evidence". A test file with no
selection marker is invisible to all of those commands: it runs only if someone
invokes pytest with no `-m` filter, which no tier does.

This guard keeps the seeding complete. It deliberately checks for *any*
non-builtin marker rather than a specific one, so a file may be seeded with
whichever tier it belongs to.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]

# pytest's own marks: they shape how a test runs without placing it in a tier.
BUILTIN_MARKS = frozenset(
    {
        "filterwarnings",
        "parametrize",
        "skip",
        "skipif",
        "tryfirst",
        "trylast",
        "usefixtures",
        "xfail",
    }
)

MARK_PATTERN = re.compile(r"pytest\.mark\.([A-Za-z_][A-Za-z0-9_]*)")


def _selection_marks(path: Path) -> set[str]:
    return {
        name
        for name in MARK_PATTERN.findall(path.read_text(encoding="utf-8"))
        if name not in BUILTIN_MARKS
    }


def _test_files() -> list[Path]:
    return sorted((ROOT / "tests").rglob("test_*.py"))


def test_every_test_file_carries_a_selection_marker() -> None:
    unseeded = [
        str(path.relative_to(ROOT))
        for path in _test_files()
        if not _selection_marks(path)
    ]

    assert not unseeded, (
        "these test files have no marker a tier command can select, so no lane "
        "runs them: " + ", ".join(unseeded)
    )


def test_marker_selection_finds_tests_in_every_tier_expression() -> None:
    """A tier whose marker selects nothing would still 'pass'."""
    marks: set[str] = set()
    for path in _test_files():
        marks |= _selection_marks(path)

    assert {"smoke", "unit", "integration"} <= marks
    assert {"distributed_cpu", "distributed_accel", "distributed_multinode"} <= marks
    assert {"benchmark_contract", "release_gate", "jax"} <= marks

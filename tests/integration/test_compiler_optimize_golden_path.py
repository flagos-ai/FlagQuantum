from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

#: The gate counts the example prints for its eight-gate circuit. They are pinned
#: because they are the example's own claim about the level surface: a change to a
#: pass set or to a level's composition has to be re-measured here rather than
#: silently re-printed. They are not host quantities -- both circuits in the
#: example are fixed, and their savings are cancellation, a `h rz h` fold, and one
#: angle addition.
_LEVEL_COUNTS = "gate count by level: 0 -> 8, 1 -> 3, 2 -> 3"


def _run_example() -> str:
    root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [sys.executable, "-m", "examples.compiler_optimize"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def test_compiler_optimize_user_example_runs_end_to_end() -> None:
    stdout = _run_example()

    assert "FlagQuantum compiler optimization check passed" in stdout
    assert "instructions: 8 -> 3" in stdout
    assert "optimized: ['rx', 'u3', 'cx']" in stdout
    assert "execution path: local_statevector" in stdout

    # The level surface is the reason this path exists, so the run has to show a
    # count for every implemented level rather than only the default's.
    assert _LEVEL_COUNTS in stdout

    # Level 2's extra reach is a claim about a second circuit, and the example
    # proves it there instead of asserting that a higher level is always shorter.
    assert "level 2 commutes past the cz where level 1 cannot: 3 -> 2 gates" in stdout


def test_the_example_shows_the_reserved_level_failing_closed() -> None:
    """A reserved level is refused with a reason, and the golden path prints both."""

    stdout = _run_example()

    assert stdout.count(": refused (") == 1
    assert "level 3: refused (" in stdout
    assert "level 3 is reserved, not implemented" in stdout
    # The refusal names the capability that is out of reach and where the same
    # capability is in reach, rather than only an unimplemented number: a reader
    # told "no such pass exists" would go looking for a pass to write that this
    # package already ships under a target-aware entry point.
    assert "target-independent by contract" in stdout
    assert "legalize_native_gates" in stdout

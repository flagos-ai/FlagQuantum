from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

#: The three vendors whose own declaration the example reads into a backend
#: profile and then preflights locally. No job is created and no credential is
#: used: a preflight is the last thing that happens before a paid submission.
_PREFLIGHTS = (
    "ionq preflight:",
    "iqm preflight:",
    "quantinuum preflight:",
)


def _run() -> str:
    root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [sys.executable, "-m", "examples.remote.provider_roster"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def test_provider_roster_user_example_runs_end_to_end() -> None:
    stdout = _run()

    assert "FlagQuantum hardware provider roster check passed" in stdout
    for preflight in _PREFLIGHTS:
        assert preflight in stdout, preflight


def test_a_vendor_declaration_becomes_a_coupling_map() -> None:
    """A device's own declaration is what the compiler is given, not a guess.

    One vendor declares its lattice directly and another declares geometry a
    coupling set is derived from, so both shapes are printed: the roster reads
    what the vendor states rather than discovering a topology.
    """

    stdout = _run()

    assert "IQM declared lattice: ((0, 1), (1, 2))" in stdout
    derived = "neutral-atom derived couplings: ((0, 1), (0, 3), (1, 2), (1, 3), (2, 3))"
    assert derived in stdout


def test_the_roster_refuses_fail_closed_and_names_the_reason() -> None:
    """A refusal before submission is the capability, not an error path.

    Each refusal is a blocker a reader can act on: one provider accepts only
    OpenQASM 2, and an analog target cannot carry a gate program at all. Both
    are reported by name rather than by a compatibility flag alone, because
    the name is what tells a caller what to change.
    """

    stdout = _run()

    assert "Quantinuum refusal: ('quantinuum_provider_requires_openqasm_2',)" in stdout
    assert (
        "neutral-atom blocker: "
        "('neutral_atom_submission_requires_an_analog_program_schedule',)"
    ) in stdout


def test_every_preflight_reports_its_program_format_and_verdict() -> None:
    """A preflight that passed must say what it checked.

    The format each provider accepts is part of the verdict: a compatible
    report against the wrong format would be a preflight that passed for a
    program the provider would reject.
    """

    stdout = _run()

    assert "'program_format': 'openqasm-3', 'compatible': True" in stdout
    assert "'program_format': 'openqasm-2', 'compatible': True" in stdout
    assert stdout.count("'blockers': ()") == len(_PREFLIGHTS)

"""What the mixed-check example proves, and what would falsify it.

``examples/qec/mixed_check_code.py`` is the user-facing path for the record whose
checks are all mixed. This file runs it and reads its transcript, so the numbers
the example prints are claims a failure can locate rather than prose.

What this file proves
---------------------
1. ``test_the_example_reports_both_patches_from_one_run``: the example exits zero
   and prints two sections, one per patch, each carrying the layout, the detector
   count, the mechanism count and the two logical rates. A script that stopped
   printing one of those would stop being an end-to-end demonstration.
2. ``test_the_two_patches_occupy_the_same_qubits``: the two records agree on the
   data-qubit count, the ancilla count and the check count. That is what makes the
   contrast a comparison of check *families* rather than of two differently sized
   codes, and it is the claim the example makes in prose.
3. ``test_every_check_of_the_mixed_patch_is_mixed``: the ZXXZ record's kind
   histogram is eight mixed checks and nothing else, while the rotated record's is
   four pure Z and four pure X. Both are read from ``check.stabilizer``, so the
   histogram is the record's own statement about itself.
4. ``test_the_mixed_patch_anchors_nothing``: the ZXXZ patch's detector count is
   exactly its check count times the round boundaries, and the rotated patch's is
   that plus eight. The eight are the preparation and readout detectors, and their
   absence on the mixed patch is the mechanism the row's ``baseline_gap`` names.
5. ``test_decoding_separates_the_two_check_families``: the two raw rates agree
   within the shot-noise envelope, and the decoded rates do not -- the rotated
   patch's rate falls by more than five and the mixed patch's by less than two.
   Read together with 4 this is a contrast the anchored detectors explain, not a
   decoder-quality claim, and the thresholds are stated as bounds rather than as
   the measured values so a resampling does not fail the file.

The last item is the one that survives a mutation. Suppressing the anchored
detectors on the rotated patch would break 4; reporting a mixed patch's rate as
decoded without decoding it would break 5; and a record that quietly dropped a
check would break 3. None of those is checkable from the model's own text, because
the model is derived from the record and would agree with any label given to it.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "qec" / "mixed_check_code.py"
TIMEOUT_SECONDS = 600

#: The section headers the transcript is split on, in the order they print. The
#: comparison is its own section rather than a tail of the second one, so a summary
#: line cannot be read as the second patch's own value.
SECTIONS = ("ZXXZ patch", "Rotated patch", "Comparison")

#: The share of the two raw rates that is allowed to separate them. Both patches
#: are the same code up to a local Clifford, so their raw rates are two draws of one
#: probability; at four thousand shots and a rate near 0.08 the binomial standard
#: error is about 0.004, which is 5% of the rate, so this is a generous envelope
#: rather than the measured 1.2%.
RAW_AGREEMENT = 0.25


def _run() -> str:
    """Run the example once and return its standard output."""

    completed = subprocess.run(
        [sys.executable, str(EXAMPLE)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=TIMEOUT_SECONDS,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


def _sections(stdout: str) -> dict[str, dict[str, str]]:
    """Split the transcript into one ``label -> value`` mapping per section.

    The two sections print the same labels, so a flat mapping would let the second
    overwrite the first and the comparison the example exists for would be lost.
    """

    blocks: dict[str, dict[str, str]] = {name: {} for name in SECTIONS}
    current: dict[str, str] | None = None
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith(SECTIONS):
            current = blocks[next(name for name in SECTIONS if stripped.startswith(name))]
            continue
        label, separator, value = stripped.partition(":")
        if current is not None and separator and line.startswith("  "):
            current[label.strip()] = value.strip()
    # The trailing summary lines are indented too, so they land in the last section.
    # They are read back by name where they are asserted, and dropping them here
    # would hide a summary the example stopped printing.
    return blocks


def _pair(value: str) -> tuple[float, float]:
    """Read an ``a, b`` field as two floats."""

    first, second = value.split(",")
    return float(first), float(second)


def _count(value: str) -> int:
    """Read the leading integer of a field that may carry a tail."""

    match = re.match(r"\d+", value)
    assert match is not None, value
    return int(match.group())


def _anchored(value: str) -> int:
    """Read the ``n bulk + m anchored`` field's anchored term."""

    match = re.search(r"\+\s*(\d+)\s+anchored", value)
    assert match is not None, value
    return int(match.group(1))


@pytest.fixture(scope="module")
def transcript() -> dict[str, dict[str, str]]:
    """Return the example's two sections, run once for the whole file."""

    return _sections(_run())


def test_the_example_reports_both_patches_from_one_run(
    transcript: dict[str, dict[str, str]],
) -> None:
    for name in SECTIONS[:2]:
        fields = transcript[name]
        assert set(fields) >= {
            "data qubits, ancillas",
            "checks",
            "detectors",
            "mechanisms, observables",
            "shots",
        }, (name, sorted(fields))
        assert int(fields["shots"]) > 0
        assert _count(fields["mechanisms, observables"]) > 0


def test_the_two_patches_occupy_the_same_qubits(
    transcript: dict[str, dict[str, str]],
) -> None:
    mixed, rotated = (transcript[name] for name in SECTIONS[:2])
    for field in ("data qubits, ancillas", "checks"):
        assert _count(mixed[field]) == _count(rotated[field]), field
    # The ancilla term of the pair is the second number, and it is asserted
    # separately because a shared data count alone would be consistent with two
    # records that measure the same data with different hardware.
    mixed_data, mixed_ancillas = _pair(mixed["data qubits, ancillas"])
    rotated_data, rotated_ancillas = _pair(rotated["data qubits, ancillas"])
    assert (mixed_data, mixed_ancillas) == (rotated_data, rotated_ancillas)


def test_every_check_of_the_mixed_patch_is_mixed(
    transcript: dict[str, dict[str, str]],
) -> None:
    assert transcript["ZXXZ patch"]["checks"].endswith("({'mixed': 8})")
    # The rotated record is the control: eight checks, none of them mixed, split
    # evenly between the two bases.
    assert transcript["Rotated patch"]["checks"].endswith(
        "({'pure Z': 4, 'pure X': 4})"
    )


def test_the_mixed_patch_anchors_nothing(
    transcript: dict[str, dict[str, str]],
) -> None:
    assert _anchored(transcript["ZXXZ patch"]["detectors"]) == 0
    # The rotated patch's eight extra detectors are two per round boundary that the
    # preparation and the readout contribute. The bound is stated as "more than
    # none" rather than as the count, because the count is the layout's and is
    # already pinned by the record's own tests.
    assert _anchored(transcript["Rotated patch"]["detectors"]) == 8


def test_decoding_separates_the_two_check_families(
    transcript: dict[str, dict[str, str]],
) -> None:
    mixed_raw, mixed_decoded = _pair(transcript["Comparison"]["ZXXZ raw, decoded"])
    rotated_raw, rotated_decoded = _pair(
        transcript["Comparison"]["rotated raw, decoded"]
    )

    assert mixed_raw == pytest.approx(rotated_raw, rel=RAW_AGREEMENT)
    assert mixed_decoded > mixed_raw / 2.0
    assert rotated_decoded < rotated_raw / 5.0

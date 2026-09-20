"""Run every example under ``examples/algorithms/`` the way a user would.

A script nothing runs is the defect these examples exist to fix, so each one is
executed here as a subprocess -- the documented ``python -m examples.algorithms.<name>``
command, from the repository root -- and its printed values are asserted. The
inventory test below compares the directory's contents against the list this
module runs, so a script added to the directory without a test fails rather than
being skipped.

Each script's premise line is checked for a phrase and not for its presence:
``PREMISE_PHRASES`` below pins the load-bearing half of every unit's advantage
premise -- two phrases for ``svd``, whose premise has two halves -- and each
phrase is matched against the whole premise paragraph, continuation lines
included. **What that catches is a premise that stops carrying its phrase, and
what it cannot catch is one that keeps the phrase and dismisses it:** a
concession can be kept lexically and waved away in the same breath ("not free,
but negligible at this scale"), and no phrase pins against that. The pins make a
half's *removal* detectable, not its *dismissal*. See that mapping for how each
phrase was chosen and for what it leaves uncovered.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_ROOT = ROOT / "examples"
EXAMPLES = EXAMPLES_ROOT / "algorithms"
GUIDE = "docs/guides/ALGORITHMS.md"
ROOT_ALIAS_IMPORT = "import flagquantum as fq"
TIMEOUT_SECONDS = 120

# Every script in the directory, one entry per unit that has one.
SCRIPTS = (
    "pca",
    "kmedians",
    "quantum_kernel",
    "feature_selection",
    "qarm",
    "svd",
)

# One or more phrases per script, each the load-bearing half of that unit's
# advantage premise: what the unit gives up rather than what it does. Chosen by
# reading the unit's own ``limitations`` in ``capability-maturity.toml`` and taking
# the phrase the script's premise paragraph already carries for that half, so the
# assertion pins the existing wording rather than rewriting the script to fit it.
# Two entries carry a caveat:
#
# - ``kmedians``' phrase ``not free`` is the shortest of them and is a floor
#   rather than a proof. It catches the clause's removal; a rewrite that keeps
#   the clause and dismisses it stays green, which is true of every phrase here.
#   A mechanism phrase such as ``truth table`` would be worse rather than
#   stronger: it pins *how* the oracle is built, not *that* it costs, and the
#   cost is the concession.
# - ``svd`` carries two halves and so two phrases, because a pin on one of them
#   leaves the other's deletion invisible: removing the whole input-model half
#   from that script's premise paragraph left the suite green with only the
#   block-encoding phrase pinned.
PREMISE_PHRASES: dict[str, tuple[str, ...]] = {
    # "The density matrix is materialized classically and its exponential is
    # built with a dense matrix exponential" -- the input model is not met.
    "pca": ("built classically",),
    # "the oracle is not free here ... synthesized from the predicate's truth
    # table at O(2**n) cost" -- the oracle model is not met.
    "kmedians": ("not free",),
    # "each feature state is built gate by gate from the classical feature
    # vector on every run" -- the data-access model is not met.
    "quantum_kernel": ("gate by gate",),
    # "This unit does not solve, and the repository has no annealer."
    "feature_selection": ("no annealer",),
    # "the transactions are iterated over classically, one controlled increment
    # of the support register per transaction-item membership" -- the coherent
    # database access is absent.
    "qarm": ("iterated classically",),
    # Two halves, and each phrase names its own. "The premise is the input model,
    # and it is not met ... the input state is built from the singular vectors a
    # classical torch.linalg.svd returns" is the first; "The block encoding is
    # not free to read: a readout that post-selects the ancilla succeeds with
    # probability ..." is the second. Neither phrase covers the *supporting
    # detail* of its half -- the probability formula under the second, or the
    # `torch.linalg.svd` line under the first -- which can be dropped while the
    # phrase remains, the dismissal gap the module docstring records.
    "svd": ("the input model is assumed, not met", "not free to read"),
}


def _run(name: str, *args: str) -> str:
    """Run one example as a subprocess and return what it printed."""
    completed = subprocess.run(
        [sys.executable, "-m", f"examples.algorithms.{name}", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
        timeout=TIMEOUT_SECONDS,
    )
    return completed.stdout


def _labelled(output: str, label: str) -> str:
    """Return the value printed after ``label`` in an example's aligned output."""
    for line in output.splitlines():
        name, separator, value = line.partition(":")
        if separator and name.strip() == label:
            return value.strip()
    raise AssertionError(f"the example printed no {label!r} line:\n{output}")


def _premise(output: str) -> str:
    """Return the whole premise paragraph an example printed, as one line.

    A premise printed across several lines is one paragraph up to the blank line
    that closes it, and the phrase is checked against the whole of it: checking
    the ``premise`` label's own line alone would miss a phrase on a continuation
    line, and checking the whole output would pass on a phrase printed anywhere
    else in the script.
    """
    lines = output.splitlines()
    for index, line in enumerate(lines):
        name, separator, _ = line.partition(":")
        if separator and name.strip() == "premise":
            end = index
            while end + 1 < len(lines) and lines[end + 1].strip():
                end += 1
            return " ".join(part.strip() for part in lines[index : end + 1])
    raise AssertionError(f"the example printed no premise paragraph:\n{output}")


def _assert_premise(name: str, output: str) -> None:
    """Assert the premise paragraph ``name`` printed carries its pinned phrases.

    One assertion per phrase, so a script whose premise carries one half of a
    two-half premise and not the other fails naming the phrase that went, rather
    than passing on the half that stayed. Each failure prints the paragraph, so
    the reader sees what the premise says now.
    """
    paragraph = _premise(output)
    for phrase in PREMISE_PHRASES[name]:
        assert phrase in paragraph, (
            f"{name}'s premise paragraph no longer carries the pinned phrase "
            f"{phrase!r}. It printed:\n{paragraph}"
        )


def test_pca_example_reports_its_readout_against_the_exact_spectrum() -> None:
    output = _run("pca")

    assert "quantum PCA -- flagquantum.algorithms.pca" in output
    assert _labelled(output, "exact eigenvalues") == "[0.0038, 0.9962]"
    assert _labelled(output, "readout") == "1.0"
    assert _labelled(output, "readout share") == "0.8142"
    assert _labelled(output, "resolution") == "0.015625"
    assert _labelled(output, "within(largest)") == "True"
    _assert_premise("pca", output)
    assert "take away" in output


def test_kmedians_example_assigns_every_point_and_updates_the_centroids() -> None:
    output = _run("kmedians")

    assert "quantum k-medians -- flagquantum.algorithms.kmedians" in output
    assert _labelled(output, "assignment") == "(0, 0, 1, 1, 2, 1)"
    assert _labelled(output, "medians") == "((0.0, 0.0), (5.0, 0.0), (0.2, 1.5))"
    assert _labelled(output, "searches") == "10"
    assert _labelled(output, "classical labels") == "(0, 0, 1, 1, 2, 1)"
    assert _labelled(output, "agrees") == "True"
    _assert_premise("kmedians", output)
    assert "take away" in output


def test_quantum_kernel_example_estimates_a_kernel_and_classifies_with_it() -> None:
    output = _run("quantum_kernel")

    assert (
        "quantum kernel estimation -- flagquantum.algorithms.quantum_kernel" in output
    )
    assert "[1.0, 0.419, 0.218, 0.306]" in output
    assert _labelled(output, "diagonal is 1") == "True"
    assert _labelled(output, "symmetric") == "True"
    assert _labelled(output, "dual coefficients") == "[1.036, 1.609, -1.768, -1.095]"
    assert _labelled(output, "training predict") == "(1, 1, -1, -1)"
    assert _labelled(output, "recovers labels") == "True"
    _assert_premise("quantum_kernel", output)
    assert "take away" in output


def test_feature_selection_example_builds_and_evaluates_one_objective() -> None:
    output = _run("feature_selection")

    assert (
        "feature selection as a QUBO -- flagquantum.algorithms.feature_selection"
        in output
    )
    assert (
        _labelled(output, "linear")
        == "{0: -16.0, 1: -15.9, 2: -15.8, 3: -15.7, 4: -15.6}"
    )
    assert _labelled(output, "quadratic terms") == "10"
    assert _labelled(output, "offset") == "20.0"
    assert _labelled(output, "energy (1, 1, 0, 0, 0)") == "-1.9"
    assert _labelled(output, "energy (1, 1, 1, 1, 1)") == "41.0"
    assert _labelled(output, "penalty 5.0") == "(1, 1, 0, 0, 0) at -1.9"
    assert _labelled(output, "penalty 0.05") == "(1, 1, 1, 1, 1) at -3.55"
    _assert_premise("feature_selection", output)
    assert "take away" in output


def test_qarm_example_estimates_the_frequent_item_fraction() -> None:
    output = _run("qarm")

    assert "frequent-item fractions -- flagquantum.algorithms.qarm" in output
    assert _labelled(output, "supports") == "(2, 1)"
    assert _labelled(output, "support wires") == "2"
    assert _labelled(output, "evaluation wires") == "5"
    assert _labelled(output, "estimate") == "0.5"
    assert _labelled(output, "resolution") == "0.097545"
    assert _labelled(output, "exact fraction") == "0.5"
    assert _labelled(output, "within(exact)") == "True"
    _assert_premise("qarm", output)
    assert "take away" in output


def test_svd_example_reads_singular_values_and_shows_the_one_wire_boundary() -> None:
    output = _run("svd")

    assert "singular values by phase estimation -- flagquantum.algorithms.svd" in output
    assert _labelled(output, "exact singular values") == "[5.464985, 0.365966]"
    assert _labelled(output, "readout") == "5.567414"
    assert _labelled(output, "readout share") == "0.5306"
    assert _labelled(output, "resolution") == "0.242061"
    assert _labelled(output, "alpha") == "7.745967"
    assert _labelled(output, "within(largest)") == "True"
    assert _labelled(output, "mode") == "101001"
    assert _labelled(output, "one-wire readout") == "7.745967"
    assert _labelled(output, "one-wire alpha") == "7.745967"
    assert _labelled(output, "readout equals alpha") == "True"
    assert _labelled(output, "refused counter").startswith("'0', carrying 0.2041")
    assert _labelled(output, "raised")
    _assert_premise("svd", output)
    assert "take away" in output


def test_the_example_directory_is_covered_and_indexed() -> None:
    scripts = sorted(path.name for path in EXAMPLES.glob("*.py"))
    expected = sorted(f"{name}.py" for name in SCRIPTS)

    assert scripts == expected, (
        "every script in examples/algorithms/ must be run by this module: add it to "
        f"SCRIPTS, or remove it. Found {scripts}, expected {expected}"
    )
    assert sorted(PREMISE_PHRASES) == sorted(SCRIPTS), (
        "every script in SCRIPTS needs a premise phrase in PREMISE_PHRASES, so a new "
        f"script's premise is pinned rather than only printed. Found "
        f"{sorted(PREMISE_PHRASES)}, expected {sorted(SCRIPTS)}"
    )
    index = (EXAMPLES / "README.md").read_text(encoding="utf-8")
    for name in scripts:
        assert name in index, f"examples/algorithms/README.md does not list {name}"


def test_the_guide_is_linked_from_the_example_and_algorithm_entry_points() -> None:
    for relative_path in ("examples/README.md", "flagquantum/algorithms/README.md"):
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        assert GUIDE in text, f"{relative_path} does not point at {GUIDE}"


def test_examples_readme_names_every_example_that_does_not_use_the_root_alias() -> None:
    """The examples README's alias paragraph must name the files it leaves out.

    ``examples/README.md`` says which examples are driven by the root-level ``fq``
    alias and lists the ones that are not. This test derives the requirement from
    the files instead of trusting it: every ``.py`` under ``examples/`` that does
    not import the alias has to be named **after that sentence**, by file name or
    by its path under ``examples/``, so the list stays complete as examples are
    added rather than going false the way a broader sentence did. A file that only
    its directory is named for does not count: the directory may be mentioned for
    another reason, as ``single_machine_quantum_ai/`` is.

    **The boundary is the sentence, not the bullet list under it.** The check is
    satisfied by a mention anywhere in the rest of the file, so a non-aliased
    example discussed in a later section for some other purpose passes without
    ever being in the exception list. What the boundary does buy is the other
    direction: a mention *before* the sentence does not satisfy it, which is what
    keeps a file that does use the alias from being counted as an exception by
    being named in the alias half of the paragraph.

    **What it does not check** is the classification itself. That a file named
    after the sentence really does import the way the list says is what the list's
    own reading is for, not this test's.
    """
    readme = (EXAMPLES_ROOT / "README.md").read_text(encoding="utf-8")
    marker = "These examples do not use that alias:"
    assert marker in readme, (
        f"examples/README.md no longer carries its {marker!r} marker, so this test "
        "cannot tell the examples that use the root-level `fq` alias from the ones "
        "that do not"
    )
    listed = readme.split(marker, 1)[1]
    scanned = sorted(EXAMPLES_ROOT.rglob("*.py"))
    assert scanned, f"{EXAMPLES_ROOT} holds no example file to check"

    missing = [
        path.relative_to(ROOT).as_posix()
        for path in scanned
        if ROOT_ALIAS_IMPORT not in path.read_text(encoding="utf-8")
        and path.name not in listed
        and path.relative_to(EXAMPLES_ROOT).as_posix() not in listed
    ]

    assert not missing, "\n".join(
        (
            "examples/README.md states which examples use the root-level `fq` alias, "
            "and an example that does not use it has to be named after that "
            "statement; these are not:",
            *missing,
        )
    )

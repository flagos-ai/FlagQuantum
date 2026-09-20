"""Run every example under ``examples/algorithms/`` the way a user would.

A script nothing runs is the defect these examples exist to fix, so each one is
executed here as a subprocess -- the documented ``python -m examples.algorithms.<name>``
command, from the repository root -- and its printed values are asserted. The
inventory test below compares the directory's contents against the list this
module runs, so a script added to the directory without a test fails rather than
being skipped.
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


def test_pca_example_reports_its_readout_against_the_exact_spectrum() -> None:
    output = _run("pca")

    assert "quantum PCA -- flagquantum.algorithms.pca" in output
    assert _labelled(output, "exact eigenvalues") == "[0.0038, 0.9962]"
    assert _labelled(output, "readout") == "1.0"
    assert _labelled(output, "readout share") == "0.8142"
    assert _labelled(output, "resolution") == "0.015625"
    assert _labelled(output, "within(largest)") == "True"
    assert _labelled(output, "premise")
    assert "take away" in output


def test_kmedians_example_assigns_every_point_and_updates_the_centroids() -> None:
    output = _run("kmedians")

    assert "quantum k-medians -- flagquantum.algorithms.kmedians" in output
    assert _labelled(output, "assignment") == "(0, 0, 1, 1, 2, 1)"
    assert _labelled(output, "medians") == "((0.0, 0.0), (5.0, 0.0), (0.2, 1.5))"
    assert _labelled(output, "searches") == "10"
    assert _labelled(output, "classical labels") == "(0, 0, 1, 1, 2, 1)"
    assert _labelled(output, "agrees") == "True"
    assert _labelled(output, "premise")
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
    assert _labelled(output, "premise")
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
    assert _labelled(output, "premise")
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
    assert _labelled(output, "premise")
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
    assert _labelled(output, "premise")
    assert "take away" in output


def test_the_example_directory_is_covered_and_indexed() -> None:
    scripts = sorted(path.name for path in EXAMPLES.glob("*.py"))
    expected = sorted(f"{name}.py" for name in SCRIPTS)

    assert scripts == expected, (
        "every script in examples/algorithms/ must be run by this module: add it to "
        f"SCRIPTS, or remove it. Found {scripts}, expected {expected}"
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
    alias and lists the ones that are not. This test derives that list from the
    files instead of trusting it: every ``.py`` under ``examples/`` that does not
    import the alias has to be named in the README, by file name or by its path
    under ``examples/``, so the sentence stays true of the files it covers as
    examples are added, rather than going false the way a broader one did. A file
    that only its directory is named for does not count: the directory may be
    mentioned for another reason, as ``single_machine_quantum_ai/`` is.
    """
    readme = (EXAMPLES_ROOT / "README.md").read_text(encoding="utf-8")
    scanned = sorted(EXAMPLES_ROOT.rglob("*.py"))
    assert scanned, f"{EXAMPLES_ROOT} holds no example file to check"

    missing = [
        path.relative_to(ROOT).as_posix()
        for path in scanned
        if ROOT_ALIAS_IMPORT not in path.read_text(encoding="utf-8")
        and path.name not in readme
        and path.relative_to(EXAMPLES_ROOT).as_posix() not in readme
    ]

    assert not missing, (
        "examples/README.md states which examples use the root-level `fq` alias and "
        "lists the ones that do not, but these are in neither list: "
        "\n".join(missing)
    )

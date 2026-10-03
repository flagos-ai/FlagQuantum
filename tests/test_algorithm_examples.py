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
    "error_mitigation",
    "pec",
    "spsa_optimizer",
    "trotter",
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
    # Two halves again. "the curve is a polynomial in the scale factor of degree
    # at most the fit order, which is not checkable from the measurements alone"
    # is the assumption the method rests on, and "the estimate carries no measured
    # uncertainty because it is Tr(O rho)" is what the unit gives up to reach it.
    # Neither phrase covers its half's supporting detail -- the residual as the
    # diagnostic, or the readout refusal that keeps the curve a state-preparation
    # estimate -- both of which can be dropped while the phrase remains.
    "error_mitigation": (
        "not checkable from the measurements",
        "no measured uncertainty",
    ),
    # Two halves. "the noise the program experiences is exactly the channel the
    # model declares, at the location the model declares it" is the assumption the
    # inversion rests on, and "this path consumes no shots" is what it gives up to
    # be exact. Pinning only the first would leave the cost's origin deletable,
    # which is the half that keeps a gamma read as a measurement of this path
    # rather than as the price a sampled implementation would pay.
    "pec": (
        "exactly the channel the model declares",
        "consumes no shots",
    ),
    # Two halves. "the estimate is an estimate rather than a gradient" is the
    # estimator's bias, and "an objective with an exact gradient is served
    # cheaper and exact without it" is what that bias costs. Pinning only the
    # first would leave the recommendation deletable, which is the half that
    # keeps this unit from being read as a preferred optimizer.
    "spsa_optimizer": (
        "an estimate rather than a gradient",
        "cheaper and exact without it",
    ),
    # Two halves. "the product approximates exp(-i t H) and nothing here bounds how
    # far apart they are" is the approximation's own concession, and "a commutator
    # norm belongs to the caller" is who owns the bound that would close it.
    # Pinning only the first would leave the ownership deletable, and a defect read
    # as a bound on the unit's accuracy rather than as a number this script
    # measured is exactly the misreading the second half prevents.
    "trotter": (
        "nothing here bounds",
        "belongs to the caller",
    ),
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
    assert _labelled(output, "exact singular values") == "[5.464986, 0.365966]"
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


def test_error_mitigation_example_shows_the_fits_and_both_refusals() -> None:
    output = _run("error_mitigation")

    assert (
        "zero-noise extrapolation -- flagquantum.algorithms.error_mitigation" in output
    )
    # The reference is measured at the runs' own dtype, so the distances below are
    # distances at one precision rather than a complex64 floor reported as a
    # complex128 extrapolation error.
    assert _labelled(output, "noiseless value") == "0.99999999999999978"
    assert _labelled(output, "scale factors") == "(1.0, 3.0, 5.0, 7.0)"
    # The four ordinates, read off the exact noise-scaled curve.
    assert _labelled(output, "scale 1") == "0.871111109257"
    assert _labelled(output, "scale 3") == "0.639999987284"
    assert _labelled(output, "scale 5") == "0.444444444444"
    assert _labelled(output, "scale 7") == "0.284444452922"
    # Three fits of the same curve: an underfit with a residual three orders
    # above the degree-two floor, the degree-two fit, and a square Richardson fit
    # whose residual is absent because the fit interpolates.
    assert "max residual 1.7778e-02" in output
    assert "max residual 4.1723e-09" in output
    assert "max residual none: the fit interpolates" in output
    assert _labelled(output, "unmitigated value") == "0.871111109257"
    # Distance from the same-dtype noiseless read: 0.1288888907432557 unmitigated
    # against 3.0299036613001817e-09 extrapolated.
    assert _labelled(output, "degree 2 improved by") == "4.254e+07x"
    assert _labelled(output, "degree 1 improved by") == "2.636e+00x"
    # The declared scaling refuses a channel whose parameter is an angle, and the
    # caller's own scaling keeps a residual two orders above the floor.
    assert _labelled(output, "declared scaling refuses").startswith(
        "cannot scale the 'coherent_overrotation' channel"
    )
    assert _labelled(output, "caller scaling, degree 1").startswith(
        "estimate 1.271993498172"
    )
    assert "max residual 8.9330e-02" in output
    assert _labelled(output, "caller scaling, degree 2").startswith(
        "estimate 1.105716958201"
    )
    assert "max residual 2.8865e-02" in output
    # And the readout boundary: Tr(O rho) is read before measurement.
    assert _labelled(output, "readout rule refused").startswith(
        "the noise model declares a readout rule"
    )
    _assert_premise("error_mitigation", output)
    assert "take away" in output


def test_pec_example_inverts_a_channel_and_shows_every_refusal() -> None:
    output = _run("pec")

    assert "probabilistic error cancellation -- flagquantum.algorithms.pec" in output
    # The channels are built at the estimate's own dtype, so the mitigated error
    # below is the density simulation's floor rather than a complex64 gap reported
    # as a complex128 result.
    assert _labelled(output, "noiseless value") == "0.99999999999999978"
    assert _labelled(output, "locations") == "['bit_flip(0,)', 'bit_flip(1,)']"
    assert _labelled(output, "estimate dtype") == "torch.complex128"
    # Two locations, each inverted, composed: the unmitigated read is 0.36 short
    # and the mitigated one is on the noiseless value to the arithmetic's floor.
    assert _labelled(output, "unmitigated") == "0.639999995231628"
    assert _labelled(output, "mitigated") == "1.000000000000000"
    assert _labelled(output, "unmitigated error") == "3.600e-01"
    assert _labelled(output, "mitigated error") == "1.110e-16"
    assert _labelled(output, "gamma") == "1.562500011642"
    assert _labelled(output, "sampling overhead").startswith("2.441406286380")
    assert _labelled(output, "exact programs") == "17 = 16 terms + 1"
    assert _labelled(output, "shots consumed") == "0"
    # The cost composes: one location's gamma squared is the two-location gamma.
    assert _labelled(output, "one location") == "gamma 1.250000004657, 4 terms"
    assert _labelled(output, "two locations") == "gamma 1.562500011642, 16 terms"
    assert _labelled(output, "product of the parts") == "1.562500011642"
    # Every admitted family is diagonal at this dtype, and the five are the ones
    # the unit's own tolerance admits.
    assert _labelled(output, "bit_flip").startswith("gamma 1.250000005")
    assert _labelled(output, "phase_flip").startswith("gamma 1.250000005")
    assert _labelled(output, "depolarizing").startswith("gamma 1.230769235")
    assert _labelled(output, "phase_damping").startswith("gamma 1.054092554")
    assert _labelled(output, "two_qubit_depolarizing").startswith(
        "gamma 1.223880601, words 16"
    )
    # The four channel families whose transfer matrix is not diagonal, each with
    # the magnitude the refusal measured.
    assert _labelled(output, "amplitude_damping").startswith(
        "refused -- channel 'amplitude_damping' is not a Pauli channel"
    )
    assert "1.000e-01" in output
    assert _labelled(output, "coherent_overrotation").startswith(
        "refused -- channel 'coherent_overrotation' is not a Pauli channel"
    )
    assert "1.987e-01" in output
    assert _labelled(output, "reset_error").startswith(
        "refused -- channel 'reset_error' is not a Pauli channel"
    )
    assert "5.000e-02" in output
    assert _labelled(output, "thermal_relaxation").startswith(
        "refused -- channel 'thermal_relaxation' is not a Pauli channel"
    )
    assert "1.000e+00" in output
    # A channel whose inverse does not exist at all, as distinct from one that is
    # merely outside the family.
    assert _labelled(output, "bit_flip at 0.5").startswith(
        "refused -- channel 'bit_flip' has a vanishing Pauli transfer eigenvalue at Y"
    )
    assert _labelled(output, "readout rule refused").startswith(
        "the noise model declares a readout rule"
    )
    _assert_premise("pec", output)
    assert "take away" in output


def test_spsa_example_measures_its_cost_and_converges() -> None:
    output = _run("spsa_optimizer")

    assert "SPSA optimization -- flagquantum.algorithms.spsa" in output
    assert _labelled(output, "parameter shift") == "4 circuit evaluations"
    assert _labelled(output, "SPSA") == "2 circuit evaluations"
    assert _labelled(output, "exact gradient") == "[-0.374048, -0.63987]"
    assert _labelled(output, "initial energy") == "-1.769414"
    assert _labelled(output, "final energy") == "-1.999686"
    assert _labelled(output, "final parameters") == "[0.004207, 0.024341]"
    assert _labelled(output, "objective calls") == "240"
    assert _labelled(output, "sampled calls") == "401"
    assert _labelled(output, "in-place write").startswith(
        "the objective modified the tensor it was given"
    )
    _assert_premise("spsa_optimizer", output)
    assert "take away" in output


def test_trotter_example_measures_the_defect_and_shows_every_refusal() -> None:
    output = _run("trotter")

    assert "Trotter product formula -- flagquantum.algorithms.trotter" in output
    # The terms are printed in the order they were declared, and the count is
    # pinned because a silently sorted or deduplicated term list would change the
    # formula while leaving every downstream number plausible.
    assert _labelled(output, "term count") == "5 in declared order"
    assert _labelled(output, "wires") == "3"
    # The primitive's residual is measured against the dense exponential of the
    # same word after dividing out one global phase, so every entry is at the
    # dtype's own floor. The gate lists are the decomposition itself: no basis
    # change for a diagonal word, a Hadamard for x, sdg-h forward and h-s backward
    # for y, and one CX per extra supported wire either side of the rotation.
    assert _labelled(output, "Z on (0,)") == "residual 1.110e-16, gates 1 ['rz']"
    assert (
        _labelled(output, "X on (0,)") == "residual 0.000e+00, gates 3 ['h', 'rz', 'h']"
    )
    assert _labelled(output, "Y on (0,)") == (
        "residual 1.475e-17, gates 5 ['sdg', 'h', 'rz', 'h', 's']"
    )
    assert _labelled(output, "ZZ on (0, 1)") == (
        "residual 1.110e-16, gates 3 ['cx', 'rz', 'cx']"
    )
    assert _labelled(output, "XZY on (2, 0, 1)").startswith(
        "residual 1.120e-16, gates 11 ['h', 'sdg', 'h', 'cx', 'cx', 'rz', 'cx', 'cx'"
    )
    # The same word with a target in the middle that carries I: the identity
    # consumes its wire and contributes no support, so this is the ZZ ladder.
    assert _labelled(output, "ZIZ on (0, 1, 2)") == (
        "residual 1.110e-16, gates 3 ['cx', 'rz', 'cx']"
    )
    # The defect against torch.matrix_exp, at three step counts per order.
    assert _labelled(output, "order 1, 2 steps") == "defect 5.085577e-02"
    assert _labelled(output, "order 1, 4 steps") == "defect 2.533908e-02"
    assert _labelled(output, "order 1, 8 steps") == "defect 1.265847e-02"
    assert _labelled(output, "order 2, 2 steps") == "defect 3.442249e-03"
    assert _labelled(output, "order 2, 4 steps") == "defect 8.546989e-04"
    assert _labelled(output, "order 2, 8 steps") == "defect 2.133117e-04"
    # And the rate, which is the composition's rather than a bound: halving the
    # step halves the first order's defect and quarters the second's.
    assert _labelled(output, "order 1 rate").startswith("2.007 per doubling")
    assert "(the defect falls off as step**1)" in output
    assert _labelled(output, "order 2 rate").startswith("4.027 per doubling")
    assert "(the defect falls off as step**2)" in output
    # Commuting terms are reached exactly, which is what makes the defect above a
    # statement about non-commutation rather than about the method being inexact.
    assert _labelled(output, "two commuting z terms") == "defect 2.220e-16"
    # The declared order is not sorted: the same two terms in the other order emit
    # a different gate sequence and reach a different state.
    assert _labelled(output, "z then x gates") == "['rz', 'h', 'rz', 'h']"
    assert _labelled(output, "x then z gates") == "['h', 'rz', 'h', 'rz']"
    assert _labelled(output, "same circuit") == "False"
    assert _labelled(output, "state distance") == "0.303293"
    # The result is an ordinary circuit: the static estimator reads gates out of
    # it, which is the whole reason exp_pauli is not an opcode here.
    assert _labelled(output, "basis") == "static_instruction_sequence"
    assert _labelled(output, "operations") == "4 over 1 of 1 wires"
    assert _labelled(output, "counts") == "{'h': 2, 'rz': 2}"
    assert _labelled(output, "t count") == "0"
    # The coefficient reaches the rotation as a tensor, so it differentiates.
    assert _labelled(output, "readout <Z>") == "0.715813373694"
    assert _labelled(output, "d<Z>/d coefficient") == "-0.161660517061"
    # Five refusals, each by name and each before a circuit exists.
    assert _labelled(output, "a term that is a multiple of the identity").startswith(
        "refused -- CapabilityError: term 0 is a multiple of the identity"
    )
    assert _labelled(output, "a coefficient with an imaginary part").startswith(
        "refused -- CapabilityError: term 0 has coefficient (1+2j)"
    )
    assert _labelled(output, "a term past the declared register").startswith(
        "refused -- ValueError: Pauli word 'Z' addresses wire(s) [2] outside a "
        "2-wire circuit"
    )
    assert _labelled(output, "an order the module does not build").startswith(
        "refused -- ValueError: order must be one of 1, 2, got 3"
    )
    assert _labelled(output, "a step count that is not a positive integer").startswith(
        "refused -- ValueError: steps must be a positive integer, got 0"
    )
    assert "NOT REFUSED" not in output
    _assert_premise("trotter", output)
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

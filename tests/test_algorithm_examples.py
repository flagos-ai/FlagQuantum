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
    "cdr",
    "spsa_optimizer",
    "nelder_mead_optimizer",
    "trotter",
    "block_encoding",
    "linear_combination",
    "logical_resources",
    "arithmetic",
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
    # Two halves. "the ideal expectation is affine in the noisy one over the
    # region the training circuits span" is the relation the fit rests on, and
    # "Clifford membership is enforced here and never exploited" is what the unit
    # gives up to reach it: the training circuits are Clifford, and this path
    # still pays to simulate them. Pinning only the first would leave the
    # second deletable, and a Clifford training set read as a cheaper one is
    # exactly the misreading the second half prevents.
    "cdr": (
        "affine in the noisy one",
        "never exploited",
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
    # Two halves, and they are halves of one premise rather than two
    # concessions: "the objective has to be deterministic" is the input the
    # method needs, and "a local minimum rather than the global one" is what
    # a converged run therefore reports. Pinning only the first would leave
    # the flag's meaning deletable, and a run whose flag reads as a global
    # claim is exactly the misreading the second half prevents.
    "nelder_mead_optimizer": (
        "has to be deterministic",
        "a local minimum rather than the global one",
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
    # Two halves. "the eigendecomposition is dense and exact, so the gate is one
    # dense matrix on n + 1 qubits and the cost is diagonalising H" is the input
    # model's own cost, and "nothing here bounds it" with the error left to the
    # caller is who owns the accuracy. Pinning only the first would leave the
    # ownership deletable, and a dense exact construction read as scale-free is
    # the misreading the second half prevents.
    "block_encoding": (
        "nothing here bounds",
        "belongs to the caller",
    ),
    # Two halves. "the preparation is a dense classical precomputation over 2 ** k
    # amplitudes, exponential in the index register's width" is the input model's
    # own cost -- the term count decides a register width, not a gate count -- and
    # "belongs to the caller's 2 ** k amplitudes rather than to any bound this unit
    # states" is who owns the accuracy and the cost the module declines to bound.
    # Pinning only the first would leave the ownership deletable, and an
    # exponential-in-the-index-register construction read as a cost claim is the
    # misreading the second half prevents.
    "linear_combination": (
        "exponential in the index",
        "belongs to the caller's",
    ),
    # Two halves. "the report counts rather than measures: nothing runs, no
    # wall-clock time, memory, or allocation is read" is the input model the
    # unit is honest about -- a report over a program text rather than an
    # execution -- and "that number belongs to the device rather than to this
    # unit" is who owns the failure rate it therefore cannot state. Pinning only
    # the first would leave the ownership deletable, and a footprint read as a
    # reliability claim is exactly the misreading the second half prevents.
    "logical_resources": (
        "counts rather than measures",
        "belongs to the device",
    ),
    # Two halves. "belongs to the compiler's own Toffoli rule rather than to this
    # construction ... no seven-T expansion is written here, because a second
    # Toffoli cost would be a second source of truth for the one number the two
    # must agree on" is who owns the cost the module declines to compute, and
    # "Only the inputs the ancilla contract names are promised a sum" is the
    # domain restriction that keeps an off-contract input from being read as an
    # addition. Pinning only the first would leave the restriction deletable, and a
    # reversible map read as an adder on every input is exactly the misreading the
    # second half prevents.
    "arithmetic": (
        "compiler's own Toffoli rule",
        "Only the inputs the ancilla contract names",
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


def test_cdr_example_fits_the_noise_and_reads_the_residual() -> None:
    output = _run("cdr")

    assert "clifford data regression -- flagquantum.algorithms.cdr" in output
    # Every labelled line, in order, because six labels repeat across the four
    # training runs and the last-wins helper below would read the wrong one.
    rows: dict[str, list[str]] = {}
    for line in output.splitlines():
        name, separator, value = line.partition(":")
        if separator and name.strip() and not line.startswith(" " * 4):
            rows.setdefault(name.strip(), []).append(value.strip())
    # The training set is a rewrite of the target. Both rotations of the
    # three-point target sit under the first quarter turn, so the nearest variant
    # drops them -- residue zero is the empty word -- and the ladder puts each
    # back as its own named word.
    assert rows["snappable rotations"] == ["['phase', 'rx', 'ry', 'rz', 'u1']"]
    assert rows["variants"] == [
        "['nearest', 'ry-at-instruction-1', 'ry-at-instruction-2']"
    ]
    assert rows["angle shifts"] == ["['0.700000', '0.870796', '1.170796']"]
    assert rows["nearest variant"] == ["['h', 'cx']"]
    # The two-point fit: the residual is absent rather than zero, and the slope is
    # the reciprocal of the shrinkage the model applied to the second point. The
    # first point's ideal is 0.0 and its noisy read is the arithmetic's own floor
    # rather than exactly zero, which is why the intercept is not exactly zero and
    # rounds to it at three digits.
    assert rows["observable"] == ["1.0 * z(0)", "1.0 * z(0) + 1.0 * x(1)"]
    assert rows["training points"] == [
        "[(0.0, -5.551115123125783e-17), (-0.9999999999999998, -0.9360000014305114)]",
        "[(0.0, 0.0), (-0.9999999999999998, -0.8985600022101402), "
        "(0.9999999999999998, 0.9360000014305113)]",
    ]
    assert rows["slope"] == ["1.068376066743", "1.090028329682"]
    assert rows["intercept"] == ["0.000e+00", "-1.360e-02"]
    assert rows["residual"] == ["None", "1.360355e-02", "1.110e-16"]
    assert rows["degrees of freedom"] == ["0", "1"]
    assert rows["unmitigated"] == ["-0.528505355905480", "-0.214372677510147"]
    assert rows["mitigated"] == ["-0.564642473395035", "-0.247275844867047"]
    assert rows["unmitigated error"] == ["3.614e-02", "4.043e-02"]
    assert rows["mitigated error"] == ["4.441e-16", "7.524e-03", "1.665e-16"]
    assert rows["distinct noisy values"] == ["3"]
    assert rows["exact value"] == ["-0.564642473395035", "-0.254799344929041"]
    # The residual separates a premise that held from one that did not, on the
    # same three training circuits: the two-noise-source run leaves one that is
    # 1.2e14 times the one-noise-source run's, and only the latter's mitigated
    # value is on the exact value.
    assert abs(float(rows["residual"][1]) / float(rows["residual"][2])) > 1e13
    # And the two-point run's closeness is an identity rather than a check: its
    # mitigated error is at the arithmetic's own floor, 1.7e13 times smaller than
    # the three-point run's, which is the run whose premise was actually tested.
    assert float(rows["mitigated error"][0]) < float(rows["mitigated error"][1]) / 1e13
    # Two refusals, each by name: the readout rule, and a program with no
    # snappable rotation at all -- refused by the stabilizer engine because the
    # rewrite cannot reach the operation, not by a second gate list here.
    assert rows["readout rule refused"][0].startswith(
        "the noise model declares a readout rule"
    )
    assert rows["ccx refused"][0].startswith(
        "CapabilityError -- instruction 1 'ccx' is not a Clifford gate"
    )
    _assert_premise("cdr", output)
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


def test_nelder_mead_example_is_exact_and_shows_what_converged_means() -> None:
    output = _run("nelder_mead_optimizer")

    assert "Nelder-Mead simplex search -- flagquantum.algorithms.nelder_mead" in output
    assert _labelled(output, "iterations") == "67"
    assert _labelled(output, "objective calls") == "132"
    assert _labelled(output, "final energy") == "-2.0"
    assert _labelled(output, "converged") == "True"
    assert _labelled(output, "Nelder-Mead") == "-2.0"
    assert _labelled(output, "SPSA, seed 13") == "-1.996896 in 132 calls"
    assert _labelled(output, "SPSA, seed 5") == "-1.993104 in 132 calls"
    assert _labelled(output, "SPSA, seed 11") == "-1.99134 in 132 calls"
    assert _labelled(output, "spread across seeds") == "0.005556"
    assert _labelled(output, "reported value") == "0.099366985524"
    assert _labelled(output, "reported converged") == "True"
    assert _labelled(output, "global minimum") == "-0.100617376638"
    assert _labelled(output, "distance from the global") == "0.199984362162"
    assert _labelled(output, "collinear simplex") == (
        "the vertices must span all 2 parameter directions"
    )
    assert _labelled(output, "in-place objective").startswith(
        "the objective modified the tensor it was given"
    )
    assert _labelled(output, "budget") == "maxiter must be a positive integer"
    assert _labelled(output, "non-finite value") == (
        "the objective returned a non-finite value"
    )
    _assert_premise("nelder_mead_optimizer", output)
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


def test_block_encoding_example_reads_its_block_back_out_of_the_circuit() -> None:
    output = _run("block_encoding")

    assert (
        "Block encoding -- flagquantum.algorithms.primitives.block_encoding" in output
    )
    assert _labelled(output, "num_system") == "2"
    assert _labelled(output, "num_ancilla") == "1"
    # The default factor is the Frobenius norm, and the two are printed side by
    # side so a factor that quietly became the spectral norm would show.
    assert _labelled(output, "Frobenius norm") == _labelled(output, "alpha")
    assert _labelled(output, "alpha clears the spectral norm") == "True"
    # The block is read out of the circuit and checked against both `H / alpha`
    # and `H`, because a construction that returned its argument agrees with the
    # second and not the first. Both residuals are at the statevector's own floor.
    assert float(_labelled(output, "circuit block vs H / alpha")) < 1e-6
    assert float(_labelled(output, "dense block vs circuit block")) < 1e-6
    assert float(_labelled(output, "circuit block vs H")) > 1.0
    # The reflection is a reflection identically, not to within a truncated series.
    assert float(_labelled(output, "U is Hermitian")) < 1e-12
    assert float(_labelled(output, "U squared is the identity")) < 1e-12
    assert float(_labelled(output, "U is unitary")) < 1e-12
    # The walk identity: the observed cosines come from `eigvals` of the circuit's
    # unitary and the expected ones from `eigvalsh` of the matrix, so the two sides
    # are different routines and the agreement is the claim.
    assert _labelled(output, "expected cosines") == _labelled(
        output, "observed cosines"
    )
    assert float(_labelled(output, "agreement")) < 1e-6
    # And the phases are non-trivial, which is what makes the identity more than a
    # statement about a Hermitian matrix's own phases.
    assert _labelled(output, "cosines off the endpoints") == "True"
    # Without the phase flip the same unitary has cosines on the endpoints alone,
    # and its spectrum is nowhere near the encoded one.
    assert (
        _labelled(output, "plain U cosines")
        == "[-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0]"
    )
    assert float(_labelled(output, "plain vs encoded spectrum")) > 1.0
    # The step's definition, not only its spectrum: `W` is compared against the flag
    # flip composed with the encoding, which is a matrix identity the cosines above
    # cannot express because a step and its own encoding have the same kind of spectrum.
    assert float(_labelled(output, "walk step vs flag flip of the encoding")) < 1e-6
    # `W^dagger W` against the identity rather than the adjoint's cosines, which are
    # `W`'s: this construction's step is Hermitian, so only the composition tells the two
    # apart. The residual is the register's own floor, which is 5e-8 rather than 1e-15
    # because the unitary is built by applying the gate to each basis state.
    assert float(_labelled(output, "adjoint step inverts the step")) < 1e-6
    # Two implementations, one consumer: the second one holds no matrix at all and
    # the same protocol reads it, so the surface is a boundary rather than a class.
    assert _labelled(output, "spectral is a BlockEncoding") == "True"
    assert _labelled(output, "spectral is a WalkEncoding") == "True"
    assert _labelled(output, "built is a WalkEncoding") == "True"
    assert _labelled(output, "built source") == "two named gates, no matrix anywhere"
    assert _labelled(output, "built walk cosines") == "[-1.0, -1.0, 1.0, 1.0]"
    # And the second implementation is held to the same composition as the first, which
    # is the part of the protocol a name alone cannot pin: it holds no matrix, so the
    # composed unitary is the only witness that its walk step is the flag flip of its
    # encoding rather than some other unitary with the same spectrum.
    assert float(_labelled(output, "built walk vs flag flip of the encoding")) < 1e-6
    # Ten construction refusals, each by name.
    assert _labelled(output, "a matrix that is not a tensor").startswith(
        "refused -- ValueError: matrix must be a torch.Tensor"
    )
    assert _labelled(output, "a matrix that is not square").startswith(
        "refused -- ValueError: matrix must be square, got 2 row(s) and 3 column(s)"
    )
    assert _labelled(output, "a dimension that is not a power of two").startswith(
        "refused -- ValueError: matrix must have a power-of-two dimension"
    )
    assert _labelled(output, "an integer matrix").startswith(
        "refused -- ValueError: matrix must be a floating-point or complex matrix"
    )
    assert _labelled(output, "the zero matrix").startswith(
        "refused -- ValueError: the zero matrix has no block to normalise"
    )
    assert _labelled(output, "a matrix with a non-finite entry").startswith(
        "refused -- ValueError: matrix must hold only finite values"
    )
    assert _labelled(output, "a matrix that is not Hermitian").startswith(
        "refused -- ValueError: the block encoding is built from the matrix's own "
        "eigenbasis, and eigh reads one triangle"
    )
    assert _labelled(output, "a factor below the spectral norm").startswith(
        "refused -- ValueError: the subnormalisation must be at least the matrix's "
        "spectral norm"
    )
    assert _labelled(output, "a factor that is zero").startswith(
        "refused -- ValueError: the subnormalisation must be positive and finite, got 0.0"
    )
    assert _labelled(output, "a factor that is not a real number").startswith(
        "refused -- ValueError: the subnormalisation must be a real number, got True"
    )
    # Three register refusals, each taken before the first gate, so the circuit is
    # empty rather than half-built: the gate is one dense matrix and there is no
    # partial form of it.
    assert _labelled(output, "the flag inside the register").startswith(
        "refused -- ValueError: ancilla 1 is one of the operator's own wires"
    )
    assert _labelled(output, "one qubit for a two-qubit operator").startswith(
        "refused -- ValueError: the register must be 2 wire(s) for this encoding, got 1"
    )
    assert _labelled(output, "a repeated qubit").startswith(
        "refused -- ValueError: the register must be distinct, got [1, 1]"
    )
    for label in (
        "the flag inside the register",
        "one qubit for a two-qubit operator",
        "a repeated qubit",
    ):
        assert _labelled(output, f"gates after {label}") == "0"
    assert "NOT REFUSED" not in output
    _assert_premise("block_encoding", output)
    assert "take away" in output


def test_linear_combination_example_encodes_the_pauli_sum_from_its_coefficients() -> (
    None
):
    output = _run("linear_combination")

    # The factor is read off the coefficients and is checked against the two norms it is
    # deliberately not. A construction that silently defaulted to the Frobenius norm would
    # agree with the block and disagree with the coefficients, so both are pinned -- and
    # the Frobenius norm is pinned as *different* rather than as larger or smaller, because
    # a one-wire Pauli product has Frobenius norm `2 ** (n / 2)` and the two are not ordered
    # against each other at all.
    assert _labelled(output, "alpha is the sum of the magnitudes") == "True"
    assert _labelled(output, "alpha clears the spectral norm") == "True"
    assert _labelled(output, "alpha is the Frobenius norm") == "False"
    assert (
        abs(
            float(_labelled(output, "Frobenius norm"))
            - float(_labelled(output, "sum of the coefficient magnitudes"))
        )
        > 0.1
    )
    # One index wire per factor of two in the term count, plus the ladder the select
    # needs: six terms is three index wires and one ladder ancilla, not six of either.
    assert _labelled(output, "index_width") == "3"
    assert _labelled(output, "ladder ancillas") == "1"
    # The block is the *signed* sum over alpha, and the two residuals beside it are what
    # separate the three candidate readings of it: the sum itself, and the same sum with
    # the signs dropped, which is what a preparation carrying no relative phase gives.
    assert float(_labelled(output, "circuit block vs H / alpha")) < 1e-6
    assert float(_labelled(output, "circuit block vs H")) > 1.0
    assert float(_labelled(output, "circuit block vs the sum of magnitudes")) > 0.1
    assert float(_labelled(output, "the signed and unsigned sums differ")) > 0.1
    # The encoding is a unitary and is not its own adjoint. The deviation is pinned as a
    # lower bound rather than an exact value: it is the property that makes the adjoint's
    # term reversal necessary, and a construction that was accidentally Hermitian would
    # make the reversal untestable rather than wrong.
    assert float(_labelled(output, "U is unitary")) < 1e-6
    assert float(_labelled(output, "U is Hermitian")) > 1.0
    # The walk identity is claimed on the ladder-zero subspace rather than on the whole
    # register, and both halves are pinned: every encoded eigenvalue is attained there,
    # and the subspace carries more eigenvalues than the encoding does, which is what
    # "restricted" means rather than "equal".
    assert _labelled(output, "encoded cosines attained") == "4 of 4"
    assert int(_labelled(output, "subspace eigenvalues")) > 2 * 4
    assert _labelled(output, "the subspace holds more than the encoding") == "True"
    assert _labelled(output, "cosines off the endpoints") == "True"
    assert _labelled(output, "subspace cosines").startswith("[-1.0, -1.0")
    # The adjoint is checked as a composition over the whole register, because this
    # construction's step is not its own adjoint and a spectrum cannot tell the two orders
    # apart. `W^dagger W` is the assertion the reversal is what satisfies.
    assert float(_labelled(output, "W^dagger W vs the identity")) < 1e-6
    assert float(_labelled(output, "W W^dagger vs the identity")) < 1e-6
    assert float(_labelled(output, "U^dagger U vs the identity")) < 1e-6
    # Two implementations, one consumer, and the second one is the one that holds a dense
    # matrix while this one never forms it: the same block read out of both is the
    # replacement evidence, and the register shapes are what tell them apart.
    assert _labelled(output, "this is a BlockEncoding") == "True"
    assert _labelled(output, "this is a WalkEncoding") == "True"
    assert _labelled(output, "spectral num_ancilla") == "1"
    assert _labelled(output, "spectral alpha") == "4.5"
    assert float(_labelled(output, "the two blocks agree")) < 1e-6
    _assert_premise("linear_combination", output)


def test_logical_resources_example_costs_one_program_and_refuses_the_rest() -> None:
    output = _run("logical_resources")

    assert "Logical resources -- flagquantum.algorithms.logical_resources" in output
    # The reuse, which is the property the whole slice rests on: the report's copy
    # of the compiler's record is printed beside the compiler's own, so a second
    # tally would print two different numbers rather than agreeing twice.
    for compiler_label, report_label in (
        ("compiler n_operations", "report estimate n_operations"),
        ("compiler depth", "report estimate depth"),
        ("compiler t_count", "report t_count"),
        ("compiler t_depth", "report t_depth"),
        ("compiler per_wire_depth", "report per_wire_depth"),
    ):
        assert _labelled(output, compiler_label) == _labelled(
            output, report_label
        ), compiler_label
    assert _labelled(output, "compiler n_operations") == "7"
    assert _labelled(output, "compiler depth") == "5"
    assert _labelled(output, "compiler per_wire_depth") == "(5, 3, 5)"
    assert _labelled(output, "the two records are equal") == "True"
    # The partition: four Cliffords and three Ts over seven operations, and the T
    # depth is not the T count.
    assert _labelled(output, "clifford_count") == "4"
    assert _labelled(output, "report t_count") == "3"
    assert _labelled(output, "n_clifford_t") == "7"
    assert _labelled(output, "report t_depth") == "2"
    assert _labelled(output, "clifford + t == operations") == "7"
    # The footprint at the default distance, against the model's own arithmetic.
    assert _labelled(output, "code_distance") == "5"
    assert _labelled(output, "physical qubits per logical") == "49"
    assert _labelled(output, "report n_qubits") == "3"
    assert _labelled(output, "logical_depth") == "5"
    assert _labelled(output, "surface_code_cycles") == "25"
    assert _labelled(output, "physical_qubits") == "147"
    assert _labelled(output, "spacetime_volume") == "3675"
    # The distance sweep, and the model's closed form beside it, so a row that
    # stopped agreeing with `2 d^2 - 1` would show rather than being recomputed.
    assert _labelled(output, "qubits/logical") == "17      49      97     161     241"
    assert _labelled(output, "2 d^2 - 1") == _labelled(output, "qubits/logical")
    assert _labelled(output, "cycles") == "15      25      35      45      55"
    assert _labelled(output, "physical qubits") == "51     147     291     483     723"
    assert (
        _labelled(output, "spacetime volume") == "765    3675   10185   21735   39765"
    )
    # A measurement is outside the schedule and is charged one further layer; the
    # tally and the qubit count are not touched by it.
    assert _labelled(output, "n_measurements") == "0"
    assert _labelled(output, "n_measurements, measured") == "2"
    assert _labelled(output, "logical_depth, none") == "5"
    assert _labelled(output, "logical_depth, measured") == "6"
    assert _labelled(output, "cycles, measured") == "30"
    assert _labelled(output, "physical qubits, measured") == "147"
    assert _labelled(output, "tally, measured") == "(4, 3)"
    # The serialized record, under the maturity schema's own field name.
    assert _labelled(output, "kind") == "flagquantum.logical_resource_report"
    assert _labelled(output, "basis") == "clifford_t_tally_times_surface_code_distance"
    assert _labelled(output, "surface_code_model") == "rotated_surface_code_2d"
    assert _labelled(output, "estimate kind") == "flagquantum.resource_estimate"
    assert _labelled(output, "capability_evidence keys") == "['limitations']"
    # Each refusal, and the two kinds of argument refusal distinguished by type:
    # a distance or a logical count that is out of range is a ValueError, and one
    # that is not an integer at all is a TypeError.
    assert "NOT REFUSED" not in output
    for label in (
        "one parametric rotation",
        "several parametric rotations",
        "a Toffoli",
        "a controlled swap",
        "a Toffoli beside a rotation",
        "a Trotter step",
        "a lowered noise channel",
    ):
        assert "CapabilityError" in _labelled(output, label), label
    for label in ("distance below three", "an even distance"):
        assert "ValueError" in _labelled(output, label), label
    assert "TypeError" in _labelled(output, "a distance that is not an integer")
    assert "ValueError" in _labelled(
        output, "a distance below three, at the entry point"
    )
    assert "ValueError" in _labelled(output, "zero logical qubits")
    assert "TypeError" in _labelled(output, "a logical count that is not an integer")
    # The logical ledger is the caller's, not inferred from the register.
    assert _labelled(output, "charged for one") == "49"
    assert _labelled(output, "charged for the register") == "147"
    assert _labelled(output, "charged for eight") == "392"
    _assert_premise("logical_resources", output)
    assert "take away" in output


def test_arithmetic_example_sums_and_hands_the_cost_to_the_compiler() -> None:
    output = _run("arithmetic")

    assert "Reversible addition -- flagquantum.algorithms.arithmetic" in output
    # The construction's own shape, at the default width and across the sweep: the
    # per-width row is printed against the closed form the module documents, so a
    # construction that changed would print two different rows rather than one
    # agreeing pair.
    assert _labelled(output, "wires") == "8"
    assert _labelled(output, "operations") == "25"
    assert _labelled(output, "ccx") == "6"
    assert _labelled(output, "cx") == "19"
    assert _labelled(output, "opcodes present") == "['ccx', 'cx']"
    assert _labelled(output, "register a") == "wires (0, 1, 2) (most significant first)"
    assert _labelled(output, "working carry wire") == "6"
    assert _labelled(output, "carry-out wire") == "7"
    # The sweep's labels repeat, so the last one printed wins: the per-width table
    # and the lowered table below it both use `n_bits`.
    assert (
        _labelled(output, "operations, measured")
        == "9     17     25     33     41     65     73"
    )
    assert (
        _labelled(output, "working ancillas")
        == "1      1      1      1      1      1      1"
    )
    # The sum, read out of the runtime's state vector rather than asserted: the
    # first addend is unchanged, the second holds the sum modulo 2**n, both
    # ancillas are clean, and the weight is one because the map is a permutation.
    assert _labelled(output, "1 + 1") == "a=1 b=2 carry=0 carry_out=0 weight=1.000000"
    assert _labelled(output, "7 + 1") == "a=7 b=0 carry=0 carry_out=1 weight=1.000000"
    assert _labelled(output, "7 + 7") == "a=7 b=6 carry=0 carry_out=1 weight=1.000000"
    # Off the ancilla contract the two wires do not behave alike, and both facts
    # are measured over every input rather than described: a dirty carry-out wire
    # cannot change the sum, and a dirty carry wire is neither the sum nor the
    # sum-with-carry-in.
    assert _labelled(output, "a=1 b=1, carry wire dirty") == "reached (1, 3, 1, 0)"
    assert (
        _labelled(output, "a dirty carry-out wire, over every input")
        == "0 of 64 do not sum, and the top wire is XORed"
    )
    assert (
        _labelled(output, "a dirty carry wire, over every input")
        == "64 of 64 are neither the sum nor the sum-with-carry-in"
    )
    # The Toffoli rule read from the compiler, and the linear cost that follows: an
    # expansion written into the arithmetic module would print a different
    # `operation counts` here.
    assert _labelled(output, "one ccx, operations") == "15"
    assert (
        _labelled(output, "one ccx, operation counts")
        == "{'h': 2, 'cx': 6, 'tdg': 3, 't': 4}"
    )
    assert _labelled(output, "one ccx, t_count") == "7"
    assert (
        _labelled(output, "t_count") == "14     28     42     56     70    112    126"
    )
    assert _labelled(output, "14 n") == _labelled(output, "t_count")
    assert (
        _labelled(output, "operations, lowered")
        == "37     73    109    145    181    289    325"
    )
    # The logical figure is a floor, and the surface-code model's own product is
    # what the rows have to agree with.
    assert _labelled(output, "distance") == "5"
    assert _labelled(output, "n_bits=2") == (
        "t_count=28 physical_qubits=294 cycles=275"
    )
    assert _labelled(output, "n_bits=4") == (
        "t_count=56 physical_qubits=490 cycles=545"
    )
    # The conversion bound refuses by name and then yields to the caller's own
    # bound, so the refusal is a bound rather than a limit of the construction.
    assert (
        _labelled(output, "n_bits=10 at the default")
        == "refused -- BasisConversionError: named-basis decomposition exceeds "
        "max_added_operations"
    )
    assert _labelled(output, "n_bits=10, bound raised to 1024") == "t_count=140"
    # Both entry points refuse the same widths, and for the same two reasons.
    for label in ("zero bits", "a negative width"):
        for entry in ("adder_circuit", "adder_wires"):
            assert "ValueError" in _labelled(output, f"{label}, {entry}"), label
    for label in (
        "a float width",
        "a width given as text",
        "a width given as a truth value",
    ):
        for entry in ("adder_circuit", "adder_wires"):
            assert "TypeError" in _labelled(output, f"{label}, {entry}"), label
    _assert_premise("arithmetic", output)
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

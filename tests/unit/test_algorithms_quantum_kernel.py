"""What the quantum-kernel unit computes, checked against a classical reference.

Every expected value in this file was produced by a real run of the construction it
checks. Two reference paths are kept apart from the module under test:

* the **classical overlap** -- ``<Phi(x)|Phi(z)>`` for the documented angle encoding,
  summed over the ``2**n`` basis states with the encoding's own phase, and its squared
  modulus as the kernel entry. It is arithmetic over the documented formula and shares
  no code with the module and no value with it.
* the **swap test's own relation** -- ``P(ancilla = 1) = 1/2 - 1/2 |<a|b>|^2`` -- read
  off a circuit's exact probabilities, which is what makes the sign blindness of the
  helper checkable without a sample in the way.

The sample sizes are part of every value: 16384 is where the accuracy assertions are
made and 4096 is the module's own default, used where the test is not about the sample.
"""

from __future__ import annotations

import math

import pytest
import torch

from flagquantum.algorithms.primitives.state_preparation import append_arbitrary_state
from flagquantum.algorithms.quantum_kernel import (
    KernelMatrixResult,
    KernelRidgeClassifier,
    _append_feature_state,
    _swap_test_overlap,
    kernel_ridge_classifier,
    kernel_ridge_regression,
    quantum_kernel_matrix,
)
from flagquantum.circuit import Circuit

pytestmark = pytest.mark.unit

# The module's own default sample size, used wherever the test is not about the sample.
_SHOTS = 4096

# The sample size the accuracy assertions are made at, so their tolerance has room:
# one entry's estimator has a standard deviation of about 1/sqrt(shots) there.
_WIDE_SHOTS = 16384

_SEED = 5
_HALF = 2.0**-0.5

# Instance A: four points in two features, chosen so that every entry of its exact
# kernel is well away from zero, which is what lets the range clause of the invariant
# check be asserted on the sampled matrix as well. Its exact kernel is
# [[1.0, 0.189252, 0.261003, 0.092623],
#  [0.189252, 1.0, 0.240819, 0.287198],
#  [0.261003, 0.240819, 1.0, 0.202146],
#  [0.092623, 0.287198, 0.202146, 1.0]]
_FAR = [[0.3, 2.9], [1.1, 0.4], [4.7, 1.2], [2.0, 3.3]]

# Instance B: three points in three features, the widest register this unit carries.
# Its entry (0, 1) is the near-zero one, and it is the reason no test here asserts the
# range clause on this instance's sampled matrix. Its exact kernel is
# [[1.0, 0.010026, 0.08796], [0.010026, 1.0, 0.179355], [0.08796, 0.179355, 1.0]]
_NEAR = [[0.6, 1.1, 2.4], [3.3, 0.2, 5.0], [2.7, 4.4, 1.0]]

# The classifier's training set, four points in two features labelled in pairs, and
# three held-out points. The exact-kernel classifier's decision values for the held-out
# points are 1.394457, -0.995729 and 0.667018 at a ridge weight of 1e-2, measured here
# rather than assumed.
_TRAIN = [[0.4, 0.5], [1.2, 1.7], [4.6, 1.1], [5.4, 1.9]]
_LABELS = (1, 1, -1, -1)
_HELD_OUT = [[1.3, 1.8], [4.5, 1.2], [0.5, 0.6]]
_HELD_OUT_LABELS = (1, -1, 1)

# The ridge weight the classifier tests use. Measured on this training set: at 1e-2 the
# dual coefficients reach 1.688 in magnitude, the in-sample margins are 0.98 or more,
# and the fit is stable under sampling; at 1e-6 they reach 1.718 with a residual of
# 1.7e-6, which is a tighter fit bought with a more sensitive system.
_RIDGE = 1e-2


def _signs(index: int, n_features: int) -> list[float]:
    """Return the ``+1``/``-1`` eigenvalue signs of one basis state, wire 0 first.

    ``Z`` on a wire is ``+1`` on ``|0>`` and ``-1`` on ``|1>``, and the basis index
    carries wire 0 as its most significant bit, so bit ``k`` of the index read from the
    top is wire ``k``'s outcome.
    """
    return [
        1.0 if ((index >> (n_features - 1 - wire)) & 1) == 0 else -1.0
        for wire in range(n_features)
    ]


def _phase(features: list[float], signs: list[float], n_features: int) -> float:
    """Return the encoding's accumulated phase on one basis state.

    This is the ``sum_k x_k s_k + sum_{j<k} (pi - x_j)(pi - x_k) s_j s_k`` of the
    documented feature map's exponent, written out here from the formula rather than
    from the module.
    """
    total = sum(features[wire] * signs[wire] for wire in range(n_features))
    total += sum(
        (math.pi - features[left])
        * (math.pi - features[right])
        * signs[left]
        * signs[right]
        for left in range(n_features)
        for right in range(left + 1, n_features)
    )
    return total


def _overlap(left: list[float], right: list[float], n_features: int) -> complex:
    """Return ``<Phi(left)|Phi(right)>`` summed over every basis state.

    Both states are a uniform superposition up to the encoding's phase, so the overlap
    is the mean of the phase difference over the ``2**n`` basis states, computed here
    in complex arithmetic.
    """
    total = 0j
    for index in range(2**n_features):
        signs = _signs(index, n_features)
        angle = _phase(right, signs, n_features) - _phase(left, signs, n_features)
        total += complex(math.cos(angle), math.sin(angle))
    return total / 2**n_features


def _feature_amplitudes(features: list[float], n_features: int) -> torch.Tensor:
    """Return the amplitude vector of the documented encoding, from the formula."""
    amplitudes = [
        2.0 ** (-n_features / 2)
        * complex(
            math.cos(_phase(features, _signs(index, n_features), n_features)),
            math.sin(_phase(features, _signs(index, n_features), n_features)),
        )
        for index in range(2**n_features)
    ]
    return torch.tensor(amplitudes, dtype=torch.complex64)


def _exact_kernel(rows: list[list[float]], n_features: int) -> list[list[float]]:
    """Return the kernel matrix of ``rows`` from the exact overlaps."""
    return [
        [abs(_overlap(left, right, n_features)) ** 2 for right in rows] for left in rows
    ]


def _signed_kernel(rows: list[list[float]], n_features: int) -> list[list[float]]:
    """Return the matrix of *signed* overlaps, the kernel a sign-carrying path gives.

    A Hadamard test estimates the real part of ``<Phi(x)|Phi(z)>``, which is a perfectly
    good inner product and can be negative. This is the matrix that alternative returns,
    built from the same overlaps the module's kernel is built from.
    """
    return [[_overlap(left, right, n_features).real for right in rows] for left in rows]


def _invariant_violations(matrix: list[list[float]]) -> list[str]:
    """Return the kernel-matrix invariants ``matrix`` breaks, by name.

    The three are checked separately and named separately, because they do not have the
    same strength and they are not broken by the same matrices: symmetry and a unit
    diagonal hold for any Gram matrix of unit feature vectors, signed or not, while the
    entry range holds only for the squared modulus.
    """
    size = len(matrix)
    violations: list[str] = []
    if any(
        abs(matrix[i][j] - matrix[j][i]) > 1e-12
        for i in range(size)
        for j in range(size)
    ):
        violations.append("symmetry")
    if any(matrix[i][i] != 1.0 for i in range(size)):
        violations.append("unit diagonal")
    if any(not 0.0 <= value <= 1.0 for row in matrix for value in row):
        violations.append("entry range")
    return violations


def _exact_decisions(
    train_rows: list[list[float]],
    labels: tuple[int, ...],
    test_rows: list[list[float]],
    n_features: int,
    regularization: float,
) -> tuple[float, ...]:
    """Return the decision values of an exact-kernel ridge classifier.

    This is the independent reference path for the classifier: the training kernel comes
    from the exact overlaps above and the solve is a direct linear solve of the
    documented system, so the module's sampled classifier is compared against a route
    that shares neither its kernel nor its fit.
    """
    kernel = torch.tensor(_exact_kernel(train_rows, n_features), dtype=torch.float64)
    targets = torch.tensor([float(label) for label in labels], dtype=torch.float64)
    size = len(train_rows)
    solved = torch.linalg.solve(
        kernel + regularization * torch.eye(size, dtype=torch.float64), targets
    )
    cross = torch.tensor(
        [
            [abs(_overlap(test, train, n_features)) ** 2 for train in train_rows]
            for test in test_rows
        ],
        dtype=torch.float64,
    )
    return tuple(float(value) for value in cross @ solved)


def _signed_states(n_wires: int) -> tuple[torch.Tensor, ...]:
    """Return four states whose overlaps are the sign-blindness triple plus a control.

    ``zero`` is ``|0...0>`` and the other three differ only in the phase of their
    ``|1...1>`` component, so every one of them has the same squared overlap with
    ``zero`` -- one half -- while the overlaps themselves are ``+1/sqrt(2)``,
    ``-1/sqrt(2)`` and ``+i/sqrt(2)``.
    """
    zero = torch.zeros(2**n_wires, dtype=torch.complex64)
    zero[0] = 1.0
    plus = torch.zeros(2**n_wires, dtype=torch.complex64)
    plus[0] = _HALF
    plus[-1] = _HALF
    minus = torch.zeros(2**n_wires, dtype=torch.complex64)
    minus[0] = -_HALF
    minus[-1] = _HALF
    imaginary = torch.zeros(2**n_wires, dtype=torch.complex64)
    imaginary[0] = 1j * _HALF
    imaginary[-1] = _HALF
    return zero, plus, minus, imaginary


def _swap_estimate(
    left: torch.Tensor,
    right: torch.Tensor,
    n_wires: int,
    *,
    shots: int,
    seed: int,
    dirty: bool = False,
) -> tuple[float, float]:
    """Run the module's swap test on two states and return its estimate and the exact one.

    The circuit is the caller's, as the helper's contract requires: the two registers
    are prepared first and the swap test is appended on top. The second value is read
    off the same circuit's exact probabilities afterwards as the ancilla's one-branch
    marginal, which is the relation the helper inverts -- and it is the same circuit
    object the helper sampled, so the two values describe one construction.

    ``dirty`` starts the ancilla in ``|1>`` instead of ``|0>``, which the helper's
    contract forbids and does not check. The flag exists so that what the forbidden
    entry does can be measured rather than described.
    """
    wires = list(range(1, 1 + n_wires))
    other = list(range(1 + n_wires, 1 + 2 * n_wires))
    circuit = Circuit(1 + 2 * n_wires)
    if dirty:
        circuit.gate("x", 0)
    append_arbitrary_state(circuit, left, wires)
    append_arbitrary_state(circuit, right, other)
    estimate = _swap_test_overlap(
        circuit,
        ancilla=0,
        left=wires,
        right=other,
        shots=shots,
        generator=torch.Generator().manual_seed(seed),
    )
    probabilities = circuit.probabilities().reshape(-1)
    exact = float(probabilities[2 ** (2 * n_wires) :].sum())
    return estimate, exact


def test_the_swap_test_is_sign_blind_in_the_three_relative_phases() -> None:
    """The triple the contract names: ``+1/sqrt(2)``, ``-1/sqrt(2)`` and ``+i/sqrt(2)``.

    The three states differ only in the phase of their ``|1...1>`` component, so their
    signed overlaps with ``|0...0>`` are ``+1/sqrt(2)``, ``-1/sqrt(2)`` and
    ``+i/sqrt(2)`` while all three squared overlaps are one half. The swap test is
    blind to that phase, so all three must return the same statistic -- and here they
    return it **to the last digit**, because with one seed the three circuits are the
    same experiment. Measured at 20000 shots over seeds 0, 1 and 2: the three estimates
    are ``0.4955``, ``0.4982`` and ``0.5089``, identical across the triple at each seed
    and within 0.009 of one half. The ancilla's exact one-branch marginal, read off each
    circuit's probabilities, is 0.25 for all three to float32 precision, which is the
    precision a statevector is held at.

    This is the test a sign-carrying implementation fails, and the assertion that
    catches it is the **magnitude** one, first and on its own: a signed path returns
    ``0.7071``, ``-0.7071`` and ``0.0`` for the triple, and the first of those is
    ``0.207`` from one half, well outside the ``0.05`` tolerance. Measured by running
    this test body with its estimating helper replaced by one that returns the signed
    overlap: the first failure is at the magnitude assertion, so the equality below is
    **never reached** by that implementation.

    The equality is a second net for a different implementation, and that is why it is
    asserted at all: one whose three readings all land inside the magnitude tolerance
    but differ from each other passes every ``*_exact`` and magnitude assertion and is
    caught here alone. Measured on a stand-in whose three readings are ``0.53``,
    ``0.47`` and ``0.47`` -- the first two assertions pass, and this equality is the
    only one that fires.

    The **identical-states** pair cannot separate the two paths: both read one there, so
    a test built on that pair alone would pass for either. That is a statement about
    that pair and not about every pair -- the test below measures a half-overlap pair,
    whose two paths read ``0.5`` and ``0.707`` and are separated by the magnitude
    assertion. The triple is what the contract names, and it is also what the equality
    needs: two states give one equality, three give one that a per-state error can break.
    """
    for n_wires in (1, 2):
        zero, plus, minus, imaginary = _signed_states(n_wires)

        for seed in (0, 1, 2):
            estimates = []
            for state in (plus, minus, imaginary):
                estimate, exact = _swap_estimate(
                    zero, state, n_wires, shots=20000, seed=seed
                )
                assert exact == pytest.approx(0.25, abs=1e-6), (n_wires, seed)
                assert estimate == pytest.approx(0.5, abs=0.05), (n_wires, seed)
                estimates.append(estimate)

            assert estimates[0] == estimates[1] == estimates[2], (n_wires, seed)

    # The control: a pair with no overlap at all is not near one half, so the equality
    # above is not the equality of three numbers that all sit at the same place.
    zero, _, _, _ = _signed_states(1)
    orthogonal = torch.tensor([0.0, 1.0], dtype=torch.complex64)
    estimate, exact = _swap_estimate(zero, orthogonal, 1, shots=20000, seed=0)
    assert exact == pytest.approx(0.5, abs=1e-6)
    assert estimate == pytest.approx(0.0, abs=0.05)


def test_the_swap_test_estimates_the_squared_overlap_and_not_the_overlap() -> None:
    """One, one half and zero squared overlap, at both register widths.

    Equal feature vectors have a squared overlap of one, and the circuit never finds that
    ancilla set, so the estimate is exactly one at every seed rather than merely close to
    it. Orthogonal ones have a squared overlap of zero and an ancilla found set exactly
    half the time. A pair whose overlap is ``1/sqrt(2)`` -- the sign-blindness triple's
    states, read as a squared overlap instead of as a sign -- has a squared overlap of
    one half and an ancilla found set a quarter of the time. Measured at 20000 shots:
    identical states give ``1.0`` exactly at every seed, orthogonal ones give estimates
    within ``0.014`` of zero, and the half-overlap pair gives estimates within ``0.009``
    of ``0.5``.

    What this pins is the readout's arithmetic and not its sign blindness. A run that
    read the share rather than inverting it would return ``0.0``, ``0.5`` and ``0.25``
    here, and the three **estimate** assertions below fire on that -- ``identical``,
    ``orthogonal`` and ``half``. The three ``*_exact`` assertions do not, and cannot:
    they are read from the circuit's own probabilities, which the module's readout never
    touches. A run that returned the *signed* overlap would leave the first two of these
    three pairs reading exactly what the squared overlap reads -- ``1.0`` and ``0.0`` --
    and would read ``0.707`` on the third, which the third estimate assertion fires on.
    The triple test above is the one that catches the sign, because a signed path
    returns three different numbers there while this circuit returns one.
    """
    for n_wires in (1, 2):
        zero, plus, _, _ = _signed_states(n_wires)
        one = torch.zeros(2**n_wires, dtype=torch.complex64)
        one[-1] = 1.0

        for seed in (0, 1, 2):
            identical, identical_exact = _swap_estimate(
                zero, zero, n_wires, shots=20000, seed=seed
            )
            orthogonal, orthogonal_exact = _swap_estimate(
                zero, one, n_wires, shots=20000, seed=seed
            )
            half, half_exact = _swap_estimate(
                zero, plus, n_wires, shots=20000, seed=seed
            )

            # The second value is the ancilla's exact one-branch marginal, so the pair's
            # three cases read 0, one half and one quarter there, and 1, 0 and one half
            # in the estimate that inverts the relation.
            assert identical_exact == 0.0, (n_wires, seed)
            assert identical == 1.0, (n_wires, seed)
            assert orthogonal_exact == pytest.approx(0.5, abs=1e-6), (n_wires, seed)
            assert orthogonal == pytest.approx(0.0, abs=0.05), (n_wires, seed)
            assert half_exact == pytest.approx(0.25, abs=1e-6), (n_wires, seed)
            assert half == pytest.approx(0.5, abs=0.05), (n_wires, seed)


def test_a_dirty_ancilla_negates_the_readout_rather_than_shifting_it() -> None:
    """The violation the helper's contract forbids, measured instead of described.

    The contract says the ancilla must enter in ``|0>``, and the helper does not check
    it. What a ``|1>`` does is a **negation** and not an offset: with the ancilla
    flipped, the two Hadamards leave it found set with probability
    ``1/2 + 1/2 |<a|b>|^2``, so the estimate is ``-|<a|b>|^2``, the negation of the
    clean reading. The distinction is not cosmetic, because a caller who reads the sign
    can undo a negation and cannot undo an offset.

    Measured at 20000 shots over seeds 0, 1 and 2 at both register widths. A pair of
    equal states reads ``+1.0`` with a clean ancilla and ``-1.0`` with a dirty one, both
    exactly, and the ancilla's exact one-branch marginal is ``0.0`` and ``1.0``
    respectively -- the second to float32 precision, ``0.9999998807907104``, which is
    the precision a statevector is held at. The clean value is not merely negated on
    average, it is negated at every sample, because the circuit never finds the ancilla
    set on a clean run and always finds it on a dirty one. A half-overlap pair moves from
    ``0.25`` to ``0.75`` in that marginal and its estimate from about ``+0.5`` to about
    ``-0.5``.
    """
    for n_wires in (1, 2):
        zero, plus, _, _ = _signed_states(n_wires)

        for seed in (0, 1, 2):
            clean, clean_exact = _swap_estimate(
                zero, zero, n_wires, shots=20000, seed=seed
            )
            dirty, dirty_exact = _swap_estimate(
                zero, zero, n_wires, shots=20000, seed=seed, dirty=True
            )

            assert clean == 1.0, (n_wires, seed)
            assert clean_exact == 0.0, (n_wires, seed)
            assert dirty == -1.0, (n_wires, seed)
            assert dirty_exact == pytest.approx(1.0, abs=1e-6), (n_wires, seed)

            clean_half, clean_half_exact = _swap_estimate(
                zero, plus, n_wires, shots=20000, seed=seed
            )
            dirty_half, dirty_half_exact = _swap_estimate(
                zero, plus, n_wires, shots=20000, seed=seed, dirty=True
            )

            assert clean_half == pytest.approx(0.5, abs=0.05), (n_wires, seed)
            assert dirty_half == pytest.approx(-0.5, abs=0.05), (n_wires, seed)
            assert clean_half_exact == pytest.approx(0.25, abs=1e-6), (n_wires, seed)
            assert dirty_half_exact == pytest.approx(0.75, abs=1e-6), (n_wires, seed)


def test_the_swap_test_refuses_registers_of_different_widths() -> None:
    """A swap is one wire of each register, so a mismatch is refused rather than run.

    Without the check the shorter register's extra wires would be left unswapped and the
    estimate would silently be of a different pair of states.
    """
    circuit = Circuit(4)
    with pytest.raises(ValueError, match="same width"):
        _swap_test_overlap(
            circuit,
            ancilla=0,
            left=[1],
            right=[2, 3],
            shots=16,
            generator=torch.Generator().manual_seed(0),
        )


def test_the_feature_state_is_the_angle_encoding_the_module_documents() -> None:
    """The gate sequence and the documented formula are the same state, wire for wire.

    The module docstring writes the feature map as a product of phase rotations, so a
    formula that the circuit does not implement would be a claim about the code that the
    code does not carry. Measured, the largest amplitude difference is ``0.0`` at one
    feature, ``7.0e-08`` at two and ``1.4e-07`` at three, which is float32 precision.

    The comparison is against the formula's own amplitudes, not against the module's
    overlap of it: the phase is accumulated over every basis state here, and the state
    vector is read back from a circuit the module only appended gates to.
    """
    for features, n_features in (([0.3], 1), ([0.3, 2.9], 2), ([0.3, 2.9, 5.1], 3)):
        circuit = Circuit(n_features)
        _append_feature_state(circuit, features, list(range(n_features)))
        state = circuit.state().reshape(-1)

        assert float(
            torch.abs(state - _feature_amplitudes(features, n_features)).max()
        ) < (1e-6), n_features


def test_the_kernel_matrix_matches_the_classical_overlaps() -> None:
    """Both instances, every entry, against the exact overlap computed here.

    Measured at 16384 shots: the worst deviation over seeds 0 through 9 is ``0.0233``
    on the four-point instance and ``0.0240`` on the three-feature one, and the
    tolerance below is twice that. One entry's estimator has a standard deviation of
    about ``1/sqrt(shots)`` there, so the assertion is a bound with room rather than a
    value: it is the same check at 4096 shots that shows the sample, and that is the
    test below.
    """
    for rows, n_features in ((_FAR, 2), (_NEAR, 3)):
        data = torch.tensor(rows, dtype=torch.float64)
        exact = _exact_kernel(rows, n_features)

        for seed in range(10):
            matrix = quantum_kernel_matrix(data, shots=_WIDE_SHOTS, seed=seed).matrix

            deviation = max(
                abs(matrix[i][j] - exact[i][j])
                for i in range(len(rows))
                for j in range(len(rows))
            )
            assert deviation < 0.05, (n_features, seed)


def test_the_kernel_matrix_is_symmetric_with_a_unit_diagonal() -> None:
    """The two invariants the module's construction fixes, checked exactly.

    Symmetry is exact here and not approximate: a kernel entry is one quantity, so the
    two triangles are the same sample rather than two samples of the same quantity, and
    the difference between them is ``0.0`` at every seed and width. The diagonal is
    sampled like every other entry -- the module runs the swap test of a state with
    itself rather than writing one into the matrix -- and it comes back ``1.0`` exactly.
    That is the circuit's arithmetic: the overlap of a state with itself is one, so the
    circuit never finds that ancilla set.

    Symmetry, the unit diagonal and the **upper** bound of the entry range run inside the
    loop, so all three are asserted on both instances. The **lower** bound runs on the
    four-point instance only, through ``_invariant_violations`` under the width gate
    below: its exact kernel's smallest entry is ``0.0926``, well above the sampler's
    error at this width, so the range clause holds there as well; on the three-feature
    instance the smallest exact entry is ``0.010026`` and the readout can cross zero,
    which the test below measures instead of denying.
    """
    for rows, n_features in ((_FAR, 2), (_NEAR, 3)):
        data = torch.tensor(rows, dtype=torch.float64)

        for seed in range(4):
            matrix = quantum_kernel_matrix(data, shots=_WIDE_SHOTS, seed=seed).matrix

            assert all(
                matrix[i][j] == matrix[j][i]
                for i in range(len(rows))
                for j in range(len(rows))
            ), (n_features, seed)
            assert all(matrix[i][i] == 1.0 for i in range(len(rows))), (
                n_features,
                seed,
            )
            assert all(value <= 1.0 for row in matrix for value in row), (
                n_features,
                seed,
            )

        if n_features == 2:
            assert _invariant_violations(list(matrix)) == []


def test_a_kernel_from_signed_overlaps_breaks_the_entry_range_invariant() -> None:
    """The bidirectional half: the invariant separates the kernel from its alternative.

    Two matrices are built here from the same exact overlaps: the squared modulus, which
    is the kernel this unit estimates, and the real part of the overlap itself, which is
    what a sign-carrying path returns. Measured on the four-point instance, **both are
    symmetric to within ``0.0`` and both have a diagonal of exactly ``1.0``**, because a
    Gram matrix of unit feature vectors has both properties whatever its entries' signs
    are. The clause that separates them is the entry range: the squared kernel's
    smallest entry is ``0.0926`` and the signed one's is ``-0.528179``, at ``(1, 3)``,
    with ten of its sixteen entries below zero.

    So the check is stated as it is and not as it is usually written: the invariant that
    fires on a signed kernel is the range clause, and a check of symmetry and the unit
    diagonal alone -- which is what "the kernel matrix is symmetric with a diagonal of
    one" names -- would pass it. The last assertion is that the module's own sampled
    matrix does not break the range clause on this instance, so the separation is
    between two matrices built here and one the module produced.
    """
    exact = _exact_kernel(_FAR, 2)
    signed = _signed_kernel(_FAR, 2)

    assert _invariant_violations(exact) == []
    assert _invariant_violations(signed) == ["entry range"]
    assert signed[1][3] == pytest.approx(-0.528179, abs=1e-6)
    assert min(value for row in signed for value in row) == pytest.approx(
        -0.528179, abs=1e-6
    )
    assert min(value for row in exact for value in row) == pytest.approx(
        0.092623, abs=1e-6
    )

    sampled = quantum_kernel_matrix(
        torch.tensor(_FAR, dtype=torch.float64), shots=_WIDE_SHOTS, seed=0
    ).matrix
    assert _invariant_violations(list(sampled)) == []


def test_the_readout_is_not_clamped_when_the_overlap_it_estimates_is_near_zero() -> (
    None
):
    """An estimate of a near-zero entry can come back below zero, and it is reported.

    The three-feature instance's entry ``(0, 1)`` has an exact squared overlap of
    ``0.010026``, which is inside the sampler's own error at these widths. Measured at
    16384 shots over seeds 0 through 5, that entry comes back as ``0.011841``,
    ``0.001587``, ``0.034058``, ``-0.000610``, ``0.015869`` and ``0.004395``: the
    fourth of them is negative. The estimate is ``1 - 2 * share`` with ``share`` in
    ``[0, 1]``, so a negative value is what an over-counted sample gives and the module
    reports it rather than projecting it onto zero, which is why no test here asserts
    the range clause on this instance's sampled matrix.

    The entry is also the one no accuracy assertion in this file is made about narrowly
    enough to matter: at the same width the four-point instance's smallest entry is
    ``0.08337`` over twelve seeds, and the invariant check is made there.
    """
    data = torch.tensor(_NEAR, dtype=torch.float64)

    estimates = [
        quantum_kernel_matrix(data, shots=_WIDE_SHOTS, seed=seed).matrix[0][1]
        for seed in range(6)
    ]

    assert estimates[3] == pytest.approx(-0.000610, abs=1e-6)
    assert min(estimates) < 0.0
    assert max(estimates) < 0.05

    far = torch.tensor(_FAR, dtype=torch.float64)
    assert all(
        min(
            value
            for row in quantum_kernel_matrix(far, shots=_WIDE_SHOTS, seed=seed).matrix
            for value in row
        )
        > 0.0
        for seed in range(12)
    )


def test_the_sample_size_is_visible_in_every_entry() -> None:
    """Every entry is a count over exactly ``shots`` samples, so ``shots`` reaches it.

    The estimate is ``1 - 2 * share``, so ``(1 - entry) / 2 * shots`` is the number of
    samples that found the ancilla set: a whole number between zero and ``shots``. Both
    hold for every entry of the matrix, which is what makes the argument load-bearing
    rather than decorative.

    Measured under a run that ignored the caller's ``shots`` and used the module's
    default 4096 instead: at a requested 100 the first row reads ``39.7705``, ``36.499``
    and ``45.581`` where whole counts belong, twelve of the sixteen entries are not whole
    numbers and the worst sits ``0.4990`` from one; at 1024 twelve of sixteen are still
    fractional; and at 4096 every entry is whole, so the largest requested width is the
    one width that would not have caught it.
    """
    data = torch.tensor(_FAR, dtype=torch.float64)

    for shots in (100, 1024, _SHOTS):
        matrix = quantum_kernel_matrix(data, shots=shots, seed=_SEED).matrix
        counts = [(1.0 - value) / 2.0 * shots for row in matrix for value in row]

        assert all(abs(count - round(count)) < 1e-9 for count in counts), shots
        assert all(0 <= round(count) <= shots for count in counts), shots
        assert sum(round(count) for count in counts) % 2 == 0, shots


def test_a_small_sample_leaves_every_entry_short() -> None:
    """The estimate is a sample, and at one shot the sample is visibly not enough.

    Measured on the four-point instance, the worst deviation from the exact kernel over
    seeds 0 through 5 is ``1.0926``, ``1.2872``, ``1.2872``, ``1.2408``, ``1.2610`` and
    ``1.2021`` at one shot, ``0.1342`` to ``0.2646`` at 64, ``0.0439`` to ``0.0850`` at
    1024, and ``0.0099`` to ``0.0233`` at 16384. The floors and ceilings below are the
    measured values with room on each side, and the point of the test is what the one
    shot figure means: the readout is a sample and not a reading, so an implementation
    that computed the entry instead of estimating it would return the exact kernel here
    and the floor would fire.
    """
    data = torch.tensor(_FAR, dtype=torch.float64)
    exact = _exact_kernel(_FAR, 2)

    def worst(shots: int, seed: int) -> float:
        matrix = quantum_kernel_matrix(data, shots=shots, seed=seed).matrix
        return max(abs(matrix[i][j] - exact[i][j]) for i in range(4) for j in range(4))

    for seed in range(6):
        assert worst(1, seed) > 1.0, seed
        assert worst(4, seed) > 0.3, seed
        assert worst(64, seed) < 0.5, seed
        assert worst(1024, seed) < 0.2, seed
        assert worst(_WIDE_SHOTS, seed) < 0.05, seed


def test_the_kernel_matrix_refuses_more_than_three_features() -> None:
    """The unit's own bound, refused here rather than by the width of a register.

    Four features is a nine-wire swap test, and the message names the bound and the
    reason. The check is this module's: without it the entry would be built and sampled,
    which is what makes the bound a decision rather than an impossibility.
    """
    data = torch.zeros((2, 4), dtype=torch.float64)

    with pytest.raises(ValueError, match="bounded at 3 features"):
        quantum_kernel_matrix(data, shots=16, seed=0)


def test_the_data_and_the_shot_count_are_validated() -> None:
    """Each check has its own case, matched on this module's wording.

    The feature-range case is the one a caller meets first and the one whose message
    carries the reason: a feature is an angle here, and the entangling angle is built
    from the two features' complements to pi, so the encoding is written for features in
    ``[0, 2*pi]``. The match strings are this module's own, so a deleted guard cannot be
    covered up by another layer's message.
    """
    good = torch.zeros((2, 2), dtype=torch.float64)

    with pytest.raises(ValueError, match="torch.Tensor"):
        quantum_kernel_matrix([[0.0, 0.0]], shots=16)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="two-dimensional"):
        quantum_kernel_matrix(torch.zeros(3, dtype=torch.float64), shots=16)
    with pytest.raises(ValueError, match="floating-point"):
        quantum_kernel_matrix(torch.zeros((2, 2), dtype=torch.int64), shots=16)
    with pytest.raises(ValueError, match="at least one feature vector, got none"):
        quantum_kernel_matrix(torch.zeros((0, 2), dtype=torch.float64), shots=16)
    with pytest.raises(ValueError, match="at least one feature, got none"):
        quantum_kernel_matrix(torch.zeros((2, 0), dtype=torch.float64), shots=16)
    with pytest.raises(ValueError, match="finite"):
        quantum_kernel_matrix(torch.tensor([[0.0, float("nan")]]), shots=16)
    with pytest.raises(ValueError, match=r"must lie in \[0, 2\*pi\]"):
        quantum_kernel_matrix(torch.tensor([[0.0, 7.0]]), shots=16)
    with pytest.raises(ValueError, match="at least one shot"):
        quantum_kernel_matrix(good, shots=0)


def test_the_ridge_solve_is_the_system_the_module_documents() -> None:
    """The dual solve, against a hand-computed instance and against its own system.

    A diagonal kernel makes the system diagonal too, so its solution is written out by
    hand here: ``alpha_i = y_i / (K_ii + lambda)``. The second half checks the identity
    the solve claims rather than the arithmetic that produced it -- ``(K + lambda I)
    alpha = y`` -- which is the property a caller relies on when the kernel is not
    diagonal.
    """
    diagonal = torch.diag(torch.tensor([2.0, 0.5, -1.0], dtype=torch.float64))
    targets = torch.tensor([1.0, 2.0, -3.0], dtype=torch.float64)

    for regularization, want in (
        (0.5, (0.4, 2.0, 6.0)),
        (2.0, (0.25, 0.8, -3.0)),
    ):
        coefficients = kernel_ridge_regression(
            diagonal, targets, regularization=regularization
        )

        assert coefficients == pytest.approx(want, abs=1e-12)

    kernel = torch.tensor(_exact_kernel(_FAR, 2), dtype=torch.float64)
    labels = torch.tensor([1.0, 1.0, -1.0, -1.0], dtype=torch.float64)
    identity = torch.eye(4, dtype=torch.float64)

    for regularization in (1e-6, _RIDGE, 1.0):
        coefficients = torch.tensor(
            kernel_ridge_regression(kernel, labels, regularization=regularization),
            dtype=torch.float64,
        )
        system = kernel + regularization * identity

        assert (
            float((system @ coefficients - labels).abs().max()) < 1e-12
        ), regularization


def test_a_vanishing_ridge_weight_buys_a_tighter_fit_with_larger_coefficients() -> None:
    """The ridge weight trades the fit against the coefficients, and the trade is measured.

    On this six-point training set the kernel is not singular -- its condition number is
    ``77.0``, measured -- so a vanishing weight does not blow up, but it does move the
    coefficients and the fit in opposite directions: measured, the largest coefficient is
    ``34.11`` at ``1e-6`` against ``8.64`` at ``1e-1``, while the largest in-sample
    residual is ``3.4e-05`` against ``0.864``. The same trade is why the classifier's
    tests use ``1e-2``: on that four-point set the largest coefficient is ``1.688`` and
    every in-sample margin is ``0.98`` or more, which is a fit the sampler cannot move.

    This is a measurement of these instances and not a rule about ridge weights: what
    the weight should be depends on the kernel matrix and on the caller's purpose, and
    no default is supplied anywhere in this module.
    """
    rows = [[0.4, 0.4], [1.0, 0.8], [2.6, 2.2], [3.2, 2.6], [5.5, 1.0], [6.1, 1.4]]
    kernel = torch.tensor(_exact_kernel(rows, 2), dtype=torch.float64)
    labels = torch.tensor([1.0, 1.0, -1.0, -1.0, 1.0, 1.0], dtype=torch.float64)
    condition = float(torch.linalg.cond(kernel))
    assert condition == pytest.approx(77.0, abs=0.1)

    def fitted(regularization: float) -> tuple[float, float]:
        coefficients = torch.tensor(
            kernel_ridge_regression(kernel, labels, regularization=regularization),
            dtype=torch.float64,
        )
        return (
            float(coefficients.abs().max()),
            float((kernel @ coefficients - labels).abs().max()),
        )

    small_size, small_residual = fitted(1e-6)
    large_size, large_residual = fitted(1e-1)

    assert small_size == pytest.approx(34.1065, abs=1e-3)
    assert small_residual == pytest.approx(3.411e-05, rel=1e-3)
    assert large_size == pytest.approx(8.6405, abs=1e-3)
    assert large_residual == pytest.approx(8.640e-01, rel=1e-3)
    assert small_size > large_size
    assert small_residual < large_residual


def test_the_ridge_operands_are_validated() -> None:
    """A non-square kernel, a mismatched target count and a zero weight are all refused."""
    kernel = torch.eye(2, dtype=torch.float64)
    targets = torch.ones(2, dtype=torch.float64)

    with pytest.raises(ValueError, match="torch.Tensor"):
        kernel_ridge_regression([[1.0, 0.0], [0.0, 1.0]], targets, regularization=1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="square"):
        kernel_ridge_regression(
            torch.zeros((2, 3), dtype=torch.float64), targets, regularization=1.0
        )
    with pytest.raises(ValueError, match="one target per kernel row"):
        kernel_ridge_regression(
            kernel, torch.ones(3, dtype=torch.float64), regularization=1.0
        )
    with pytest.raises(ValueError, match="floating-point"):
        kernel_ridge_regression(
            torch.eye(2, dtype=torch.int64),
            targets,
            regularization=1.0,
        )
    with pytest.raises(ValueError, match="finite"):
        kernel_ridge_regression(
            torch.tensor([[1.0, 0.0], [0.0, float("inf")]]),
            targets,
            regularization=1.0,
        )
    with pytest.raises(ValueError, match="positive and finite"):
        kernel_ridge_regression(kernel, targets, regularization=0.0)


def test_the_classifier_reproduces_its_training_labels() -> None:
    """A small ridge weight fits the training set, so the sign is the label's.

    Measured at 4096 shots: over seeds 0 through 19 the classifier's prediction for every
    one of the four training rows equals that row's label, which is 80 predictions and
    none wrong; at 16384 shots the same. The in-sample margins are what makes that a
    bound rather than a coin flip -- the smallest is ``0.983`` in the exact kernel, where
    the four decision values are ``0.989737``, ``0.984444``, ``-0.98312`` and
    ``-0.988628``, and the sampled values sit within ``0.1482`` of them over those seeds
    -- and no accuracy claim about any other set is made here.
    """
    train = torch.tensor(_TRAIN, dtype=torch.float64)
    exact = _exact_decisions(_TRAIN, _LABELS, _TRAIN, 2, _RIDGE)
    assert all(
        (value > 0.0) == (label > 0)
        for value, label in zip(exact, _LABELS, strict=False)
    )

    for seed in range(20):
        classifier = kernel_ridge_classifier(
            train, _LABELS, regularization=_RIDGE, shots=_SHOTS, seed=seed
        )
        predictions = classifier.predict(train, shots=_SHOTS, seed=seed + 50)

        assert predictions == _LABELS, seed


def test_the_classifier_decides_held_out_rows_as_the_exact_kernel_does() -> None:
    """Three held-out rows, decided against an exact-kernel fit computed here.

    The reference fits the same system on the exact kernel and decides the held-out rows
    with it, so the module's sampled classifier is compared against a route of its own.
    Its exact decision values are ``1.394457``, ``-0.995729`` and ``0.667018``, so every
    margin is more than half a unit and the sampled values sit well inside them.
    Measured at 4096 shots over seeds 0 through 19: 60 held-out predictions and none
    wrong, the largest deviation of a sampled decision value from the exact one being
    ``0.1192``.
    """
    train = torch.tensor(_TRAIN, dtype=torch.float64)
    held_out = torch.tensor(_HELD_OUT, dtype=torch.float64)
    exact = _exact_decisions(_TRAIN, _LABELS, _HELD_OUT, 2, _RIDGE)
    want = tuple(1 if value >= 0.0 else -1 for value in exact)
    assert want == _HELD_OUT_LABELS

    for seed in range(20):
        classifier = kernel_ridge_classifier(
            train, _LABELS, regularization=_RIDGE, shots=_SHOTS, seed=seed
        )
        values = classifier.decision_function(held_out, shots=_SHOTS, seed=seed + 90)
        predictions = classifier.predict(held_out, shots=_SHOTS, seed=seed + 90)

        assert predictions == want, seed
        assert (
            max(
                abs(value - reference)
                for value, reference in zip(values, exact, strict=False)
            )
            < 0.5
        ), seed


def test_the_decision_function_is_the_weighted_sum_the_module_documents() -> None:
    """The sum is checked on the one instance where one of its terms is known by hand.

    A classifier with a single training row reduces the decision value to
    ``alpha * K(x, x_0)``, and at ``x = x_0`` that entry is the diagonal, which is one:
    so the decision value there is exactly ``alpha`` however the entry was sampled, and
    the value at a different row is ``alpha`` times that row's entry. Measured: at 4096
    shots over seeds 0 through 9 the training row's value is ``2.0`` exactly at every
    one of them, and at 16384 shots over the same ten seeds the other row's value is
    within ``0.031`` of ``0.378504``, which is ``2`` times the exact squared overlap
    ``0.189252``. The first of those is a hand-computed value -- the diagonal entry is
    one by construction, so the product is ``alpha`` whatever the sample -- and the
    second is the module's own kernel entry scaled by the coefficient it was given, so
    the formula the docstring writes is the object the code computes.
    """
    row = list(_FAR[0])
    classifier = KernelRidgeClassifier(
        coefficients=(2.0,),
        training_data=(tuple(row),),
        labels=(1,),
        regularization=_RIDGE,
    )

    for seed in range(10):
        at_training = classifier.decision_function(
            torch.tensor([row], dtype=torch.float64), shots=_SHOTS, seed=seed
        )
        assert at_training == (2.0,), seed

        at_other = classifier.decision_function(
            torch.tensor([_FAR[1]], dtype=torch.float64),
            shots=_WIDE_SHOTS,
            seed=seed,
        )
        assert at_other[0] == pytest.approx(2.0 * 0.189252, abs=0.06), seed
        assert classifier.predict(
            torch.tensor([_FAR[1]], dtype=torch.float64),
            shots=_WIDE_SHOTS,
            seed=seed,
        ) == (1,), seed


def test_the_classifier_predicts_plus_one_at_a_zero_decision_value() -> None:
    """The tie rule at exactly zero, reached by constructing the boundary directly.

    Zero coefficients make every decision value exactly zero whatever the kernel
    entries are, so the boundary is reached without a search for an instance that hits
    it. The rule is that a value of zero or above is predicted ``+1``, which is what the
    module does and what its docstring says it does.
    """
    classifier = KernelRidgeClassifier(
        coefficients=(0.0,) * len(_TRAIN),
        training_data=tuple(tuple(row) for row in _TRAIN),
        labels=_LABELS,
        regularization=_RIDGE,
    )
    held_out = torch.tensor(_HELD_OUT, dtype=torch.float64)

    values = classifier.decision_function(held_out, shots=64, seed=_SEED)

    assert values == (0.0, 0.0, 0.0)
    assert classifier.predict(held_out, shots=64, seed=_SEED) == (1, 1, 1)


def test_the_classifier_validates_its_inputs() -> None:
    """A wrong feature count, a wrong label and a wrong sample size are all refused."""
    train = torch.tensor(_TRAIN, dtype=torch.float64)

    with pytest.raises(ValueError, match=r"\+1 or -1"):
        kernel_ridge_classifier(
            train, (1, 0, -1, -1), regularization=_RIDGE, shots=64, seed=0
        )
    with pytest.raises(ValueError, match="must number the same"):
        kernel_ridge_classifier(
            train, (1, 1, -1), regularization=_RIDGE, shots=64, seed=0
        )
    with pytest.raises(ValueError, match="must lie in"):
        kernel_ridge_classifier(
            torch.tensor([[0.0, 9.0]] * 4),
            _LABELS,
            regularization=_RIDGE,
            shots=64,
            seed=0,
        )
    with pytest.raises(ValueError, match="positive and finite"):
        kernel_ridge_classifier(train, _LABELS, regularization=0.0, shots=64, seed=0)

    classifier = kernel_ridge_classifier(
        train, _LABELS, regularization=_RIDGE, shots=64, seed=0
    )
    with pytest.raises(ValueError, match="feature count must match"):
        classifier.predict(torch.zeros((2, 3), dtype=torch.float64), shots=64, seed=0)
    with pytest.raises(ValueError, match="at least one shot"):
        classifier.predict(train, shots=0, seed=0)


def test_the_kernel_matrix_result_validates_its_own_fields() -> None:
    """A matrix that could not come from a swap test is refused at construction."""
    with pytest.raises(ValueError, match="at least one row"):
        KernelMatrixResult(matrix=(), shots=16)
    with pytest.raises(ValueError, match="square"):
        KernelMatrixResult(matrix=((1.0, 0.0),), shots=16)
    with pytest.raises(ValueError, match="square"):
        KernelMatrixResult(matrix=((1.0, 0.0), (0.0, 1.0), (0.0, 0.0)), shots=16)
    with pytest.raises(ValueError, match="finite"):
        KernelMatrixResult(matrix=((1.0, float("nan")), (0.0, 1.0)), shots=16)
    with pytest.raises(ValueError, match=r"lies in \[-1, 1\]"):
        KernelMatrixResult(matrix=((1.0, -1.5), (-1.5, 1.0)), shots=16)
    with pytest.raises(ValueError, match="at least one shot"):
        KernelMatrixResult(matrix=((1.0,),), shots=0)


def test_the_classifier_validates_its_own_fields() -> None:
    """A classifier whose fields do not describe a training set is refused."""
    fields: dict[str, object] = {
        "training_data": ((0.0,), (1.0,)),
        "labels": (-1, 1),
        "regularization": 1.0,
    }

    with pytest.raises(ValueError, match="at least one training row"):
        KernelRidgeClassifier(coefficients=(), training_data=(), labels=(), regularization=1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="three lengths must agree"):
        KernelRidgeClassifier(coefficients=(0.0,), **fields)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"must be \+1 or -1"):
        KernelRidgeClassifier(
            coefficients=(0.0, 0.0),
            training_data=((0.0,), (1.0,)),
            labels=(1, 0),
            regularization=1.0,
        )
    # A ragged row is the case the factory path cannot produce and the decision function
    # does not catch: it compares the incoming feature count against the first row's
    # alone, so a narrower row reaches the encoding and raises IndexError from it --
    # measured, on a training set of widths 2 and 1 -- while a wider one has its extra
    # features silently dropped: measured, a set of widths 1 and 2 decides a width-1 row
    # to 1.5362548828125, exactly as the truncated pair does. Neither is reported by any
    # other check, which is why the width belongs here.
    with pytest.raises(ValueError, match="same number of features"):
        KernelRidgeClassifier(
            coefficients=(0.0, 0.0),
            training_data=((0.0,), (1.0, 2.0)),
            labels=(1, -1),
            regularization=1.0,
        )
    with pytest.raises(ValueError, match="coefficient must be finite"):
        KernelRidgeClassifier(
            coefficients=(0.0, float("nan")),
            training_data=((0.0,), (1.0,)),
            labels=(1, -1),
            regularization=1.0,
        )
    with pytest.raises(ValueError, match="training feature must be finite"):
        KernelRidgeClassifier(
            coefficients=(0.0, 0.0),
            training_data=((0.0,), (float("inf"),)),
            labels=(1, -1),
            regularization=1.0,
        )
    with pytest.raises(ValueError, match="ridge weight must be positive"):
        KernelRidgeClassifier(
            coefficients=(0.0, 0.0),
            training_data=((0.0,), (1.0,)),
            labels=(1, -1),
            regularization=-1.0,
        )


def test_the_same_seed_replays_the_same_result_field_for_field() -> None:
    """One generator serves a run's entries in a fixed order, so a seed replays it."""
    data = torch.tensor(_FAR, dtype=torch.float64)

    first = quantum_kernel_matrix(data, shots=512, seed=_SEED)
    second = quantum_kernel_matrix(data, shots=512, seed=_SEED)

    assert first == second
    assert first.matrix == second.matrix
    assert first.shots == second.shots

    train = torch.tensor(_TRAIN, dtype=torch.float64)
    held_out = torch.tensor(_HELD_OUT, dtype=torch.float64)

    left = kernel_ridge_classifier(
        train, _LABELS, regularization=_RIDGE, shots=512, seed=_SEED
    )
    right = kernel_ridge_classifier(
        train, _LABELS, regularization=_RIDGE, shots=512, seed=_SEED
    )

    assert left == right
    assert left.decision_function(held_out, shots=512, seed=_SEED) == (
        right.decision_function(held_out, shots=512, seed=_SEED)
    )

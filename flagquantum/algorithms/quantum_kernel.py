"""Kernel entries estimated by a swap test, with a classical kernel ridge classifier on top.

A kernel entry is the squared overlap of two feature states, and this unit estimates it
with the swap test -- the kernel-matrix circuit of Havlíček, Córcoles, Temme, Harrow,
Kandala, Chow and Gambetta, "Supervised learning with quantum-enhanced feature spaces",
*Nature* **567**, 209-212 (2019), DOI 10.1038/s41586-019-0980-2. The classifier the
entries feed is a **classical** kernel ridge classifier: kernel ridge regression is a
textbook classical method, and the only quantum part of this unit is the kernel. Nothing
here pairs kernel ridge regression with a quantum kernel in the literature's name.

**The premise is the data-access model, and a gate-level implementation pays it.** The
kernel-matrix circuit estimates an entry from two feature states, and the statement that
this is cheap rests on the states being available: the data enters through a qRAM or an
amplitude-encoding unitary whose cost the kernel estimate does not count. This unit
supplies no such access model. Each feature state is built gate by gate from the
classical feature vector on every run, so the data-access cost is paid explicitly rather
than assumed away, and **no end-to-end advantage follows**. Two further statements are
theirs and not this module's, and neither is softened here. The classical hardness of
estimating these kernel entries is **a conjecture** in Havlíček et al., not a theorem.
The rigorous speed-up results for quantum kernel methods are for a **fault-tolerant
quantum computer** (Liu, Arunachalam and Temme, "A rigorous and robust quantum speed-up
in supervised machine learning", *Nature Physics* **17**, 1013-1017 (2021), DOI
10.1038/s41567-021-01287-z); the cited kernel-matrix construction itself is written for
noisy intermediate-scale devices and does not require fault tolerance.

**The feature map is this module's own angle encoding, not the cited paper's.** A
Hadamard on every wire, then a phase rotation carrying each feature on its own wire and
an entangling phase rotation on every pair of wires. The state is

    |Phi(x)> = prod_{j<k} exp(i (pi - x_j) (pi - x_k) Z_j Z_k)
               prod_k exp(i x_k Z_k) H^{tensor n} |0...0>,

so a feature is an angle in ``[0, 2*pi]`` and the pair angles are products of the two
features' complements. The map is chosen to be small, concrete and classically
computable, which is what makes the independent reference path in the tests possible;
it is not the map of Havlíček et al., whose hardness conjecture therefore says nothing
about this unit.

**The estimate is a sample, not a reading.** The swap test's ancilla is found set with
probability ``1/2 - 1/2 |<a|b>|^2``, so the readout inverts that relation on the share
of the sample that found it set. The sampling cost is the paper's own: **the cost
Havlíček et al. state is ``O(eps**-2)`` shots per kernel entry**, and an ``m`` by ``m``
kernel matrix therefore **``O(m**2 / eps**2)``** -- a concrete number rather than an
adjective. Nothing in this unit converts it into an accuracy: no error bound, no
confidence interval, no shot-selection rule and no repetition scheme is computed or
reported anywhere here, and the module does not predict how far an entry will sit from
the overlap it estimates.

**The classifier's fit is classical, and its predictions inherit the sample.** The dual
coefficients solve the ridge-regularised linear system of kernel ridge regression on the
training labels, and a prediction is the sign of the resulting decision function, which
is a weighted sum of estimated kernel entries. A point whose decision value is small
enough can therefore be decided differently by a different sample, and this module
computes no margin, no bound and no accuracy estimate. The tie rule at exactly zero is
stated on :meth:`KernelRidgeClassifier.predict`, because a sign has no value at zero and
the choice has to be made somewhere.

**Wire layout.** One kernel entry is one circuit: the ancilla on wire 0, the left feature
state on the next ``n`` wires and the right feature state on the last ``n``, where ``n``
is the feature count. Wire 0 leads every sample key as
:meth:`flagquantum.circuit.Circuit.counts` returns them, one character per wire with wire
0 the most significant.

This unit is demonstration scale. It makes no performance, capacity, or hardware claim,
and it does not select a runtime.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from ..circuit import Circuit

__all__ = [
    "KernelMatrixResult",
    "KernelRidgeClassifier",
    "kernel_ridge_classifier",
    "kernel_ridge_regression",
    "quantum_kernel_matrix",
]

# The default sample size, matching the other sampling units in this package.
_DEFAULT_SHOTS = 4096

# A kernel entry prepares two feature states plus an ancilla, so the circuit is
# ``2 * n + 1`` wires wide and the state it evolves grows as ``2 ** (2 * n + 1)``. The
# unit is bounded at three features, which is also the width the tests' classical
# reference enumerates.
_MAX_FEATURES = 3

# The angle encoding is written for features in ``[0, 2*pi]``: a feature is an angle,
# and the pair angle is the product of the two features' complements to ``pi``.
_MAX_FEATURE_ANGLE = 2.0 * math.pi


@dataclass(frozen=True, kw_only=True)
class KernelMatrixResult:
    """One sampled kernel matrix over a set of feature vectors.

    ``kw_only`` is not optional here: ``shots`` is a count and the matrix is a nested
    sequence of the same element type, so a positional spelling would let a caller
    transpose the two without an error.

    Attributes:
        matrix: One row per input feature vector, in the order the rows were given, one
            column per input feature vector. Entry ``(i, j)`` is the sampled estimate of
            ``|<Phi(x_i)|Phi(x_j)>|^2``: ``1 - 2 * share``, where ``share`` is the
            fraction of the sample whose swap-test ancilla was found set. It is an
            estimate, so it carries the sampler's error and is not a recomputation of
            the exact overlap; see the module docstring for what is and is not reported
            about that error. The range of an estimate is ``[-1, 1]`` by construction
            because ``share`` lies in ``[0, 1]``, and an estimate can be negative where
            the overlap it estimates is close to zero -- the readout is not clamped.
        shots: The number of samples every entry was estimated from.
    """

    matrix: tuple[tuple[float, ...], ...]
    shots: int

    def __post_init__(self) -> None:
        """Reject a matrix that could not be a kernel matrix of sampled entries.

        Raises:
            ValueError: If the matrix is empty, is not square, holds a non-finite
                entry, holds an entry outside ``[-1, 1]``, or if ``shots`` is not
                positive.
        """
        if not self.matrix:
            raise ValueError(
                "a kernel matrix needs at least one row, got none; with no feature "
                "vector there is no pair to estimate an entry for"
            )
        width = len(self.matrix[0])
        if any(len(row) != width for row in self.matrix) or len(self.matrix) != width:
            raise ValueError(
                "a kernel matrix is square, one row and one column per feature vector, "
                f"got {len(self.matrix)} row(s) of width(s) "
                f"{sorted({len(row) for row in self.matrix})}"
            )
        if not all(math.isfinite(value) for row in self.matrix for value in row):
            raise ValueError(
                "every kernel entry must be finite; a non-finite entry is not an "
                "estimate of a squared overlap"
            )
        if not all(-1.0 <= value <= 1.0 for row in self.matrix for value in row):
            raise ValueError(
                "every kernel entry is 1 - 2 * share with share in [0, 1], so it lies "
                "in [-1, 1]; an entry outside that range cannot come from a swap test"
            )
        if self.shots < 1:
            raise ValueError(
                f"a kernel entry is estimated from at least one shot, got "
                f"shots={self.shots}"
            )


@dataclass(frozen=True, kw_only=True)
class KernelRidgeClassifier:
    """A kernel ridge classifier fitted on a training set, with a quantum kernel.

    The classifier is classical: it holds the dual coefficients of a ridge-regularised
    kernel regression on the training labels, and it predicts by the sign of the
    decision function those coefficients define. What is quantum is the kernel the
    coefficients and the decision function are built from, which is estimated by the
    swap test -- see :func:`quantum_kernel_matrix` and the module docstring. A
    classifier built over the same training set with the same seed is the same
    classifier; a different sample is a different fit.

    ``kw_only`` is not optional here: the fields are three sequences of the same kinds
    of numbers, so a positional spelling would let a caller transpose the training data
    and the labels without an error.

    Attributes:
        coefficients: The dual coefficients, one per training feature vector, in
            training order. They solve
            ``(K + regularization * I) alpha = y`` for the training kernel ``K`` and
            the labels ``y``.
        training_data: The training feature vectors, one tuple of features per row, in
            the order the fit was given them and in the order ``coefficients`` and
            ``labels`` follow.
        labels: The training label of each row, each ``+1`` or ``-1``.
        regularization: The ridge weight the coefficients were solved at, positive.
    """

    coefficients: tuple[float, ...]
    training_data: tuple[tuple[float, ...], ...]
    labels: tuple[int, ...]
    regularization: float

    def __post_init__(self) -> None:
        """Reject a classifier whose fields do not describe one training set.

        Raises:
            ValueError: If no training row is carried; if the coefficients, the training
                rows and the labels disagree in length; if a label is not ``+1`` or
                ``-1``; if a coefficient or a feature is non-finite; or if the
                regularization is not positive.
        """
        if not self.training_data:
            raise ValueError(
                "a kernel ridge classifier is fitted on at least one training row, got "
                "none; with no row there is no dual coefficient to carry"
            )
        if not (len(self.coefficients) == len(self.training_data) == len(self.labels)):
            raise ValueError(
                "a dual coefficient, a training row and a label belong to each "
                "training point, so the three lengths must agree, got "
                f"{len(self.coefficients)} coefficient(s), {len(self.training_data)} "
                f"training row(s) and {len(self.labels)} label(s)"
            )
        if any(label not in (-1, 1) for label in self.labels):
            raise ValueError(
                "the classifier predicts by the sign of its decision function, so every "
                f"training label must be +1 or -1, got labels={self.labels}"
            )
        if not all(math.isfinite(value) for value in self.coefficients):
            raise ValueError(
                "every dual coefficient must be finite; a non-finite coefficient "
                "decides no point"
            )
        if not all(math.isfinite(value) for row in self.training_data for value in row):
            raise ValueError(
                "every training feature must be finite; a decision function evaluated "
                "at a non-finite row is not a decision"
            )
        if not math.isfinite(self.regularization) or self.regularization <= 0.0:
            raise ValueError(
                "the ridge weight must be positive and finite, got regularization="
                f"{self.regularization}; it is what keeps the training system solvable "
                "when the kernel matrix is singular"
            )

    def decision_function(
        self,
        data: torch.Tensor,
        *,
        shots: int = _DEFAULT_SHOTS,
        seed: int | None = None,
    ) -> tuple[float, ...]:
        """Return the decision value of every row of ``data``.

        The value of a row is ``sum_j alpha_j K(x, x_j)`` over the training rows, with
        every entry ``K(x, x_j)`` estimated by a swap test of its own. The values are
        therefore sampled rather than computed exactly: see the module docstring for
        what that does and does not report.

        Args:
            data: The feature vectors to decide, a real floating-point tensor of shape
                ``(m, n_features)`` with the same feature count the fit was given, at
                least one row, every feature in ``[0, 2*pi]`` and finite.
            shots: The number of samples each kernel entry is estimated from.
            seed: The sampler's seed, or ``None`` to draw from the ambient generator.
                The same seed and the same arguments replay the same values exactly.

        Returns:
            One decision value per row, in row order. The values are not probabilities
            and carry no interval.

        Raises:
            ValueError: If ``data`` fails the validation :func:`quantum_kernel_matrix`
                applies; if its feature count differs from the training feature count;
                or if ``shots`` is less than one.
        """
        rows, n_features = _validated_rows(data, shots)
        if n_features != len(self.training_data[0]):
            raise ValueError(
                "a classifier decides rows in the space it was trained in, so the "
                f"feature count must match the training set's, got {n_features} "
                f"against {len(self.training_data[0])}"
            )
        generator = _generator(seed)
        values: list[float] = []
        for row in rows:
            entries = [
                _kernel_entry(row, train, shots=shots, generator=generator)
                for train in self.training_data
            ]
            values.append(
                sum(
                    coefficient * entry
                    for coefficient, entry in zip(
                        self.coefficients, entries, strict=True
                    )
                )
            )
        return tuple(values)

    def predict(
        self,
        data: torch.Tensor,
        *,
        shots: int = _DEFAULT_SHOTS,
        seed: int | None = None,
    ) -> tuple[int, ...]:
        """Return the predicted label of every row of ``data``.

        The prediction is the sign of the decision value, and the sign of zero is not a
        label, so the rule at the boundary is stated rather than left to ``sign``: a row
        whose decision value is **zero or above** is predicted ``+1``, and a row whose
        decision value is below zero is predicted ``-1``. The rule is this module's
        choice, and the boundary can be reached, because a decision value is a sum of
        sampled entries rather than an exact quantity.

        Args:
            data: The feature vectors to predict, as in :meth:`decision_function`.
            shots: The number of samples each kernel entry is estimated from.
            seed: The sampler's seed, or ``None`` to draw from the ambient generator.

        Returns:
            One label per row, each ``+1`` or ``-1``, in row order.

        Raises:
            ValueError: If ``data`` or ``shots`` fails the validation
                :meth:`decision_function` applies.
        """
        return tuple(
            1 if value >= 0.0 else -1
            for value in self.decision_function(data, shots=shots, seed=seed)
        )


def quantum_kernel_matrix(
    data: torch.Tensor,
    *,
    shots: int = _DEFAULT_SHOTS,
    seed: int | None = None,
) -> KernelMatrixResult:
    """Estimate the kernel matrix of ``data`` by swap test, one entry per sample.

    Every entry ``(i, j)`` is estimated from a swap test of the feature states of rows
    ``i`` and ``j``, and the two triangles are the same estimate: the swap test for
    ``(i, j)`` and the one for ``(j, i)`` are the same experiment on the same two
    states, so the entry is sampled once and mirrored rather than sampled twice. The
    matrix is therefore symmetric by construction, which is not the same as being
    accurate: the estimate carries the sampler's error, and no error bound or
    confidence interval is computed or reported anywhere in this unit.

    The diagonal is sampled rather than assumed. Rows ``i`` and ``j`` that are equal
    give the same feature state twice, whose overlap is one, so the ancilla is never
    found set and the entry is one; the module runs that swap test like any other
    instead of writing one into the matrix.

    Args:
        data: The feature vectors, a real floating-point tensor of shape
            ``(m, n_features)`` with at least one row, at most three features, every
            feature finite and in ``[0, 2*pi]``.
        shots: The number of samples each entry is estimated from, at least one. Every
            entry of the returned matrix is a count over exactly this many samples.
        seed: The sampler's seed, or ``None`` to draw from the ambient generator. One
            generator serves every entry in the run, so the same seed and the same
            arguments replay the same matrix exactly.

    Returns:
        The sampled kernel matrix and the sample size it was estimated at. What the
        entries are estimates of, and what is not reported about their error, is
        stated in the module docstring and is not restated here.

    Raises:
        ValueError: If ``data`` is not a two-dimensional real floating-point tensor; if
            it holds no row; if it holds more than three features or a feature outside
            ``[0, 2*pi]``; if it holds a non-finite entry; or if ``shots`` is less than
            one.
    """
    rows, _ = _validated_rows(data, shots)
    generator = _generator(seed)
    estimates = [[0.0] * len(rows) for _ in rows]
    for left in range(len(rows)):
        for right in range(left, len(rows)):
            value = _kernel_entry(
                rows[left], rows[right], shots=shots, generator=generator
            )
            estimates[left][right] = value
            estimates[right][left] = value
    return KernelMatrixResult(
        matrix=tuple(tuple(row) for row in estimates), shots=shots
    )


def kernel_ridge_regression(
    kernel: torch.Tensor,
    targets: torch.Tensor,
    *,
    regularization: float,
) -> tuple[float, ...]:
    """Solve the dual of a ridge-regularised kernel regression.

    This is the classical textbook solve, and it is done here in double precision on
    the CPU whatever the caller holds the operands on: it is a linear system, not part
    of a state a circuit evolves. The system is ``(K + regularization * I) alpha =
    targets``, and the returned coefficients are ``alpha``.

    The ridge weight is what keeps the system solvable and what decides how closely the
    fit follows the training targets: a small weight follows them closely and makes the
    coefficients large where the kernel matrix is close to singular. This module states
    no rule for choosing it and supplies no default, because the choice belongs to the
    caller; :func:`kernel_ridge_classifier` takes it as a required argument for the same
    reason.

    Args:
        kernel: The kernel matrix, a square real floating-point tensor of ``m`` rows, as
            :func:`quantum_kernel_matrix` or a caller's own kernel returns one.
        targets: The regression targets, a real floating-point tensor of ``m`` entries,
            in the order the kernel's rows are in.
        regularization: The ridge weight, positive and finite.

    Returns:
        The ``m`` dual coefficients, in the order of the kernel's rows.

    Raises:
        ValueError: If ``kernel`` is not a square two-dimensional real floating-point
            tensor; if ``targets`` is not a one-dimensional real floating-point tensor
            of the same length; if either holds a non-finite entry; or if
            ``regularization`` is not positive and finite.
    """
    matrix, values = _validated_ridge_operands(kernel, targets, regularization)
    size = int(matrix.shape[0])
    identity = torch.eye(size, dtype=matrix.dtype, device=matrix.device)
    coefficients = torch.linalg.solve(matrix + regularization * identity, values)
    return tuple(float(coefficient) for coefficient in coefficients)


def kernel_ridge_classifier(
    training_data: torch.Tensor,
    labels: Sequence[int],
    *,
    regularization: float,
    shots: int = _DEFAULT_SHOTS,
    seed: int | None = None,
) -> KernelRidgeClassifier:
    """Fit a kernel ridge classifier on ``training_data`` and its labels.

    The fit is two steps and only the first is quantum: the training kernel is
    estimated by :func:`quantum_kernel_matrix`, and its dual coefficients are then
    solved by :func:`kernel_ridge_regression` on the labels. The returned classifier
    carries the training set, because deciding a new row needs the kernel between it and
    every training row.

    The labels are ``+1`` and ``-1`` rather than an arbitrary pair of classes, because
    the prediction is the sign of the decision function and the two classes therefore
    have to be the two signs.

    Args:
        training_data: The training feature vectors, a real floating-point tensor of
            shape ``(m, n_features)``, validated as in :func:`quantum_kernel_matrix`.
        labels: The training labels, one per row, each ``+1`` or ``-1``.
        regularization: The ridge weight, passed to
            :func:`kernel_ridge_regression`.
        shots: The number of samples each training kernel entry is estimated from.
        seed: The sampler's seed for the training kernel, or ``None`` to draw from the
            ambient generator.

    Returns:
        The fitted classifier. A different sample is a different fit: the coefficients
        come from an estimated kernel, so the same training set and the same ridge
        weight fitted from another seed can differ.

    Raises:
        ValueError: If ``training_data`` fails the validation
            :func:`quantum_kernel_matrix` applies; if ``labels`` does not carry exactly
            one ``+1`` or ``-1`` per row; or if ``regularization`` or ``shots`` fails
            the validation its own function applies.
    """
    rows, _ = _validated_rows(training_data, shots)
    training_labels = _validated_labels(labels, len(rows))
    # The ridge weight is checked before the kernel is sampled, so a weight that cannot
    # be solved at is refused without running the circuits first.
    _validated_regularization(regularization)
    estimated = quantum_kernel_matrix(training_data, shots=shots, seed=seed)
    kernel = torch.tensor(estimated.matrix, dtype=torch.float64)
    targets = torch.tensor(
        [float(label) for label in training_labels], dtype=torch.float64
    )
    coefficients = kernel_ridge_regression(
        kernel, targets, regularization=regularization
    )
    return KernelRidgeClassifier(
        coefficients=coefficients,
        training_data=rows,
        labels=training_labels,
        regularization=regularization,
    )


def _swap_test_overlap(
    circuit: Circuit,
    *,
    ancilla: int,
    left: Sequence[int],
    right: Sequence[int],
    shots: int,
    generator: torch.Generator | None,
) -> float:
    """Append the swap test to ``circuit`` and estimate the squared overlap from it.

    The swap test is a Hadamard on the ancilla, a controlled swap per wire pair, and a
    Hadamard on the ancilla again. It estimates the squared overlap of the two states
    and not their signed overlap: the ancilla is found set with probability
    ``1/2 - 1/2 |<a|b>|^2``, which is the same number for ``|<a|b>|`` and for
    ``-|<a|b>|`` and for ``i |<a|b>|``, so the readout below has no sign to report. The
    estimate is ``1 - 2 * share``, where ``share`` is the fraction of the sample that
    found the ancilla set: that is the relation above inverted on the sample's share
    rather than on the probability, so the estimate carries the sampler's error and no
    error bound.

    Two conditions are the caller's, and neither is checked here. **Both registers must
    already carry the two states** when this is called: the swaps act on whatever the
    registers hold. **The ancilla must be in ``|0>`` on entry**, the same contract
    :func:`~flagquantum.algorithms.primitives.oracle.append_multi_controlled_x` states
    for its ladder wires; an ancilla that enters in ``|1>`` shifts the readout instead
    of raising.

    Args:
        circuit: The circuit to extend.
        ancilla: The wire the Hadamards and the controlled swaps are controlled on.
        left: The wires carrying the first state, one per wire of the register.
        right: The wires carrying the second state, the same length as ``left`` and
            holding the same wires in the same order, so that ``left[k]`` and
            ``right[k]`` are the two wires of one controlled swap.
        shots: The number of samples to draw, at least one.
        generator: The sampler's generator, or ``None`` for the ambient one.

    Returns:
        The estimated squared overlap, in ``[-1, 1]``.

    Raises:
        ValueError: If the two registers are not the same width.
    """
    if len(left) != len(right):
        raise ValueError(
            "a swap test swaps one wire of the first register with one wire of the "
            "second, so the two registers must be the same width, got "
            f"{len(left)} and {len(right)}"
        )
    circuit.gate("h", ancilla)
    for first, second in zip(left, right, strict=True):
        circuit.gate("cswap", (ancilla, first, second))
    circuit.gate("h", ancilla)
    counts = circuit.counts(shots, generator=generator)[0]
    return 1.0 - 2.0 * _ancilla_share(counts, ancilla)


def _ancilla_share(counts: Mapping[str | int, int], ancilla: int) -> float:
    """Return the share of a sample whose ``ancilla`` bit is set.

    A sample key carries every wire, one character per wire with wire 0 the most
    significant, so the ancilla's outcome is one character of the key. The shares are
    taken over the sample the counts were drawn from, which is what makes the readout a
    share of exactly that many samples.

    Args:
        counts: The sample counts, keyed by the bit string of the whole register.
        ancilla: The wire to read.

    Returns:
        The fraction of the sample whose key has that wire set.
    """
    total = sum(counts.values())
    ones = sum(count for key, count in counts.items() if str(key)[ancilla] == "1")
    return ones / total


def _kernel_entry(
    left: Sequence[float],
    right: Sequence[float],
    *,
    shots: int,
    generator: torch.Generator | None,
) -> float:
    """Estimate one kernel entry, the squared overlap of two feature vectors.

    One circuit is built per entry: the ancilla on wire 0, the left feature state on the
    wires that follow it and the right feature state on the wires after those.

    Args:
        left: The first feature vector.
        right: The second feature vector.
        shots: The number of samples to draw, at least one.
        generator: The sampler's generator, or ``None`` for the ambient one.

    Returns:
        The estimated squared overlap of the two feature states.
    """
    n_features = len(left)
    left_wires = list(range(1, 1 + n_features))
    right_wires = list(range(1 + n_features, 1 + 2 * n_features))
    circuit = Circuit(1 + 2 * n_features)
    _append_feature_state(circuit, left, left_wires)
    _append_feature_state(circuit, right, right_wires)
    return _swap_test_overlap(
        circuit,
        ancilla=0,
        left=left_wires,
        right=right_wires,
        shots=shots,
        generator=generator,
    )


def _append_feature_state(
    circuit: Circuit, features: Sequence[float], wires: Sequence[int]
) -> None:
    """Append the angle-encoded feature state of ``features`` to ``circuit``.

    The state is a Hadamard on every wire, then a phase rotation carrying one feature on
    each of its own wires and an entangling phase rotation on every pair of them:

        |Phi(x)> = prod_{j<k} exp(i (pi - x_j)(pi - x_k) Z_j Z_k)
                   prod_k exp(i x_k Z_k) H^{tensor n} |0...0>.

    The gate spellings follow the layer's own conventions, ``RZ(theta) =
    exp(-i theta Z / 2)`` and ``RZZ(theta) = exp(-i theta Z_j Z_k / 2)``, so the phase
    rotation of feature ``k`` is ``rz`` at ``-2 x_k`` and the entangling one of the pair
    ``(j, k)`` is ``rzz`` at ``-2 (pi - x_j)(pi - x_k)``. Every term commutes with every
    other, so the order they are emitted in changes nothing about the state.

    Args:
        circuit: The circuit to extend.
        features: The feature vector, one feature per wire.
        wires: The wires to prepare, at most three and one per feature.
    """
    for wire in wires:
        circuit.gate("h", wire)
    for index, wire in enumerate(wires):
        circuit.gate("rz", wire, theta=-2.0 * features[index])
    for first in range(len(wires)):
        for second in range(first + 1, len(wires)):
            pair = (math.pi - features[first]) * (math.pi - features[second])
            circuit.gate("rzz", (wires[first], wires[second]), theta=-2.0 * pair)


def _validated_rows(
    data: torch.Tensor,
    shots: int,
) -> tuple[tuple[tuple[float, ...], ...], int]:
    """Validate a data matrix and return its rows and its feature count.

    The rows are returned as tuples of Python floats in double precision: the features
    are angles the gate sequence is built from, and that classical pass is where the
    whole classical cost of the unit sits.

    Args:
        data: The candidate feature vectors, one row per point.
        shots: The sample size the run will use, checked here because it is a property
            of the run rather than of the data.

    Returns:
        ``(rows, n_features)``, one tuple of features per row and the feature count.

    Raises:
        ValueError: If ``data`` is not a two-dimensional real floating-point tensor; if
            it holds no row; if it holds no feature, more than :data:`_MAX_FEATURES`
            features, a feature outside ``[0, 2*pi]`` or a non-finite feature; or if
            ``shots`` is less than one.
    """
    if not isinstance(data, torch.Tensor):
        raise ValueError(
            f"the feature vectors must be a torch.Tensor, got {type(data).__name__}; "
            "the feature states are built from feature vectors, not from a description "
            "of them"
        )
    if data.dim() != 2:
        raise ValueError(
            "the feature vectors must be a two-dimensional tensor, one row per point, "
            f"got shape {tuple(int(size) for size in data.shape)}"
        )
    if not data.is_floating_point():
        raise ValueError(
            "the feature vectors must be a real floating-point tensor, got dtype "
            f"{data.dtype}; a feature is an angle, and an integer tensor would be "
            "silently promoted"
        )
    if shots < 1:
        raise ValueError(
            f"a kernel entry is estimated from at least one shot, got shots={shots}; "
            "with no samples the ancilla is never found set and every entry would read "
            "as a complete overlap"
        )
    rows = tuple(
        tuple(float(value) for value in row)
        for row in data.detach().to(device="cpu", dtype=torch.float64)
    )
    if not rows:
        raise ValueError(
            "a kernel matrix needs at least one feature vector, got none; with no point "
            "there is no pair to estimate an entry for"
        )
    n_features = len(rows[0])
    if n_features < 1:
        raise ValueError(
            "a feature vector carries at least one feature, got none; a state with no "
            "wire encodes nothing"
        )
    if n_features > _MAX_FEATURES:
        raise ValueError(
            f"this unit is bounded at {_MAX_FEATURES} features, got {n_features}: one "
            "kernel entry prepares two feature states and an ancilla, so the circuit is "
            "2 * n + 1 wires wide and the state it evolves grows with the register's "
            "own square, which is the demonstration scale this unit is written at"
        )
    for row in rows:
        for value in row:
            if not math.isfinite(value):
                raise ValueError(
                    "every feature must be finite; a feature is an angle, and a "
                    "non-finite one makes the feature state undefined rather than "
                    "merely inaccurate"
                )
            if not 0.0 <= value <= _MAX_FEATURE_ANGLE:
                raise ValueError(
                    f"every feature must lie in [0, 2*pi], got {value}; the encoding "
                    "carries a feature as an angle and the entangling angle is the "
                    "product of two features' complements to pi"
                )
    return rows, n_features


def _validated_labels(labels: Sequence[int], n_rows: int) -> tuple[int, ...]:
    """Validate the training labels and return them as a tuple.

    Args:
        labels: The candidate labels, one per training row.
        n_rows: The number of training rows.

    Returns:
        The labels as a tuple of ``+1`` and ``-1``.

    Raises:
        ValueError: If the labels do not number one per row, or if one is not ``+1`` or
            ``-1``.
    """
    values = tuple(int(label) for label in labels)
    if len(values) != n_rows:
        raise ValueError(
            "a training label belongs to each training row, so the two must number the "
            f"same, got {len(values)} label(s) for {n_rows} row(s)"
        )
    if any(value not in (-1, 1) for value in values):
        raise ValueError(
            "the classifier predicts by the sign of its decision function, so every "
            f"training label must be +1 or -1, got {values}"
        )
    return values


def _validated_ridge_operands(
    kernel: torch.Tensor,
    targets: torch.Tensor,
    regularization: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate the ridge system's operands and return them in double precision.

    The solve is a classical linear solve, so it runs on the CPU in double precision
    whatever the caller holds the operands on.

    Args:
        kernel: The candidate kernel matrix.
        targets: The candidate targets.
        regularization: The ridge weight.

    Returns:
        ``(kernel, targets)`` as ``float64`` CPU tensors.

    Raises:
        ValueError: If ``kernel`` is not a square two-dimensional real floating-point
            tensor; if ``targets`` is not a one-dimensional real floating-point tensor
            of the same length; if either holds a non-finite entry; or if the ridge
            weight is not positive and finite.
    """
    if not isinstance(kernel, torch.Tensor) or not isinstance(targets, torch.Tensor):
        raise ValueError(
            "the ridge system is solved from a kernel matrix and a target vector, both "
            "torch.Tensor; a description of either is not one"
        )
    if not kernel.is_floating_point() or not targets.is_floating_point():
        raise ValueError(
            "the kernel matrix and the targets must both be real floating-point "
            f"tensors, got dtypes {kernel.dtype} and {targets.dtype}; an integer "
            "operand would be silently promoted"
        )
    _validated_regularization(regularization)
    matrix = kernel.detach().to(device="cpu", dtype=torch.float64)
    values = targets.detach().to(device="cpu", dtype=torch.float64)
    if matrix.dim() != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            "the kernel matrix must be a square two-dimensional tensor, one row and "
            f"one column per training point, got shape "
            f"{tuple(int(size) for size in kernel.shape)}"
        )
    if values.dim() != 1 or values.shape[0] != matrix.shape[0]:
        raise ValueError(
            "the targets must be one-dimensional, one target per kernel row, got "
            f"{tuple(int(size) for size in targets.shape)} against a kernel of "
            f"{int(matrix.shape[0])} rows"
        )
    if not bool(torch.isfinite(matrix).all()) or not bool(torch.isfinite(values).all()):
        raise ValueError(
            "the kernel matrix and the targets must be finite; a non-finite entry "
            "makes the solve fail rather than return a fitted model"
        )
    return matrix, values


def _validated_regularization(regularization: float) -> None:
    """Check the ridge weight a solve is about to use.

    The weight is checked by both callers that take one, and it is checked before the
    kernel is sampled as well as before the solve, so a weight the system cannot be
    solved at is refused without running the circuits first.

    Args:
        regularization: The ridge weight as given.

    Raises:
        ValueError: If the weight is not positive and finite.
    """
    if not math.isfinite(regularization) or regularization <= 0.0:
        raise ValueError(
            "the ridge weight must be positive and finite, got regularization="
            f"{regularization}; the weight is what keeps the system solvable, and the "
            "zero weight leaves a singular kernel matrix singular"
        )


def _generator(seed: int | None) -> torch.Generator | None:
    """Return the sampler's generator for a seed, or ``None`` for the ambient one.

    One generator serves a whole run, so a run's entries are drawn from a single stream
    in a fixed order and the same seed replays the same values exactly.

    Args:
        seed: The seed, or ``None``.

    Returns:
        A generator seeded with ``seed``, or ``None``.
    """
    return torch.Generator().manual_seed(seed) if seed is not None else None

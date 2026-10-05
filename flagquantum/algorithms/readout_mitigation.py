"""Readout-error mitigation: invert a declared classical confusion exactly.

The repository already *models* readout error and already *applies* it. A
:class:`~flagquantum.noise.model.ReadoutRule` states a classical confusion
matrix on a set of qubits, :meth:`~flagquantum.noise.model.NoiseModel`
carries a list of them, and ``apply_readout_probabilities``,
``apply_readout_expectation_z`` and ``apply_readout_samples`` push a true
distribution forward through that confusion after state evolution. What no
unit in the repository does is the other direction, and three units say so in
their own limitations: zero-noise extrapolation, probabilistic error
cancellation and Clifford data regression each estimate ``Tr(O rho)``, each
correctly refuses a noise model that declares a readout rule, and each records
that readout-error mitigation is absent. This unit is that absent half.

**The arithmetic, stated once.** A readout rule states ``P(observed | true)``
with row index ``true`` and column index ``observed``, and the forward path
computes ``p_obs = M p_true`` on the flattened index space. The correction is
therefore ``p_true = M^-1 p_obs``, and it is *exact linear algebra* rather than
an estimate: the only error the correction itself adds is that of inverting a
matrix, and the whole difficulty of the problem is in the two places the
inverse does not exist or does not help.

**Why the map factorizes, and why overlapping rules are refused.** Rules on
disjoint qubits compose into a tensor product of their maps, so the whole
``2**n_qubits`` inverse is never formed: each rule contributes one *block*, the
block's inverse is formed once at plan time, and the correction applies the
per-block inverse in the same index arrangement the forward path uses. Two
rules that name the same qubit are refused rather than composed. The forward
path would apply both, one after the other, on that qubit -- which is a
statement about a controller whose confusion was counted twice, not a
statement about a device -- and inverting the composed map would faithfully
return the inverse of a modelling mistake.

**What the inverse costs, as a number rather than a warning.** Correcting a
distribution inverts a matrix whose singular values are below one, so the
correction amplifies the sampling noise that is already in the input. The plan
reports ``sampling_overhead``, the induced 1-norm ``||M^-1||_1`` of the whole
``2**n_qubits`` map's inverse, and it bounds exactly one thing: for a measured
expectation ``w^T q`` of the corrected distribution with ``||w||_inf <= 1``,

    sd(w^T q) <= sampling_overhead / sqrt(N)

where ``N`` is the number of shots the observed frequencies were taken from.
The proof is one line -- the frequencies have covariance
``(diag(p) - p p^T) / N``, so ``w^T q = w^T M^-1 p_obs`` has variance at most
``||M^-T w||_inf^2 / N`` and ``||M^-T w||_inf <= ||w||_inf ||M^-1||_1`` -- and
it is the reason a correction is worth doing at all rather than a formality:
the same factor is what makes it useless on a device whose readout error is
too close to one half. The whole map's 1-norm is never formed directly: the map
is the tensor product of the blocks, an induced 1-norm is multiplicative under
that product, so the plan multiplies the blocks' own norms instead of building
a ``4**n_qubits`` matrix. That product is also the honest statement of where the
method stops being usable, because it grows with the number of qubits that carry
readout error rather than with the worst one: two qubits at a per-qubit
amplification of 1.35 amplify a whole-register expectation by 1.83, and the
same per-qubit confusion on sixteen qubits would amplify it by ``1.35**16``
without any single block being remarkable. That last number is arithmetic on
the per-qubit factor rather than a measurement, and no device was run to obtain
it.

**Why a near-singular block is refused by name.** A block whose smallest
singular value reaches the declared floor has an inverse whose amplification
is at least the reciprocal of that floor, so the correction returns a
combination of measured frequencies with weights larger than any statement the
shots can support. Refusing names the qubits, the singular value that was
measured and the floor that was declared, in that order, so a caller can tell
a device whose readout is broken from a tolerance that is set too tightly. The
default floor is ``1e-6``, the same value the cancellation unit checks a
vanishing transfer eigenvalue against.

**What the estimate is not.** The output is a corrected *distribution*, and
unlike a density matrix it is under no obligation to be non-negative: the
inverse of a stochastic matrix has negative entries whenever the block's
confusion is invertible and not a permutation, so a corrected distribution
routinely carries negative mass. Clipping that mass to zero would return a
different vector while keeping the name, so the negative mass is summed and
reported, and ``is_physical`` states whether the corrected vector happened to
land in the simplex rather than asserting that it must. The unit also assumes
the declared confusion is the device's confusion: a miscalibrated, drifting or
correlated-in-a-way-the-model-does-not-carry readout error survives the
inversion untouched, so a corrected distribution close to the ideal one is a
statement about the model before it is a statement about a device. Only the
computational basis is corrected, because that is the only basis a classical
confusion matrix acts in.

**What is not here.** No interpolation or smoothing of a corrected
distribution, no maximum-likelihood or constrained fit that would restore
positivity, no tensored-correction approximation for a block wider than the
ceiling, and no noise-aware readout assignment beyond the declared model.
Correcting an expectation with a correlated block is available through the
corrected distribution and is not offered as a separate shortcut, because a
shortcut would need its own statement of which observable it read.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from flagquantum.errors import CapabilityError
from flagquantum.noise.model import (
    CorrelatedReadoutError,
    NoiseModel,
    ReadoutError,
    ReadoutRule,
)

__all__ = (
    "MAX_CORRELATED_BLOCK_QUBITS",
    "NORMALIZATION_TOLERANCE",
    "READOUT_MITIGATION_ASSUMPTIONS",
    "READOUT_MITIGATION_LIMITATIONS",
    "READOUT_MITIGATION_SCHEMA",
    "SINGULAR_VALUE_FLOOR",
    "ReadoutBlock",
    "ReadoutMitigationPlan",
    "ReadoutMitigationResult",
    "plan_readout_mitigation",
    "run_readout_mitigation",
    "run_readout_mitigation_counts",
)

READOUT_MITIGATION_SCHEMA = "flagquantum.readout_mitigation_plan.v1"

#: Smallest singular value a block's confusion may have and still be inverted.
#: Chosen to match the tolerance the probabilistic-error-cancellation unit
#: checks a vanishing transfer eigenvalue against, so one device is not
#: admitted by one mitigation method and refused by another for the same
#: reason.
SINGULAR_VALUE_FLOOR = 1e-6

#: How far a supplied distribution may be from summing to one. Chosen to match
#: the completeness tolerance `KrausChannel` validates a channel against, so
#: one arithmetic standard covers both directions of the same statement, and
#: chosen no tighter than that because a caller who accumulates frequencies in
#: float32 arrives about 2e-8 from one and must not be refused for it.
NORMALIZATION_TOLERANCE = 1e-6

#: Widest correlated block this unit will invert. A block of ``k`` qubits needs
#: a ``2**k`` by ``2**k`` float64 matrix and its singular values, so the cost
#: is quartic in ``2**k`` and the ceiling is a memory statement rather than an
#: accuracy one: ten qubits is 8 MiB, and a device-wide correlated profile is
#: a different algorithm.
MAX_CORRELATED_BLOCK_QUBITS = 10

READOUT_MITIGATION_ASSUMPTIONS = (
    "The declared confusion matrices are the device's readout behaviour, in "
    "the computational basis, and they are the only source of the correction.",
    "The observed frequencies were taken from the same device the model "
    "describes, and readout confusion is independent of the state that was "
    "measured.",
    "A qubit carrying no readout rule reads perfectly, because its block's "
    "inverse is the identity and the correction leaves it untouched.",
)

READOUT_MITIGATION_LIMITATIONS = (
    "The corrected distribution is a quasi-probability and may carry negative "
    "mass; the negative mass is reported and never clipped.",
    "The correction is exact only for the declared model: a miscalibrated, "
    "drifting, or uncarried-correlated readout error survives the inversion.",
    "The sampling overhead bounds the variance amplification of a corrected "
    "expectation and is not a measured variance, and it is reported whether or "
    "not a shot count was supplied.",
    "No positivity-restoring or maximum-likelihood fit is applied, and no "
    "smoothing or interpolation of the corrected distribution is offered.",
    "Only the computational basis is corrected, and only over the qubits the "
    "plan covers.",
    "No performance, scaling, hardware or fault-tolerance claim follows from "
    "the demonstration scale of two to four qubits on the CPU path.",
)


def _probability_matrix(
    error: ReadoutError | CorrelatedReadoutError,
    owner: str,
) -> torch.Tensor:
    matrix = torch.as_tensor(error.probabilities, dtype=torch.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{owner} must be square")
    return matrix


@dataclass(frozen=True)
class ReadoutBlock:
    """One set of qubits whose readout confusion does not factorize further.

    Two matrices describe a block and the difference between them is a
    transpose that is easy to lose. ``probabilities`` is the confusion *as
    declared*, with the row index the true value and the column index the
    observed one, which is how a calibration table is written down.
    ``correction`` is the confusion as an *operator on a distribution*, and a
    distribution in this repository's flat index space is a vector whose first
    qubit is the most significant bit. The two are related by
    ``correction = inv(probabilities.T)``, and the whole correction is exact
    only because the forward path applies ``probabilities.T`` to that same
    vector.

    Attributes:
        qubits: The qubits the block covers, in the order the block's index is
            built from: the first qubit is the most significant bit, which is
            the convention :meth:`NoiseModel.apply_readout_probabilities` uses.
        probabilities: The block's forward map, row index ``true`` and column
            index ``observed``, exactly as the rule declared it.
        correction: The operator that sends an observed block distribution to a
            true one, ``inv(probabilities.T)``. It is what the mitigation
            applies, and it is what ``sampling_overhead`` is measured from.
        smallest_singular_value: The smallest singular value of
            ``probabilities``. The block is invertible only while this is
            above the floor the plan was built with, and it is the quantity the
            refusal names. A matrix and its transpose share their singular
            values, so the floor means the same thing whichever of the two is
            read.
        condition_number: ``s_max / s_min`` of ``probabilities``, which is also
            ``correction``'s.
        sampling_overhead: The induced 1-norm of ``correction``, the largest
            absolute column sum of the operator that is actually applied. It
            bounds the standard deviation of any corrected expectation with
            weights bounded by one, and it is multiplicative over blocks, which
            is what makes the plan's own figure a product.
    """

    qubits: tuple[int, ...]
    probabilities: torch.Tensor
    correction: torch.Tensor
    smallest_singular_value: float
    condition_number: float
    sampling_overhead: float

    def __post_init__(self) -> None:
        if not self.qubits:
            raise ValueError("a readout block must cover at least one qubit")
        if len(set(self.qubits)) != len(self.qubits):
            raise ValueError("a readout block cannot name a qubit twice")
        if any(not isinstance(qubit, int) or qubit < 0 for qubit in self.qubits):
            raise ValueError("readout block qubits must be non-negative integers")
        size = 2 ** len(self.qubits)
        for name in ("probabilities", "correction"):
            matrix = getattr(self, name)
            if matrix.shape != (size, size):
                raise ValueError(
                    f"a block of {len(self.qubits)} qubits needs a {size} by "
                    f"{size} {name}, got {tuple(matrix.shape)}"
                )
            if matrix.dtype != torch.float64:
                raise ValueError(f"a block's {name} must be float64")
        if not self.smallest_singular_value > 0:
            raise ValueError(
                "a block whose smallest singular value is not positive has no "
                "inverse to apply"
            )
        if not self.condition_number >= 1:
            raise ValueError("a condition number cannot be below one")
        if not self.sampling_overhead >= 1:
            raise ValueError(
                "the induced 1-norm of a correction is at least one, because "
                "every row of the inverse of a row-stochastic matrix sums to "
                "one, so the largest absolute column sum cannot be below one"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the block as plain JSON-serializable data."""

        return {
            "qubits": list(self.qubits),
            "probabilities": [
                [float(value) for value in row] for row in self.probabilities
            ],
            "smallest_singular_value": float(self.smallest_singular_value),
            "condition_number": float(self.condition_number),
            "sampling_overhead": float(self.sampling_overhead),
        }


@dataclass(frozen=True)
class ReadoutMitigationPlan:
    """How a declared model's readout confusion is to be inverted.

    The plan is a value: it is derived from the model's readout rules and the
    qubit count alone, it carries the identity of exactly those inputs, and two
    requests differing only in a confusion matrix have different identities.

    Attributes:
        n_qubits: The width the correction is defined over.
        blocks: The disjoint blocks the map factorizes into, ordered by their
            first qubit so the identity does not depend on rule order.
        covered_qubits: The qubits some rule names, in increasing order.
        uncovered_qubits: The qubits no rule names. Their inverse is the
            identity, which is an assumption the plan records rather than a
            measurement.
        sampling_overhead: The induced 1-norm of the whole map's inverse, which
            is the *product* of the blocks' ``sampling_overhead`` values, or
            one when nothing is covered. It is the quantity the standard error
            bound divides by, so it is the whole map's and not the worst
            block's: a device with readout error on many qubits is not bounded
            by its best-invertible one.
        condition_number: The whole map's condition number, the product of the
            blocks' condition numbers, or one when nothing is covered.
        singular_value_floor: The floor the blocks were admitted against.
        identity: ``sha256`` over the schema, the width, the floor and the
            blocks' qubits and forward matrices.
    """

    n_qubits: int
    blocks: tuple[ReadoutBlock, ...]
    covered_qubits: tuple[int, ...]
    uncovered_qubits: tuple[int, ...]
    sampling_overhead: float
    condition_number: float
    singular_value_floor: float
    identity: str

    def __post_init__(self) -> None:
        if not isinstance(self.n_qubits, int) or self.n_qubits < 1:
            raise ValueError("a readout plan needs a positive qubit count")
        blocks = tuple(self.blocks)
        if blocks != tuple(sorted(blocks, key=lambda block: block.qubits[0])):
            raise ValueError("readout blocks must be ordered by their first qubit")
        named: list[int] = []
        for block in blocks:
            for qubit in block.qubits:
                if qubit >= self.n_qubits:
                    raise ValueError(
                        f"readout qubit {qubit} is outside a {self.n_qubits}-qubit "
                        "plan"
                    )
            named.extend(block.qubits)
        if len(set(named)) != len(named):
            raise ValueError(
                "readout blocks must be disjoint, and a qubit appears in two " "of them"
            )
        if tuple(sorted(named)) != tuple(self.covered_qubits):
            raise ValueError("covered_qubits must be the blocks' qubits, sorted")
        expected_uncovered = tuple(
            qubit for qubit in range(self.n_qubits) if qubit not in set(named)
        )
        if tuple(self.uncovered_qubits) != expected_uncovered:
            raise ValueError(
                "uncovered_qubits must be the qubits no block names, in "
                "increasing order"
            )
        overhead = 1.0
        condition = 1.0
        for block in blocks:
            overhead *= block.sampling_overhead
            condition *= block.condition_number
        if self.sampling_overhead != overhead:
            raise ValueError(
                "a plan's sampling overhead is the product of its blocks', "
                "because the whole map's inverse is the blocks' inverses "
                "tensored together and an induced 1-norm is multiplicative "
                "under that product"
            )
        if self.condition_number != condition:
            raise ValueError(
                "a plan's condition number is the product of its blocks', for "
                "the same reason its sampling overhead is"
            )
        if not self.identity:
            raise ValueError("a plan must carry the identity of its inputs")

    @property
    def covered(self) -> bool:
        """Whether the plan corrects anything at all."""

        return bool(self.blocks)

    def to_dict(self) -> dict[str, Any]:
        """Return the plan as plain JSON-serializable data, identity included."""

        return {
            "schema": READOUT_MITIGATION_SCHEMA,
            "n_qubits": self.n_qubits,
            "blocks": [block.to_dict() for block in self.blocks],
            "covered_qubits": list(self.covered_qubits),
            "uncovered_qubits": list(self.uncovered_qubits),
            "sampling_overhead": float(self.sampling_overhead),
            "condition_number": float(self.condition_number),
            "singular_value_floor": float(self.singular_value_floor),
            "identity": self.identity,
        }

    def summary(self) -> dict[str, Any]:
        """Return the plan as a report rather than as a payload."""

        return {
            "identity": self.identity,
            "n_qubits": self.n_qubits,
            "n_blocks": len(self.blocks),
            "block_qubits": [list(block.qubits) for block in self.blocks],
            "block_sampling_overheads": [
                block.sampling_overhead for block in self.blocks
            ],
            "covered_qubits": list(self.covered_qubits),
            "uncovered_qubits": list(self.uncovered_qubits),
            "sampling_overhead": self.sampling_overhead,
            "condition_number": self.condition_number,
            "singular_value_floor": self.singular_value_floor,
            "limiting_block": (
                min(
                    self.blocks,
                    key=lambda block: block.smallest_singular_value,
                ).to_dict()
                if self.blocks
                else None
            ),
        }


@dataclass(frozen=True)
class ReadoutMitigationResult:
    """A corrected distribution together with what qualifies it.

    Attributes:
        probabilities: The corrected distribution, on the same batch shape and
            index convention as the observed one. It is float64 whatever the
            observed distribution's dtype was, because inverting a confusion
            matrix in the precision the confusion was declared in would report
            the arithmetic's error as the device's.
        observed: The distribution that was corrected, carried so the two are
            read together.
        plan: The plan that was applied, with its identity and its blocks.
        negative_mass: The sum of the corrected vector's negative entries,
            reported rather than clipped. It is a sum over the flattened index
            space, summed over the batch axis.
        total_variation: ``||corrected - observed||_1`` over the flattened
            index space, summed over the batch axis. It is how far the
            correction moved the distribution, and it is zero exactly when the
            correction was the identity.
        is_physical: Whether every corrected entry is within the normalization
            tolerance of non-negative and the vector sums to one. A method that
            always answered ``True`` would be reporting a clipped vector under
            the name of an inverted one.
        shots: The shot count the observed frequencies were taken from, or
            ``None`` when the caller supplied a distribution rather than
            counts.
        standard_error_bound: ``sampling_overhead / sqrt(shots)``, the bound on
            the standard deviation of a corrected expectation whose weights are
            bounded by one, or ``None`` when no shot count was supplied.
        assumptions: What had to hold for the correction to mean anything.
        limitations: What the correction does not carry.
    """

    probabilities: torch.Tensor
    observed: torch.Tensor
    plan: ReadoutMitigationPlan
    negative_mass: float
    total_variation: float
    is_physical: bool
    shots: int | None = None
    standard_error_bound: float | None = None
    assumptions: tuple[str, ...] = READOUT_MITIGATION_ASSUMPTIONS
    limitations: tuple[str, ...] = READOUT_MITIGATION_LIMITATIONS

    def __post_init__(self) -> None:
        if not isinstance(self.plan, ReadoutMitigationPlan):
            raise TypeError("a result needs the plan it was corrected by")
        if self.probabilities.shape != self.observed.shape:
            raise ValueError(
                "a corrected distribution has the observed one's shape, got "
                f"{tuple(self.probabilities.shape)} against "
                f"{tuple(self.observed.shape)}"
            )
        if self.probabilities.dtype != torch.float64:
            raise ValueError("a corrected distribution is float64")
        if not bool(torch.isfinite(self.probabilities).all()):
            raise ValueError("a corrected distribution must be finite")
        if self.negative_mass < 0:
            raise ValueError("a negative mass is a sum of negative entries")
        if self.total_variation < 0:
            raise ValueError("a total variation is a norm")
        if self.shots is not None and self.shots < 1:
            raise ValueError("a shot count must be positive")
        if (self.standard_error_bound is None) != (self.shots is None):
            raise ValueError(
                "a standard error bound is reported exactly when a shot count "
                "was supplied, because it is the shot count it is divided by"
            )
        if not self.assumptions:
            raise ValueError("a result must carry the assumptions it rests on")
        if not self.limitations:
            raise ValueError("a result must carry its limitations")

    def to_dict(self) -> dict[str, Any]:
        """Return the result as plain JSON-serializable data."""

        return {
            "schema": READOUT_MITIGATION_SCHEMA + ".result",
            "plan_identity": self.plan.identity,
            "probabilities": [float(value) for value in self.probabilities.reshape(-1)],
            "observed": [float(value) for value in self.observed.reshape(-1)],
            "negative_mass": self.negative_mass,
            "total_variation": self.total_variation,
            "is_physical": self.is_physical,
            "sampling_overhead": self.plan.sampling_overhead,
            "condition_number": self.plan.condition_number,
            "shots": self.shots,
            "standard_error_bound": self.standard_error_bound,
        }


def _build_block(
    qubits: tuple[int, ...],
    matrix: torch.Tensor,
    floor: float,
) -> ReadoutBlock:
    singular_values = torch.linalg.svdvals(matrix)
    smallest = float(singular_values.min())
    if smallest <= floor:
        raise CapabilityError(
            "readout confusion on qubits "
            f"{list(qubits)} has smallest singular value {smallest:.6e}, at or "
            f"below the declared floor {floor:.1e}, so its inverse would "
            "amplify sampling noise by at least "
            f"{1.0 / floor:.1e} and the correction would report weights no "
            "shot count can support"
        )
    # `matrix` is the confusion as declared -- row index true, column index
    # observed -- while the vector it is applied to carries the block's first
    # qubit as its most significant bit. So the operator that corrects a
    # distribution is the inverse of the transposed matrix.
    correction = torch.linalg.inv(matrix.T)
    return ReadoutBlock(
        qubits=qubits,
        probabilities=matrix,
        correction=correction,
        smallest_singular_value=smallest,
        condition_number=float(singular_values.max()) / smallest,
        sampling_overhead=float(correction.abs().sum(dim=0).max()),
    )


def plan_readout_mitigation(
    noise_model: NoiseModel,
    *,
    n_qubits: int,
    singular_value_floor: float = SINGULAR_VALUE_FLOOR,
) -> ReadoutMitigationPlan:
    """Build the correction a declared noise model's readout rules imply.

    Args:
        noise_model: The model whose ``readout_rules`` are inverted. Its
            channel rules are not read: this unit corrects measurement and
            says nothing about gate noise.
        n_qubits: The width the correction is defined over.
        singular_value_floor: The smallest singular value a block may have and
            still be inverted.

    Returns:
        The plan, carrying one block per maximal set of qubits whose confusion
        does not factorize further.

    Raises:
        TypeError: If ``noise_model`` is not a
            :class:`~flagquantum.noise.model.NoiseModel`.
        ValueError: If ``n_qubits`` is not a positive integer, a rule names a
            qubit outside the width, two rules name the same qubit, or the model
            declares no readout rule at all.
        CapabilityError: If a block's smallest singular value is at or below
            the floor, or a correlated block is wider than
            ``MAX_CORRELATED_BLOCK_QUBITS``.
    """

    if not isinstance(noise_model, NoiseModel):
        raise TypeError("readout mitigation reads a NoiseModel's readout rules")
    if not isinstance(n_qubits, int) or isinstance(n_qubits, bool) or n_qubits < 1:
        raise ValueError("readout mitigation needs a positive qubit count")
    if not isinstance(singular_value_floor, float) or not (
        0 < singular_value_floor < 1
    ):
        raise ValueError("a singular value floor must be a float in (0, 1)")
    if not noise_model.readout_rules:
        raise ValueError(
            "the declared model carries no readout rule, so its confusion is "
            "the identity and the correction would be the observed "
            "distribution under another name"
        )

    blocks: list[ReadoutBlock] = []
    owner: dict[int, int] = {}
    for index, rule in enumerate(noise_model.readout_rules):
        if not isinstance(rule, ReadoutRule):
            raise TypeError(
                "readout_rules must hold ReadoutRule records, got "
                f"{type(rule).__name__}"
            )
        if isinstance(rule.error, CorrelatedReadoutError):
            if len(rule.wires) != rule.error.n_wires:
                raise ValueError(
                    f"a correlated readout rule on {len(rule.wires)} qubits "
                    f"declares a {rule.error.n_wires}-qubit matrix"
                )
            if len(rule.wires) > MAX_CORRELATED_BLOCK_QUBITS:
                raise CapabilityError(
                    f"a correlated readout block of {len(rule.wires)} qubits "
                    "exceeds the "
                    f"{MAX_CORRELATED_BLOCK_QUBITS}-qubit ceiling: inverting it "
                    "needs a "
                    f"{2 ** len(rule.wires)} by {2 ** len(rule.wires)} "
                    "float64 matrix and its singular values"
                )
            if any(qubit >= n_qubits for qubit in rule.wires):
                outside = sorted({qubit for qubit in rule.wires if qubit >= n_qubits})
                raise ValueError(
                    f"correlated readout qubit(s) {outside} are outside the "
                    f"{n_qubits}-qubit circuit"
                )
            candidate = [
                _build_block(
                    tuple(rule.wires),
                    _square_matrix(
                        rule,
                        2**rule.error.n_wires,
                        f"a rule on {len(rule.wires)} qubits",
                    ),
                    singular_value_floor,
                )
            ]
        else:
            # A single-qubit confusion on several qubits is not a correlated
            # error: the forward path applies the same 2 by 2 matrix to each of
            # those qubits independently. Splitting into singleton blocks is
            # what keeps the correction the exact inverse of the forward path
            # rather than the inverse of a correlated map that was never
            # applied.
            matrix = _square_matrix(rule, 2, "a single-qubit readout rule")
            candidate = [
                _build_block((qubit,), matrix, singular_value_floor)
                for qubit in _checked_qubits(rule, n_qubits)
            ]
        for block in candidate:
            for qubit in block.qubits:
                if qubit in owner:
                    raise ValueError(
                        f"readout qubit {qubit} is named by readout rule "
                        f"{owner[qubit]} and again by readout rule {index}, so "
                        "the forward path would apply readout confusion twice "
                        "on it; the correction refuses to invert a confusion "
                        "that was counted twice"
                    )
                owner[qubit] = index
            blocks.append(block)

    blocks.sort(key=lambda block: block.qubits[0])
    covered = tuple(sorted(owner))
    uncovered = tuple(qubit for qubit in range(n_qubits) if qubit not in owner)
    frozen = tuple(blocks)
    payload = {
        "schema": READOUT_MITIGATION_SCHEMA,
        "n_qubits": n_qubits,
        "singular_value_floor": singular_value_floor,
        "blocks": [
            {
                "qubits": list(block.qubits),
                "probabilities": [
                    [float(value) for value in row] for row in block.probabilities
                ],
            }
            for block in frozen
        ],
    }
    identity = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return ReadoutMitigationPlan(
        n_qubits=n_qubits,
        blocks=frozen,
        covered_qubits=covered,
        uncovered_qubits=uncovered,
        sampling_overhead=math.prod(block.sampling_overhead for block in frozen),
        condition_number=math.prod(block.condition_number for block in frozen),
        singular_value_floor=singular_value_floor,
        identity=identity,
    )


def _square_matrix(
    rule: ReadoutRule,
    expected: int,
    owner: str,
) -> torch.Tensor:
    matrix = _probability_matrix(rule.error, "a readout confusion matrix")
    if matrix.shape != (expected, expected):
        raise ValueError(
            f"{owner} needs a {expected} by {expected} confusion matrix, got "
            f"{tuple(matrix.shape)}"
        )
    return matrix


def _checked_qubits(rule: ReadoutRule, n_qubits: int) -> tuple[int, ...]:
    if not rule.wires:
        raise ValueError("a readout rule must name at least one qubit")
    for qubit in rule.wires:
        if not isinstance(qubit, int) or isinstance(qubit, bool) or qubit < 0:
            raise ValueError("a readout rule's qubits must be non-negative integers")
        if qubit >= n_qubits:
            raise ValueError(f"readout qubit {qubit} is outside the circuit")
    return tuple(rule.wires)


def _apply_block(
    observed: torch.Tensor,
    block: ReadoutBlock,
    n_qubits: int,
) -> torch.Tensor:
    """Apply one block's correction in the forward path's index arrangement.

    The block's qubits are moved to the end of the index list, flattened in
    their declared order -- which is the block's own most-significant-first
    order -- corrected as one vector, and moved back. Qubits outside the block
    are carried along untouched, so a caller reading the code can see that an
    uncovered qubit is a spectator rather than having to trust it.
    """

    values = observed.reshape(*observed.shape[:-1], *((2,) * n_qubits))
    batch_dims = values.ndim - n_qubits
    selected = [batch_dims + qubit for qubit in block.qubits]
    unselected = [
        batch_dims + qubit for qubit in range(n_qubits) if qubit not in block.qubits
    ]
    permutation = [*range(batch_dims), *unselected, *selected]
    inverse_permutation = [permutation.index(index) for index in range(values.ndim)]
    arranged = values.permute(permutation)
    prefix = arranged.shape[: -len(block.qubits)]
    flat = arranged.reshape(*prefix, 2 ** len(block.qubits))
    corrected = flat @ block.correction.mT
    corrected = corrected.reshape(*prefix, *((2,) * len(block.qubits)))
    return corrected.permute(inverse_permutation).reshape_as(observed)


def _check_observed(
    observed: torch.Tensor, plan: ReadoutMitigationPlan
) -> torch.Tensor:
    values = torch.as_tensor(observed, dtype=torch.float64)
    if values.ndim < 1:
        raise ValueError(
            "an observed distribution ends in a flattened index dimension of "
            f"length {2 ** plan.n_qubits}"
        )
    if values.shape[-1] != 2**plan.n_qubits:
        raise ValueError(
            f"an observed distribution for {plan.n_qubits} qubits ends in "
            f"{2 ** plan.n_qubits} entries, got {int(values.shape[-1])}"
        )
    if not bool(torch.isfinite(values).all()):
        raise ValueError("an observed distribution must be finite")
    if bool(torch.any(values < 0)):
        raise ValueError(
            "an observed distribution is a frequency vector and cannot carry a "
            "negative entry"
        )
    sums = values.reshape(-1, values.shape[-1]).sum(dim=-1)
    if bool(torch.any(sums <= 0)):
        raise ValueError(
            "an observed distribution must have a positive total, and a row of "
            "it sums to zero"
        )
    if not bool(
        torch.allclose(
            sums, torch.ones_like(sums), atol=NORMALIZATION_TOLERANCE, rtol=0
        )
    ):
        worst = float((sums - 1).abs().max())
        raise ValueError(
            "an observed distribution must sum to one on every row, and the "
            f"furthest row is {worst:.3e} from one, beyond the "
            f"{NORMALIZATION_TOLERANCE:.1e} this unit admits; divide counts by "
            "their total before mitigating them rather than letting this unit "
            "renormalize silently"
        )
    return values


def run_readout_mitigation(
    observed: torch.Tensor,
    plan: ReadoutMitigationPlan,
    *,
    shots: int | None = None,
) -> ReadoutMitigationResult:
    """Correct an observed distribution by the plan's inverse.

    Args:
        observed: The measured distribution, ending in ``2 ** n_qubits``
            entries in the index convention
            :meth:`NoiseModel.apply_readout_probabilities` produces, with any
            number of leading batch dimensions. Every row must sum to one.
        plan: The plan to apply, from :func:`plan_readout_mitigation`.
        shots: The number of shots ``observed`` was taken from, when it came
            from shots. Supplying it adds ``standard_error_bound``; omitting it
            leaves the overhead reported without a variance statement.

    Returns:
        The corrected distribution with the negative mass, the total variation
        and the plan it rests on.

    Raises:
        TypeError: If ``plan`` is not a
            :class:`ReadoutMitigationPlan`.
        ValueError: If the observed distribution is not finite, not
            non-negative, not normalized, or the wrong width; or if ``shots``
            is not a positive integer.
    """

    if not isinstance(plan, ReadoutMitigationPlan):
        raise TypeError("readout mitigation applies a ReadoutMitigationPlan")
    if shots is not None and (
        not isinstance(shots, int) or isinstance(shots, bool) or shots < 1
    ):
        raise ValueError("a shot count must be a positive integer")
    values = _check_observed(observed, plan)
    corrected = values
    for block in plan.blocks:
        corrected = _apply_block(corrected, block, plan.n_qubits)
    negative_mass = float(
        corrected.clamp(max=0).abs().reshape(-1, corrected.shape[-1]).sum()
    )
    total_variation = float(
        (corrected - values).abs().reshape(-1, values.shape[-1]).sum()
    )
    rows = corrected.reshape(-1, corrected.shape[-1])
    physical = bool(
        torch.all(rows >= -NORMALIZATION_TOLERANCE)
        and torch.allclose(
            rows.sum(dim=-1),
            torch.ones(rows.shape[0], dtype=torch.float64),
            atol=NORMALIZATION_TOLERANCE,
            rtol=0,
        )
    )
    return ReadoutMitigationResult(
        probabilities=corrected,
        observed=values,
        plan=plan,
        negative_mass=negative_mass,
        total_variation=total_variation,
        is_physical=physical,
        shots=shots,
        standard_error_bound=(
            None if shots is None else plan.sampling_overhead / shots**0.5
        ),
    )


def run_readout_mitigation_counts(
    counts: Mapping[str, int],
    plan: ReadoutMitigationPlan,
    *,
    n_qubits: int | None = None,
) -> ReadoutMitigationResult:
    """Correct a shot histogram, keyed by bit string, by the plan's inverse.

    The key is the measured bit string with qubit ``0`` leftmost, which is the
    convention the flat index space uses: the key ``"010"`` is qubit ``1`` set
    and its flat index is ``2``. The histogram is normalized by its own total,
    and the total is carried into the result as ``shots``, so the standard
    error bound is stated for the shot count the histogram actually holds.

    Args:
        counts: Measured bit strings against counts. Every key must be a bit
            string of exactly the plan's width, and no count may be negative.
        plan: The plan to apply.
        n_qubits: Accepted only to check it against the plan, because the width
            is a property of the plan. Passing a different width is refused
            rather than ignored.

    Returns:
        The corrected distribution over the histogram's support.

    Raises:
        TypeError: If ``counts`` is not a mapping of strings to counts, or
            ``plan`` is not a :class:`ReadoutMitigationPlan`.
        ValueError: If a key is not a bit string of the plan's width, a count
            is not a non-negative integer, the histogram is empty, it holds no
            shot, or ``n_qubits`` disagrees with the plan.
    """

    if not isinstance(plan, ReadoutMitigationPlan):
        raise TypeError("readout mitigation applies a ReadoutMitigationPlan")
    if n_qubits is not None and n_qubits != plan.n_qubits:
        raise ValueError(
            f"the plan is {plan.n_qubits} qubits wide and {n_qubits} was passed; "
            "the width is the plan's, so the two cannot disagree"
        )
    if not isinstance(counts, Mapping):
        raise TypeError("a shot histogram is a mapping of bit strings to counts")
    width = plan.n_qubits
    frequencies = torch.zeros(2**width, dtype=torch.float64)
    total = 0
    for key, count in counts.items():
        if not isinstance(key, str):
            raise TypeError("a shot histogram is keyed by bit strings")
        if len(key) != width or any(character not in "01" for character in key):
            raise ValueError(
                f"{key!r} is not a {width}-character bit string with qubit 0 "
                "leftmost"
            )
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError(f"the count for {key!r} must be a non-negative integer")
        frequencies[int(key, 2)] += count
        total += count
    if not counts:
        raise ValueError("a shot histogram with no outcome holds no measurement")
    if total == 0:
        raise ValueError("a shot histogram whose counts are all zero holds no shot")
    return run_readout_mitigation(frequencies / total, plan, shots=total)

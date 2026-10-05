"""Readout-error mitigation: the inverse of the confusion the model applies.

The premise this file rests on is that the correction and the forward path are
the same statement read in two directions. The forward path is the repository's
own :meth:`~flagquantum.noise.model.NoiseModel.apply_readout_probabilities`, and
every exactness assertion here is taken against *that* method rather than
against a second formula written into the test: the whole ``2**n`` map is built
by asking the forward path what it does to each computational basis state, the
correction is applied to that map, and the product is required to be the
identity. A correction that inverted a map the forward path does not apply --
the single most likely failure of a unit like this one -- fails those
assertions and nothing else in the file has to know.

The assertions are deliberately of two shapes. Where a quantity is a *claim*
that could be improved (a bound, a gap), the assertion is an inequality or an
identity so it cannot go stale by being beaten. Where a quantity is the
*evidence* itself (the three-term corrected vector that carries negative mass,
the singular value the refusal names), the value is pinned, because a test that
did not pin it would not be reporting the behaviour it exists to report.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping

import pytest
import torch

import flagquantum as fq
from flagquantum.algorithms import readout_mitigation as module
from flagquantum.algorithms.readout_mitigation import (
    MAX_CORRELATED_BLOCK_QUBITS,
    NORMALIZATION_TOLERANCE,
    READOUT_MITIGATION_ASSUMPTIONS,
    READOUT_MITIGATION_LIMITATIONS,
    READOUT_MITIGATION_SCHEMA,
    SINGULAR_VALUE_FLOOR,
    ReadoutBlock,
    ReadoutMitigationPlan,
    ReadoutMitigationResult,
    plan_readout_mitigation,
    run_readout_mitigation,
    run_readout_mitigation_counts,
)
from flagquantum.errors import CapabilityError
from flagquantum.noise import (
    CorrelatedReadoutError,
    NoiseModel,
    ReadoutError,
    ReadoutRule,
)

pytestmark = pytest.mark.unit


def _confusion(probability: float) -> ReadoutError:
    return ReadoutError(
        ((1.0 - probability, probability), (probability, 1.0 - probability))
    )


def _model(*rules: ReadoutRule) -> NoiseModel:
    return NoiseModel(readout_rules=list(rules))


def _whole_space_map(model: NoiseModel, n_qubits: int) -> torch.Tensor:
    """Return the ``2**n`` map the repository's forward path actually applies."""

    size = 2**n_qubits
    matrix = torch.zeros(size, size, dtype=torch.float64)
    for true_index in range(size):
        basis = torch.zeros(size, dtype=torch.float64)
        basis[true_index] = 1.0
        matrix[:, true_index] = model.apply_readout_probabilities(
            basis, n_wires=n_qubits
        )
    return matrix


def _plan_as_whole_space_map(plan: ReadoutMitigationPlan) -> torch.Tensor:
    """Return the ``2**n`` map the correction actually applies, column by column."""

    size = 2**plan.n_qubits
    matrix = torch.zeros(size, size, dtype=torch.float64)
    for index in range(size):
        basis = torch.zeros(size, dtype=torch.float64)
        basis[index] = 1.0
        matrix[:, index] = run_readout_mitigation(basis, plan).probabilities
    return matrix


def _exact_distribution(circuit: fq.Circuit) -> torch.Tensor:
    """Return the runtime's exact computational-basis distribution."""

    state = fq.run(circuit, options=fq.ExecutionOptions()).state[0]
    return torch.as_tensor((state.conj() * state).real, dtype=torch.float64)


# --- the correction is the forward path's inverse ----------------------------


@pytest.mark.parametrize(
    ("label", "n_qubits", "rules"),
    (
        ("one singleton", 2, (ReadoutRule((0,), _confusion(0.13)),)),
        (
            "two singletons",
            2,
            (ReadoutRule((0,), _confusion(0.13)), ReadoutRule((1,), _confusion(0.07))),
        ),
        (
            "two singletons in reverse order",
            3,
            (ReadoutRule((2,), _confusion(0.13)), ReadoutRule((0,), _confusion(0.05))),
        ),
        (
            "one rule naming two qubits",
            2,
            (ReadoutRule((0, 1), _confusion(0.11)),),
        ),
        (
            "asymmetric singleton",
            3,
            (ReadoutRule((1,), ReadoutError(((0.9, 0.1), (0.2, 0.8)))),),
        ),
    ),
)
def test_the_correction_inverts_the_map_the_forward_path_applies(
    label: str, n_qubits: int, rules: tuple[ReadoutRule, ...]
) -> None:
    """``plan`` applied to the whole map must give the identity, to float64."""

    model = _model(*rules)
    plan = plan_readout_mitigation(model, n_qubits=n_qubits)
    forward = _whole_space_map(model, n_qubits)
    correction = _plan_as_whole_space_map(plan)
    identity = torch.eye(2**n_qubits, dtype=torch.float64)
    residual = float((correction @ forward - identity).abs().max())
    assert residual < 1e-14, (label, residual)
    # And it is the inverse rather than merely a left inverse at this size.
    assert torch.allclose(
        correction, torch.linalg.inv(forward), atol=1e-12, rtol=0
    ), label


def test_a_correlated_block_is_inverted_as_one_block() -> None:
    """A block that does not factorize must not be split into singletons."""

    probability = 0.06
    correlated = CorrelatedReadoutError(
        (
            (1.0 - 2 * probability, probability, probability, 0.0),
            (probability, 1.0 - 2 * probability, 0.0, probability),
            (probability, 0.0, 1.0 - 2 * probability, probability),
            (0.0, probability, probability, 1.0 - 2 * probability),
        )
    )
    model = _model(ReadoutRule((0, 1), correlated))
    plan = plan_readout_mitigation(model, n_qubits=2)
    assert len(plan.blocks) == 1
    assert plan.blocks[0].qubits == (0, 1)
    forward = _whole_space_map(model, 2)
    correction = _plan_as_whole_space_map(plan)
    identity = torch.eye(4, dtype=torch.float64)
    assert float((correction @ forward - identity).abs().max()) < 1e-14
    # No pair of per-qubit blocks could have reproduced this correction: a
    # tensor product of two 2 by 2 matrices has rank one once its four indices
    # are grouped as a (true, observed) pair per qubit, and this map does not.
    joint = (
        torch.as_tensor(correlated.probabilities, dtype=torch.float64)
        .reshape(2, 2, 2, 2)
        .permute(0, 2, 1, 3)
        .reshape(4, 4)
    )
    second_singular_value = float(torch.linalg.svdvals(joint)[1])
    assert second_singular_value > 1e-3, second_singular_value


def test_a_correlated_block_composes_with_a_singleton_beside_it() -> None:
    """A correlated two-qubit block and a singleton must compose exactly."""

    correlated = CorrelatedReadoutError(
        (
            (0.8, 0.1, 0.1, 0.0),
            (0.1, 0.8, 0.0, 0.1),
            (0.1, 0.0, 0.8, 0.1),
            (0.0, 0.1, 0.1, 0.8),
        )
    )
    model = _model(
        ReadoutRule((0, 1), correlated),
        ReadoutRule((2,), _confusion(0.09)),
    )
    plan = plan_readout_mitigation(model, n_qubits=3)
    assert [block.qubits for block in plan.blocks] == [(0, 1), (2,)]
    forward = _whole_space_map(model, 3)
    correction = _plan_as_whole_space_map(plan)
    assert (
        float((correction @ forward - torch.eye(8, dtype=torch.float64)).abs().max())
        < 1e-13
    )


def test_the_correction_recovers_the_exact_distribution_of_a_circuit() -> None:
    """A prepared state, pushed through readout and corrected, comes back."""

    model = _model(ReadoutRule((1,), _confusion(0.08)))
    plan = plan_readout_mitigation(model, n_qubits=2)
    circuit = fq.Circuit(2).x(1)
    exact = _exact_distribution(circuit)
    measured = model.apply_readout_probabilities(exact, n_wires=2)
    corrected = run_readout_mitigation(measured, plan)
    assert float((corrected.probabilities - exact).abs().max()) < 1e-15
    # The forward step genuinely moved the distribution, so the recovery above
    # is not the identity map passing through.
    assert float((measured - exact).abs().max()) > 0.05
    assert corrected.total_variation == pytest.approx(
        float((measured - exact).abs().sum()), abs=1e-15
    )


def test_an_uncovered_qubit_reads_perfectly_and_is_named_as_uncovered() -> None:
    """A qubit with no rule is left untouched and recorded as an assumption."""

    model = _model(ReadoutRule((0,), _confusion(0.2)))
    plan = plan_readout_mitigation(model, n_qubits=3)
    assert plan.covered_qubits == (0,)
    assert plan.uncovered_qubits == (1, 2)
    exact = torch.zeros(8, dtype=torch.float64)
    exact[0b101] = 1.0  # qubits 0 and 2 set
    measured = model.apply_readout_probabilities(exact, n_wires=3)
    corrected = run_readout_mitigation(measured, plan).probabilities
    assert float((corrected - exact).abs().max()) < 1e-15


def test_the_correction_is_the_transpose_inverse_of_the_declared_confusion() -> None:
    """The declared index convention and the flat index convention differ.

    A calibration table is written with the true value as the row index; a
    distribution in flat index space is a column vector whose first qubit is the
    most significant bit. The two are related by a transposition, and losing it
    leaves a correction that is still the inverse of *something* while being
    wrong on every asymmetric rule, so the relation is asserted directly and
    the other pairing is shown to differ.
    """

    declared = ReadoutError(((0.9, 0.1), (0.2, 0.8)))
    model = _model(ReadoutRule((0,), declared))
    block = plan_readout_mitigation(model, n_qubits=1).blocks[0]
    matrix = torch.as_tensor(declared.probabilities, dtype=torch.float64)
    assert torch.equal(block.correction, torch.linalg.inv(matrix.T))
    # The operator the forward path applies to a distribution is the transpose
    # of the declared table, which is why the correction is its inverse.
    assert torch.allclose(_whole_space_map(model, 1), matrix.T, atol=1e-15, rtol=0)
    untransposed = torch.linalg.inv(matrix)
    assert float((untransposed - block.correction).abs().max()) > 0.05


# --- the amplification is the whole map's, and it is honest ------------------


def test_the_reported_overhead_is_the_whole_maps_induced_one_norm() -> None:
    """The plan's overhead must equal ``||M^-1||_1`` of the map it inverts."""

    model = _model(
        ReadoutRule((0,), _confusion(0.13)),
        ReadoutRule((1,), _confusion(0.13)),
        ReadoutRule((2,), ReadoutError(((0.9, 0.1), (0.2, 0.8)))),
    )
    plan = plan_readout_mitigation(model, n_qubits=3)
    forward = _whole_space_map(model, 3)
    dense = torch.linalg.inv(forward)
    assert plan.sampling_overhead == pytest.approx(
        float(dense.abs().sum(dim=0).max()), rel=1e-12
    )
    assert plan.condition_number == pytest.approx(
        float(torch.linalg.cond(forward)), rel=1e-10
    )
    # The whole map's overhead is the product of the blocks', never the
    # largest block's: taking the maximum would understate this device.
    blocks = [block.sampling_overhead for block in plan.blocks]
    assert plan.sampling_overhead == pytest.approx(math.prod(blocks), rel=1e-15)
    assert plan.sampling_overhead > max(blocks)


def test_the_overhead_is_one_exactly_when_nothing_needs_correcting() -> None:
    """A permutation confusion has no amplification and must report as much."""

    swap = ReadoutError(((0.0, 1.0), (1.0, 0.0)))
    plan = plan_readout_mitigation(_model(ReadoutRule((0,), swap)), n_qubits=1)
    assert plan.sampling_overhead == 1.0
    assert plan.condition_number == 1.0
    flip = run_readout_mitigation(torch.tensor([0.25, 0.75]), plan)
    assert float((flip.probabilities - torch.tensor([0.75, 0.25])).abs().max()) < 1e-15
    assert flip.negative_mass == 0.0


def test_the_standard_error_bound_holds_against_measured_frequency_noise() -> None:
    """The stated bound must dominate a measured standard deviation.

    The prediction is the exact one rather than the reported bound: with
    ``v = M^-T w`` the frequencies have covariance ``(diag(d) - d d^T) / N``
    about the device distribution ``d``, so the corrected weight has standard
    deviation ``sqrt((sum_i v_i^2 d_i - (v^T d)^2) / N)``. The count of
    repetitions is high enough that the sample deviation is itself within a few
    percent.
    """

    probability = 0.2
    model = _model(ReadoutRule((0,), _confusion(probability)))
    plan = plan_readout_mitigation(model, n_qubits=1)
    forward = torch.as_tensor(
        _confusion(probability).probabilities, dtype=torch.float64
    )
    shots = 400
    repetitions = 4000
    true = torch.tensor([0.7, 0.3], dtype=torch.float64)
    # The device reports `observed`, so that is what is sampled and what the
    # correction is handed; sampling `true` and correcting it would be a
    # different and larger statistic.
    device_distribution = forward @ true
    weight = torch.tensor([1.0, -0.6], dtype=torch.float64)
    generator = torch.Generator().manual_seed(20240617)
    values = []
    for _ in range(repetitions):
        draws = torch.multinomial(
            device_distribution, shots, replacement=True, generator=generator
        )
        observed = torch.bincount(draws, minlength=2).to(torch.float64) / shots
        result = run_readout_mitigation(observed, plan, shots=shots)
        values.append(float(weight @ result.probabilities))
    measured_sd = float(torch.tensor(values).std())
    pullback = torch.linalg.solve(forward.mT, weight)
    predicted_sd = math.sqrt(
        (
            float((pullback**2 * device_distribution).sum())
            - float(pullback @ device_distribution) ** 2
        )
        / shots
    )
    bound = plan.sampling_overhead / math.sqrt(shots)
    assert predicted_sd <= bound
    assert measured_sd == pytest.approx(predicted_sd, rel=0.05)
    assert measured_sd <= bound


def test_the_standard_error_bound_is_reported_exactly_when_shots_are_known() -> None:
    """The bound is ``overhead / sqrt(N)``, and absent when N is absent."""

    plan = plan_readout_mitigation(
        _model(ReadoutRule((0,), _confusion(0.1)), ReadoutRule((1,), _confusion(0.1))),
        n_qubits=2,
    )
    distribution = torch.tensor([0.4, 0.1, 0.1, 0.4], dtype=torch.float64)
    without = run_readout_mitigation(distribution, plan)
    assert without.shots is None
    assert without.standard_error_bound is None
    with_shots = run_readout_mitigation(distribution, plan, shots=2500)
    assert with_shots.standard_error_bound == pytest.approx(
        plan.sampling_overhead / 50.0, rel=1e-15
    )
    # Doubling the shot count must halve the bound, which is the only thing it
    # claims and the reason it is quoted per shot.
    doubled = run_readout_mitigation(distribution, plan, shots=10000)
    bound = with_shots.standard_error_bound
    assert bound is not None
    assert doubled.standard_error_bound == pytest.approx(bound / 2.0, rel=1e-15)


# --- negative mass is reported, never clipped -------------------------------


def test_a_corrected_distribution_carries_negative_mass_and_says_so() -> None:
    """The inverse of a stochastic matrix is not stochastic; that is the point."""

    probability = 0.25
    plan = plan_readout_mitigation(
        _model(ReadoutRule((0,), _confusion(probability))), n_qubits=1
    )
    result = run_readout_mitigation(torch.tensor([0.9, 0.1], dtype=torch.float64), plan)
    assert [float(value) for value in result.probabilities] == pytest.approx(
        [1.3, -0.3], abs=1e-15
    )
    assert result.negative_mass == pytest.approx(0.3, abs=1e-15)
    assert result.is_physical is False
    # It still sums to one, so the negative entry is the only unphysical part.
    assert float(result.probabilities.sum()) == pytest.approx(1.0, abs=1e-15)
    # Clipping would have produced this, and it is a different vector.
    clipped = result.probabilities.clamp(min=0)
    clipped = clipped / clipped.sum()
    assert float((clipped - result.probabilities).abs().max()) > 0.2


def test_a_corrected_distribution_inside_the_simplex_is_called_physical() -> None:
    """A minor confusion corrected on a smooth distribution stays physical."""

    plan = plan_readout_mitigation(
        _model(ReadoutRule((0,), _confusion(0.02))), n_qubits=1
    )
    result = run_readout_mitigation(torch.tensor([0.6, 0.4]), plan)
    assert result.negative_mass == 0.0
    assert result.is_physical is True
    assert result.total_variation > 0.0


def test_the_result_carries_the_caveats_it_rests_on() -> None:
    """The result's assumptions and limitations are the module's, not empty."""

    plan = plan_readout_mitigation(
        _model(ReadoutRule((0,), _confusion(0.1))), n_qubits=1
    )
    result = run_readout_mitigation(torch.tensor([0.5, 0.5]), plan)
    assert result.assumptions == READOUT_MITIGATION_ASSUMPTIONS
    assert result.limitations == READOUT_MITIGATION_LIMITATIONS
    assert any("negative" in line for line in result.limitations)
    assert result.observed is not None
    assert float((result.observed - torch.tensor([0.5, 0.5])).abs().max()) == 0.0


# --- refusals ---------------------------------------------------------------


def test_a_block_at_the_floor_is_refused_by_name() -> None:
    """A maximally confused qubit has no inverse and must be refused."""

    half = ReadoutError(((0.5, 0.5), (0.5, 0.5)))
    with pytest.raises(CapabilityError) as caught:
        plan_readout_mitigation(_model(ReadoutRule((0,), half)), n_qubits=1)
    message = str(caught.value)
    assert "qubits [0]" in message
    assert "smallest singular value" in message
    assert f"{SINGULAR_VALUE_FLOOR:.1e}" in message


def test_the_floor_is_the_callers_and_a_tighter_one_refuses_more() -> None:
    """A block the default admits must be refused when the floor is raised."""

    model = _model(ReadoutRule((0,), _confusion(0.13)))
    plan_readout_mitigation(model, n_qubits=1)
    with pytest.raises(CapabilityError) as caught:
        plan_readout_mitigation(model, n_qubits=1, singular_value_floor=0.9)
    assert "9.0e-01" in str(caught.value)
    with pytest.raises(ValueError):
        plan_readout_mitigation(model, n_qubits=1, singular_value_floor=0.0)
    with pytest.raises(ValueError):
        plan_readout_mitigation(model, n_qubits=1, singular_value_floor=1.0)


def test_two_rules_on_one_qubit_are_refused_rather_than_composed() -> None:
    """Inverting a confusion the forward path applied twice is not a correction."""

    model = _model(
        ReadoutRule((0,), _confusion(0.1)),
        ReadoutRule((1,), _confusion(0.1)),
        ReadoutRule((0,), _confusion(0.2)),
    )
    with pytest.raises(ValueError) as caught:
        plan_readout_mitigation(model, n_qubits=2)
    message = str(caught.value)
    assert "readout qubit 0" in message
    assert "twice" in message


def test_a_model_with_no_readout_rule_is_refused() -> None:
    """Correcting nothing would return the measurement under another name."""

    with pytest.raises(ValueError) as caught:
        plan_readout_mitigation(NoiseModel(), n_qubits=2)
    assert "no readout rule" in str(caught.value)
    with pytest.raises(TypeError):
        plan_readout_mitigation("a model", n_qubits=2)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        plan_readout_mitigation(_model(ReadoutRule((0,), _confusion(0.1))), n_qubits=0)
    with pytest.raises(ValueError):
        plan_readout_mitigation(_model(ReadoutRule((4,), _confusion(0.1))), n_qubits=2)


def test_a_correlated_block_wider_than_the_ceiling_is_refused() -> None:
    """The ceiling is a memory statement and it refuses rather than thrashing."""

    width = MAX_CORRELATED_BLOCK_QUBITS + 1
    size = 2**width
    correlated = CorrelatedReadoutError(
        tuple(
            tuple(1.0 if row == column else 0.0 for column in range(size))
            for row in range(size)
        )
    )
    with pytest.raises(CapabilityError) as caught:
        plan_readout_mitigation(
            _model(ReadoutRule(tuple(range(width)), correlated)), n_qubits=width
        )
    assert "exceeds the 10-qubit ceiling" in str(caught.value)


def test_a_correlated_rule_whose_width_disagrees_with_its_matrix_is_refused() -> None:
    """A matrix is either the rule's width or the rule is wrong."""

    correlated = CorrelatedReadoutError(((0.9, 0.1), (0.1, 0.9)))
    with pytest.raises(ValueError) as caught:
        plan_readout_mitigation(_model(ReadoutRule((0, 1), correlated)), n_qubits=2)
    assert "declares a 1-qubit matrix" in str(caught.value)


def test_an_observed_distribution_that_is_not_a_distribution_is_refused() -> None:
    """Silently renormalizing would report a different vector under the name."""

    plan = plan_readout_mitigation(
        _model(ReadoutRule((0,), _confusion(0.1))), n_qubits=1
    )
    with pytest.raises(ValueError) as caught:
        run_readout_mitigation(torch.tensor([0.9, 0.3]), plan)
    assert "sum to one" in str(caught.value)
    with pytest.raises(ValueError):
        run_readout_mitigation(torch.tensor([1.1, -0.1]), plan)
    with pytest.raises(ValueError):
        run_readout_mitigation(torch.tensor([0.0, 0.0]), plan)
    with pytest.raises(ValueError):
        run_readout_mitigation(torch.tensor([0.25, 0.25, 0.25, 0.25]), plan)
    with pytest.raises(ValueError):
        run_readout_mitigation(torch.tensor([float("nan"), 1.0]), plan)
    with pytest.raises(TypeError):
        run_readout_mitigation(torch.tensor([0.5, 0.5]), "a plan")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        run_readout_mitigation(torch.tensor([0.5, 0.5]), plan, shots=0)


def test_an_input_the_normalization_tolerance_admits_is_corrected() -> None:
    """The tolerance is wide enough for accumulated float32 frequencies."""

    plan = plan_readout_mitigation(
        _model(ReadoutRule((0,), _confusion(0.1))), n_qubits=1
    )
    drifted = torch.tensor([0.60000002, 0.40000003], dtype=torch.float32)
    assert abs(float(drifted.sum()) - 1.0) < NORMALIZATION_TOLERANCE
    result = run_readout_mitigation(drifted, plan)
    assert result.probabilities.dtype == torch.float64
    assert float((result.probabilities - torch.tensor([0.5, 0.5])).abs().max()) > 0.1


# --- the plan is a value ----------------------------------------------------


def test_the_identity_tracks_the_confusion_the_width_and_nothing_else() -> None:
    """Two plans differing only in what they invert must not share an identity."""

    first = plan_readout_mitigation(
        _model(ReadoutRule((0,), _confusion(0.1))), n_qubits=2
    )
    same = plan_readout_mitigation(
        _model(ReadoutRule((0,), _confusion(0.1))), n_qubits=2
    )
    other_matrix = plan_readout_mitigation(
        _model(ReadoutRule((0,), _confusion(0.2))), n_qubits=2
    )
    other_width = plan_readout_mitigation(
        _model(ReadoutRule((0,), _confusion(0.1))), n_qubits=3
    )
    assert first.identity == same.identity
    assert first.identity != other_matrix.identity
    assert first.identity != other_width.identity
    # The qubit count is inside the identity even when the uncovered part of it
    # changes nothing about the correction.
    assert first.uncovered_qubits != other_width.uncovered_qubits


def test_the_identity_does_not_depend_on_the_order_the_rules_arrived_in() -> None:
    """Rule order is not a property of the map and must not reach the identity."""

    left = plan_readout_mitigation(
        _model(ReadoutRule((0,), _confusion(0.1)), ReadoutRule((1,), _confusion(0.2))),
        n_qubits=2,
    )
    right = plan_readout_mitigation(
        _model(ReadoutRule((1,), _confusion(0.2)), ReadoutRule((0,), _confusion(0.1))),
        n_qubits=2,
    )
    assert left.identity == right.identity
    assert [block.qubits for block in left.blocks] == [
        block.qubits for block in right.blocks
    ]


def test_blocks_are_ordered_by_their_first_qubit_and_disjoint() -> None:
    """The block list is a partition, and its order is the plan's, not the input's."""

    plan = plan_readout_mitigation(
        _model(
            ReadoutRule((5,), _confusion(0.1)),
            ReadoutRule((2, 3), _confusion(0.1)),
            ReadoutRule((0,), _confusion(0.1)),
        ),
        n_qubits=8,
    )
    assert [block.qubits for block in plan.blocks] == [(0,), (2,), (3,), (5,)]
    named = [qubit for block in plan.blocks for qubit in block.qubits]
    assert len(set(named)) == len(named)
    assert plan.covered_qubits == (0, 2, 3, 5)
    assert plan.uncovered_qubits == (1, 4, 6, 7)
    assert plan.covered is True


def test_the_plan_payload_is_json_and_carries_its_schema() -> None:
    """The plan is a serializable value rather than a bag of tensors."""

    correlated = CorrelatedReadoutError(
        (
            (0.9, 0.1, 0.0, 0.0),
            (0.1, 0.9, 0.0, 0.0),
            (0.0, 0.0, 0.9, 0.1),
            (0.0, 0.0, 0.1, 0.9),
        )
    )
    plan = plan_readout_mitigation(_model(ReadoutRule((0, 1), correlated)), n_qubits=3)
    payload = plan.to_dict()
    assert payload["schema"] == READOUT_MITIGATION_SCHEMA
    assert payload["identity"] == plan.identity
    assert payload["covered_qubits"] == [0, 1]
    assert payload["uncovered_qubits"] == [2]
    assert isinstance(payload["blocks"][0]["probabilities"][0][0], float)
    restored = json.loads(json.dumps(payload))
    assert restored == payload
    assert restored["blocks"][0]["qubits"] == [0, 1]
    # The identity survives a JSON round trip, which is what makes it usable
    # as a key in a record written by something else.
    assert restored["identity"] == plan.identity


def test_the_summary_reports_the_product_and_the_limiting_block() -> None:
    """The report exposes the number that matters and the block behind it."""

    plan = plan_readout_mitigation(
        _model(
            ReadoutRule((0,), _confusion(0.02)),
            ReadoutRule((1,), _confusion(0.3)),
        ),
        n_qubits=2,
    )
    summary = plan.summary()
    assert summary["n_blocks"] == 2
    assert summary["sampling_overhead"] == pytest.approx(
        math.prod(summary["block_sampling_overheads"]), rel=1e-15
    )
    assert summary["limiting_block"]["qubits"] == [1]
    assert summary["limiting_block"]["smallest_singular_value"] < 0.5


def test_the_result_payload_is_json_and_names_the_plan_it_used() -> None:
    """A result is auditable without the plan object being to hand."""

    plan = plan_readout_mitigation(
        _model(ReadoutRule((0,), _confusion(0.1))), n_qubits=1
    )
    result = run_readout_mitigation(torch.tensor([0.6, 0.4]), plan, shots=100)
    payload = result.to_dict()
    assert payload["plan_identity"] == plan.identity
    assert payload["schema"] == READOUT_MITIGATION_SCHEMA + ".result"
    assert payload["shots"] == 100
    assert payload["standard_error_bound"] == pytest.approx(
        plan.sampling_overhead / 10.0, rel=1e-15
    )
    assert json.loads(json.dumps(payload)) == payload


# --- counts -----------------------------------------------------------------


def test_a_bit_string_is_read_with_qubit_zero_leftmost() -> None:
    """The key convention must be the index convention, not the reverse."""

    plan = plan_readout_mitigation(
        _model(ReadoutRule((1,), _confusion(0.1))), n_qubits=2
    )
    result = run_readout_mitigation_counts({"10": 30, "00": 70}, plan)
    expected = torch.zeros(4, dtype=torch.float64)
    expected[2] = 0.3  # qubit 0 set is index 2
    expected[0] = 0.7
    assert (
        float(
            (
                result.probabilities
                - run_readout_mitigation(expected, plan).probabilities
            )
            .abs()
            .max()
        )
        == 0.0
    )
    assert result.shots == 100
    # Reversing the key convention would put the 0.3 on index 1 instead.
    reversed_reading = torch.tensor([0.7, 0.3, 0.0, 0.0], dtype=torch.float64)
    assert (
        float(
            (
                run_readout_mitigation(reversed_reading, plan).probabilities
                - result.probabilities
            )
            .abs()
            .max()
        )
        > 0.05
    )


def test_a_histogram_is_normalized_by_its_own_total() -> None:
    """The shot count a histogram holds is the one its bound is quoted for."""

    plan = plan_readout_mitigation(
        _model(ReadoutRule((0,), _confusion(0.1))), n_qubits=1
    )
    result = run_readout_mitigation_counts({"0": 910, "1": 90}, plan)
    assert result.shots == 1000
    assert result.standard_error_bound == pytest.approx(
        plan.sampling_overhead / math.sqrt(1000), rel=1e-15
    )
    assert float(result.observed.sum()) == pytest.approx(1.0, abs=1e-15)
    assert result.observed[1] == pytest.approx(0.09, abs=1e-15)


def test_a_width_that_disagrees_with_the_plan_is_refused() -> None:
    """The width belongs to the plan, so a second opinion is an error."""

    plan = plan_readout_mitigation(
        _model(ReadoutRule((0,), _confusion(0.1))), n_qubits=2
    )
    with pytest.raises(ValueError) as caught:
        run_readout_mitigation_counts({"00": 1}, plan, n_qubits=3)
    assert "the width is the plan's" in str(caught.value)
    assert float(
        run_readout_mitigation_counts({"00": 1}, plan, n_qubits=2).probabilities.sum()
    ) == pytest.approx(1.0, abs=1e-15)


@pytest.mark.parametrize(
    "counts",
    (
        {"000": 1},
        {"0": 1, "x": 1},
        {"0": -1},
        {"0": 1.5},
        {},
        {"0": 0},
    ),
)
def test_a_histogram_that_is_not_a_histogram_is_refused(
    counts: Mapping[str, object],
) -> None:
    """Every malformed histogram is refused rather than guessed at."""

    plan = plan_readout_mitigation(
        _model(ReadoutRule((0,), _confusion(0.1))), n_qubits=1
    )
    with pytest.raises((ValueError, TypeError)):
        run_readout_mitigation_counts(counts, plan)  # type: ignore[arg-type]


def test_a_histogram_reads_the_shot_count_it_was_given() -> None:
    """Two histograms of the same frequency at different totals differ in bound."""

    plan = plan_readout_mitigation(
        _model(ReadoutRule((0,), _confusion(0.1))), n_qubits=1
    )
    few = run_readout_mitigation_counts({"0": 91, "1": 9}, plan)
    many = run_readout_mitigation_counts({"0": 9100, "1": 900}, plan)
    assert float((few.probabilities - many.probabilities).abs().max()) < 1e-15
    few_bound = few.standard_error_bound
    many_bound = many.standard_error_bound
    assert few_bound is not None and many_bound is not None
    assert many_bound < few_bound


# --- invariants of the records ----------------------------------------------


def test_a_block_that_is_not_a_block_is_refused() -> None:
    """The block's own invariants are checked rather than assumed."""

    matrix = torch.as_tensor(_confusion(0.1).probabilities, dtype=torch.float64)
    correction = torch.linalg.inv(matrix.T)
    with pytest.raises(ValueError):
        ReadoutBlock((), matrix, correction, 0.8, 1.25, 1.25)
    with pytest.raises(ValueError):
        ReadoutBlock((0, 0), matrix, correction, 0.8, 1.25, 1.25)
    with pytest.raises(ValueError):
        ReadoutBlock((0,), matrix, matrix.to(torch.float32), 0.8, 1.25, 1.25)
    with pytest.raises(ValueError):
        ReadoutBlock((0,), matrix, torch.eye(3, dtype=torch.float64), 0.8, 1.25, 1.25)
    with pytest.raises(ValueError):
        ReadoutBlock((0,), matrix, correction, 0.0, 1.25, 1.25)
    with pytest.raises(ValueError):
        ReadoutBlock((0,), matrix, correction, 0.8, 0.5, 1.25)
    with pytest.raises(ValueError):
        ReadoutBlock((0,), matrix, correction, 0.8, 1.25, 0.5)
    with pytest.raises(ValueError):
        ReadoutBlock((-1,), matrix, correction, 0.8, 1.25, 1.25)


def test_a_result_that_is_not_a_result_is_refused() -> None:
    """The result's invariants are checked rather than assumed."""

    plan = plan_readout_mitigation(
        _model(ReadoutRule((0,), _confusion(0.1))), n_qubits=1
    )
    observed = torch.tensor([0.5, 0.5], dtype=torch.float64)
    corrected = run_readout_mitigation(observed, plan).probabilities
    with pytest.raises(TypeError):
        ReadoutMitigationResult(corrected, observed, "a plan", 0.0, 0.0, True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ReadoutMitigationResult(corrected.reshape(1, 2), observed, plan, 0.0, 0.0, True)
    with pytest.raises(ValueError):
        ReadoutMitigationResult(
            corrected.to(torch.float32), observed, plan, 0.0, 0.0, True
        )
    with pytest.raises(ValueError):
        ReadoutMitigationResult(corrected, observed, plan, -0.1, 0.0, True)
    with pytest.raises(ValueError):
        ReadoutMitigationResult(corrected, observed, plan, 0.0, -0.1, True)
    with pytest.raises(ValueError):
        ReadoutMitigationResult(corrected, observed, plan, 0.0, 0.0, True, shots=0)
    with pytest.raises(ValueError):
        ReadoutMitigationResult(
            corrected,
            observed,
            plan,
            0.0,
            0.0,
            True,
            shots=10,
            standard_error_bound=None,
        )
    with pytest.raises(ValueError):
        ReadoutMitigationResult(
            corrected, observed, plan, 0.0, 0.0, True, assumptions=()
        )
    with pytest.raises(ValueError):
        ReadoutMitigationResult(
            corrected, observed, plan, 0.0, 0.0, True, limitations=()
        )


def test_the_module_declares_the_names_it_exports() -> None:
    """Every exported name must exist, and the list must be the module's own."""

    assert module.__all__ == (
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
    algorithms = fq.algorithms
    for name in module.__all__:
        assert hasattr(module, name), name
        assert getattr(algorithms, name) is getattr(module, name), name
        assert name in algorithms.__all__, name

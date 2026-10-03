"""Classifying a Kraus channel as a mixture of unitaries, or as something else.

A channel whose Kraus operators are all non-negative real scales times unitaries
is a probability distribution over unitary operators. That reading is
executable, not descriptive: a sampler that needs a pure error per trajectory -
a stabilizer frame, a Pauli twirl, a trajectory ensemble - can use the mixture
directly, while a channel that is merely trace preserving cannot be sampled that
way at all. The distinction is decided here, once, from the operators, so every
consumer that needs it reads the same answer instead of re-deriving one.
"""

from __future__ import annotations

import pytest
import torch

from flagquantum.noise import (
    KrausChannel,
    UnitaryMixture,
    amplitude_damping_channel,
    bit_flip_channel,
    coherent_overrotation_channel,
    depolarizing_channel,
    phase_damping_channel,
    phase_flip_channel,
    reset_error_channel,
    thermal_relaxation_channel,
    two_qubit_depolarizing_channel,
)
from flagquantum.noise.channels import _scaled_unitary

pytestmark = pytest.mark.unit

IDENTITY = torch.eye(2, dtype=torch.complex128)
PAULI_X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
PAULI_Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
PAULI_Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)


def _mixture(channel: KrausChannel) -> UnitaryMixture:
    """Return the channel's mixture, failing the test if it has none."""

    mixture = channel.unitary_mixture
    assert mixture is not None, f"{channel.name} is a mixture of unitaries"
    return mixture


def _completeness(mixture: UnitaryMixture) -> torch.Tensor:
    """Return ``sum_i p_i U_i^† U_i``, which is the identity for a mixture."""

    total = torch.zeros_like(mixture.unitaries[0])
    for probability, unitary in zip(
        mixture.probabilities, mixture.unitaries, strict=True
    ):
        total = total + float(probability) * unitary.mH @ unitary
    return total


def _probabilities(channel: KrausChannel) -> list[float]:
    """Return the branch probabilities, at the precision the channel carries.

    A factory builds its Kraus operators in the runtime's default complex dtype,
    so a channel's weights are the square roots of its probabilities rounded to
    that dtype and are not the decimals the factory was called with. The
    comparison is therefore approximate rather than exact, and the tolerance is
    the dtype's rather than the test's.
    """

    return [float(value) for value in _mixture(channel).probabilities]


def _assert_close(left: torch.Tensor, right: torch.Tensor) -> None:
    """Compare two operators across the two dtypes this module builds them in."""

    torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-6, check_dtype=False)


@pytest.mark.parametrize(
    ("channel", "expected"),
    [
        (bit_flip_channel(0.25), [0.75, 0.25]),
        (phase_flip_channel(0.25), [0.75, 0.25]),
        (depolarizing_channel(0.3), [0.7, 0.1, 0.1, 0.1]),
    ],
)
def test_a_one_wire_pauli_channel_is_a_mixture_of_its_frames(
    channel: KrausChannel, expected: list[float]
) -> None:
    """Three of the four declared channel opcodes are Pauli mixtures.

    A bit flip is two frames and a depolarizing channel is four, so the branch
    count is part of the answer: a consumer that sampled the identity branch of a
    depolarizing channel as though it were one of three would be wrong by a
    factor the caller could never see in the samples.
    """

    assert _probabilities(channel) == pytest.approx(expected, abs=1e-6)


def test_the_two_qubit_depolarizing_channel_lists_all_fifteen_frames() -> None:
    """Fifteen non-identity frames is the whole Pauli group on two wires minus one.

    The count is asserted rather than described because the channel is the
    widest mixture the built-in factories produce, and a table that dropped a
    frame would still reconstruct a trace-preserving channel - just not this one.
    """

    mixture = _mixture(two_qubit_depolarizing_channel(0.3))

    assert len(mixture.unitaries) == 16
    assert _probabilities(two_qubit_depolarizing_channel(0.3))[0] == pytest.approx(
        0.7, abs=1e-6
    )
    assert _probabilities(two_qubit_depolarizing_channel(0.3))[1:] == pytest.approx(
        [0.02] * 15, abs=1e-6
    )
    _assert_close(_completeness(mixture), torch.eye(4, dtype=torch.complex128))


def test_a_single_branch_unitary_channel_is_a_mixture_of_one() -> None:
    """A perfectly coherent error is still a mixture, of one branch.

    Zero rotation is the identity branch and it is a mixture rather than a
    refusal, because the classification answers a question about the operators
    and not about whether the channel is interesting.
    """

    mixture = _mixture(coherent_overrotation_channel(0.0))

    assert len(mixture.unitaries) == 1
    assert float(mixture.probabilities[0]) == 1.0
    _assert_close(mixture.unitaries[0], torch.eye(2, dtype=torch.complex128))


@pytest.mark.parametrize(
    "channel",
    [
        amplitude_damping_channel(0.3),
        phase_damping_channel(0.3),
        reset_error_channel(0.3),
        thermal_relaxation_channel(0.4, 0.8, 0.05),
    ],
)
def test_a_channel_that_is_not_a_mixture_says_so(channel: KrausChannel) -> None:
    """These four are trace preserving and are not mixtures of unitaries.

    Each one's Kraus operators do not stay on the unitary group: a damping
    channel's first operator has unequal singular values, and a reset channel's
    operators are not normalised at all. No probability distribution over
    unitaries reproduces any of them, and the answer is `None` rather than an
    error, because being a general channel is a fact about the channel and not a
    failure of the caller.
    """

    assert channel.unitary_mixture is None
    assert channel.is_unitary_mixture is False


def test_a_damping_channel_is_refused_by_its_unequal_singular_values() -> None:
    """The refusal is decided by the whole Gram matrix, not by its first entry.

    The amplitude damping operator at ``gamma = 0.5`` has the diagonal Gram
    matrix ``diag(1, 0.5)``. A check that read only the leading entry would take
    the scale ``1`` from it and call the operator unitary, which would put a
    non-unitary branch into a mixture whose consumers apply it as a pure error.
    """

    damping = 0.5
    first = torch.tensor([[1.0, 0.0], [0.0, damping**0.5]], dtype=torch.complex128)
    second = torch.tensor([[0.0, damping**0.5], [0.0, 0.0]], dtype=torch.complex128)

    assert torch.real((first.mH @ first)[0, 0]) == 1.0
    assert KrausChannel("hand_built_damping", (first, second)).unitary_mixture is None


def test_a_scale_that_misses_one_is_refused_where_trace_preservation_allows_it() -> (
    None
):
    """The mixture's weights sum to one, and its tolerance is the tighter one.

    Trace preservation already forces the squared scales to sum to one, so this
    step is a boundary rather than a second derivation: the channel check admits
    a completeness error of ``1e-6`` relative to the identity and this one admits
    a total weight ``1e-6`` away from one, and a channel between the two
    tolerances is a channel whose probabilities do not add up. A sampler that
    used those weights would drop an error at a rate no one asked for, so the
    boundary is refused instead of renormalised.
    """

    overshoot = 1.0 + 1.5e-6
    channel = KrausChannel("overweight", (overshoot**0.5 * IDENTITY,))

    # The channel is trace preserving within its own tolerance and is not a
    # mixture whose probabilities sum to one within the mixture's.
    assert channel.unitary_mixture is None


def test_a_zero_operator_is_a_branch_of_probability_zero_not_a_refusal() -> None:
    """The edge of a parameter range is still inside it.

    Every depolarizing channel at ``p = 1`` and every bit flip at ``p = 0`` carries
    a Kraus operator that is the zero matrix, because a factory takes a square
    root of a probability that reached an endpoint. Reading that zero as a
    non-unitary operator would refuse a channel that is a Pauli error with
    certainty, and the branch it stands for has no weight to apply anyway, so it
    is dropped rather than kept as a zero matrix under a name that promises a
    unitary.
    """

    saturated = _mixture(depolarizing_channel(1.0))
    silent = _mixture(bit_flip_channel(0.0))

    assert _probabilities(depolarizing_channel(1.0)) == pytest.approx(
        [1 / 3, 1 / 3, 1 / 3], abs=1e-6
    )
    assert len(saturated.unitaries) == 3
    assert all(bool(torch.any(unitary)) for unitary in saturated.unitaries)
    assert _probabilities(bit_flip_channel(0.0)) == [1.0]
    assert len(silent.unitaries) == 1
    _assert_close(silent.unitaries[0], IDENTITY)


def test_every_branch_of_a_mixture_is_unitary() -> None:
    """The claim in the type's name is checked on the operators themselves."""

    mixture = _mixture(two_qubit_depolarizing_channel(0.3))

    for unitary in mixture.unitaries:
        _assert_close(unitary.mH @ unitary, torch.eye(4, dtype=torch.complex128))
    _assert_close(_completeness(mixture), torch.eye(4, dtype=torch.complex128))


def test_a_branch_is_a_frame_up_to_its_global_phase() -> None:
    """A phase on a branch is part of the branch, not a second branch.

    ``-i X`` is one unitary with probability one, so the mixture holds one
    branch rather than an identity/X pair, and the stored operator is the one
    that was given. A consumer that needs a phase-free frame removes the phase
    itself; a classifier that removed it here would be answering a question the
    caller did not ask.
    """

    phased = -1j * PAULI_X
    channel = KrausChannel("phased_flip", (phased,))

    mixture = _mixture(channel)

    assert len(mixture.unitaries) == 1
    _assert_close(mixture.unitaries[0], phased)


def test_the_probabilities_carry_the_channel_dtype_and_device() -> None:
    """A caller builds the next operator from these, so they follow the channel."""

    channel = depolarizing_channel(0.3, dtype=torch.complex64)

    mixture = _mixture(channel)

    assert mixture.probabilities.dtype == torch.float32
    assert mixture.probabilities.device == channel.kraus[0].device


def test_a_mixture_with_a_missing_probability_is_refused() -> None:
    """One probability per unitary is the type's invariant, and it is enforced."""

    with pytest.raises(ValueError, match="one probability per unitary"):
        UnitaryMixture(torch.tensor([1.0, 0.0]), (IDENTITY,))


def test_is_unitary_mixture_agrees_with_the_mixture_it_reports() -> None:
    """The boolean is a convenience over the same answer, not a second one."""

    for channel in (
        bit_flip_channel(0.25),
        depolarizing_channel(0.3),
        two_qubit_depolarizing_channel(0.3),
    ):
        assert channel.is_unitary_mixture is (channel.unitary_mixture is not None)
    for channel in (
        amplitude_damping_channel(0.3),
        thermal_relaxation_channel(0.4, 0.8, 0.05),
    ):
        assert channel.is_unitary_mixture is False


def test_a_mixture_branch_that_is_not_normalised_is_not_a_branch() -> None:
    """Reconstruction is the other direction of the same claim.

    The mixture is rebuilt from its probabilities and its unitaries and compared
    against the channel's own operators, so a classifier that returned the right
    count with the wrong operators would fail here rather than downstream.
    """

    channel = bit_flip_channel(0.25)
    mixture = _mixture(channel)

    rebuilt = tuple(
        float(probability) ** 0.5 * unitary
        for probability, unitary in zip(
            mixture.probabilities, mixture.unitaries, strict=True
        )
    )

    for original, operator in zip(channel.kraus, rebuilt, strict=True):
        _assert_close(original, operator)


def test_the_scaled_unitary_scale_is_the_root_of_the_gram_entry() -> None:
    """The helper returns the scale, not the Gram matrix entry it came from.

    A branch is stored as ``k U`` and a consumer divides by the scale to get
    ``U``, so the answer has to be ``k``. Returning the Gram entry would hand a
    caller ``k ** 2`` and every branch would come back scaled by the channel's
    own weight.
    """

    assert _scaled_unitary(0.5 * PAULI_X) == 0.5
    assert _scaled_unitary(IDENTITY) == 1.0
    assert _scaled_unitary(torch.zeros(2, 2, dtype=torch.complex128)) == 0.0


def test_a_gram_matrix_that_overflows_is_not_a_scaled_unitary() -> None:
    """A finite matrix whose Gram matrix is not finite has no scale to report.

    The operators a channel can hold are checked for finiteness when the channel
    is built, so this input arrives only through the helper. It is refused rather
    than answered with an infinite scale, because the caller's next step is to
    divide by the scale and every branch of an infinite scale is a zero matrix
    under a name that promises a unitary.
    """

    huge = torch.tensor([[1e200, 0.0], [0.0, 1e200]], dtype=torch.complex128)

    assert bool(torch.all(torch.isfinite(huge)))
    assert not bool(torch.all(torch.isfinite(huge.mH @ huge)))
    assert _scaled_unitary(huge) is None


def test_a_gram_matrix_that_underflows_is_not_a_scaled_unitary() -> None:
    """A nonzero matrix whose Gram matrix rounds to zero has no positive scale.

    The scale is the square root of a Gram entry, and an entry that underflowed
    to zero would make the answer zero - which the classifier reads as a branch
    of probability zero and *drops*. Silently dropping a nonzero operator is the
    one outcome worse than refusing it.
    """

    underflowing = torch.tensor([[0.0, 0.0], [1e-200, 0.0]], dtype=torch.complex128)

    assert bool(torch.any(underflowing))
    assert float((underflowing.mH @ underflowing).abs().max()) == 0.0
    assert _scaled_unitary(underflowing) is None


def test_an_operator_whose_gram_matrix_is_a_rank_one_projector_is_refused() -> None:
    """Equal diagonal Gram entries are necessary for a scaled unitary and not sufficient.

    These two operators have Gram matrices whose diagonal is ``(0.5, 0.5)`` and
    whose off-diagonal is ``+/- 0.5``. A test that read only the diagonal would
    take each scale to be ``sqrt(0.5)``, find the squared scales summing to one,
    and report a two-branch mixture of unitaries - for a channel whose operators
    are not unitary and whose branches no sampler could apply as errors.
    """

    first = 0.5**0.5 * torch.tensor([[1.0, 1.0], [0.0, 0.0]], dtype=torch.complex128)
    second = 0.5 * torch.tensor([[1.0, -1.0], [-1.0, 1.0]], dtype=torch.complex128)

    channel = KrausChannel("phase_incoherent", (first, second))

    diagonal = torch.diagonal(first.mH @ first)
    assert torch.allclose(
        diagonal, torch.tensor([0.5, 0.5], dtype=diagonal.dtype), atol=1e-12
    )
    assert abs(float(torch.real((first.mH @ first)[0, 1]))) > 1e-6
    assert channel.unitary_mixture is None


def test_an_operator_with_unequal_gram_diagonal_entries_is_refused() -> None:
    """The whole diagonal has to be one scalar, not just its leading entry.

    These two diagonal operators have Gram matrices ``diag(0.8, 0.6)`` and
    ``diag(0.2, 0.4)``. Both leading entries are positive and the two of them sum
    to one, so a classifier that read only the leading entry would report a
    two-branch mixture whose branches are the operators divided by
    ``sqrt(0.8)`` and ``sqrt(0.2)`` - two matrices that are not unitary, under a
    type whose name promises that they are.
    """

    first = torch.diag(torch.tensor([0.8**0.5, 0.6**0.5], dtype=torch.complex128))
    second = torch.diag(torch.tensor([0.2**0.5, 0.4**0.5], dtype=torch.complex128))

    channel = KrausChannel("anisotropic_damping", (first, second))

    assert float(torch.real((first.mH @ first)[0, 0])) != pytest.approx(
        float(torch.real((first.mH @ first)[1, 1]))
    )
    assert channel.unitary_mixture is None


def test_a_scaled_unitary_within_the_tolerance_is_classified() -> None:
    """The tolerance is the channel's own, so the boundary matches the input it accepts.

    `KrausChannel` admits a completeness error of ``1e-6`` and this classifier
    admits the same, which is what keeps a channel at the edge of that range
    inside the class a consumer needs it to be in rather than one step outside
    it. A zero tolerance here would refuse a channel the channel type already
    calls trace preserving, and the caller would have no operator to fix.
    """

    nearly_identity = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0 - 5e-8]], dtype=torch.complex128
    )

    mixture = _mixture(KrausChannel("nearly_identity", (nearly_identity,)))

    assert len(mixture.unitaries) == 1
    assert float(mixture.probabilities[0]) == pytest.approx(1.0, abs=1e-6)


def test_the_classification_is_a_property_of_the_operators_not_the_name() -> None:
    """A channel named for one thing and holding another is classified by what it holds.

    The name is carried through to the error messages a consumer writes, so it
    has to stay the caller's; the classification reads only the operators.
    """

    named_for_a_flip = KrausChannel(
        "bit_flip", two_qubit_depolarizing_channel(0.3).kraus
    )
    named_for_nothing = KrausChannel("not_a_pauli", phase_damping_channel(0.3).kraus)

    assert named_for_a_flip.name == "bit_flip"
    assert len(_mixture(named_for_a_flip).unitaries) == 16
    assert named_for_nothing.unitary_mixture is None

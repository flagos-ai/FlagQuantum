"""The Pauli twirl, the refusals it rests on, and the value it recovers.

Every numeric assertion states the measured margin and the divergence a
plausible wrong rule produces, so the tolerance is evidence rather than a
formality. The decomposition is double-precision arithmetic over a channel whose
Kraus operators are stored in the runtime's single precision, so the closed-form
checks run against the parameter the channel actually declares rather than
against the round number it was built from; the end-to-end checks run in
``complex128``, where the only error left is that stored parameter.
"""

from __future__ import annotations

import math

import pytest
import torch

from flagquantum.algorithms import Hamiltonian, HamiltonianTerm
from flagquantum.algorithms.pec import (
    PEC_ASSUMPTIONS,
    PEC_LIMITATIONS,
    PauliTwirlDecomposition,
    PecLocation,
    PecResult,
    _word_operators,
    pauli_twirl_decomposition,
    run_pec,
)
from flagquantum.circuit import Circuit
from flagquantum.noise import (
    KrausChannel,
    NoiseModel,
    ReadoutError,
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

pytestmark = pytest.mark.unit

# The channel factories store their operators in the runtime's single precision,
# which leaves a declared probability of 0.1 accurate to about 1.5e-8 and every
# closed form derived from it accurate to about 1e-7. The exact-recovery checks
# below run in complex128, where the composite map is the identity to the
# arithmetic's own floor: the measured distance from the noiseless value is 0.0
# for bit_flip and phase_flip, 2.2e-16 for depolarizing, 1.1e-16 for a two-location
# program and 2.2e-16 for two_qubit_depolarizing. A tolerance of 1e-9 sits seven
# orders above that floor and five orders below the smallest divergence a wrong
# rule produces: dropping one location's correction leaves the two-location read
# at 0.799999932699200, which is 2.0e-1 away, and omitting the signed weight of a
# single term moves the single-qubit read by 2.0e-1.
_EXACT_TOLERANCE = 1e-9
# The closed forms are evaluated at the probability the channel declares, so the
# only gap left is the transfer matrix's own arithmetic. The largest measured gap
# over bit_flip, phase_flip, depolarizing and two_qubit_depolarizing is 1.2e-6 in
# gamma before the declared parameter is accounted for and 2.3e-7 after, and the
# divergences a wrong normalization produces -- 1/N instead of 1/N**2, or a
# weighted sum of the spectrum instead of of its reciprocal -- are of order 0.5
# to 3.0.
_CLOSED_FORM_TOLERANCE = 1e-6
# The channels this unit must refuse carry an off-diagonal Pauli transfer entry of
# 1.000e-01 (amplitude_damping at 0.1), 1.987e-01 (coherent_overrotation at 0.2),
# 5.000e-02 (reset_error at 0.05) and 1.000e+00 (thermal_relaxation), while the
# largest the admitted families show is 3.5e-08 -- complex64 rounding in
# phase_damping, whose exact value is zero. The tolerance has to sit between them,
# and these two numbers pin that it does.
_ADMITTED_OFFDIAGONAL_CEILING = 1e-7
_SMALLEST_REFUSED_OFFDIAGONAL = 5.0e-2


def _declared_probability(channel: KrausChannel, name: str) -> float:
    """The parameter the channel actually carries, not the one it was built from."""

    declared = dict(channel.parameters)
    return float(declared[name])


def _z(wire: int = 0) -> Hamiltonian:
    return Hamiltonian([HamiltonianTerm(1.0, "z", (wire,))])


def _zz() -> Hamiltonian:
    return Hamiltonian([HamiltonianTerm(1.0, "zz", (0, 1))])


def test_the_channel_is_rebuilt_from_the_declaration_the_lowering_wrote() -> None:
    """A lowered instruction carries the parameter and the operators together."""

    from flagquantum.algorithms.pec import _declared_channel
    from flagquantum.compiler import lower_noise_model
    from flagquantum.core import Instruction

    declared = bit_flip_channel(0.1)
    lowered = lower_noise_model(
        Circuit(1, dtype=torch.complex128).x(0), NoiseModel().add("x", declared)
    )
    channel_instruction = next(
        instruction
        for instruction in lowered.instructions
        if instruction.metadata.get("is_channel")
    )
    rebuilt = _declared_channel(channel_instruction)
    # The parameter is read from the declaration rather than inverted back out of
    # the operators, so a rebuilt channel is the one the model named to within the
    # precision the operators were stored at: 7.0e-08 in the probability, which is
    # the same figure the decomposition's sum-to-one identity carries.
    assert rebuilt.name == declared.name
    assert rebuilt.n_wires == declared.n_wires
    assert dict(rebuilt.parameters)["probability"] == pytest.approx(
        dict(declared.parameters)["probability"], abs=1e-7
    )
    assert len(rebuilt.kraus) == len(declared.kraus)
    # An instruction that carries no operators has no channel to invert, and the
    # refusal names that rather than failing later inside a matrix product.
    no_operators = {"probability": 0.1}
    with pytest.raises(ValueError, match="carries no Kraus operators"):
        _declared_channel(
            Instruction(name="bit_flip", wires=(0,), params=no_operators, matrix=None)
        )
    with pytest.raises(ValueError, match="carries no Kraus operators"):
        _declared_channel(
            Instruction(name="bit_flip", wires=(0,), params=no_operators, matrix=())
        )


def test_the_tolerance_sits_between_the_admitted_and_the_refused_families() -> None:
    """The one number the whole unit turns on, pinned from both sides."""

    admitted = {
        "bit_flip": bit_flip_channel(0.1),
        "phase_flip": phase_flip_channel(0.1),
        "depolarizing": depolarizing_channel(0.1),
        "two_qubit_depolarizing": two_qubit_depolarizing_channel(0.1),
        "phase_damping": phase_damping_channel(0.1),
    }
    measured: dict[str, float] = {}
    for name, channel in admitted.items():
        decomposition = pauli_twirl_decomposition(channel)
        measured[name] = decomposition.largest_offdiagonal
        assert decomposition.largest_offdiagonal <= _ADMITTED_OFFDIAGONAL_CEILING, (
            f"{name} is admitted by the tolerance and its off-diagonal margin is "
            f"{decomposition.largest_offdiagonal:.3e}, above the "
            f"{_ADMITTED_OFFDIAGONAL_CEILING:.0e} ceiling this test pins"
        )
    # phase_damping is the family that sets the ceiling: its exact transfer matrix
    # is diagonal and its measured 3.5e-08 is the channel's own precision.
    assert measured["phase_damping"] == pytest.approx(3.499e-08, rel=0.05)
    refused = {
        "amplitude_damping": amplitude_damping_channel(0.1),
        "coherent_overrotation": coherent_overrotation_channel(0.2),
        "reset_error": reset_error_channel(0.05),
        "thermal_relaxation": thermal_relaxation_channel(0.05, 0.05, 1.0),
    }
    for name, channel in refused.items():
        with pytest.raises(ValueError) as refusal:
            pauli_twirl_decomposition(channel)
        assert "is not a Pauli channel" in str(refusal.value)
        # The message carries the magnitude it measured, so a reader can see the
        # margin rather than take the verdict on trust.
        assert "off-diagonal entry of magnitude" in str(refusal.value)
    assert max(measured.values()) < 1e-6 < _SMALLEST_REFUSED_OFFDIAGONAL


def test_the_refused_magnitudes_are_the_ones_the_message_reports() -> None:
    """Each refusal quotes a number, and the number is the one it measured."""

    for channel, expected in (
        (amplitude_damping_channel(0.1), 1.000e-01),
        (coherent_overrotation_channel(0.2), 1.987e-01),
        (reset_error_channel(0.05), 5.000e-02),
        (thermal_relaxation_channel(0.05, 0.05, 1.0), 1.000e00),
    ):
        with pytest.raises(ValueError) as refusal:
            pauli_twirl_decomposition(channel)
        message = str(refusal.value)
        assert f"{expected:.3e}" in message, (
            f"{channel.name} refused with {message!r}, which does not quote the "
            f"{expected:.3e} this test measured"
        )


def test_bit_flip_reproduces_its_closed_form_at_the_declared_probability() -> None:
    """gamma = 1/(1-2p), c_I = (1-p)/(1-2p), and the flipped weight is its complement."""

    channel = bit_flip_channel(0.1)
    probability = _declared_probability(channel, "probability")
    decomposition = pauli_twirl_decomposition(channel)
    assert decomposition.pauli_words == ("I", "X", "Y", "Z")
    spectrum = dict(
        zip(decomposition.pauli_words, decomposition.transfer_spectrum, strict=True)
    )
    assert spectrum["I"] == pytest.approx(1.0, abs=_CLOSED_FORM_TOLERANCE)
    assert spectrum["X"] == pytest.approx(1.0, abs=_CLOSED_FORM_TOLERANCE)
    assert spectrum["Y"] == pytest.approx(1.0 - 2.0 * probability, abs=1e-7)
    assert spectrum["Z"] == pytest.approx(1.0 - 2.0 * probability, abs=1e-7)
    weights = dict(
        zip(decomposition.pauli_words, decomposition.coefficients, strict=True)
    )
    assert weights["I"] == pytest.approx(
        (1.0 - probability) / (1.0 - 2.0 * probability), abs=_CLOSED_FORM_TOLERANCE
    )
    assert weights["X"] == pytest.approx(
        -probability / (1.0 - 2.0 * probability), abs=_CLOSED_FORM_TOLERANCE
    )
    assert weights["Y"] == pytest.approx(0.0, abs=_CLOSED_FORM_TOLERANCE)
    assert weights["Z"] == pytest.approx(0.0, abs=_CLOSED_FORM_TOLERANCE)
    assert decomposition.gamma == pytest.approx(
        1.0 / (1.0 - 2.0 * probability), abs=1e-6
    )
    # What a wrong normalization produces: dividing by N**2 instead of N halves
    # every weight and breaks the sum-to-one identity this unit checks. The sum is
    # checked at the module's own tolerance rather than at the arithmetic's alone,
    # because the channel's operators are stored in single precision: the measured
    # gap here is 7.0e-08, six orders below the 0.125 a halved identity weight
    # would leave behind.
    assert abs(sum(decomposition.coefficients) - 1.0) <= 1e-6
    assert decomposition.sampling_overhead == pytest.approx(
        decomposition.gamma**2, rel=1e-12
    )


def test_depolarizing_reproduces_its_closed_form_and_equal_weights() -> None:
    """lambda = 1-4p/3 on every non-identity word, and one shared weight."""

    channel = depolarizing_channel(0.1)
    probability = _declared_probability(channel, "probability")
    decomposition = pauli_twirl_decomposition(channel)
    eigenvalue = 1.0 - 4.0 * probability / 3.0
    spectrum = dict(
        zip(decomposition.pauli_words, decomposition.transfer_spectrum, strict=True)
    )
    assert spectrum["I"] == pytest.approx(1.0, abs=_CLOSED_FORM_TOLERANCE)
    for word in ("X", "Y", "Z"):
        assert spectrum[word] == pytest.approx(eigenvalue, abs=1e-7)
    weights = dict(
        zip(decomposition.pauli_words, decomposition.coefficients, strict=True)
    )
    assert weights["I"] == pytest.approx((1.0 + 3.0 / eigenvalue) / 4.0, abs=1e-6)
    for word in ("X", "Y", "Z"):
        # A wrong sign convention gives +p/(3*lambda) here, which is 0.038 rather
        # than -0.038 -- the same magnitude, the opposite correction.
        assert weights[word] == pytest.approx((1.0 - 1.0 / eigenvalue) / 4.0, abs=1e-7)
        assert weights[word] < 0.0
    assert decomposition.gamma == pytest.approx(
        (3.0 / eigenvalue - 1.0) / 2.0, abs=1e-6
    )


def test_two_qubit_depolarizing_weights_fifteen_words_equally() -> None:
    """Sixteen words, one identity weight, and fifteen identical corrections."""

    channel = two_qubit_depolarizing_channel(0.1)
    probability = _declared_probability(channel, "probability")
    decomposition = pauli_twirl_decomposition(channel)
    assert len(decomposition.pauli_words) == 16
    assert decomposition.pauli_words[0] == "II"
    assert decomposition.pauli_words[1] == "IX"
    eigenvalue = 1.0 - 16.0 * probability / 15.0
    identity = decomposition.coefficients[0]
    others = decomposition.coefficients[1:]
    assert identity == pytest.approx((1.0 + 15.0 / eigenvalue) / 16.0, abs=1e-6)
    for other in others:
        assert other == pytest.approx((1.0 - 1.0 / eigenvalue) / 16.0, abs=1e-8)
    assert decomposition.gamma == pytest.approx(
        (30.0 / eigenvalue - 14.0) / 16.0, abs=1e-6
    )


def test_phase_damping_stays_inside_the_pauli_family_it_is_admitted_at() -> None:
    """Its exact transfer matrix is diagonal; only its stored precision is not."""

    decomposition = pauli_twirl_decomposition(phase_damping_channel(0.1))
    assert 0.0 < decomposition.largest_offdiagonal <= _ADMITTED_OFFDIAGONAL_CEILING
    weights = dict(
        zip(decomposition.pauli_words, decomposition.coefficients, strict=True)
    )
    assert weights["X"] == pytest.approx(0.0, abs=1e-6)
    assert weights["Y"] == pytest.approx(0.0, abs=1e-6)
    assert weights["Z"] < 0.0 < weights["I"]
    assert decomposition.gamma == pytest.approx(1.0540925, abs=1e-6)


@pytest.mark.parametrize(
    "channel",
    [
        bit_flip_channel(0.0),
        bit_flip_channel(0.25),
        phase_flip_channel(0.1),
        depolarizing_channel(0.0),
        depolarizing_channel(0.3),
        phase_damping_channel(0.1),
    ],
)
def test_the_weights_always_sum_to_one_however_the_channel_is_parameterized(
    channel: KrausChannel,
) -> None:
    """Trace preservation survives the inversion; it is checked, not assumed."""

    decomposition = pauli_twirl_decomposition(channel)
    # The identity the inverse must satisfy, checked at the module's own tolerance
    # because the channel's operators are stored in single precision: the largest
    # measured gap over these six channels is 7.0e-08, six orders below the 0.125
    # that a halved identity weight, or a 1/N**2 normalization, would leave.
    assert abs(math.fsum(decomposition.coefficients) - 1.0) <= 1e-6
    assert decomposition.gamma >= 1.0
    assert decomposition.dominant_word[0].strip("IXYZ") == ""
    assert abs(decomposition.dominant_word[1]) == max(
        abs(value) for value in decomposition.coefficients
    )


def test_a_channel_with_no_inverse_is_refused_by_name() -> None:
    """bit_flip at one half sends two words to zero, so the weights diverge."""

    with pytest.raises(ValueError) as refusal:
        pauli_twirl_decomposition(bit_flip_channel(0.5))
    message = str(refusal.value)
    assert "vanishing Pauli transfer eigenvalue" in message
    assert "at Y" in message
    # The measurement behind the refusal, quoted: lambda_Y is exactly zero, and a
    # caller who tried to invert it would divide by zero rather than get a large
    # but finite weight.
    assert "0.000e+00" in message


def test_the_decomposition_refuses_something_that_is_not_a_channel() -> None:
    with pytest.raises(TypeError, match="needs a Kraus channel"):
        pauli_twirl_decomposition("bit_flip")  # type: ignore[arg-type]


def test_a_word_whose_length_disagrees_with_its_qubit_count_is_refused() -> None:
    """A word is read position by position, so its length is the qubit count."""

    # The positive control: one word, its non-identity factors, and the wire each
    # factor sits on. An identity factor is what the channel's wire order is
    # read against, so it contributes nothing and must not be emitted.
    assert _word_operators("IY", 2) == ((1, "y"),)
    assert _word_operators("XI", 2) == ((0, "x"),)
    assert _word_operators("II", 2) == ()
    # A word one character short reads a qubit that is not there, and a word one
    # character long reads a factor off the end, so both are refused by name
    # rather than silently truncated.
    with pytest.raises(ValueError, match="2 qubit\\(s\\) needs 2 character\\(s\\)"):
        _word_operators("X", 2)
    with pytest.raises(ValueError, match="1 qubit\\(s\\) needs 1 character\\(s\\)"):
        _word_operators("XX", 1)


def test_the_refusal_names_the_row_and_the_column_in_that_order() -> None:
    """The message is asymmetric, and it says which word goes where."""

    # The offending entry is (row Z, column I): the channel sends the identity
    # partly to Z, and the transposed entry is 3.5e-08 -- the channel's own
    # precision. Naming the pair the other way round would describe a different
    # matrix, so the direction is pinned rather than left to the reader.
    for channel, magnitude in (
        (amplitude_damping_channel(0.1), "1.000e-01"),
        (reset_error_channel(0.05), "5.000e-02"),
    ):
        with pytest.raises(ValueError) as refusal:
            pauli_twirl_decomposition(channel)
        message = str(refusal.value)
        assert "sends I partly to Z" in message, (
            f"{channel.name} refused with {message!r}, which does not name the "
            "column word first"
        )
        assert "sends Z partly to I" not in message
        assert magnitude in message


def test_the_decomposition_record_refuses_a_non_finite_off_diagonal_margin() -> None:
    """A record that cannot be read against a tolerance is refused, not stored."""

    good = pauli_twirl_decomposition(bit_flip_channel(0.1))
    for margin in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError, match="largest_offdiagonal must be finite"):
            PauliTwirlDecomposition(
                channel_name=good.channel_name,
                n_wires=good.n_wires,
                pauli_words=good.pauli_words,
                coefficients=good.coefficients,
                transfer_spectrum=good.transfer_spectrum,
                largest_offdiagonal=margin,
            )


def test_the_dominant_word_is_read_from_the_weights_rather_than_the_spectrum() -> None:
    """The two sequences are not ordered alike, so the wrong one names the wrong word."""

    good = pauli_twirl_decomposition(bit_flip_channel(0.1))
    # Coefficients sum to exactly 1.0 and the identity carries the largest
    # magnitude, 1.5; the spectrum's largest entry sits on X instead. Reading the
    # key from the spectrum would return X while the weight belongs to I.
    weighted = PauliTwirlDecomposition(
        channel_name=good.channel_name,
        n_wires=good.n_wires,
        pauli_words=good.pauli_words,
        coefficients=(1.5, 0.0, -0.5, 0.0),
        transfer_spectrum=(1.0, 5.0, 1.0, 1.0),
        largest_offdiagonal=good.largest_offdiagonal,
    )
    assert weighted.dominant_word == ("I", 1.5)
    assert weighted.gamma == pytest.approx(2.0, abs=1e-12)


def test_the_decomposition_record_refuses_a_basis_it_cannot_be_read_against() -> None:
    """A record's words and weights must be one basis, not two."""

    good = pauli_twirl_decomposition(bit_flip_channel(0.1))
    with pytest.raises(ValueError, match="identity\\s+first"):
        PauliTwirlDecomposition(
            channel_name=good.channel_name,
            n_wires=good.n_wires,
            pauli_words=("X", "I", "Y", "Z"),
            coefficients=good.coefficients,
            transfer_spectrum=good.transfer_spectrum,
            largest_offdiagonal=good.largest_offdiagonal,
        )
    with pytest.raises(ValueError, match="one coefficient and one transfer"):
        PauliTwirlDecomposition(
            channel_name=good.channel_name,
            n_wires=good.n_wires,
            pauli_words=good.pauli_words,
            coefficients=good.coefficients[:2],
            transfer_spectrum=good.transfer_spectrum,
            largest_offdiagonal=good.largest_offdiagonal,
        )
    with pytest.raises(ValueError, match="must be finite"):
        PauliTwirlDecomposition(
            channel_name=good.channel_name,
            n_wires=good.n_wires,
            pauli_words=good.pauli_words,
            coefficients=(1.0, math.nan, 0.0, 0.0),
            transfer_spectrum=good.transfer_spectrum,
            largest_offdiagonal=good.largest_offdiagonal,
        )
    with pytest.raises(ValueError, match="must sum to one"):
        PauliTwirlDecomposition(
            channel_name=good.channel_name,
            n_wires=good.n_wires,
            pauli_words=good.pauli_words,
            coefficients=(1.5, -0.125, 0.0, 0.0),
            transfer_spectrum=good.transfer_spectrum,
            largest_offdiagonal=good.largest_offdiagonal,
        )
    with pytest.raises(ValueError, match="must name the channel"):
        PauliTwirlDecomposition(
            channel_name="",
            n_wires=good.n_wires,
            pauli_words=good.pauli_words,
            coefficients=good.coefficients,
            transfer_spectrum=good.transfer_spectrum,
            largest_offdiagonal=good.largest_offdiagonal,
        )
    with pytest.raises(ValueError, match="at least one wire"):
        PauliTwirlDecomposition(
            channel_name=good.channel_name,
            n_wires=0,
            pauli_words=(),
            coefficients=(),
            transfer_spectrum=(),
            largest_offdiagonal=0.0,
        )


def test_a_location_refuses_a_decomposition_of_a_different_channel() -> None:
    decomposition = pauli_twirl_decomposition(bit_flip_channel(0.1))
    with pytest.raises(ValueError, match="must name the same channel"):
        PecLocation(0, "phase_flip", (0,), decomposition)
    with pytest.raises(ValueError, match="cannot be negative"):
        PecLocation(-1, decomposition.channel_name, (0,), decomposition)
    with pytest.raises(ValueError, match="names 2"):
        PecLocation(0, decomposition.channel_name, (0, 1), decomposition)
    with pytest.raises(TypeError, match="needs the decomposition"):
        PecLocation(0, "bit_flip", (0,), "bit_flip")  # type: ignore[arg-type]
    location = PecLocation(3, decomposition.channel_name, (0,), decomposition)
    assert location.gamma == decomposition.gamma


def test_the_mitigated_value_is_the_noiseless_one_for_one_qubit_location() -> None:
    """The composite map is the identity, so the read is the noiseless read."""

    circuit = Circuit(1, dtype=torch.complex128).x(0)
    model = NoiseModel().add("x", bit_flip_channel(0.1))
    result = run_pec(circuit, _z(), noise_model=model, dtype=torch.complex128)
    # The unmitigated read is 1.999999326992e-01 away from where it should be;
    # the mitigated one is exactly on it.
    assert result.unmitigated == pytest.approx(-0.799999932699200, abs=1e-12)
    assert abs(result.estimate + 1.0) <= _EXACT_TOLERANCE
    assert result.gamma == pytest.approx(1.250000105158, abs=1e-9)
    assert result.sampling_overhead == pytest.approx(result.gamma**2, rel=1e-12)
    assert result.term_count == 4
    assert result.executions == 5
    assert len(result.locations) == 1
    assert result.locations[0].channel_name == "bit_flip"
    assert result.locations[0].wires == (0,)
    assert result.applied_correction == pytest.approx(
        result.estimate - result.unmitigated, abs=1e-15
    )


@pytest.mark.parametrize(
    "channel",
    [bit_flip_channel(0.1), depolarizing_channel(0.1)],
)
def test_the_correction_is_exact_for_the_families_that_move_this_observable(
    channel: KrausChannel,
) -> None:
    """The two families whose flip actually perturbs the z read are put back."""

    circuit = Circuit(1, dtype=torch.complex128).x(0)
    result = run_pec(
        circuit,
        _z(),
        noise_model=NoiseModel().add("x", channel),
        dtype=torch.complex128,
    )
    assert abs(result.estimate + 1.0) <= 1e-15
    assert abs(result.unmitigated + 1.0) > 1e-3


@pytest.mark.parametrize(
    "channel", [phase_flip_channel(0.1), phase_damping_channel(0.1)]
)
def test_a_phase_family_leaves_this_observable_where_it_was_and_is_still_corrected(
    channel: KrausChannel,
) -> None:
    """The mitigation returns the same value the unmitigated read had, and says so."""

    result = run_pec(
        Circuit(1, dtype=torch.complex128).x(0),
        _z(),
        noise_model=NoiseModel().add("x", channel),
        dtype=torch.complex128,
    )
    # A phase fault on |1> does not move a population, so the unmitigated read is
    # already 1.0 to 7.0e-08 and the correction has nothing to add. The unit still
    # inverts the channel and still reports the cost, which is the honest reading:
    # an unmoved value is not evidence that the inversion worked. bit_flip at the
    # same probability moves this same read by 2.0e-01, which is the contrast that
    # makes this test mean something.
    assert abs(result.unmitigated + 1.0) < 1e-7
    assert abs(result.estimate + 1.0) <= 1e-15 or abs(result.estimate + 1.0) < 1e-7
    moved = run_pec(
        Circuit(1, dtype=torch.complex128).x(0),
        _z(),
        noise_model=NoiseModel().add("x", bit_flip_channel(0.1)),
        dtype=torch.complex128,
    )
    assert abs(moved.unmitigated + 1.0) > 1e-1
    assert abs(moved.estimate + 1.0) <= 1e-15


def test_a_precision_the_channel_declares_limits_the_correction() -> None:
    """The floor is the channel's own stored precision, not the simulation's."""

    observable = _z()
    channel = phase_damping_channel(0.1)
    model = NoiseModel().add("x", channel)
    double = run_pec(
        Circuit(1, dtype=torch.complex128).x(0),
        observable,
        noise_model=model,
        dtype=torch.complex128,
    )
    single = run_pec(
        Circuit(1, dtype=torch.complex64).x(0), observable, noise_model=model
    )
    # phase_damping's transfer matrix is diagonal but its Kraus operators are
    # stored in single precision, so even the exact path stops 3.5e-8 short. The
    # single-precision run is no worse, because it is not the simulation that
    # sets the floor here.
    assert abs(double.estimate + 1.0) == pytest.approx(3.499e-08, rel=0.05)
    assert abs(single.estimate + 1.0) < 1e-6
    # A channel whose declared transfer matrix is stored exactly enough does reach
    # the simulation's own floor, which is why the two are reported separately.
    exact = run_pec(
        Circuit(1, dtype=torch.complex128).x(0),
        observable,
        noise_model=NoiseModel().add("x", bit_flip_channel(0.1)),
        dtype=torch.complex128,
    )
    assert abs(exact.estimate + 1.0) == 0.0
    assert abs(float(exact.estimate) - float(single.estimate)) < 1e-6


def test_two_locations_compose_and_each_one_carries_its_own_cost() -> None:
    """One inverted channel per location, and gamma is the product of the two."""

    circuit = Circuit(2, dtype=torch.complex128).x(0).x(1)
    model = NoiseModel().add("x", bit_flip_channel(0.1))
    result = run_pec(circuit, _zz(), noise_model=model, dtype=torch.complex128)
    assert [location.wires for location in result.locations] == [(0,), (1,)]
    assert [location.program_index for location in result.locations] == [1, 3]
    assert result.term_count == 16
    assert result.executions == 17
    assert result.unmitigated == pytest.approx(0.639999892318725, abs=1e-12)
    assert abs(result.estimate - 1.0) <= _EXACT_TOLERANCE
    assert result.gamma == pytest.approx(1.562500262894, abs=1e-9)
    assert result.gamma == pytest.approx(result.locations[0].gamma ** 2, rel=1e-12)


def test_a_model_naming_one_wire_costs_what_it_declares_and_no_more() -> None:
    """The cost is the declared model's; a location the model omits is not inverted."""

    circuit = Circuit(2, dtype=torch.complex128).x(0).x(1)
    observable = _zz()
    one = run_pec(
        circuit,
        observable,
        noise_model=NoiseModel().add("x", bit_flip_channel(0.1), wires=(0,)),
        dtype=torch.complex128,
    )
    both = run_pec(
        circuit,
        observable,
        noise_model=NoiseModel().add("x", bit_flip_channel(0.1)),
        dtype=torch.complex128,
    )
    # Both models describe a program whose only noisy wire is wire 0, so both put
    # the read back on 1.0: the composite map is the identity of each declared
    # model, not of the device. What differs is what the run spends, and the
    # unmitigated read the single-location model starts from, which is 1.6e-1
    # better than the two-location one because wire 1 is not noisy under it. A read
    # that recovers the ideal is therefore evidence about the declared model before
    # it is evidence about anything else.
    assert one.locations[0].wires == (0,)
    assert one.locations[0].program_index == 1
    assert one.term_count == 4
    assert one.unmitigated == pytest.approx(0.799999932699200, abs=1e-12)
    assert abs(one.estimate - 1.0) <= _EXACT_TOLERANCE
    assert abs(both.estimate - 1.0) <= _EXACT_TOLERANCE
    assert both.unmitigated == pytest.approx(0.639999892318725, abs=1e-12)
    assert both.unmitigated < one.unmitigated
    assert both.gamma == pytest.approx(one.gamma**2, rel=1e-12)
    assert both.term_count == 4 * one.term_count


def test_the_correction_also_removes_a_drift_on_an_untouched_observable() -> None:
    """Noise on another wire still perturbs the read; inverting it removes that too."""

    circuit = Circuit(2, dtype=torch.complex128).x(0).x(1)
    observable = _z(0)
    model = NoiseModel().add("x", bit_flip_channel(0.1), wires=(1,))
    result = run_pec(circuit, observable, noise_model=model, dtype=torch.complex128)
    assert result.locations[0].wires == (1,)
    # The channel's operators are stored in single precision, so applying it is
    # not exactly trace-preserving at double precision: z(0) drifts 7.0e-8 while
    # the observable's own value never changes. Inverting the channel puts it back
    # exactly, which is a statement about the stored channel rather than about a
    # device.
    assert result.unmitigated == pytest.approx(-1.0, abs=1e-6)
    assert result.unmitigated > -1.0
    assert result.estimate == -1.0


def test_a_two_wire_channel_is_inverted_at_the_location_its_gate_occupies() -> None:
    circuit = Circuit(2, dtype=torch.complex128).h(0).cx(0, 1)
    model = NoiseModel().add("cx", two_qubit_depolarizing_channel(0.05))
    result = run_pec(circuit, _zz(), noise_model=model, dtype=torch.complex128)
    assert result.locations[0].wires == (0, 1)
    assert result.locations[0].decomposition.n_wires == 2
    assert result.term_count == 16
    assert result.unmitigated == pytest.approx(0.946666619956318, abs=1e-12)
    assert abs(result.estimate - 1.0) <= _EXACT_TOLERANCE
    assert result.gamma == pytest.approx(1.105633857998, abs=1e-9)
    assert result.sampling_overhead == pytest.approx(1.222426227953, abs=1e-9)


def test_two_programs_carrying_the_same_channel_are_corrected_the_same_way() -> None:
    """The decomposition is the channel's; the program index is the program's."""

    observable = _zz()
    channel = bit_flip_channel(0.1)
    explicit = run_pec(
        Circuit(2, dtype=torch.complex128).h(0).cx(0, 1),
        observable,
        noise_model=NoiseModel().add("cx", channel),
        dtype=torch.complex128,
    )
    spread = run_pec(
        Circuit(2, dtype=torch.complex128).x(0).x(1),
        observable,
        noise_model=NoiseModel().add("x", channel),
        dtype=torch.complex128,
    )
    # The same two-location gamma, because both programs carry the same channel
    # twice; the same mitigated value, because both start from the same
    # computational basis state. What the program decides is where the locations
    # sit: the entangling program's `cx` is a single instruction at index 2, while
    # the product program's two `x` gates sit at 1 and 3.
    assert explicit.gamma == pytest.approx(spread.gamma, rel=1e-12)
    assert explicit.term_count == spread.term_count == 16
    assert abs(explicit.estimate - spread.estimate) <= _EXACT_TOLERANCE
    assert explicit.unmitigated == pytest.approx(spread.unmitigated, abs=1e-15)
    assert [location.program_index for location in explicit.locations] == [2, 3]
    assert [location.program_index for location in spread.locations] == [1, 3]
    assert [location.decomposition.coefficients for location in explicit.locations] == [
        location.decomposition.coefficients for location in spread.locations
    ]


def test_the_result_refuses_a_term_count_that_does_not_match_its_locations() -> None:
    location = PecLocation(
        0, "bit_flip", (0,), pauli_twirl_decomposition(bit_flip_channel(0.1))
    )
    with pytest.raises(ValueError, match="product of the locations' word counts"):
        PecResult(
            estimate=1.0,
            unmitigated=0.5,
            locations=(location,),
            term_count=5,
        )
    with pytest.raises(ValueError, match="at least one inverted location"):
        PecResult(estimate=1.0, unmitigated=0.5, locations=(), term_count=1)
    with pytest.raises(ValueError, match="estimate must be finite"):
        PecResult(
            estimate=math.nan,
            unmitigated=0.5,
            locations=(location,),
            term_count=4,
        )
    with pytest.raises(ValueError, match="unmitigated must be finite"):
        PecResult(
            estimate=1.0,
            unmitigated=math.inf,
            locations=(location,),
            term_count=4,
        )
    with pytest.raises(ValueError, match="assumptions must state"):
        PecResult(
            estimate=1.0,
            unmitigated=0.5,
            locations=(location,),
            term_count=4,
            assumptions=(),
        )
    with pytest.raises(ValueError, match="limitations must state"):
        PecResult(
            estimate=1.0,
            unmitigated=0.5,
            locations=(location,),
            term_count=4,
            limitations=("  ",),
        )
    with pytest.raises(TypeError, match="needs the locations"):
        PecResult(
            estimate=1.0,
            unmitigated=0.5,
            locations=(1,),  # type: ignore[arg-type]
            term_count=4,
        )


def test_the_declared_limitations_name_what_the_unit_does_not_do() -> None:
    text = " ".join(PEC_LIMITATIONS)

    # Phrase pins, in the spirit of the algorithm example suite: what must not
    # quietly disappear is that the combination is not a physical state, that the
    # overhead is a sampled estimate's cost rather than this path's, that the
    # inverse only cancels what the model declares, and that the two remaining
    # techniques are still absent.
    assert "not a physical state" in text
    assert "sampling_overhead is gamma squared" in text
    assert "survives the combination" in text
    assert "four terms per single-qubit noise location" in text
    assert "Clifford data regression and readout-error mitigation are absent" in text
    assert "no gate-folding scale factor is offered" in text
    assumptions = " ".join(PEC_ASSUMPTIONS)
    assert "exactly the channel the model declares" in assumptions
    assert "Each declared channel is a Pauli channel" in assumptions
    assert "is not part of rho" in assumptions
    assert len(PEC_LIMITATIONS) >= 5 and len(PEC_ASSUMPTIONS) >= 3


def test_the_noise_model_type_is_checked_before_anything_is_simulated() -> None:
    with pytest.raises(TypeError, match="needs the noise model to invert"):
        run_pec(  # type: ignore[arg-type]
            Circuit(1).x(0), _z(), noise_model={"bit_flip": 0.1}
        )


def test_a_readout_rule_is_refused_rather_than_left_unused() -> None:
    """Tr(O rho) is read before measurement, so a readout rule cannot apply."""

    model = NoiseModel()
    model.add("x", bit_flip_channel(0.1))
    model.add_readout(0, ReadoutError(((0.95, 0.05), (0.05, 0.95))))
    with pytest.raises(ValueError, match="declares a readout rule"):
        run_pec(Circuit(1).x(0), _z(), noise_model=model)


def test_a_circuit_batch_is_refused_because_one_expectation_is_read_per_term() -> None:
    model = NoiseModel().add("x", bit_flip_channel(0.1))
    with pytest.raises(ValueError, match="batch size must be one"):
        run_pec(Circuit(1, bsz=2).x(0), _z(), noise_model=model)


def test_a_declared_input_state_is_refused_because_the_ir_starts_at_zero() -> None:
    circuit = Circuit(1, inputs=torch.tensor([[1.0, 0.0]])).x(0)
    model = NoiseModel().add("x", bit_flip_channel(0.1))
    with pytest.raises(ValueError, match="declares its own input state"):
        run_pec(circuit, _z(), noise_model=model)


def test_a_model_that_puts_no_channel_in_the_program_is_refused() -> None:
    with pytest.raises(ValueError, match="no effect on this program"):
        run_pec(
            Circuit(1).x(0),
            _z(),
            noise_model=NoiseModel().add("h", bit_flip_channel(0.1)),
        )
    with pytest.raises(ValueError, match="puts no channel in this program"):
        run_pec(Circuit(1).x(0), _z(), noise_model=NoiseModel())


def test_every_refusal_happens_before_the_first_expectation_is_read() -> None:
    """A mis-specified request costs no simulation time, which is checkable."""

    class _Unreachable(Hamiltonian):
        def expectation(self, target: object, **kwargs: object) -> torch.Tensor:
            raise AssertionError(
                "the request should have been refused before any term was simulated"
            )

    observable = _Unreachable([HamiltonianTerm(1.0, "z", (0,))])
    for model in (
        NoiseModel().add("x", amplitude_damping_channel(0.1)),
        NoiseModel().add("x", bit_flip_channel(0.5)),
    ):
        with pytest.raises(ValueError):
            run_pec(
                Circuit(1, dtype=torch.complex128).x(0), observable, noise_model=model
            )
    wide = Circuit(6, dtype=torch.complex128)
    for wire in range(6):
        wide.x(wire)
    with pytest.raises(ValueError, match="above\\s+this unit's cap"):
        run_pec(
            wide,
            _Unreachable([HamiltonianTerm(1.0, "z", (0,))]),
            noise_model=NoiseModel().add("x", bit_flip_channel(0.1)),
        )


def test_the_term_cap_admits_five_locations_and_refuses_six() -> None:
    five = Circuit(5, dtype=torch.complex128)
    for wire in range(5):
        five.x(wire)
    model = NoiseModel().add("x", bit_flip_channel(0.1))
    result = run_pec(five, _z(), noise_model=model, dtype=torch.complex128)
    assert result.term_count == 1024
    assert result.gamma == pytest.approx(1.250000105158**5, abs=1e-6)
    assert abs(result.estimate + 1.0) <= 1e-6


def test_the_unit_is_reachable_from_its_package_and_not_from_the_root() -> None:
    """The algorithms package adds no root-level ``fq.`` name."""

    import flagquantum
    import flagquantum.algorithms as algorithms

    assert algorithms.run_pec is run_pec
    assert algorithms.pauli_twirl_decomposition is pauli_twirl_decomposition
    assert "run_pec" in algorithms.__all__
    assert algorithms.PecResult is PecResult
    assert not hasattr(flagquantum, "run_pec")

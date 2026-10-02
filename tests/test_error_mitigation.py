"""Zero-noise extrapolation against the density simulator, end to end.

Every number pinned here was measured on this configuration with
``dtype=torch.complex128`` and no sampling, so the only spread between runs is
floating-point summation order. The four scale factors ``1, 3, 5, 7`` are used
throughout because the curve they produce has a measured degree of two for the
single-``cx`` depolarizing family and of three for a coherent over-rotation, and
because a degree-two curve fitted at degree one leaves a residual that is seven
orders above the degree-two floor -- which is the diagnostic this unit exists to
report.

The scale factors are the abscissas; the expectations below are the ordinates,
listed here once so each assertion can name the curve it is reading:

    scale factor   depolarizing(0.05)   coherent over-rotation(0.15 x)
    1              0.871111109           0.955336489
    3              0.639999987           0.621609968
    5              0.444444444           0.070737202
    7              0.284444453          -0.504846105

The noiseless value of the same observable is 0.9999999403953552, which is what
``zz`` on ``h(0); cx(0, 1)`` evaluates to in this environment.
"""

from __future__ import annotations

import math

import pytest
import torch

import flagquantum as fq
from flagquantum.algorithms import Hamiltonian, HamiltonianTerm, run_zne
from flagquantum.algorithms.error_mitigation import scale_noise_model
from flagquantum.noise import (
    NoiseModel,
    coherent_overrotation_channel,
    depolarizing_channel,
)

pytestmark = pytest.mark.integration

_NOISELESS = 0.9999999403953552
_SCALE_FACTORS = (1.0, 3.0, 5.0, 7.0)
_DEPOLARIZING_CURVE = (
    0.8711111092567441,
    0.6399999872843424,
    0.44444444444444414,
    0.2844444529215494,
)
_OVREROTATION_CURVE = (
    0.9553364891256061,
    0.6216099682706645,
    0.07073720166770281,
    -0.5048461045998573,
)
_DEPOLARIZING_PROBABILITY = 0.05
_OVREROTATION_ANGLE = 0.15

# A mitigation claim has two halves: the estimate must land near the noiseless
# value, and it must land nearer to it than the unmitigated point did. The
# measured distances to the noiseless value are 0.1288888907432559 for the
# unmitigated point and 3.03e-9 for the degree-two estimate, a ratio of 4.3e7.
# Requiring a ratio of 1e6 leaves the claim 43x of headroom while still being six
# orders out of reach for a method that returns the unmitigated point, or the
# mean of the curve, or a fit evaluated at the wrong abscissa.
_MINIMUM_IMPROVEMENT_FACTOR = 1e6
# The channel parameters keep the runtime's single precision, so the recovered
# estimate floors at 3.0e-9 from the ideal rather than at a rounding floor. The
# tolerance is 1e-6: 330x above that measured floor, and 1.3e5 below the
# unmitigated error, so it cannot be passed by a rule that fails to extrapolate.
_ESTIMATE_TOLERANCE = 1e-6
# The pinned curve values are reproducible to the last printed digit across
# runs; a tolerance of 1e-9 leaves four orders of headroom while still being far
# below the 1.9e-1 spacing of the curve's own points, so a mis-scaled channel or
# a wrong abscissa cannot reproduce them.
_CURVE_TOLERANCE = 1e-9


@pytest.fixture
def bell_pair() -> fq.Circuit:
    """A circuit whose ``zz`` expectation moves only through the noise."""

    return fq.Circuit(2).h(0).cx(0, 1)


@pytest.fixture
def zz_observable() -> Hamiltonian:
    return Hamiltonian([HamiltonianTerm(1.0, "zz", (0, 1))])


def _depolarizing_model(*, dtype: torch.dtype | None) -> NoiseModel:
    return NoiseModel().add(
        "cx", depolarizing_channel(_DEPOLARIZING_PROBABILITY, dtype=dtype)
    )


def test_the_noiseless_reference_is_what_the_fixture_claims(
    bell_pair: fq.Circuit, zz_observable: Hamiltonian
) -> None:
    """Anchor: the ideal signal is one, so every distance below is measured from it."""

    measured = float(zz_observable.expectation(bell_pair.density_matrix()))

    assert measured == pytest.approx(_NOISELESS, abs=1e-12)
    assert measured == pytest.approx(1.0, abs=1e-6)


def test_a_degree_two_fit_recovers_the_noiseless_expectation(
    bell_pair: fq.Circuit, zz_observable: Hamiltonian
) -> None:
    result = run_zne(
        bell_pair,
        zz_observable,
        noise_model=_depolarizing_model(dtype=torch.complex128),
        scale_factors=_SCALE_FACTORS,
        order=2,
        dtype=torch.complex128,
    )

    assert [item.scale_factor for item in result.measurements] == list(_SCALE_FACTORS)
    for measured, pinned in zip(result.measurements, _DEPOLARIZING_CURVE, strict=True):
        assert measured.expectation == pytest.approx(pinned, abs=_CURVE_TOLERANCE)

    assert result.executions == 4
    assert result.estimate == pytest.approx(_NOISELESS, abs=_ESTIMATE_TOLERANCE)
    assert result.unmitigated == pytest.approx(
        _DEPOLARIZING_CURVE[0], abs=_CURVE_TOLERANCE
    )
    assert result.applied_correction is not None
    assert result.applied_correction == pytest.approx(
        result.estimate - result.unmitigated, abs=1e-12
    )

    improvement = abs(result.unmitigated - _NOISELESS) / abs(
        result.estimate - _NOISELESS
    )
    assert improvement > _MINIMUM_IMPROVEMENT_FACTOR

    # The residual is the evidence that the fitted degree was the right one: the
    # measured floor for this degree is 4.2e-9, and a degree-one fit of the same
    # curve leaves 1.8e-2.
    assert result.fit.max_residual is not None
    assert result.fit.max_residual < 1e-7
    assert result.fit.degrees_of_freedom == 1
    assert not result.fit.interpolates
    assert result.variance_amplification == pytest.approx(2.940625, abs=1e-9)


def test_an_underfit_returns_a_wrong_number_and_a_residual_that_says_so(
    bell_pair: fq.Circuit, zz_observable: Hamiltonian
) -> None:
    """A plausible wrong choice, kept as a test so the diagnostic stays load-bearing."""

    result = run_zne(
        bell_pair,
        zz_observable,
        noise_model=_depolarizing_model(dtype=torch.complex128),
        scale_factors=_SCALE_FACTORS,
        order=1,
        dtype=torch.complex128,
    )

    # Measured: estimate 0.951111100845866, so 4.9e-2 away from the noiseless
    # value rather than 3.0e-9, and the residual is 1.78e-2 -- not a floor.
    assert result.estimate == pytest.approx(0.951111100845866, abs=1e-9)
    assert abs(result.estimate - _NOISELESS) > 1e-2
    assert result.fit.max_residual is not None
    assert result.fit.max_residual == pytest.approx(0.01777779, abs=1e-7)

    # The residual separates the two fits by six orders of magnitude while the
    # estimates differ by five, so a reader who checks the residual first cannot
    # be misled by the agreement of neither.
    assert result.fit.max_residual > 1e-3
    assert result.fit.max_residual / 4.172325152040912e-09 > 1e6
    assert abs(result.estimate - _NOISELESS) / 3.03e-9 > 1e5


def test_richardson_and_the_interpolating_polynomial_fit_agree(
    bell_pair: fq.Circuit, zz_observable: Hamiltonian
) -> None:
    """The same four points, two implementations of the same interpolation."""

    common = {
        "noise_model": _depolarizing_model(dtype=torch.complex128),
        "scale_factors": _SCALE_FACTORS,
        "order": 3,
        "dtype": torch.complex128,
    }
    richardson = run_zne(bell_pair, zz_observable, extrapolation="richardson", **common)
    polynomial = run_zne(
        bell_pair, zz_observable, extrapolation="polynomial_least_squares", **common
    )

    # Both are square fits, so both recover the same interpolating weights from
    # the same measurements and agree to the last printed digit.
    assert richardson.estimate == pytest.approx(polynomial.estimate, abs=1e-12)
    assert richardson.estimate == pytest.approx(_NOISELESS, abs=_ESTIMATE_TOLERANCE)
    assert richardson.variance_amplification == pytest.approx(
        polynomial.variance_amplification, abs=1e-12
    )
    for result in (richardson, polynomial):
        assert result.fit.interpolates
        assert result.fit.degrees_of_freedom == 0
        assert result.fit.max_residual is None


def test_the_fourth_point_buys_accuracy_that_the_third_degree_does_not(
    bell_pair: fq.Circuit, zz_observable: Hamiltonian
) -> None:
    """The cost of a higher degree, measured rather than asserted."""

    common = {
        "noise_model": _depolarizing_model(dtype=torch.complex128),
        "scale_factors": _SCALE_FACTORS,
        "dtype": torch.complex128,
    }
    degree_two = run_zne(bell_pair, zz_observable, order=2, **common)
    degree_three = run_zne(bell_pair, zz_observable, order=3, **common)

    # Measured: degree two lands 3.0e-9 from the noiseless value at a variance
    # amplification of 2.940625, while degree three lands 2.1e-8 away at
    # 11.390625. The higher degree is 3.9x more costly and an order of magnitude
    # less accurate here, which is why the degree is chosen by residual and not
    # by ambition.
    assert degree_two.estimate == pytest.approx(_NOISELESS, abs=_ESTIMATE_TOLERANCE)
    assert degree_three.estimate == pytest.approx(_NOISELESS, abs=_ESTIMATE_TOLERANCE)
    assert degree_three.variance_amplification / degree_two.variance_amplification == (
        pytest.approx(11.390625 / 2.940625, abs=1e-9)
    )
    assert abs(degree_three.estimate - _NOISELESS) > abs(
        degree_two.estimate - _NOISELESS
    )
    assert degree_two.fit.max_residual is not None
    assert degree_two.fit.max_residual < 1e-7


def test_the_degree_is_a_property_of_the_circuit_and_channel_not_a_constant(
    bell_pair: fq.Circuit, zz_observable: Hamiltonian
) -> None:
    """Amplitude damping needs a second degree where depolarizing needs one."""

    from flagquantum.noise import amplitude_damping_channel

    model = NoiseModel().add(
        "cx", amplitude_damping_channel(0.02, dtype=torch.complex128)
    )
    scales = (1.0, 2.0, 3.0, 4.0, 5.0)

    degree_one = run_zne(
        bell_pair,
        zz_observable,
        noise_model=model,
        scale_factors=scales,
        order=1,
        dtype=torch.complex128,
    )
    degree_two = run_zne(
        bell_pair,
        zz_observable,
        noise_model=model,
        scale_factors=scales,
        order=2,
        dtype=torch.complex128,
    )

    # Measured residual 1.6e-3 at degree one against 1.5e-9 at degree two, and
    # estimates 5.6e-3 and 3.6e-9 from the noiseless value.
    assert degree_one.fit.max_residual is not None
    assert degree_two.fit.max_residual is not None
    assert degree_one.fit.max_residual > 1e-4
    assert degree_two.fit.max_residual < 1e-7
    assert degree_two.fit.max_residual < degree_one.fit.max_residual / 1e5
    assert degree_one.estimate == pytest.approx(0.9943999979, abs=1e-8)
    assert degree_two.estimate == pytest.approx(_NOISELESS, abs=_ESTIMATE_TOLERANCE)
    assert degree_two.estimate != pytest.approx(degree_one.estimate, abs=1e-4)


def test_the_default_scaling_refuses_a_family_it_cannot_scale(
    bell_pair: fq.Circuit, zz_observable: Hamiltonian
) -> None:
    model = NoiseModel().add(
        "cx", coherent_overrotation_channel(_OVREROTATION_ANGLE, axis="x")
    )

    with pytest.raises(
        ValueError, match="cannot scale the 'coherent_overrotation' channel"
    ):
        run_zne(
            bell_pair,
            zz_observable,
            noise_model=model,
            scale_factors=_SCALE_FACTORS,
            order=2,
        )


def _overrotation_scaling(model: NoiseModel, factor: float) -> NoiseModel:
    """The caller's claim: the over-rotation angle grows with the scale factor."""

    scaled = NoiseModel()
    for rule in model.rules:
        declared = dict(rule.channel.parameters)
        scaled.add(
            rule.gate_names,
            coherent_overrotation_channel(
                float(declared["angle"]) * factor,
                axis=str(declared["axis"]),
                dtype=rule.channel.kraus[0].dtype,
            ),
            wires=rule.wires,
        )
    return scaled


def test_a_non_polynomial_family_keeps_its_residual_and_its_bias(
    bell_pair: fq.Circuit, zz_observable: Hamiltonian
) -> None:
    """The honesty test: a wrong assumption must not look like a small residual."""

    model = NoiseModel().add(
        "cx",
        coherent_overrotation_channel(
            _OVREROTATION_ANGLE, axis="x", dtype=torch.complex128
        ),
    )
    fits = {
        order: run_zne(
            bell_pair,
            zz_observable,
            noise_model=model,
            scale_factors=_SCALE_FACTORS,
            order=order,
            scaling=_overrotation_scaling,
            dtype=torch.complex128,
        )
        for order in (1, 2)
    }

    for measured, pinned in zip(fits[1].measurements, _OVREROTATION_CURVE, strict=True):
        assert measured.expectation == pytest.approx(pinned, abs=_CURVE_TOLERANCE)

    # Measured: the residual falls from 8.9e-2 to 2.9e-2, so it stays three to
    # five orders above the polynomial family's 1.8e-2-to-4.2e-9 range, and the
    # estimate is 2.7e-1 and then 1.1e-1 away from the noiseless value. A higher
    # degree does not converge, and the residual is what shows it.
    for order, result in fits.items():
        assert result.fit.max_residual is not None, order
        assert result.fit.max_residual > 1e-2, order
        assert abs(result.estimate - _NOISELESS) > 5e-2, order

    assert fits[1].fit.max_residual > fits[2].fit.max_residual
    assert abs(fits[2].estimate - _NOISELESS) > 1e-1
    # Ten thousand times the error the polynomial family reached at the same
    # degree, so this is not a precision floor.
    assert abs(fits[2].estimate - _NOISELESS) / 3.03e-9 > 1e4

    # The claim that the scaled model differs only in noise strength is the
    # caller's, and the result says so.
    assert fits[1].assumptions == fits[2].assumptions
    assert "The scaling callable is the caller's" in fits[1].assumptions[-1]
    assert all(
        "declared scaling multiplies" not in item for item in fits[1].assumptions
    )


def test_the_unmitigated_value_is_reported_only_when_the_curve_measures_it(
    bell_pair: fq.Circuit, zz_observable: Hamiltonian
) -> None:
    model = _depolarizing_model(dtype=torch.complex128)

    without = run_zne(
        bell_pair,
        zz_observable,
        noise_model=model,
        scale_factors=(3.0, 5.0, 7.0),
        order=1,
        dtype=torch.complex128,
    )
    assert without.executions == 3
    assert without.unmitigated is None
    assert without.applied_correction is None

    with_unit = run_zne(
        bell_pair,
        zz_observable,
        noise_model=model,
        scale_factors=_SCALE_FACTORS,
        order=2,
        dtype=torch.complex128,
    )
    assert with_unit.unmitigated == with_unit.measurements[0].expectation
    assert with_unit.unmitigated == pytest.approx(
        _DEPOLARIZING_CURVE[0], abs=_CURVE_TOLERANCE
    )


def test_every_point_names_the_model_that_produced_it(
    bell_pair: fq.Circuit, zz_observable: Hamiltonian
) -> None:
    model = _depolarizing_model(dtype=torch.complex128)

    result = run_zne(
        bell_pair,
        zz_observable,
        noise_model=model,
        scale_factors=_SCALE_FACTORS,
        order=2,
        dtype=torch.complex128,
    )

    identities = [item.noise_model_identity for item in result.measurements]
    assert len(set(identities)) == len(_SCALE_FACTORS)
    for item in result.measurements:
        expected = scale_noise_model(model, item.scale_factor)
        assert item.noise_model_identity == expected.identity


def test_the_curve_is_monotone_in_the_scale_factor(
    bell_pair: fq.Circuit, zz_observable: Hamiltonian
) -> None:
    """A curve that did not decay would make the extrapolation meaningless."""

    result = run_zne(
        bell_pair,
        zz_observable,
        noise_model=_depolarizing_model(dtype=torch.complex128),
        scale_factors=_SCALE_FACTORS,
        order=2,
        dtype=torch.complex128,
    )
    curve = [item.expectation for item in result.measurements]

    assert curve == sorted(curve, reverse=True)
    assert curve[0] - curve[-1] > 0.5


def test_a_scale_factor_beyond_the_channel_bound_is_refused_before_simulating(
    bell_pair: fq.Circuit, zz_observable: Hamiltonian
) -> None:
    # 0.05 scaled by 25 is 1.25, which is not a probability.
    with pytest.raises(ValueError, match="which is above one"):
        run_zne(
            bell_pair,
            zz_observable,
            noise_model=_depolarizing_model(dtype=torch.complex128),
            scale_factors=(1.0, 25.0),
            order=1,
            dtype=torch.complex128,
        )


def test_the_same_configuration_produces_the_same_estimate(
    bell_pair: fq.Circuit, zz_observable: Hamiltonian
) -> None:
    model = _depolarizing_model(dtype=torch.complex128)
    runs = [
        run_zne(
            bell_pair,
            zz_observable,
            noise_model=model,
            scale_factors=_SCALE_FACTORS,
            order=2,
            dtype=torch.complex128,
        )
        for _ in range(2)
    ]

    assert runs[0].estimate == runs[1].estimate
    assert runs[0].fit.weights == runs[1].fit.weights
    assert [item.noise_model_identity for item in runs[0].measurements] == [
        item.noise_model_identity for item in runs[1].measurements
    ]


def test_the_default_runtime_precision_still_recovers_the_noiseless_value(
    bell_pair: fq.Circuit, zz_observable: Hamiltonian
) -> None:
    """Single precision moves the floor, not the conclusion."""

    result = run_zne(
        bell_pair,
        zz_observable,
        noise_model=_depolarizing_model(dtype=None),
        scale_factors=_SCALE_FACTORS,
        order=2,
    )

    # Measured in the default single precision: the estimate is 1.3e-7 from the
    # noiseless value and the residual is 6.3e-8, against an unmitigated
    # distance of 1.288888907432559e-1 -- an improvement of 1.0e6.
    assert result.estimate == pytest.approx(_NOISELESS, abs=1e-5)
    assert result.fit.max_residual is not None
    assert result.fit.max_residual < 1e-6
    assert result.unmitigated is not None
    assert abs(result.unmitigated - _NOISELESS) > 1e-2
    improvement = abs(result.unmitigated - _NOISELESS) / abs(
        result.estimate - _NOISELESS
    )
    assert improvement > 1e5


def test_a_non_finite_scale_factor_is_refused_before_the_first_simulation(
    bell_pair: fq.Circuit, zz_observable: Hamiltonian
) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        run_zne(
            bell_pair,
            zz_observable,
            noise_model=_depolarizing_model(dtype=torch.complex128),
            scale_factors=(1.0, math.inf),
            order=1,
            dtype=torch.complex128,
        )

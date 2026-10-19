"""Extrapolation arithmetic, scaling refusals, and result invariants.

Every numeric assertion here states the measured margin and the divergence a
plausible wrong rule produces, so the tolerance is evidence rather than a
formality. The extrapolations run on exactly-known polynomials in double
precision, where the arithmetic is the only source of error, and the
divergence a wrong rule produces is stated per assertion.
"""

from __future__ import annotations

import math

import pytest
import torch

from flagquantum.algorithms import Hamiltonian, HamiltonianTerm
from flagquantum.algorithms.error_mitigation import (
    EXTRAPOLATION_METHODS,
    ZNE_ASSUMPTIONS,
    ZNE_LIMITATIONS,
    ExtrapolationFit,
    ZneMeasurement,
    ZneResult,
    extrapolate_polynomial,
    extrapolate_richardson,
    richardson_weights,
    run_zne,
    scale_noise_model,
)
from flagquantum.circuit import Circuit
from flagquantum.noise import (
    CorrelatedReadoutError,
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

# The extrapolations below are exact polynomial algebra in float64. The
# vanishing quantity is a rounding floor of order 1e-15 to 1e-13, and the
# divergences a wrong rule produces -- evaluating the curve at the mean scale
# factor, using weights that do not sum to one, or fitting the wrong degree --
# are of order 0.1 to 1. A tolerance of 1e-11 therefore sits about two orders
# above every measured floor recorded here and eleven orders below the smallest
# wrong-rule divergence, so it cannot be satisfied by a wrong rule.
_EXACT_TOLERANCE = 1e-11
# The Richardson weights of a square system solve a small linear system whose
# condition number grows with the spread of the abscissas; the largest measured
# disagreement with the least-squares weights over the grids below is 1.9e-13.
_WEIGHT_TOLERANCE = 1e-11


def _quadratic(scales: tuple[float, ...]) -> tuple[float, ...]:
    """An exactly degree-two curve, so any higher-degree term is zero."""

    return tuple(2.0 + 3.0 * x - 0.5 * x * x for x in scales)


def test_richardson_weights_cancel_every_lower_order_and_sum_to_one() -> None:
    weights = richardson_weights((1.0, 3.0, 5.0), 2)

    assert weights == pytest.approx((1.875, -1.25, 0.375), abs=_EXACT_TOLERANCE)
    # The two defining equations, checked as equations rather than as numbers.
    assert math.fsum(weights) == pytest.approx(1.0, abs=_EXACT_TOLERANCE)
    for order in (1, 2):
        cancel = math.fsum(
            w * x**order for w, x in zip(weights, (1.0, 3.0, 5.0), strict=True)
        )
        assert cancel == pytest.approx(0.0, abs=_EXACT_TOLERANCE)


def test_richardson_weights_refuse_a_point_count_that_is_not_order_plus_one() -> None:
    with pytest.raises(ValueError, match="Richardson extrapolation of order 2"):
        richardson_weights((1.0, 3.0), 2)
    with pytest.raises(ValueError, match="Richardson extrapolation of order 1"):
        richardson_weights((1.0, 3.0, 5.0), 1)


def test_the_richardson_weights_are_the_interpolating_polynomial_weights() -> None:
    """Two methods, one answer, on the point counts where both are defined."""

    for scales in ((1.0, 3.0), (1.0, 2.0, 4.0), (2.0, 3.0, 5.0, 7.0)):
        order = len(scales) - 1
        values = _quadratic(scales)
        richardson = extrapolate_richardson(scales, values, order=order)
        polynomial = extrapolate_polynomial(scales, values, order=order)

        # Measured disagreement over these grids is at most 1.9e-13; a rule that
        # took the least-squares weights where the interpolating ones are meant,
        # or the reverse, differs by order 0.1.
        for left, right in zip(richardson.weights, polynomial.weights, strict=True):
            assert left == pytest.approx(right, abs=_WEIGHT_TOLERANCE)
        assert richardson.estimate == pytest.approx(
            polynomial.estimate, abs=_EXACT_TOLERANCE
        )


def test_a_least_squares_fit_recovers_the_constant_term_of_an_exact_curve() -> None:
    scales = (1.0, 2.0, 3.0, 4.0, 5.0)
    fit = extrapolate_polynomial(scales, _quadratic(scales), order=2)

    # Measured error 2.2e-15 and measured residual 2.7e-15 against a rounding
    # floor near 1e-15; a rule that read off a measured point instead, or fitted
    # a shifted abscissa, is off by order 1.
    assert fit.estimate == pytest.approx(2.0, abs=_EXACT_TOLERANCE)
    assert fit.max_residual is not None
    assert fit.max_residual == pytest.approx(0.0, abs=_EXACT_TOLERANCE)
    assert fit.degrees_of_freedom == 2
    assert not fit.interpolates
    assert fit.variance_amplification == pytest.approx(
        math.fsum(weight * weight for weight in fit.weights), abs=_EXACT_TOLERANCE
    )
    assert fit.weights == pytest.approx(
        (1.8, 0.0, -0.8, -0.6, 0.6), abs=_EXACT_TOLERANCE
    )


def test_a_square_fit_reports_no_residual_instead_of_an_arithmetic_zero() -> None:
    fit = extrapolate_polynomial((1.0, 2.0, 3.0), _quadratic((1.0, 2.0, 3.0)), order=2)

    assert fit.interpolates
    assert fit.degrees_of_freedom == 0
    assert fit.max_residual is None


def test_an_underfit_of_a_quadratic_curve_is_biased_and_its_residual_says_so() -> None:
    """The residual is the only thing standing between a fit and a wrong number."""

    scales = (1.0, 2.0, 3.0, 4.0, 5.0)
    fit = extrapolate_polynomial(scales, _quadratic(scales), order=1)

    # A straight line through a downward-bending curve crosses zero noise well
    # above the true constant term: the measured estimate is 5.5 against 2.0,
    # and the measured residual is 1.0 -- not a rounding floor.
    assert fit.estimate == pytest.approx(5.5, abs=1e-9)
    assert fit.max_residual is not None
    assert fit.max_residual == pytest.approx(1.0, abs=1e-9)
    assert fit.degrees_of_freedom == 3


def test_the_fit_order_must_be_a_positive_integer() -> None:
    with pytest.raises(ValueError, match="must be at least one"):
        extrapolate_polynomial((1.0, 2.0), (1.0, 2.0), order=0)
    with pytest.raises(ValueError, match="must be an integer"):
        extrapolate_polynomial(
            (1.0, 2.0), (1.0, 2.0), order=1.0  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="must be an integer"):
        extrapolate_richardson(
            (1.0, 2.0), (1.0, 2.0), order=True  # type: ignore[arg-type]
        )


def test_a_polynomial_fit_needs_at_least_order_plus_one_points() -> None:
    with pytest.raises(ValueError, match="needs at least 4 scale factors"):
        extrapolate_polynomial((1.0, 2.0, 3.0), (1.0, 2.0, 3.0), order=3)


def test_the_scale_factors_must_be_distinct_positive_and_finite() -> None:
    with pytest.raises(ValueError, match="must be distinct"):
        extrapolate_polynomial((1.0, 1.0, 3.0), (1.0, 2.0, 3.0), order=1)
    with pytest.raises(ValueError, match="must be positive"):
        extrapolate_polynomial((0.0, 1.0, 3.0), (1.0, 2.0, 3.0), order=1)
    with pytest.raises(ValueError, match="must be finite"):
        extrapolate_polynomial((1.0, 2.0, math.inf), (1.0, 2.0, 3.0), order=1)
    with pytest.raises(ValueError, match="must not be empty"):
        extrapolate_polynomial((), (), order=1)


def test_an_extrapolation_needs_one_expectation_per_scale_factor() -> None:
    with pytest.raises(ValueError, match="one expectation per scale factor"):
        extrapolate_polynomial((1.0, 2.0, 3.0), (1.0, 2.0), order=1)
    with pytest.raises(ValueError, match="one expectation per scale factor"):
        extrapolate_richardson((1.0, 2.0), (1.0, 2.0, 3.0), order=1)


def _fit(**overrides: object) -> ExtrapolationFit:
    """A valid degree-one fit of two points, with one attribute replaced.

    The default is the interpolating case, so its residual is legitimately
    absent. Tests that need a fit with freedom to leave a residual override the
    point count, the weights, and the degree of freedom together.
    """

    fields: dict[str, object] = {
        "method": "polynomial_least_squares",
        "order": 1,
        "scale_factors": (1.0, 3.0),
        "expectations": (4.0, 2.0),
        "estimate": 5.0,
        "weights": (1.5, -0.5),
        "variance_amplification": 2.5,
        "max_residual": None,
        "degrees_of_freedom": 0,
    }
    fields.update(overrides)
    return ExtrapolationFit(**fields)  # type: ignore[arg-type]


# A degree-one fit of three points: weights 1.5, -1.0, 0.5 sum to one and give an
# estimate of 1.5*4 - 1.0*2 + 0.5*1 = 4.5 over expectations (4, 2, 1).
_THREE_POINT_FIT: dict[str, object] = {
    "scale_factors": (1.0, 3.0, 5.0),
    "expectations": (4.0, 2.0, 1.0),
    "estimate": 4.5,
    "weights": (1.5, -1.0, 0.5),
    "variance_amplification": 3.5,
    "max_residual": 0.0,
    "degrees_of_freedom": 1,
}


def test_a_fit_refuses_a_self_inconsistent_weight_set() -> None:
    with pytest.raises(ValueError, match="must sum to one"):
        _fit(weights=(0.5, 0.4))
    with pytest.raises(ValueError, match="sum of the squared weights"):
        _fit(variance_amplification=2.0)
    with pytest.raises(ValueError, match="one weight per scale factor"):
        _fit(weights=(1.0, 0.5, -0.5))
    with pytest.raises(ValueError, match="points left over after the fit"):
        _fit(degrees_of_freedom=1)
    with pytest.raises(ValueError, match="unsupported extrapolation method"):
        _fit(method="probabilistic_error_cancellation")


def test_a_fit_reports_a_residual_exactly_when_it_has_freedom_to_leave_one() -> None:
    with pytest.raises(ValueError, match="arithmetic identity rather than a check"):
        _fit(degrees_of_freedom=0, max_residual=0.0)
    with pytest.raises(ValueError, match="must report the residual it left"):
        _fit(**{**_THREE_POINT_FIT, "max_residual": None})
    with pytest.raises(ValueError, match="finite non-negative number"):
        _fit(**{**_THREE_POINT_FIT, "max_residual": -1.0})
    with pytest.raises(ValueError, match="finite non-negative number"):
        _fit(**{**_THREE_POINT_FIT, "max_residual": math.nan})
    with pytest.raises(ValueError, match="the estimate must be finite"):
        _fit(estimate=math.nan)


def test_the_result_must_show_the_curve_the_fit_was_given() -> None:
    fit = extrapolate_polynomial((1.0, 3.0, 5.0), (4.0, 2.0, 1.0), order=1)
    measurements = (
        ZneMeasurement(1.0, 4.0, "identity-a"),
        ZneMeasurement(3.0, 2.0, "identity-b"),
        ZneMeasurement(5.0, 1.0, "identity-c"),
    )
    result = ZneResult(fit.estimate, fit, 4.0, measurements)

    assert result.unmitigated == 4.0
    assert result.executions == 3
    assert result.variance_amplification == fit.variance_amplification
    assert result.applied_correction == pytest.approx(
        fit.estimate - 4.0, abs=_EXACT_TOLERANCE
    )

    swapped = (measurements[1], measurements[0], measurements[2])
    with pytest.raises(ValueError, match="the curve the fit was given"):
        ZneResult(fit.estimate, fit, 4.0, swapped)
    with pytest.raises(ValueError, match="the curve the fit was given"):
        ZneResult(fit.estimate, fit, 4.0, measurements[:2])
    with pytest.raises(ValueError, match="at least one measurement"):
        ZneResult(fit.estimate, fit, None, ())


def test_the_unmitigated_value_must_be_the_measurement_at_one() -> None:
    fit = extrapolate_polynomial((1.0, 3.0, 5.0), (4.0, 2.0, 1.0), order=1)
    measurements = (
        ZneMeasurement(1.0, 4.0, "identity-a"),
        ZneMeasurement(3.0, 2.0, "identity-b"),
        ZneMeasurement(5.0, 1.0, "identity-c"),
    )

    with pytest.raises(ValueError, match="must be reported rather than"):
        ZneResult(fit.estimate, fit, None, measurements)
    with pytest.raises(ValueError, match="must be the measurement at a scale"):
        ZneResult(fit.estimate, fit, 2.0, measurements)

    without_one = (
        ZneMeasurement(3.0, 2.0, "identity-b"),
        ZneMeasurement(5.0, 1.0, "identity-c"),
    )
    scaled = extrapolate_polynomial((3.0, 5.0), (2.0, 1.0), order=1)
    result = ZneResult(scaled.estimate, scaled, None, without_one)
    assert result.unmitigated is None
    assert result.applied_correction is None


def test_a_result_refuses_to_carry_no_caveats() -> None:
    fit = extrapolate_polynomial((1.0, 3.0, 5.0), (4.0, 2.0, 1.0), order=1)
    measurements = (
        ZneMeasurement(1.0, 4.0, "identity-a"),
        ZneMeasurement(3.0, 2.0, "identity-b"),
        ZneMeasurement(5.0, 1.0, "identity-c"),
    )

    with pytest.raises(ValueError, match="assumptions must state at least one"):
        ZneResult(fit.estimate, fit, 4.0, measurements, assumptions=())
    with pytest.raises(ValueError, match="limitations must state at least one"):
        ZneResult(fit.estimate, fit, 4.0, measurements, limitations=())
    with pytest.raises(ValueError, match="assumptions must state at least one"):
        ZneResult(fit.estimate, fit, 4.0, measurements, assumptions=("  ",))


def test_a_measurement_must_name_the_model_that_produced_it() -> None:
    with pytest.raises(ValueError, match="identity is empty"):
        ZneMeasurement(1.0, 0.5, "")
    with pytest.raises(ValueError, match="the scale factor must be positive"):
        ZneMeasurement(0.0, 0.5, "identity")
    with pytest.raises(ValueError, match="the expectation must be finite"):
        ZneMeasurement(1.0, math.nan, "identity")


def test_the_declared_limitations_name_what_the_unit_does_not_do() -> None:
    text = " ".join(ZNE_LIMITATIONS)

    # Phrase pins, in the spirit of the algorithm example suite: what must not
    # quietly disappear is the statement that the estimate carries no measured
    # uncertainty, that the other mitigation techniques are absent and where the
    # one that is not absent lives, and that readout confusion is outside the
    # observable this unit reads.
    assert "no confidence interval" in text
    assert (
        "Probabilistic error cancellation is provided beside it by "
        "flagquantum.algorithms.run_pec" in text
    )
    assert (
        "Probabilistic error cancellation is provided beside it by "
        "flagquantum.algorithms.run_pec" in text
    )
    assert (
        "Clifford data regression is provided beside it by "
        "flagquantum.algorithms.run_cdr" in text
    )
    # The phrase pin is that gate and circuit folding is *separate* rather than
    # missing, and that its unit of scale is a length ratio rather than this
    # unit's channel parameter; pinning the old wording would pin the absence of
    # the folding unit instead of this unit's boundary with it.
    assert (
        "Gate and circuit folding is provided beside it by "
        "flagquantum.algorithms.fold_program" in text
    )
    assert "different units of scale and are refused together" in text
    assert "max_residual is the diagnostic that exposes it" in text
    assert (
        "a model that declares a readout rule is refused rather than measured" in text
    )
    assert "No other process depends on the scale factor" in " ".join(ZNE_ASSUMPTIONS)
    assert len(EXTRAPOLATION_METHODS) == 2


def test_the_result_records_which_scaling_claim_it_rests_on() -> None:
    """The default scaling makes the claim here; a caller's callable makes it there."""

    circuit = Circuit(2).h(0).cx(0, 1)
    observable = Hamiltonian([HamiltonianTerm(1.0, "zz", (0, 1))])
    model = NoiseModel().add("cx", depolarizing_channel(0.05))

    declared = run_zne(
        circuit, observable, noise_model=model, scale_factors=(1.0, 3.0), order=1
    )
    text = " ".join(declared.assumptions)
    assert "The declared scaling multiplies the one error-probability parameter" in text
    assert "The scaling callable is the caller's" not in text

    caller = run_zne(
        circuit,
        observable,
        noise_model=model,
        scale_factors=(1.0, 3.0),
        order=1,
        scaling=lambda noise_model, factor: scale_noise_model(noise_model, factor),
    )
    text = " ".join(caller.assumptions)
    assert "The scaling callable is the caller's" in text
    assert "The declared scaling multiplies" not in text
    # The two runs measured the same curve, so the only difference the result
    # records is which of the two claims the estimate rests on.
    assert caller.measurements == declared.measurements


def test_run_zne_refuses_a_model_whose_readout_rule_it_could_not_apply() -> None:
    """Readout confusion is applied after measurement, so Tr(O rho) cannot carry it."""

    model = NoiseModel()
    model.add("cx", depolarizing_channel(0.05))
    model.add_readout(0, ReadoutError(((0.95, 0.05), (0.05, 0.95))))
    calls: list[float] = []

    def _scaling(inner: NoiseModel, factor: float) -> NoiseModel:
        calls.append(factor)
        return inner

    with pytest.raises(ValueError, match="declares a readout rule"):
        run_zne(
            Hamiltonian,  # type: ignore[arg-type]
            Hamiltonian([HamiltonianTerm(1.0, "z", (0,))]),
            noise_model=model,
            scale_factors=(1.0, 2.0),
            order=1,
            scaling=_scaling,
        )
    assert calls == []


def _scalable_models() -> dict[str, tuple[NoiseModel, str, float]]:
    """One model per scalable channel family, with its parameter and its value."""

    return {
        "bit_flip": (
            NoiseModel().add("h", bit_flip_channel(0.02)),
            "probability",
            0.02,
        ),
        "phase_flip": (
            NoiseModel().add("h", phase_flip_channel(0.02)),
            "probability",
            0.02,
        ),
        "depolarizing": (
            NoiseModel().add("cx", depolarizing_channel(0.02)),
            "probability",
            0.02,
        ),
        "two_qubit_depolarizing": (
            NoiseModel().add("cx", two_qubit_depolarizing_channel(0.02)),
            "probability",
            0.02,
        ),
        "amplitude_damping": (
            NoiseModel().add("h", amplitude_damping_channel(0.02)),
            "gamma",
            0.02,
        ),
        "phase_damping": (
            NoiseModel().add("h", phase_damping_channel(0.02)),
            "gamma",
            0.02,
        ),
    }


def test_scaling_multiplies_the_one_strength_parameter_each_family_declares() -> None:
    for name, (model, parameter, value) in _scalable_models().items():
        scaled = scale_noise_model(model, 3.0)

        declared = dict(scaled.rules[0].channel.parameters)
        assert tuple(declared) == (parameter,), name
        # The channel stores its parameter in the runtime real dtype, and the
        # default here is float32, so the comparison is against the value the
        # channel actually holds rather than against the literal 0.02.
        stored = float(dict(model.rules[0].channel.parameters)[parameter])
        assert declared[parameter] == pytest.approx(stored * 3.0, rel=1e-6), name
        assert declared[parameter] != pytest.approx(stored, rel=1e-6), name
        assert scaled.rules[0].channel.name == name
        assert scaled.rules[0].gate_names == model.rules[0].gate_names
        assert scaled.rules[0].wires == model.rules[0].wires
        assert value == pytest.approx(stored, rel=1e-6)
        assert scaled.identity != model.identity


def test_scaling_leaves_the_original_model_untouched() -> None:
    model = NoiseModel().add("cx", depolarizing_channel(0.05, dtype=torch.complex128))
    before = dict(model.rules[0].channel.parameters)["probability"]

    scale_noise_model(model, 5.0)

    assert dict(model.rules[0].channel.parameters)["probability"] == before
    assert len(model.rules) == 1


def test_scaling_copies_readout_rules_because_readout_error_is_not_gate_noise() -> None:
    model = NoiseModel()
    model.add("cx", depolarizing_channel(0.05))
    model.add_readout(0, ReadoutError(((0.9, 0.1), (0.2, 0.8))))
    model.add_correlated_readout(
        (1, 2),
        CorrelatedReadoutError(
            ((0.9, 0.1, 0.0, 0.0),) * 2 + ((0.0, 0.0, 0.1, 0.9),) * 2
        ),
    )

    scaled = scale_noise_model(model, 3.0)

    assert scaled.readout_rules == model.readout_rules
    assert [rule.wires for rule in scaled.readout_rules] == [(0,), (1, 2)]


def test_scaling_refuses_a_channel_whose_parameter_is_not_an_error_probability() -> (
    None
):
    """The refusal names the family, because the caller must choose what to do."""

    for channel, name in (
        (coherent_overrotation_channel(0.2, axis="x"), "coherent_overrotation"),
        (reset_error_channel(0.02, 0.01), "reset_error"),
        (thermal_relaxation_channel(50.0, 70.0, 1.0), "thermal_relaxation"),
    ):
        model = NoiseModel().add("h", channel)
        with pytest.raises(ValueError, match=f"cannot scale the '{name}' channel"):
            scale_noise_model(model, 2.0)


def test_scaling_refuses_to_drive_a_probability_above_one() -> None:
    model = NoiseModel().add("cx", depolarizing_channel(0.25))

    with pytest.raises(ValueError, match="which is above one"):
        scale_noise_model(model, 5.0)
    # Exactly one is the boundary, and it is reachable rather than refused.
    assert dict(scale_noise_model(model, 4.0).rules[0].channel.parameters) == {
        "probability": pytest.approx(1.0, abs=1e-6)
    }


def test_scaling_refuses_a_scale_factor_that_is_not_positive() -> None:
    model = NoiseModel().add("cx", depolarizing_channel(0.02))

    for factor in (0.0, -1.0, math.inf, math.nan):
        with pytest.raises(
            ValueError, match="scale factor must be positive|must be finite"
        ):
            scale_noise_model(model, factor)
    with pytest.raises(TypeError, match="scaling needs a NoiseModel"):
        scale_noise_model(object(), 2.0)  # type: ignore[arg-type]


def test_scaling_refuses_the_whole_model_when_one_rule_cannot_be_scaled() -> None:
    """All or nothing: a partly scaled model would mix two curves into one."""

    model = NoiseModel()
    model.add("h", depolarizing_channel(0.02))
    model.add("x", coherent_overrotation_channel(0.2, axis="x"))

    with pytest.raises(ValueError, match="coherent_overrotation"):
        scale_noise_model(model, 2.0)


def test_run_zne_validates_the_extrapolation_before_it_simulates_anything() -> None:
    circuit = Hamiltonian  # any object; the simulation must never be reached
    calls: list[float] = []

    def _scaling(model: NoiseModel, factor: float) -> NoiseModel:
        calls.append(factor)
        return model

    with pytest.raises(
        ValueError, match="unsupported extrapolation 'clifford_data_regression'"
    ):
        run_zne(
            circuit,  # type: ignore[arg-type]
            Hamiltonian([HamiltonianTerm(1.0, "z", (0,))]),
            noise_model=NoiseModel(),
            scale_factors=(1.0, 2.0),
            order=1,
            extrapolation="clifford_data_regression",  # type: ignore[arg-type]
            scaling=_scaling,
        )
    assert calls == []

    with pytest.raises(ValueError, match="Richardson extrapolation of order 2"):
        run_zne(
            circuit,  # type: ignore[arg-type]
            Hamiltonian([HamiltonianTerm(1.0, "z", (0,))]),
            noise_model=NoiseModel(),
            scale_factors=(1.0, 2.0),
            order=2,
            extrapolation="richardson",
            scaling=_scaling,
        )
    assert calls == []

    with pytest.raises(TypeError, match="needs the noise model to scale"):
        run_zne(
            circuit,  # type: ignore[arg-type]
            Hamiltonian([HamiltonianTerm(1.0, "z", (0,))]),
            noise_model=None,  # type: ignore[arg-type]
            scale_factors=(1.0, 2.0),
            order=1,
            scaling=_scaling,
        )
    assert calls == []

"""Zero-noise extrapolation over scaled noise models.

The unit measures one observable at several noise strengths and continues the
resulting curve to zero noise. What makes the estimate trustworthy is not the
polynomial fit but the two diagnostics the fit reports: the largest absolute
residual it left, and the factor by which it amplifies the variance of the
points it was given. A fit whose residual is not at the rounding floor is a fit
of a curve that is not a polynomial of that degree, and the extrapolated value
is then a number the method has no claim on.

**The method's one assumption, stated as an obligation.** The estimate is
unbiased exactly when the measured curve is a polynomial of degree at most
``order`` in the scale factor. Everything else follows from it: the ideal
signal must not depend on the scale factor, and no other process may depend on
it either. Neither condition is checked here, because neither is checkable from
the measurements alone.

**Scaling is the caller's claim, and one implementation is offered.** Noise is
scaled by multiplying the single error-probability parameter a channel declares,
which keeps the channel in its own family at a different strength. A channel
whose parameters are not error probabilities -- ``coherent_overrotation``'s
angle, ``reset_error``'s two independent reset probabilities,
``thermal_relaxation``'s time constants and duration, which its factory does not
take as one leading parameter -- is refused with a stated reason rather than
scaled, because multiplying such a parameter does not scale the noise. A caller
who wants one of those supplies a ``scaling`` callable and owns the claim that
the scaled model differs from the original only in noise strength.

**The curve is ``Tr(O rho)``, which is before measurement.** Readout confusion is
a classical misassignment applied after measurement, so it is not in ``rho`` and
this path cannot represent it. A model that declares a readout rule is refused
rather than measured without it, because the alternative is a state-preparation
estimate reported under the name of a measured one.

**What is not here.** Probabilistic error cancellation, Clifford data
regression, circuit folding, shot-based execution, and readout-error mitigation
are all absent. The estimate is a point value with no confidence interval,
because this slice extrapolates exact state expectations rather than samples.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch

from ..noise import (
    KrausChannel,
    NoiseModel,
    amplitude_damping_channel,
    bit_flip_channel,
    depolarizing_channel,
    phase_damping_channel,
    phase_flip_channel,
    two_qubit_depolarizing_channel,
)
from .core import Hamiltonian

if TYPE_CHECKING:  # pragma: no cover - import cycle guard for typing only
    from ..circuit import Circuit

__all__ = (
    "EXTRAPOLATION_METHODS",
    "ExtrapolationFit",
    "ZNE_ASSUMPTIONS",
    "ZNE_LIMITATIONS",
    "ZneMeasurement",
    "ZneResult",
    "extrapolate_polynomial",
    "extrapolate_richardson",
    "richardson_weights",
    "run_zne",
    "scale_noise_model",
)

ExtrapolationMethod = Literal["polynomial_least_squares", "richardson"]

#: The extrapolations this unit implements, in the order the CLI-facing literal
#: in :func:`run_zne` accepts them. One vocabulary: the value a caller passes is
#: the value :attr:`ExtrapolationFit.method` reports.
EXTRAPOLATION_METHODS: tuple[ExtrapolationMethod, ...] = (
    "polynomial_least_squares",
    "richardson",
)

#: What must hold for the extrapolated value to be closer to the noiseless value
#: than the unmitigated one. These are the method's obligations, not the unit's
#: properties, and every result carries a copy so a reader sees them where the
#: number is.
ZNE_ASSUMPTIONS: tuple[str, ...] = (
    "The expectation value is a polynomial of degree at most the fit order in "
    "the scale factor, so the fitted curve's value at zero is the noiseless "
    "value.",
    "The scale factor changes only the noise strength. An ideal signal that "
    "itself depends on the scale factor is extrapolated to the wrong point.",
    "No other process depends on the scale factor. A distortion that is present "
    "at one point of the curve and absent at another is extrapolated as if it "
    "were part of the noise, and nothing in the fit detects that.",
)

_ZERO_ARGUMENT_LIMITATIONS: tuple[str, ...] = (
    "This unit extrapolates exact state expectations, Tr(O rho). It consumes no "
    "shots, reports no confidence interval, and its estimate carries no measured "
    "uncertainty; variance_amplification is the factor a shot-based estimate "
    "would inherit from the same weights.",
    "Readout-error mitigation is absent, and a model that declares a readout rule "
    "is refused rather than measured without it: readout confusion is a classical "
    "misassignment applied after measurement, so it is not part of rho and this "
    "path cannot see it, while extrapolating a curve that omits it would return a "
    "state-preparation estimate under the name of a measured one.",
    "Only zero-noise extrapolation is provided. Probabilistic error cancellation "
    "and Clifford data regression are absent.",
    "Noise is scaled through a channel parameter rather than by folding the "
    "circuit, so no gate-folding scale factor is offered.",
    "Zero-noise extrapolation removes no bias from a family that is not "
    "polynomial in the scale factor. max_residual is the diagnostic that exposes "
    "it, and a square fit's residual is zero by construction and is therefore not "
    "evidence: fit again with more points at a lower order before reading a "
    "residual as a check.",
)

#: The limitations every result carries.
ZNE_LIMITATIONS: tuple[str, ...] = _ZERO_ARGUMENT_LIMITATIONS

_DECLARED_SCALING_ASSUMPTION = (
    "The declared scaling multiplies the one error-probability parameter each "
    "scalable channel declares, so a scaled channel is the same family at a "
    "different strength. A channel whose parameters are not error probabilities "
    "is refused rather than scaled."
)

_CALLER_SCALING_ASSUMPTION = (
    "The scaling callable is the caller's, so the claim that a scaled model "
    "differs from the original only in noise strength is the caller's to make. "
    "This unit does not check it."
)

# Channels whose noise strength really is one probability-valued parameter, with
# the name that parameter carries. Scaling anything else would change the shape
# of the noise rather than only its amount, so anything else is refused. Kept as
# a name-keyed table because the parameter is what the factory's first argument
# means, and there is no other place that fact is recorded.
_SCALABLE_CHANNELS: Mapping[str, tuple[Callable[..., KrausChannel], str]] = {
    "bit_flip": (bit_flip_channel, "probability"),
    "phase_flip": (phase_flip_channel, "probability"),
    "depolarizing": (depolarizing_channel, "probability"),
    "two_qubit_depolarizing": (two_qubit_depolarizing_channel, "probability"),
    "amplitude_damping": (amplitude_damping_channel, "gamma"),
    "phase_damping": (phase_damping_channel, "gamma"),
}


def _finite_floats(values: Iterable[float], *, owner: str) -> tuple[float, ...]:
    """Return ``values`` as finite floats, refusing an empty or non-finite set."""

    numbers: list[float] = []
    for value in values:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{owner} must be finite, and {number!r} was given")
        numbers.append(number)
    if not numbers:
        raise ValueError(f"{owner} must not be empty")
    return tuple(numbers)


def _as_scale_factors(scale_factors: Iterable[float]) -> tuple[float, ...]:
    """Return the scale factors as distinct positive finite floats."""

    factors = _finite_floats(scale_factors, owner="scale factors")
    for factor in factors:
        if factor <= 0.0:
            raise ValueError(
                "a scale factor must be positive, because a scaled channel is "
                f"built by multiplying a parameter, and {factor!r} was given"
            )
    if len(set(factors)) != len(factors):
        raise ValueError(
            "every scale factor must be distinct, because repeated abscissas "
            "make the extrapolation's linear system singular"
        )
    return factors


def _require_order(order: object) -> int:
    """Return ``order`` as a positive int, refusing zero and non-integers."""

    if type(order) is not int:
        raise ValueError("the extrapolation order must be an integer")
    if order < 1:
        raise ValueError(
            "the extrapolation order must be at least one, because order zero "
            "returns the scale-factor-one measurement under the name of a "
            "mitigated estimate without extrapolating anything"
        )
    return order


def _require_point_count(
    scale_factors: Sequence[float],
    order: int,
    method: ExtrapolationMethod,
) -> None:
    """Refuse a point set that cannot support the requested fit.

    One definition, used by both extrapolators and by :func:`run_zne`, so a
    caller who asks for an unsupported fit learns it before any simulation runs.
    """

    if method == "richardson":
        if len(scale_factors) != order + 1:
            raise ValueError(
                f"Richardson extrapolation of order {order} interpolates exactly "
                f"through {order + 1} points, and {len(scale_factors)} were given; "
                "extrapolate with the polynomial least squares method to use more "
                "points than the degree requires"
            )
        return
    if len(scale_factors) < order + 1:
        raise ValueError(
            f"a degree-{order} extrapolation needs at least {order + 1} scale "
            f"factors, and {len(scale_factors)} were given"
        )


def _vandermonde(scale_factors: Sequence[float], order: int) -> torch.Tensor:
    """Return the design matrix whose column ``k`` is the scale factors to ``k``."""

    factors = torch.tensor(scale_factors, dtype=torch.float64)
    return torch.stack([factors**power for power in range(order + 1)], dim=1)


def richardson_weights(
    scale_factors: Sequence[float],
    order: int,
) -> tuple[float, ...]:
    """Return the linear weights whose combination cancels orders one to ``order``.

    The weights solve ``sum_i w_i x_i**k = 0`` for every ``k`` from one to
    ``order`` together with ``sum_i w_i = 1``, so they are the coefficients of
    the interpolating polynomial's constant term and they always sum to one.
    They are also the sensitivity of the estimate to each measurement, which is
    why :attr:`ExtrapolationFit.variance_amplification` is the sum of their
    squares over the same points.

    Raises:
        ValueError: if fewer or more than ``order + 1`` scale factors are given,
            if they are not distinct positive finite values, or if ``order`` is
            not a positive integer.
    """

    factors = _as_scale_factors(scale_factors)
    degree = _require_order(order)
    _require_point_count(factors, degree, "richardson")
    design = _vandermonde(factors, degree)
    target = torch.zeros(degree + 1, dtype=torch.float64)
    target[0] = 1.0
    solution: torch.Tensor = torch.linalg.solve(design.mT, target)
    return tuple(float(weight) for weight in solution)


def _weighted(weights: Sequence[float], values: Sequence[float]) -> float:
    """Return ``sum_i weights[i] * values[i]`` in double precision."""

    weight_tensor = torch.tensor(weights, dtype=torch.float64)
    value_tensor = torch.tensor(values, dtype=torch.float64)
    return float(torch.dot(weight_tensor, value_tensor))


@dataclass(frozen=True, slots=True)
class ExtrapolationFit:
    """One extrapolation of a scaled-expectation curve to zero scale factor.

    Attributes:
        method: Which extrapolation produced this fit, one of
            :data:`EXTRAPOLATION_METHODS`.
        order: The polynomial degree that was fitted.
        scale_factors: The abscissas, in the order they were measured.
        expectations: The ordinates, in the same order.
        estimate: The value the fit assigns to a scale factor of zero.
        weights: The linear weights over ``expectations`` that produce
            ``estimate``, so ``estimate == sum(weights * expectations)``.
        variance_amplification: The sum of the squared weights. It is the factor
            by which the fit scales the variance of one measurement, and it is
            at least one because the weights sum to one.
        max_residual: The largest absolute difference between an ordinate and
            the fitted curve at its abscissa, or ``None`` when the fit
            interpolates a square system. A square fit's residual is zero by
            construction, and reporting it as zero would present an arithmetic
            identity as a check, so it is reported as absent instead.
        degrees_of_freedom: The number of points beyond those the fit consumes,
            which is the number of ways it can fail to pass through the data.
    """

    method: str
    order: int
    scale_factors: tuple[float, ...]
    expectations: tuple[float, ...]
    estimate: float
    weights: tuple[float, ...]
    variance_amplification: float
    max_residual: float | None
    degrees_of_freedom: int

    def __post_init__(self) -> None:
        if self.method not in EXTRAPOLATION_METHODS:
            raise ValueError(
                f"unsupported extrapolation method {self.method!r}; expected one "
                f"of {', '.join(EXTRAPOLATION_METHODS)}"
            )
        _require_order(self.order)
        factors = _as_scale_factors(self.scale_factors)
        object.__setattr__(self, "scale_factors", factors)
        values = _finite_floats(self.expectations, owner="expectations")
        object.__setattr__(self, "expectations", values)
        if len(factors) != len(values):
            raise ValueError(
                "a fit needs one expectation per scale factor, and "
                f"{len(factors)} scale factors carry {len(values)} expectations"
            )
        _require_point_count(factors, self.order, self.method)
        weights = _finite_floats(self.weights, owner="weights")
        if len(weights) != len(factors):
            raise ValueError(
                "a fit needs one weight per scale factor, and "
                f"{len(factors)} scale factors carry {len(weights)} weights"
            )
        object.__setattr__(self, "weights", weights)
        if not math.isfinite(self.estimate):
            raise ValueError(
                f"the estimate must be finite, and {self.estimate!r} is not"
            )
        total = math.fsum(weights)
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                "the extrapolation weights must sum to one so the fit reproduces "
                f"a constant curve exactly, and they sum to {total!r}"
            )
        amplification = math.fsum(weight * weight for weight in weights)
        if abs(amplification - self.variance_amplification) > 1e-9:
            raise ValueError(
                "variance_amplification must be the sum of the squared weights, "
                f"{amplification!r}, and {self.variance_amplification!r} was given"
            )
        if self.degrees_of_freedom != len(factors) - (self.order + 1):
            raise ValueError(
                "degrees_of_freedom must be the points left over after the fit, "
                f"{len(factors) - (self.order + 1)}, and "
                f"{self.degrees_of_freedom} was given"
            )
        if self.degrees_of_freedom == 0:
            if self.max_residual is not None:
                raise ValueError(
                    "a fit that consumes every point interpolates it exactly, so "
                    "its residual is an arithmetic identity rather than a check; "
                    "report it as absent instead of as a number"
                )
        elif self.max_residual is None:
            raise ValueError(
                f"a fit with {self.degrees_of_freedom} degrees of freedom must "
                "report the residual it left, and none was given"
            )
        elif not math.isfinite(self.max_residual) or self.max_residual < 0.0:
            raise ValueError(
                "a residual must be a finite non-negative number, and "
                f"{self.max_residual!r} is not"
            )

    @property
    def interpolates(self) -> bool:
        """Whether the fit consumed every point instead of approximating them."""

        return self.degrees_of_freedom == 0


def extrapolate_richardson(
    scale_factors: Sequence[float],
    expectations: Sequence[float],
    *,
    order: int,
) -> ExtrapolationFit:
    """Interpolate exactly through ``order + 1`` points and read off zero.

    Raises:
        ValueError: if the point counts disagree, if there are not exactly
            ``order + 1`` of them, or if the abscissas are not distinct positive
            finite values.
    """

    factors = _as_scale_factors(scale_factors)
    values = _finite_floats(expectations, owner="expectations")
    if len(factors) != len(values):
        raise ValueError(
            "an extrapolation needs one expectation per scale factor, and "
            f"{len(factors)} scale factors carry {len(values)} expectations"
        )
    degree = _require_order(order)
    _require_point_count(factors, degree, "richardson")
    weights = richardson_weights(factors, degree)
    return ExtrapolationFit(
        method="richardson",
        order=degree,
        scale_factors=factors,
        expectations=values,
        estimate=_weighted(weights, values),
        weights=weights,
        variance_amplification=math.fsum(weight * weight for weight in weights),
        max_residual=None,
        degrees_of_freedom=0,
    )


def extrapolate_polynomial(
    scale_factors: Sequence[float],
    expectations: Sequence[float],
    *,
    order: int,
) -> ExtrapolationFit:
    """Fit a polynomial of degree ``order`` to every point and read off zero.

    The fit is the least-squares one when there are more points than the degree
    requires, and the interpolating one when there are exactly that many. The
    residual it leaves is reported, which is what makes a curve that is not a
    polynomial of this degree visible instead of silent.

    Raises:
        ValueError: if the point counts disagree, if there are fewer than
            ``order + 1`` points, or if the abscissas are not distinct positive
            finite values.
    """

    factors = _as_scale_factors(scale_factors)
    values = _finite_floats(expectations, owner="expectations")
    if len(factors) != len(values):
        raise ValueError(
            "an extrapolation needs one expectation per scale factor, and "
            f"{len(factors)} scale factors carry {len(values)} expectations"
        )
    degree = _require_order(order)
    _require_point_count(factors, degree, "polynomial_least_squares")
    design = _vandermonde(factors, degree)
    ordinates = torch.tensor(values, dtype=torch.float64)
    solution: torch.Tensor = torch.linalg.lstsq(design, ordinates).solution
    residual: torch.Tensor = design @ solution - ordinates
    weights = tuple(float(weight) for weight in torch.linalg.pinv(design)[0])
    remaining = len(factors) - (degree + 1)
    return ExtrapolationFit(
        method="polynomial_least_squares",
        order=degree,
        scale_factors=factors,
        expectations=values,
        estimate=_weighted(weights, values),
        weights=weights,
        variance_amplification=math.fsum(weight * weight for weight in weights),
        max_residual=None if remaining == 0 else float(residual.abs().max()),
        degrees_of_freedom=remaining,
    )


def _scaled_channel(channel: KrausChannel, factor: float) -> KrausChannel:
    """Return ``channel`` with its one error-probability parameter multiplied.

    Raises:
        ValueError: if the channel's family has no single error-probability
            parameter to multiply, or if the product leaves the unit interval
            where a probability has to live.
    """

    name = channel.name
    entry = _SCALABLE_CHANNELS.get(name)
    if entry is None:
        raise ValueError(
            f"cannot scale the {name!r} channel: its parameters are not error "
            "probabilities, so multiplying one would change the shape of the "
            "noise rather than only its strength. Supply a scaling callable that "
            "states what the scaled model is."
        )
    factory, parameter_name = entry
    declared = dict(channel.parameters)
    if tuple(declared) != (parameter_name,):
        raise ValueError(
            f"cannot scale the {name!r} channel: it declares parameters "
            f"{', '.join(sorted(declared))} where one error probability named "
            f"{parameter_name!r} was expected, so the scale factor's meaning "
            "would be ambiguous"
        )
    strength = float(declared[parameter_name]) * factor
    if strength > 1.0:
        raise ValueError(
            f"scaling the {name!r} channel by {factor!r} gives it an error "
            f"probability of {strength!r}, which is above one; the scale factor "
            "is bounded by the channel it scales"
        )
    reference = channel.kraus[0]
    return factory(strength, dtype=reference.dtype, device=reference.device)


def scale_noise_model(model: NoiseModel, factor: float) -> NoiseModel:
    """Return ``model`` with every scalable channel's error strength multiplied.

    Readout rules are copied unchanged. Readout error is a classical
    misassignment rather than a gate error, so scaling it would tie two
    independent noise sources to one scale factor and break the assumption that
    the scale factor changes only the gate noise. A readout-confused expectation
    therefore stays readout-confused after extrapolation.

    Raises:
        ValueError: if ``factor`` is not a positive finite number, or if any rule
            names a channel :func:`_scaled_channel` refuses. The refusal is all
            or nothing: one unscalable rule refuses the whole model rather than
            leaving it partly scaled, which would silently mix two scale factors
            into one curve.
    """

    if not isinstance(model, NoiseModel):
        raise TypeError("scaling needs a NoiseModel")
    number = _finite_floats((factor,), owner="the scale factor")[0]
    if number <= 0.0:
        raise ValueError(f"the scale factor must be positive, and {number!r} was given")
    scaled = NoiseModel(device_profile=model.device_profile)
    for rule in model.rules:
        scaled.add(
            rule.gate_names, _scaled_channel(rule.channel, number), wires=rule.wires
        )
    # Copied verbatim rather than rebuilt: a readout rule is a classical
    # misassignment, not a gate error, and re-deriving one here would be a second
    # constructor for the same record.
    scaled.readout_rules.extend(model.readout_rules)
    return scaled


@dataclass(frozen=True, slots=True)
class ZneMeasurement:
    """One observable measured against one scaled noise model.

    Attributes:
        scale_factor: The factor the noise model was scaled by.
        expectation: The observable's exact expectation under that model.
        noise_model_identity: The scaled model's content hash, so a measurement
            names the model that produced it rather than the model it came from.
    """

    scale_factor: float
    expectation: float
    noise_model_identity: str

    def __post_init__(self) -> None:
        factor = _finite_floats((self.scale_factor,), owner="the scale factor")[0]
        if factor <= 0.0:
            raise ValueError(
                f"the scale factor must be positive, and {factor!r} is not"
            )
        object.__setattr__(self, "scale_factor", factor)
        value = _finite_floats((self.expectation,), owner="the expectation")[0]
        object.__setattr__(self, "expectation", value)
        if not self.noise_model_identity:
            raise ValueError(
                "a measurement must name the model that produced it, and its "
                "identity is empty"
            )


@dataclass(frozen=True, slots=True)
class ZneResult:
    """An extrapolated estimate together with the curve and the caveats it rests on.

    Attributes:
        estimate: The extrapolated value at zero noise.
        fit: The fit the estimate came from, carrying the residual and the
            variance amplification.
        unmitigated: The measurement at a scale factor of exactly one, or
            ``None`` when the scale factors do not include one. It is what the
            extrapolation corrected, and it is not a noiseless value.
        measurements: The measured curve, in the order it was fitted.
        assumptions: What had to hold for the estimate to mean anything.
        limitations: What the estimate does not carry.
    """

    estimate: float
    fit: ExtrapolationFit
    unmitigated: float | None
    measurements: tuple[ZneMeasurement, ...]
    assumptions: tuple[str, ...] = ZNE_ASSUMPTIONS
    limitations: tuple[str, ...] = ZNE_LIMITATIONS

    def __post_init__(self) -> None:
        if not math.isfinite(self.estimate):
            raise ValueError(
                f"the estimate must be finite, and {self.estimate!r} is not"
            )
        if not isinstance(self.fit, ExtrapolationFit):
            raise TypeError("a result needs the fit its estimate came from")
        if not self.measurements:
            raise ValueError("a result needs at least one measurement")
        measured_factors = tuple(item.scale_factor for item in self.measurements)
        measured_values = tuple(item.expectation for item in self.measurements)
        if measured_factors != self.fit.scale_factors or measured_values != (
            self.fit.expectations
        ):
            raise ValueError(
                "the measurements must be the curve the fit was given, in the "
                "same order, so a result cannot report a fit of a different "
                "curve than the one it shows"
            )
        unmitigated = tuple(
            item.expectation for item in self.measurements if item.scale_factor == 1.0
        )
        if self.unmitigated is None:
            if unmitigated:
                raise ValueError(
                    "the curve measures a scale factor of one, so the "
                    "unmitigated value is known and must be reported rather than "
                    "left absent"
                )
        elif len(unmitigated) != 1 or unmitigated[0] != self.unmitigated:
            raise ValueError(
                "the unmitigated value must be the measurement at a scale "
                "factor of one, and it is not"
            )
        for field_name in ("assumptions", "limitations"):
            declared = getattr(self, field_name)
            if not declared or any(not item.strip() for item in declared):
                raise ValueError(
                    f"{field_name} must state at least one non-empty entry, "
                    "because an estimate whose caveats are empty reads as a "
                    "claim the method has not made"
                )

    @property
    def executions(self) -> int:
        """How many scaled models were simulated to produce the estimate."""

        return len(self.measurements)

    @property
    def variance_amplification(self) -> float:
        """The factor the extrapolation scales a measurement's variance by."""

        return self.fit.variance_amplification

    @property
    def applied_correction(self) -> float | None:
        """How far the estimate moved from the unmitigated measurement."""

        if self.unmitigated is None:
            return None
        return self.estimate - self.unmitigated


def _measure(
    circuit: Circuit,
    hamiltonian: Hamiltonian,
    model: NoiseModel,
    factor: float,
    *,
    device: torch.device | str,
    dtype: torch.dtype | None,
) -> ZneMeasurement:
    """Simulate one scaled model exactly and read the observable off it."""

    from ..runtime.noise_registry import noisy_density_matrix

    density = noisy_density_matrix(circuit, model, device=device, dtype=dtype)
    values = torch.as_tensor(hamiltonian.expectation(density)).reshape(-1)
    if values.numel() != 1:
        raise ValueError(
            "zero-noise extrapolation needs one expectation value per model, and "
            f"the Hamiltonian returned {values.numel()}; run it at batch size one"
        )
    return ZneMeasurement(factor, float(values[0]), model.identity)


def run_zne(
    circuit: Circuit,
    hamiltonian: Hamiltonian,
    *,
    noise_model: NoiseModel,
    scale_factors: Sequence[float],
    order: int,
    extrapolation: ExtrapolationMethod = "polynomial_least_squares",
    scaling: Callable[[NoiseModel, float], NoiseModel] = scale_noise_model,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | None = None,
) -> ZneResult:
    """Measure ``hamiltonian`` at each scaled noise model and extrapolate to zero.

    Args:
        circuit: The program whose observable is measured. It is executed
            unchanged once per scale factor, so the ideal signal is the same at
            every point by construction.
        hamiltonian: The observable, evaluated exactly on each scaled model's
            density matrix. Its expectation is a single number, so the circuit's
            batch size is one.
        noise_model: The model at a scale factor of one. It must not declare a
            readout rule: the observable is read as ``Tr(O rho)``, which is
            before measurement, so a classical readout confusion would be
            silently unused.
        scale_factors: The factors to scale it by. Distinct positive finite
            numbers; include ``1.0`` to have the unmitigated measurement
            reported beside the estimate.
        order: The polynomial degree to fit. It also fixes how many points each
            method needs, which is checked before anything is simulated.
        extrapolation: The fit to use, one of :data:`EXTRAPOLATION_METHODS`.
        scaling: How a scaled model is built. The default multiplies each
            channel's error probability and refuses a channel where that
            multiplication has no meaning; a caller-supplied callable takes over
            the claim that the scaled model differs only in noise strength.
        device: The device the density simulation runs on.
        dtype: The density simulation's complex dtype. The default is the runtime
            configuration's, which is single precision; the extrapolation's own
            arithmetic is double precision regardless.

    Returns:
        The estimate, the fit, the measured curve, and the assumptions and
        limitations that go with them.

    Raises:
        ValueError: if the scale factors, the order, or the point count cannot
            support the requested fit, if the model declares a readout rule the
            observable path cannot apply, or if the scaling refuses a channel.
            All of it is checked before the first simulation runs, so a
            mis-specified extrapolation costs no simulation time.
    """

    factors = _as_scale_factors(scale_factors)
    degree = _require_order(order)
    if extrapolation not in EXTRAPOLATION_METHODS:
        raise ValueError(
            f"unsupported extrapolation {extrapolation!r}; expected one of "
            f"{', '.join(EXTRAPOLATION_METHODS)}"
        )
    if not isinstance(noise_model, NoiseModel):
        raise TypeError("zero-noise extrapolation needs the noise model to scale")
    _require_point_count(factors, degree, extrapolation)
    if noise_model.readout_rules:
        raise ValueError(
            "the noise model declares a readout rule, and this unit evaluates the "
            "observable as Tr(O rho) before measurement, where classical readout "
            "confusion is not represented; measuring anyway would report a "
            "state-preparation estimate while the rule stayed silently unused, so "
            "the model is refused instead. Drop the readout rules to extrapolate "
            "the state-preparation estimate, or mitigate readout separately."
        )
    measurements = tuple(
        _measure(
            circuit,
            hamiltonian,
            scaling(noise_model, factor),
            factor,
            device=device,
            dtype=dtype,
        )
        for factor in factors
    )
    fit = (
        extrapolate_richardson
        if extrapolation == "richardson"
        else extrapolate_polynomial
    )(factors, tuple(item.expectation for item in measurements), order=degree)
    unmitigated = next(
        (item.expectation for item in measurements if item.scale_factor == 1.0),
        None,
    )
    assumptions = ZNE_ASSUMPTIONS + (
        (
            _DECLARED_SCALING_ASSUMPTION
            if scaling is scale_noise_model
            else _CALLER_SCALING_ASSUMPTION
        ),
    )
    return ZneResult(
        estimate=fit.estimate,
        fit=fit,
        unmitigated=unmitigated,
        measurements=measurements,
        assumptions=assumptions,
    )

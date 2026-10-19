"""Simultaneous perturbation stochastic approximation for noisy objectives.

SPSA estimates a gradient from two objective evaluations regardless of how many
parameters the objective has.  That constant cost is what makes it usable on a
shot-based or otherwise stochastic objective, where an exact per-parameter
gradient would cost evaluations proportional to the parameter count and would
still be noisy.

At iteration ``k`` the optimizer perturbs every coordinate at once by one random
sign vector ``delta_k`` drawn from the symmetric Bernoulli distribution and
forms

.. math::

    \\hat{g}_k = \\frac{y(\\theta_k + c_k \\delta_k) - y(\\theta_k - c_k \\delta_k)}{2 c_k} \\delta_k^{-1},
    \\qquad \\theta_{k+1} = \\theta_k - a_k \\hat{g}_k,

with the two gain sequences ``c_k = c / k**gamma`` and
``a_k = a / (A + k)**alpha``.  The estimate is biased for every finite ``c_k``
and converges as ``c_k`` shrinks, so the update it produces is an approximation
of a gradient step rather than a gradient: an objective that already has an
exact gradient available is optimized more cheaply and more accurately by
:func:`torch.autograd` or by parameter shift, and this module does not pretend
otherwise.  It reports the number of objective evaluations it has spent so a
caller can compare that cost against a gradient-based alternative.

This module is deliberately torch-native: the parameters are one flat
:class:`torch.Tensor`, and the perturbation is drawn from a
:class:`torch.Generator`, so a run with a seeded generator replays exactly and
on the same device as the parameters.  A caller who does not supply a generator
gets one created for the parameter device, and the draw is then not reproducible
from the call site.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

from ..errors import ValidationError

__all__ = [
    "SPSA_ASSUMPTIONS",
    "SPSA_LIMITATIONS",
    "SPSAOptimizer",
    "SPSAResult",
]

Objective = Callable[[torch.Tensor], torch.Tensor]

#: What has to hold for a run's reported point to mean anything.
SPSA_ASSUMPTIONS: tuple[str, ...] = (
    "The objective returns one finite scalar per call and does not modify the "
    "tensor it is given. Every quantity this unit reports -- the estimate, the "
    "update, and the cost in the trajectory -- is read from those calls, so an "
    "objective that returns a vector or a not-a-number is refused rather than "
    "reduced.",
    "The perturbation is the direction of the finite difference and nothing "
    "else. The estimate divides by the same sign vector it perturbed along, so "
    "every coordinate is treated as having the same scale; a parameter whose "
    "useful step is a thousand times another's is served worse here than by a "
    "method that scales its coordinates, and no such scaling is applied.",
    "The objective may be stochastic. This is the one unit in the package whose "
    "estimate is built to survive a noisy value, which is why it spends two "
    "evaluations per step whatever the parameter count instead of one per "
    "coordinate. A deterministic objective is served more cheaply and more "
    "accurately by autograd or by the simplex search beside this unit.",
)

#: What a result does not carry, and what its trajectory does not say.
SPSA_LIMITATIONS: tuple[str, ...] = (
    "The estimate is biased for every finite perturbation and is not a "
    "gradient: its expectation reaches the objective's gradient only as the "
    "perturbation shrinks, so one estimate is not a descent direction and the "
    "update built from it is an approximation of a gradient step. Nothing in a "
    "result's cost sequence is evidence that the objective decreased, and the "
    "sequence is a record rather than a convergence claim.",
    "The gain sequences are the caller's. `parameter_gain`, `stability`, "
    "`perturbation`, and the two exponents are read as given and are not tuned "
    "to the objective, because a perturbation that is right for one scale of "
    "parameter is wrong for another. The measured consequence is that one "
    "constant is not best on every objective: on a three-qubit MaxCut cost the "
    "measured best energy over four hundred steps moves from -0.977 with the "
    "default perturbation of 0.2 to -1.000 with 0.05, so a run whose cost "
    "sequence stalls is a statement about those constants before it is a "
    "statement about the optimizer.",
    "There is no convergence flag and no stopping rule beyond the step budget. "
    "The perturbation this optimizer divides by shrinks by construction, so a "
    "difference between two calls that are close together is divided by a small "
    "number and looks large; an iterate-spread threshold like the simplex "
    "search's would therefore report a settled point as unsettled. A caller who "
    "needs a convergence statement reads the cost sequence or runs the "
    "deterministic unit instead.",
    "The cost is a count of objective evaluations and not a latency, a device "
    "cost, or a parallel schedule. The two evaluations of one step are "
    "sequential calls, and no batching, vectorization, or asynchronous "
    "evaluation is attempted.",
    "Reproducibility is a property of the generator, not of the optimizer. A "
    "caller who does not supply a `torch.Generator` gets one made for the "
    "parameter device, and the run is then not replayable from the call site.",
)

#: Default stability constant as a fraction of ``maxiter`` when only ``maxiter``
#: is given.  It is Spall's own starting recommendation.
_STABILITY_FRACTION = 0.1

#: Default first step size, chosen so that the initial update is 0.05 of the
#: estimate for every ``maxiter``.
_DEFAULT_FIRST_STEP = 0.05


class SPSAOptimizer:
    """Minimize a scalar objective with two evaluations per iteration.

    The optimizer owns a flat parameter vector and never inspects the objective's
    internals, so the objective may be a shot-based circuit measurement, a
    hardware submission, or any other callable that returns one scalar.  It does
    not use autograd: the objective's return value is read with ``float()`` and
    any graph attached to it is ignored.

    Args:
        maxiter: The number of iterations the caller expects to perform.  Used to
            derive ``stability`` when that is not supplied, and otherwise
            ignored.  Exactly one of ``maxiter`` and ``stability`` is required,
            because the step-size sequence is undefined without a stability
            constant.
        stability: The ``A`` of ``a_k = a / (A + k)**alpha``.  Non-negative; a
            larger value keeps the early steps closer to the initial step size.
        parameter_gain: The ``a`` of the step-size sequence.  Derived from
            ``stability`` and ``parameter_gain_exponent`` when omitted, so that
            the first step is 0.05.
        perturbation: The ``c`` of the perturbation-size sequence, in the units
            of the parameters.  It is the initial perturbation only; later
            iterations use ``c / k**gamma``.
        parameter_gain_exponent: The ``alpha`` of the step-size sequence.
        perturbation_exponent: The ``gamma`` of the perturbation-size sequence.
            It is slightly larger than ``alpha / 2`` so that the perturbation
            vanishes faster than the step size, which is the standard choice.

    Raises:
        ValidationError: If a gain, an exponent, a count, or the generator is not
            what its name promises, or if neither ``maxiter`` nor ``stability``
            is supplied.  Every refusal happens at construction or before the
            first evaluation, never part-way through an optimization.

    Examples:
        Drive a two-qubit circuit to the minimum of a Pauli energy with no
        gradient and two circuit evaluations per step:

        >>> import torch
        >>> import flagquantum as fq
        >>> from flagquantum import algorithms as fqa
        >>> def energy(values):
        ...     circuit = fq.Circuit(2, dtype=torch.complex128)
        ...     circuit = circuit.ry(0, values[0]).ry(1, values[1]).cx(0, 1)
        ...     outputs = fq.expectation(fq.Z(0) + fq.Z(1))
        ...     return -fq.run(circuit, outputs=outputs).expectation().sum()
        >>> optimizer = fqa.SPSAOptimizer(
        ...     maxiter=120, perturbation=0.25, generator=torch.Generator().manual_seed(13)
        ... )
        >>> values = torch.full((2,), 0.4, dtype=torch.float64)
        >>> for _ in range(120):
        ...     values = optimizer.step(energy, values)
        >>> round(float(energy(values)), 2)
        -2.0
        >>> optimizer.evaluations
        240
    """

    def __init__(
        self,
        *,
        maxiter: int | None = None,
        stability: float | None = None,
        parameter_gain: float | None = None,
        perturbation: float = 0.2,
        parameter_gain_exponent: float = 0.602,
        perturbation_exponent: float = 0.101,
        generator: torch.Generator | None = None,
    ) -> None:
        if maxiter is None and stability is None:
            raise ValidationError(
                "SPSAOptimizer needs one of maxiter or stability, because the step "
                "size a / (A + k)**alpha has no value without a stability constant"
            )
        if maxiter is not None and (
            isinstance(maxiter, bool) or not isinstance(maxiter, int) or maxiter <= 0
        ):
            raise ValidationError("maxiter must be a positive integer or None")
        if stability is not None and (
            isinstance(stability, bool)
            or not isinstance(stability, (int, float))
            or not math.isfinite(float(stability))
            or float(stability) < 0.0
        ):
            raise ValidationError(
                "stability must be a finite non-negative number or None"
            )
        for name, value in (
            ("perturbation", perturbation),
            ("parameter_gain_exponent", parameter_gain_exponent),
            ("perturbation_exponent", perturbation_exponent),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValidationError(f"{name} must be a finite number")
        if float(perturbation) <= 0.0:
            raise ValidationError(
                "perturbation must be positive, because a zero perturbation makes "
                "the difference quotient 0/0 and a negative one reverses the "
                "finite-difference direction it is supposed to be symmetric about"
            )
        if parameter_gain is not None and (
            isinstance(parameter_gain, bool)
            or not isinstance(parameter_gain, (int, float))
            or not math.isfinite(float(parameter_gain))
            or float(parameter_gain) <= 0.0
        ):
            raise ValidationError(
                "parameter_gain must be a finite positive number or None"
            )
        if generator is not None and not isinstance(generator, torch.Generator):
            raise ValidationError(
                "generator must be a torch.Generator or None, because the "
                "perturbation is drawn with torch's own random-number generator"
            )

        if stability is None:
            assert maxiter is not None
            resolved_stability = float(maxiter * _STABILITY_FRACTION)
        else:
            resolved_stability = float(stability)
        if parameter_gain is None:
            resolved_gain = float(
                _DEFAULT_FIRST_STEP
                * math.pow(resolved_stability + 1.0, parameter_gain_exponent)
            )
        else:
            resolved_gain = float(parameter_gain)

        self.maxiter = maxiter
        self.stability = resolved_stability
        self.parameter_gain = resolved_gain
        self.perturbation = float(perturbation)
        self.parameter_gain_exponent = float(parameter_gain_exponent)
        self.perturbation_exponent = float(perturbation_exponent)
        self._generator = generator
        self._generator_device: str | None = (
            str(generator.device) if generator is not None else None
        )
        self._index = 1
        self._evaluations = 0

    @property
    def steps(self) -> int:
        """The number of parameter updates this optimizer has applied."""

        return self._index - 1

    @property
    def evaluations(self) -> int:
        """The number of objective evaluations this optimizer has spent.

        One :meth:`step` costs two, whether the objective has one parameter or a
        million.  :meth:`step_and_cost` costs one more for the reported cost.
        """

        return self._evaluations

    @property
    def perturbation_size(self) -> float:
        """The ``c_k`` the next estimate will use."""

        return self.perturbation / math.pow(self._index, self.perturbation_exponent)

    @property
    def step_size(self) -> float:
        """The ``a_k`` the next update will use."""

        return self.parameter_gain / math.pow(
            self.stability + self._index, self.parameter_gain_exponent
        )

    def _generator_for(self, device: torch.device) -> torch.Generator:
        """Return a generator that can draw on ``device``.

        A caller-supplied generator is used as given and a device mismatch is
        refused rather than worked around, because silently drawing the
        perturbation somewhere else would make the replay property depend on
        where the parameters happen to live.  The internal generator follows the
        parameters and is recreated when they move.
        """

        if self._generator is not None:
            if self._generator_device != device.type:
                raise ValidationError(
                    f"the perturbation generator draws on {self._generator_device!r} "
                    f"but the parameters are on {device.type!r}; pass a generator for "
                    "the parameter device, or omit it to let the optimizer create one"
                )
            return self._generator
        if self._generator_device != device.type:
            self._generator = torch.Generator(device=device.type)
            self._generator_device = device.type
        assert self._generator is not None
        return self._generator

    def _perturbation(self, parameters: torch.Tensor) -> torch.Tensor:
        """Draw one symmetric Bernoulli sign vector shaped like ``parameters``."""

        generator = self._generator_for(parameters.device)
        draws = torch.randint(
            0,
            2,
            parameters.shape,
            generator=generator,
            device=parameters.device,
            dtype=torch.int8,
        )
        return draws.to(parameters.dtype) * 2.0 - 1.0

    def _evaluate(self, objective: Objective, argument: torch.Tensor) -> torch.Tensor:
        """Return one finite scalar objective value, or refuse.

        ``argument`` is a tensor this optimizer created, never the caller's own
        parameters, so an objective that writes into what it is handed cannot
        corrupt the caller's values.  It is refused rather than tolerated: the
        two evaluations of one estimate must describe the same base point, and an
        in-place objective makes the second one start from the first one's
        leftovers.
        """

        before = argument.detach().clone()
        value = objective(argument)
        self._evaluations += 1
        if not torch.equal(argument.detach(), before):
            raise ValidationError(
                "the objective modified the tensor it was given; SPSA evaluates "
                "the objective at two perturbed points and reports the cost at the "
                "unperturbed point, so an in-place objective makes those "
                "evaluations describe different parameters"
            )
        if not isinstance(value, torch.Tensor):
            raise ValidationError(
                "the objective must return a torch.Tensor, because SPSA needs a "
                f"scalar it can difference; it returned {type(value).__name__}"
            )
        if value.numel() != 1:
            raise ValidationError(
                "the objective must return exactly one value, because a "
                "simultaneous-perturbation estimate over a vector return has no "
                f"single direction to follow; it returned {value.numel()} values"
            )
        scalar = value.detach().reshape(()).to(torch.float64)
        if not bool(torch.isfinite(scalar)):
            raise ValidationError(
                "the objective returned a non-finite value, so the estimate it "
                "would produce is meaningless; fix the objective rather than "
                "letting a not-a-number into the gain sequences"
            )
        return scalar

    def estimate_gradient(
        self, objective: Objective, parameters: torch.Tensor
    ) -> torch.Tensor:
        """Return the simultaneous-perturbation estimate at ``parameters``.

        Does not advance the iteration counter, so repeated calls at the same
        point use the same ``c_k`` and different perturbations.  Use
        :meth:`step` to estimate and update together.

        Args:
            objective: A callable taking the flat parameter tensor and returning
                one scalar.  It must not modify the tensor it is given.
            parameters: A finite, floating-point, one-dimensional-or-shaped
                tensor of trainable values.

        Returns:
            The estimate, shaped like ``parameters`` and cast to its dtype.

        Raises:
            ValidationError: If the objective is not callable, if the parameters
                are not a finite floating-point tensor, if the objective returns
                something other than one finite scalar, or if the objective
                modified the tensor it was given.
        """

        if not callable(objective):
            raise ValidationError("objective must be callable")
        if not isinstance(parameters, torch.Tensor):
            raise ValidationError(
                f"parameters must be a torch.Tensor; got {type(parameters).__name__}"
            )
        if not parameters.is_floating_point():
            raise ValidationError(
                "parameters must have a floating dtype, because the perturbation "
                f"is added to them; got {parameters.dtype}"
            )
        if parameters.numel() == 0:
            raise ValidationError("parameters must contain at least one value")
        if not bool(torch.isfinite(parameters).all()):
            raise ValidationError("parameters must all be finite")

        size = self.perturbation_size
        signs = self._perturbation(parameters)
        plus = self._evaluate(objective, parameters + size * signs)
        minus = self._evaluate(objective, parameters - size * signs)
        estimate = (plus - minus) / (2.0 * size * signs.to(torch.float64))
        return estimate.to(parameters.dtype)

    def step(self, objective: Objective, parameters: torch.Tensor) -> torch.Tensor:
        """Apply one SPSA update and return the new parameters.

        Costs two objective evaluations regardless of the parameter count.
        """

        estimate = self.estimate_gradient(objective, parameters)
        updated = parameters.to(torch.float64) - self.step_size * estimate
        self._index += 1
        return updated.to(parameters.dtype)

    def step_and_cost(
        self, objective: Objective, parameters: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply one SPSA update and return it with the cost before the update.

        Costs three objective evaluations: two for the estimate and one for the
        reported cost, which is read at ``parameters`` before the update is
        applied.  The reported cost is a detached scalar tensor.
        """

        estimate = self.estimate_gradient(objective, parameters)
        cost = self._evaluate(objective, parameters.clone())
        updated = parameters.to(torch.float64) - self.step_size * estimate
        self._index += 1
        return updated.to(parameters.dtype), cost

    def minimize(
        self,
        objective: Objective,
        parameters: torch.Tensor,
        *,
        steps: int | None = None,
    ) -> SPSAResult:
        """Run the recursion for a fixed budget and return the record.

        This is the shape :meth:`NelderMeadOptimizer.minimize` has, so the two
        gradient-free units can be handed the same objective and compared, which
        is what :func:`~flagquantum.algorithms.variational.maxcut_objective`
        exposes the QAOA cost for.  The step-wise surface stays: a caller
        adapting a shot-based run wants to read a cost between updates, and one
        that wants a fixed budget wants this.

        The step-size and perturbation sequences are indexed by the optimizer's
        own step counter, so a second call on the same object *continues* the
        recursion rather than replaying it.  Both the counter and the generator
        are part of the optimizer's state; a run that has to be replayable is
        made by building a fresh optimizer with the same seed.

        Args:
            objective: A callable taking the flat parameter tensor and returning
                one finite scalar. It must not modify the tensor it is given.
            parameters: The starting point, a finite floating-point tensor of any
                shape with at least one value.
            steps: The number of updates to apply, defaulting to ``maxiter`` when
                the optimizer was built with one. A run has to be given a budget
                one way or the other, because an unbounded loop over a recursive
                step-size sequence terminates on nothing.

        Returns:
            The last point the run reached, the cost at it, the cost before every
            update, and the evaluations this run spent.

        Raises:
            ValidationError: If the objective is not callable, if the starting
                point or the budget is not what its name promises, or if the
                objective returns something other than one finite scalar.
        """

        if steps is None:
            steps = self.maxiter
        if steps is None:
            raise ValidationError(
                "minimize needs a step budget: this optimizer was built without "
                "maxiter, so pass steps or rebuild it with maxiter, because a "
                "recursive step-size sequence has no other stopping rule"
            )
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValidationError(
                f"steps must be a positive integer, and {steps!r} was given"
            )

        point = parameters
        spent_before = self._evaluations
        history: list[float] = []
        for _ in range(steps):
            point, cost = self.step_and_cost(objective, point)
            history.append(float(cost))
        value = self._evaluate(objective, point.clone())
        return SPSAResult(
            parameters=point.detach().clone(),
            value=float(value),
            history=tuple(history),
            evaluations=self._evaluations - spent_before,
            steps=len(history),
        )


@dataclass(frozen=True, slots=True)
class SPSAResult:
    """The point one :meth:`SPSAOptimizer.minimize` call reached, and its costs.

    Attributes:
        parameters: The final point, shaped like the one the run started from.
            Unlike the simplex search's result this is the *last* iterate and not
            a best-so-far: a biased steepest-descent-like recursion has no
            comparison that would make one iterate better than another, and
            keeping a running minimum would report a point the run did not
            return to.
        value: The objective at :attr:`parameters`, as a Python float, read with
            one extra evaluation after the last update.
        history: The objective value at each point *before* the update applied
            there, so it has one entry per update and its first entry is the
            starting cost. It is a record and not a descent: Adam and SPSA alike
            can raise it.
        evaluations: The objective evaluations the run spent, which is
            ``3 * steps + 1``: two for the estimate, one for the reported cost,
            and one for the final value.
        steps: The number of parameter updates the run applied.
        assumptions: What had to hold for the reported point to mean anything.
        limitations: What the point and the trajectory do not carry.
    """

    parameters: torch.Tensor
    value: float
    history: tuple[float, ...]
    evaluations: int
    steps: int
    assumptions: tuple[str, ...] = SPSA_ASSUMPTIONS
    limitations: tuple[str, ...] = SPSA_LIMITATIONS

    def __post_init__(self) -> None:
        if not isinstance(self.parameters, torch.Tensor):
            raise TypeError("parameters must be the tensor the final point is")
        if self.parameters.numel() == 0:
            raise ValueError("parameters must contain at least one value")
        if not math.isfinite(self.value):
            raise ValueError(f"value must be finite, and {self.value!r} is not")
        for name in ("steps", "evaluations"):
            number = getattr(self, name)
            if isinstance(number, bool) or not isinstance(number, int) or number < 1:
                raise ValueError(f"{name} must be a positive integer")
        if len(self.history) != self.steps:
            raise ValueError(
                "history must carry one cost per update, and it holds "
                f"{len(self.history)} entries for {self.steps} steps"
            )
        if not all(math.isfinite(entry) for entry in self.history):
            raise ValueError("history must hold finite costs")
        if self.evaluations != 3 * self.steps + 1:
            raise ValueError(
                "a run spends two evaluations per estimate, one per reported "
                f"cost, and one on the final value, so {self.steps} steps cost "
                f"{3 * self.steps + 1} evaluations and not {self.evaluations}"
            )

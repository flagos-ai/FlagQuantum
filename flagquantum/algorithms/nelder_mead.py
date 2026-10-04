"""Derivative-free minimization by a moving simplex.

Nelder-Mead keeps ``n + 1`` points in ``n`` dimensions, orders them by objective
value, and replaces the worst one each iteration by reflecting it through the
centroid of the others, expanding the reflection when it is the best point so
far, contracting it when it is not, and shrinking the whole simplex toward the
best vertex when even the contraction fails.  It reads nothing but the order of
objective values, so it needs no gradient, no parameter-shift rule, and no
autograd graph, and it is the method of choice when the objective is a black box
whose derivative is unavailable and whose evaluation is deterministic.

Two properties of the method are what make it usable and what make it dangerous,
and both are stated rather than implied.  The first is that it is a *local*
method: it follows the basin its initial simplex was placed in and has no
mechanism for leaving it, so a run reports the best vertex of one basin and not a
global minimum.  The second is that its convergence theory is incomplete in more
than one dimension -- the widely used simplex variant is not guaranteed to
converge to a local minimum even on smooth functions -- so the two spread
thresholds this unit terminates on say that the simplex has collapsed, not that
the point it collapsed onto is a minimum.  Both are carried on the returned
:class:`NelderMeadResult` as ``assumptions`` and ``limitations``.

The contrast with :class:`~flagquantum.algorithms.spsa.SPSAOptimizer` beside it
is the point of having both.  SPSA is for a *stochastic* objective, which it
handles by spending two evaluations per step on a random finite difference, and
it is honest that the estimate is biased.  Nelder-Mead is for a *deterministic*
objective, where comparing two values is meaningful, and it is then exactly
reproducible: nothing here draws a random number, so two runs of the same
objective and the same start return the same point bit for bit.  An objective
that is noisy defeats Nelder-Mead rather than being tolerated by it, because
every decision it makes is a comparison of two noisy values.

This module is deliberately torch-native: the simplex is one tensor of shape
``(n + 1, n)``, so a vertex is a row, the centroid is a column mean, and the
whole iteration is a handful of tensor operations that run wherever the
parameters live.  The objective is a plain callable returning one finite scalar
tensor and is never inspected, so it may be a circuit measurement, a submitted
job, or any other black box.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from ..errors import ValidationError

__all__ = [
    "NELDER_MEAD_ASSUMPTIONS",
    "NELDER_MEAD_LIMITATIONS",
    "NelderMeadOptimizer",
    "NelderMeadResult",
]

Objective = Callable[[torch.Tensor], torch.Tensor]

#: What has to hold for a run's reported vertex to mean anything.
NELDER_MEAD_ASSUMPTIONS: tuple[str, ...] = (
    "The objective is deterministic at the resolution the simplex resolves. "
    "Every decision the method makes is a comparison of two objective values, "
    "so an objective whose value moves between two evaluations of the same "
    "point turns that comparison into noise and the trajectory into a walk "
    "rather than a descent. A stochastic objective is the subject of the SPSA "
    "unit beside this one.",
    "The objective is continuous and the region the simplex explores contains "
    "the point it converges to. Nelder-Mead is a local method: it follows the "
    "basin its initial simplex was placed in, and no restart, multi-start, or "
    "basin-hopping mechanism is applied here.",
    "The ordering of objective values is what the caller means by better. No "
    "feasibility, constraint, or penalty notion enters, so a constrained "
    "problem is only as well posed as the caller's own transformation of it.",
)

#: What a result does not carry, and what its ``converged`` flag does not say.
NELDER_MEAD_LIMITATIONS: tuple[str, ...] = (
    "Convergence is not guaranteed, and `converged` does not mean the vertex is "
    "a local minimum. This method stops when two measured spreads -- the "
    "objective values across the simplex and the vertex positions across the "
    "simplex -- both fall under the caller's thresholds, and a simplex can "
    "collapse onto a point that is not a minimum. A simplex that does not span "
    "the parameter directions is refused, which makes the collinear "
    "construction the published counterexample uses inexpressible here rather "
    "than reproducible; what remains reachable is a local minimum, measured in "
    "the tests as a two-well quartic started on the shallow side reporting "
    "`converged=True` 0.199984362162 above the global minimum. The standard "
    "remedy for both is a restart from the result or several starts compared "
    "against each other, and neither is performed here.",
    "No derivative, curvature, or uncertainty is estimated or reported. The "
    "method reads only the order of objective values, so it carries no "
    "information about the objective's slope at the point it returns, and two "
    "objectives with the same ordering produce the same trajectory.",
    "The cost is a count of objective evaluations and not a latency, a device "
    "cost, or a parallel schedule. Vertices are evaluated one at a time, one "
    "evaluation per objective call, and no vectorization of the simplex is "
    "attempted.",
    "Simplex construction, the four coefficients, the termination thresholds, "
    "and the iteration budget are the caller's. This unit applies no scaling of "
    "the parameters beyond the initial simplex it builds, applies no bounds, and "
    "performs no constraint handling, restart, or parallel evaluation.",
)


@dataclass(frozen=True, slots=True)
class NelderMeadResult:
    """The best vertex of a finished simplex, with the spreads it stopped on.

    Attributes:
        parameters: The best vertex, shaped like the point the run started from.
        value: The objective value at that vertex, as a Python float.
        simplex: The final simplex as a tensor of shape ``(n + 1, n)``, ordered
            by objective value, so row 0 is ``parameters``. It is the evidence a
            run leaves behind: the spreads below are read from it, so it is
            reported rather than discarded.
        value_spread: The largest gap between the best vertex's value and any
            other vertex's value, in objective units.
        simplex_spread: The largest coordinate-wise distance between the best
            vertex and any other vertex, in parameter units.
        tolerance: The simplex-spread threshold the run was given.
        atol: The value-spread threshold the run was given.
        iterations: The number of simplex replacements the run applied.
        evaluations: The number of objective evaluations the run spent.
        converged: Whether both spreads fell under their thresholds. ``False``
            means the iteration budget ran out first, which is a bounded run
            rather than a failed one.
        assumptions: What had to hold for the vertex to mean anything.
        limitations: What the vertex and the flag do not carry.
    """

    parameters: torch.Tensor
    value: float
    simplex: torch.Tensor
    value_spread: float
    simplex_spread: float
    tolerance: float
    atol: float
    iterations: int
    evaluations: int
    converged: bool
    assumptions: tuple[str, ...] = NELDER_MEAD_ASSUMPTIONS
    limitations: tuple[str, ...] = NELDER_MEAD_LIMITATIONS

    def __post_init__(self) -> None:
        if not isinstance(self.parameters, torch.Tensor):
            raise TypeError("parameters must be the tensor the best vertex is")
        if self.parameters.numel() == 0:
            raise ValueError("parameters must contain at least one value")
        if not math.isfinite(self.value):
            raise ValueError(f"the value must be finite, and {self.value!r} is not")
        if not isinstance(self.simplex, torch.Tensor):
            raise TypeError("simplex must be the tensor the vertices are in")
        expected = (self.parameters.numel() + 1, self.parameters.numel())
        if tuple(self.simplex.shape) != expected:
            raise ValueError(
                f"a result over {self.parameters.numel()} parameters needs a "
                f"simplex of shape {expected}, and it has "
                f"{tuple(self.simplex.shape)}"
            )
        if not torch.equal(self.simplex[0], self.parameters.reshape(-1)):
            raise ValueError(
                "the parameters must be the simplex's own best vertex, because "
                "the result reports one point and one simplex and they have to "
                "be the same run's"
            )
        for name in ("value_spread", "simplex_spread", "tolerance", "atol"):
            number = getattr(self, name)
            if not math.isfinite(number) or number < 0.0:
                raise ValueError(f"{name} must be a finite non-negative number")
        for name in ("iterations", "evaluations"):
            number = getattr(self, name)
            if isinstance(number, bool) or not isinstance(number, int) or number < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.iterations == 0 and not self.converged:
            raise ValueError(
                "a run that applied no simplex replacement ended because its "
                "initial simplex already met both thresholds, so it cannot "
                "report that it did not converge"
            )
        if self.evaluations < self.parameters.numel() + 1:
            raise ValueError(
                "a result needs at least one evaluation per vertex, because the "
                "simplex cannot have been ordered without them"
            )
        collapsed = self.value_spread <= self.atol and (
            self.simplex_spread <= self.tolerance
        )
        if self.converged != collapsed:
            raise ValueError(
                "converged must report the two spreads against their own "
                f"thresholds: the spreads are {self.value_spread!r} and "
                f"{self.simplex_spread!r}, the thresholds are {self.atol!r} and "
                f"{self.tolerance!r}, and converged is {self.converged!r}"
            )


def _finite_positive(value: object, *, name: str) -> float:
    """Return ``value`` as a finite positive float, or refuse."""

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValidationError(f"{name} must be a finite positive number")
    return float(value)


class NelderMeadOptimizer:
    """Minimize a scalar objective with no derivative and no random draw.

    The optimizer owns nothing between calls: one :meth:`minimize` call builds a
    simplex from the starting point, runs it, and returns the record.  It does
    not use autograd -- the objective's return value is read with ``float()`` and
    any graph attached to it is ignored.

    Args:
        maxiter: The number of simplex replacements to attempt before returning
            the best vertex found. It is a budget, not a convergence condition.
        initial_step: The offset applied to each coordinate when the initial
            simplex is built from a single starting point, as a fraction. The
            offset for coordinate ``i`` is ``initial_step * max(|x_i|, 1)``, so a
            parameter far from the origin is perturbed proportionally and a
            parameter near it is perturbed by ``initial_step``.
        reflection: The ``rho`` of the reflection through the centroid of the
            other vertices. It must be positive, and should not exceed
            ``expansion``.
        expansion: The ``chi`` of the expansion applied when the reflection is
            better than every vertex. It must exceed 1.
        contraction: The ``gamma`` of the contraction applied when the
            reflection is worse than the second-worst vertex, on the same side
            of the centroid as the reflection when the reflection still beats the
            worst vertex and on the opposite side when it does not. It must lie
            strictly between 0 and 1.
        shrink: The ``sigma`` of the shrink toward the best vertex, applied when
            even the contraction fails to improve on the worst vertex. It must
            lie strictly between 0 and 1.
        tolerance: The simplex-spread threshold. The run stops once no vertex is
            farther than this from the best one in any coordinate.
        atol: The value-spread threshold. The run stops once no vertex's
            objective value differs from the best one's by more than this. Both
            thresholds have to be met, because a flat objective collapses the
            values without moving the vertices and a stalled simplex collapses
            the vertices without agreeing on a value.

    Raises:
        ValidationError: If a coefficient is not what its name promises, if the
            budget or a threshold is not a positive number, or if an initial
            simplex is not what it has to be. Every refusal about the objective
            happens before a vertex is stored beside a value, so a refused run
            leaves no half-described simplex behind.

    Examples:
        Drive a two-qubit circuit to the minimum of a Pauli energy with no
        gradient and no random draw:

        >>> import torch
        >>> import flagquantum as fq
        >>> from flagquantum import algorithms as fqa
        >>> def energy(values):
        ...     circuit = fq.Circuit(2, dtype=torch.complex128)
        ...     circuit = circuit.ry(0, values[0]).ry(1, values[1]).cx(0, 1)
        ...     outputs = fq.expectation(fq.Z(0) + fq.Z(1))
        ...     return -fq.run(circuit, outputs=outputs).expectation().sum()
        >>> result = fqa.NelderMeadOptimizer(maxiter=200).minimize(
        ...     energy, torch.tensor([0.4, 0.4], dtype=torch.float64)
        ... )
        >>> round(result.value, 10)
        -2.0
        >>> result.converged
        True
    """

    def __init__(
        self,
        *,
        maxiter: int = 200,
        initial_step: float = 0.05,
        reflection: float = 1.0,
        expansion: float = 2.0,
        contraction: float = 0.5,
        shrink: float = 0.5,
        tolerance: float = 1e-8,
        atol: float = 1e-8,
    ) -> None:
        if isinstance(maxiter, bool) or not isinstance(maxiter, int) or maxiter < 1:
            raise ValidationError("maxiter must be a positive integer")
        self.initial_step = _finite_positive(initial_step, name="initial_step")
        self.reflection = _finite_positive(reflection, name="reflection")
        self.expansion = _finite_positive(expansion, name="expansion")
        self.contraction = _finite_positive(contraction, name="contraction")
        self.shrink = _finite_positive(shrink, name="shrink")
        if self.expansion <= 1.0:
            raise ValidationError(
                "expansion must exceed 1, because an expansion that does not "
                "reach past the reflection is not an expansion"
            )
        if self.expansion < self.reflection:
            raise ValidationError(
                "expansion must be at least reflection, because an expansion "
                "closer to the centroid than the reflection would move the "
                "simplex back toward the vertex it is replacing"
            )
        if not 0.0 < self.contraction < 1.0:
            raise ValidationError(
                "contraction must lie strictly between 0 and 1, because it "
                "places the contracted vertex between the centroid and the "
                "vertex it is compared against"
            )
        if not 0.0 < self.shrink < 1.0:
            raise ValidationError(
                "shrink must lie strictly between 0 and 1, because it moves "
                "every vertex toward the best one without passing it"
            )
        self.tolerance = _finite_positive(tolerance, name="tolerance")
        self.atol = _finite_positive(atol, name="atol")
        self.maxiter = maxiter
        self._evaluations = 0

    @property
    def evaluations(self) -> int:
        """The number of objective evaluations the last run spent.

        Read after :meth:`minimize`; it is reset at the start of every run, so it
        describes one run rather than a lifetime.
        """

        return self._evaluations

    def initial_simplex(self, parameters: torch.Tensor) -> torch.Tensor:
        """Return the ``(n + 1, n)`` simplex a run would start from.

        Exposed because the starting simplex is the caller's problem and not
        this unit's: it decides which basin a local method can reach at all, and
        a caller who knows the scales and correlations of their parameters should
        build it rather than accept the coordinate-wise default.  The default
        places the given point first and offsets coordinate ``i`` by
        ``initial_step * max(|x_i|, 1)``.
        """

        point = self._checked_parameters(parameters)
        return self._default_simplex(point)

    def _default_simplex(self, point: torch.Tensor) -> torch.Tensor:
        """Return the coordinate-wise starting simplex for a checked point."""

        count = point.numel()
        offsets = torch.zeros(
            (count + 1, count), dtype=point.dtype, device=point.device
        )
        diagonal = torch.arange(count, device=point.device)
        scale = torch.clamp(point.abs(), min=1.0)
        offsets[1 + diagonal, diagonal] = self.initial_step * scale
        return point.reshape(1, count) + offsets

    def minimize(
        self,
        objective: Objective,
        parameters: torch.Tensor,
        *,
        simplex: torch.Tensor | None = None,
    ) -> NelderMeadResult:
        """Run the simplex down to a vertex or to the iteration budget.

        Args:
            objective: A callable taking one vertex tensor of shape ``(n,)`` and
                returning one finite scalar. It must not modify the tensor it is
                given.
            parameters: The starting point, a finite floating-point tensor of any
                shape with at least one value.
            simplex: An optional ``(n + 1, n)`` starting simplex, or a sequence of
                ``n + 1`` vertices. Pass it to place the first simplex yourself,
                which is the only control this unit offers over which basin the
                run can reach. A tensor has to already carry the parameters'
                dtype and start at the parameters; a sequence is built in the
                parameters' dtype.

        Returns:
            The finished run's record, whose ``converged`` says whether the two
            spreads met their thresholds and whose ``iterations`` says what was
            spent when they did not.

        Raises:
            ValidationError: If the objective is not callable, if the starting
                point or the simplex is not a finite floating-point tensor of the
                right shape, if the objective returns something other than one
                finite scalar, or if the objective modified the tensor it was
                given.
        """

        if not callable(objective):
            raise ValidationError("objective must be callable")
        point = self._checked_parameters(parameters)
        vertices = self._checked_simplex(point, simplex)
        self._evaluations = 0
        values = torch.stack([self._evaluate(objective, vertex) for vertex in vertices])

        iterations = 0
        for _ in range(self.maxiter):
            order = torch.argsort(values, stable=True)
            vertices = vertices[order]
            values = values[order]
            if self._spreads_settled(vertices, values):
                break
            vertices, values = self._replace_worst(objective, vertices, values)
            iterations += 1

        order = torch.argsort(values, stable=True)
        vertices = vertices[order]
        values = values[order]
        value_spread, simplex_spread = self._spreads(vertices, values)
        return NelderMeadResult(
            parameters=vertices[0].clone(),
            value=float(values[0]),
            simplex=vertices,
            value_spread=value_spread,
            simplex_spread=simplex_spread,
            tolerance=self.tolerance,
            atol=self.atol,
            iterations=iterations,
            evaluations=self._evaluations,
            converged=(value_spread <= self.atol and simplex_spread <= self.tolerance),
        )

    def _replace_worst(
        self,
        objective: Objective,
        vertices: torch.Tensor,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the simplex with the worst vertex replaced, or shrunk.

        The five branches are the method: reflect, expand a good reflection,
        contract outside a reflection that is still better than the worst,
        contract inside one that is not, and shrink when neither contraction
        improves on the worst vertex.
        """

        best = values[0]
        second_worst = values[-2]
        worst = values[-1]
        centroid = vertices[:-1].mean(dim=0)
        reflected = centroid + self.reflection * (centroid - vertices[-1])
        reflected_value = self._evaluate(objective, reflected)

        if reflected_value < best:
            expanded = centroid + self.expansion * (reflected - centroid)
            expanded_value = self._evaluate(objective, expanded)
            if expanded_value < reflected_value:
                return self._substitute(vertices, values, -1, expanded, expanded_value)
            return self._substitute(vertices, values, -1, reflected, reflected_value)
        if reflected_value < second_worst:
            return self._substitute(vertices, values, -1, reflected, reflected_value)

        if reflected_value < worst:
            contracted = centroid + self.contraction * (reflected - centroid)
            contracted_value = self._evaluate(objective, contracted)
            if contracted_value <= reflected_value:
                return self._substitute(
                    vertices, values, -1, contracted, contracted_value
                )
            return self._shrink(objective, vertices, values)

        contracted = centroid - self.contraction * (centroid - vertices[-1])
        contracted_value = self._evaluate(objective, contracted)
        if contracted_value < worst:
            return self._substitute(vertices, values, -1, contracted, contracted_value)
        return self._shrink(objective, vertices, values)

    def _shrink(
        self,
        objective: Objective,
        vertices: torch.Tensor,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Move every vertex but the best one toward the best one."""

        moved = vertices[0] + self.shrink * (vertices[1:] - vertices[0])
        moved_values = torch.stack(
            [self._evaluate(objective, vertex) for vertex in moved]
        )
        return torch.cat([vertices[:1], moved], dim=0), torch.cat(
            [values[:1], moved_values]
        )

    @staticmethod
    def _substitute(
        vertices: torch.Tensor,
        values: torch.Tensor,
        index: int,
        vertex: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the simplex with one vertex and its value replaced."""

        replaced_vertices = vertices.clone()
        replaced_vertices[index] = vertex
        replaced_values = values.clone()
        replaced_values[index] = value
        return replaced_vertices, replaced_values

    def _spreads(
        self, vertices: torch.Tensor, values: torch.Tensor
    ) -> tuple[float, float]:
        """Return the two measured spreads the termination rule reads."""

        value_spread = float((values - values[0]).abs().max())
        simplex_spread = float((vertices - vertices[0]).abs().max())
        return value_spread, simplex_spread

    def _spreads_settled(self, vertices: torch.Tensor, values: torch.Tensor) -> bool:
        """Whether both spreads have fallen under their thresholds."""

        value_spread, simplex_spread = self._spreads(vertices, values)
        return value_spread <= self.atol and simplex_spread <= self.tolerance

    def _checked_parameters(self, parameters: torch.Tensor) -> torch.Tensor:
        """Return the starting point as a finite float tensor, or refuse."""

        if not isinstance(parameters, torch.Tensor):
            raise ValidationError(
                f"parameters must be a torch.Tensor; got {type(parameters).__name__}"
            )
        if not parameters.is_floating_point():
            raise ValidationError(
                "parameters must have a floating dtype, because the simplex is "
                f"built by adding offsets to them; got {parameters.dtype}"
            )
        if parameters.numel() == 0:
            raise ValidationError("parameters must contain at least one value")
        if not bool(torch.isfinite(parameters).all()):
            raise ValidationError("parameters must all be finite")
        return parameters.detach().reshape(-1).clone()

    def _checked_simplex(
        self, point: torch.Tensor, simplex: torch.Tensor | Sequence[object] | None
    ) -> torch.Tensor:
        """Return the starting simplex as a ``(n + 1, n)`` tensor, or refuse."""

        count = point.numel()
        if simplex is None:
            candidate = self._default_simplex(point)
        elif isinstance(simplex, torch.Tensor):
            candidate = simplex
        else:
            rows = [torch.as_tensor(row, dtype=point.dtype) for row in simplex]
            if len(rows) != count + 1:
                raise ValidationError(
                    f"a simplex over {count} parameters needs {count + 1} "
                    f"vertices, and {len(rows)} were given"
                )
            candidate = torch.stack(rows)
        if not candidate.is_floating_point():
            raise ValidationError(
                "a simplex must have a floating dtype, because vertices are "
                f"moved by scaled differences; got {candidate.dtype}"
            )
        if tuple(candidate.shape) != (count + 1, count):
            raise ValidationError(
                f"a simplex over {count} parameters must have shape "
                f"({count + 1}, {count}), and it has {tuple(candidate.shape)}"
            )
        if candidate.dtype != point.dtype:
            raise ValidationError(
                "a simplex must carry the parameters' own dtype, because its "
                "vertices are moved in it and the run reports them back; the "
                f"parameters are {point.dtype} and the simplex is {candidate.dtype}"
            )
        if not bool(torch.isfinite(candidate).all()):
            raise ValidationError("every vertex of a simplex must be finite")
        if not torch.equal(candidate[0], point):
            raise ValidationError(
                "the simplex's first vertex must be the starting point, because "
                "a result reports one point and one simplex and they have to be "
                "the same run's"
            )
        degenerate = torch.linalg.matrix_rank(
            (candidate[1:] - candidate[0]).to(torch.float64)
        )
        if int(degenerate) < count:
            raise ValidationError(
                "the vertices must span all "
                f"{count} parameter directions, and they span only "
                f"{int(degenerate)}; a simplex flat in one direction cannot "
                "move in it"
            )
        return candidate.detach().clone()

    def _evaluate(self, objective: Objective, vertex: torch.Tensor) -> torch.Tensor:
        """Return one finite scalar objective value, or refuse.

        ``vertex`` is a tensor this optimizer created, never the caller's own
        parameters, so an objective that writes into what it is handed cannot
        corrupt the caller's values.  It is refused rather than tolerated: every
        decision in this method compares two stored objective values, and an
        in-place objective makes a stored vertex describe values that no longer
        belong to it.
        """

        before = vertex.detach().clone()
        value = objective(vertex)
        self._evaluations += 1
        if not torch.equal(vertex.detach(), before):
            raise ValidationError(
                "the objective modified the tensor it was given; Nelder-Mead "
                "stores a vertex and its objective value together, so an "
                "in-place objective makes the stored pair describe different "
                "parameters"
            )
        if not isinstance(value, torch.Tensor):
            raise ValidationError(
                "the objective must return a torch.Tensor, because Nelder-Mead "
                f"orders vertices by value; it returned {type(value).__name__}"
            )
        if value.numel() != 1:
            raise ValidationError(
                "the objective must return exactly one value, because a simplex "
                f"is ordered by one number per vertex; it returned "
                f"{value.numel()} values"
            )
        scalar = value.detach().reshape(()).to(torch.float64)
        if not bool(torch.isfinite(scalar)):
            raise ValidationError(
                "the objective returned a non-finite value, so the vertex it "
                "describes cannot be ordered against the others; fix the "
                "objective rather than letting a not-a-number decide which "
                "vertex is discarded"
            )
        return scalar

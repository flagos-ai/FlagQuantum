"""Constrained minimization by linear interpolation inside a shrinking trust region.

This unit models the objective and every constraint by affine interpolation over
a coordinate simplex, then takes the step that solves one linear program:
minimize the linearized objective plus a priced violation, inside a box of radius
``rho`` around the current point.  Everything it knows about a function comes
from those affine models, and the box is how far it is willing to act on them,
which is why ``rho`` is called the trust-region radius: it is the distance over
which a straight-line interpolation of a curved function is taken seriously.
When a step decreases the merit function, ``rho`` either stays or grows back
toward its starting value, and when a step does not decrease it, ``rho`` halves
and every model is rebuilt at the new scale.

Constraints are the reason the unit exists.  The subproblem carries one extra
variable ``t``, the largest violation the linearized constraints are allowed to
have at the answer, priced by the penalty; because a step of zero always
satisfies the box and ``t`` is bounded below by zero, the subproblem is always
feasible and always bounded, so an infeasible start needs neither a refusal nor a
separate repair phase.  What restores feasibility is the merit function
``objective + penalty * violation`` that decides acceptance, and the penalty
doubles whenever the trial point's true violation exceeds the one the linear
models predicted -- the case where the models were optimistic about feasibility
and the constraint must therefore be priced above the disagreement they declared.
Feasibility is reported as a measurement and not asserted: the result carries the
constraint values at the point it returns, the largest of them, and whether that
largest one fell under the tolerance the caller gave.

The neighbours are the reason to read this docstring before reaching for it.
:class:`~flagquantum.algorithms.nelder_mead.NelderMeadOptimizer` decides by
comparing two objective values, so it has no direction in which to restore
feasibility and no notion of admissibility at all.
:class:`~flagquantum.algorithms.spsa.SPSAOptimizer` does build a descent
direction, from a random finite difference at two objective evaluations per step,
but it prices no violation either, so a constrained problem posed to it is only
as well posed as the caller's own transformation of it.  Both are cheaper per
iteration than this unit: the affine models here are rebuilt from a fresh
coordinate simplex whenever the base point or the radius moves, so an iteration
that moves costs up to ``(m + 1) * (n + 1)`` calls for ``n`` parameters and ``m``
constraints, where an evolving-simplex method reuses its stored points and spends
about one.  That cost is reported as a count of calls and is not presented here
as a latency, a device cost, or a comparison against any other implementation.

Two properties of the method are stated rather than implied, and both are carried
on the returned :class:`CobylaResult`.  The first is that the subproblem is convex
while the problem is not: each step is the exact minimizer of a linear program, so
a step is unique in cost even though the affine models it was built from are only
valid at the scale ``rho``, and the point a run returns is where the trust region
collapsed rather than a global minimum.  The second is that ``converged`` reports
that ``rho`` reached its floor, which is a statement about the trust region and not
about optimality: a run whose steps stop improving shrinks ``rho`` until either it
finds a scale at which they improve again or it runs out of radius.

The module is deliberately torch-native in the same way as the simplex unit
beside it: the parameter vector is one tensor, the points and steps are tensors,
and the linear program is solved in Python floats over data that was read out of
those tensors, so the answer does not depend on the dtype or the device the
parameters live on.  The objective and each constraint are plain callables
returning one finite scalar tensor and are never inspected, so a constraint may
be a circuit measurement or a submitted job exactly as an objective may.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from ..errors import ValidationError

__all__ = [
    "COBYLA_ASSUMPTIONS",
    "COBYLA_LIMITATIONS",
    "CobylaOptimizer",
    "CobylaResult",
]

Objective = Callable[[torch.Tensor], torch.Tensor]
Constraint = Callable[[torch.Tensor], torch.Tensor]

#: What has to hold for a run's reported point to mean anything.
COBYLA_ASSUMPTIONS: tuple[str, ...] = (
    "Each function is deterministic at the resolution the coordinate simplex "
    "resolves, and its value at a point is the same on every call. Every model "
    "here is a difference of two evaluations at two points within one radius, so "
    "a function whose value moves between two evaluations of the same point turns "
    "its own gradient estimate into noise while this unit reuses the first value "
    "it saw. A stochastic objective is the subject of the SPSA unit beside this "
    "one.",
    "The functions are continuous and the region the trust region explores "
    "contains the point it converges to. The method is local: it follows the basin "
    "its starting point sits in, and no restart, multi-start, or basin-hopping "
    "mechanism is applied here.",
    "A constraint is admissible exactly when its returned value is non-negative, "
    "and the caller means every one of them. A feasible point is one where all of "
    "them are, the merit function adds their violations without weights, and no "
    "equality constraint is expressible except as a pair of inequalities the "
    "caller writes.",
    "The caller's own parameter scale is the unit of the trust region: the "
    "coordinate offset, the box, and the radius floor are all in the parameters' "
    "own units, and no scaling, normalization, or per-coordinate radius is "
    "applied here.",
)

#: What a result does not carry, and what its ``converged`` flag does not say.
COBYLA_LIMITATIONS: tuple[str, ...] = (
    "`converged` says the trust region reached its floor and nothing more. Each "
    "step is the exact minimizer of a linear program, but that program is built "
    "from affine interpolations of functions that are not affine, so the point a "
    "run returns is where the radius ran out rather than a minimum of the "
    "objective over the feasible region. A curved constraint is the clearest "
    "case: the linearization admits points the constraint does not, the merit "
    "function rejects them, and the run then shrinks until the disagreement is "
    "below the scale it is willing to act on. Neither a local nor a global "
    "optimality certificate is computed, no Karush-Kuhn-Tucker condition is "
    "checked, and no convergence rate, iteration bound, or confidence interval is "
    "reported.",
    "Feasibility is measured and not achieved by construction. `residual` is the "
    "largest violation at the returned point, `feasible` is that number compared "
    "against the caller's own tolerance, and a run can return a point whose "
    "`feasible` is `False` -- a bounded run, and not a failure of the arithmetic. "
    "A point that is feasible is feasible only to the precision the functions "
    "were evaluated at and the tolerance the caller chose; nothing here is a "
    "proof that a constraint holds off the sampled points.",
    "The penalty is a schedule and not a calibrated weight. It starts at the "
    "value the caller gave, doubles when the trial violation exceeds the "
    "linearized prediction of it, is capped by the caller's ceiling, and never "
    "decreases within a run. A run that reaches the ceiling is reported through "
    "the final `penalty` rather than by a separate flag, and the exact penalty "
    "that would make the constrained problem equivalent to an unconstrained one "
    "is not computed: that equivalence holds only above a threshold the caller "
    "has not supplied and this unit does not estimate.",
    "The cost is a count of calls and not a latency, a device cost, or a parallel "
    "schedule. Every function is called one point at a time, one value per call, "
    "and no vectorization of the simplex, of the constraint set, or of the linear "
    "program is attempted. An iteration that moves rebuilds one affine model per "
    "function from a coordinate simplex, so it spends up to `(m + 1) * (n + 1)` "
    "calls for `n` parameters and `m` constraints, against about one for a method "
    "that carries its stored points forward.",
    "The starting radius, its floor, the iteration budget, the initial penalty, "
    "its ceiling, and the feasibility tolerance are the caller's, and the quality "
    "of the answer depends on them in a way this unit does not certify. A "
    "starting radius small against the curvature of a function makes every model "
    "a poor one and the run slow; a large one makes each model wrong over the "
    "region it is trusted in. A step is formed in the parameters' own dtype, so a "
    "low-precision parameter vector rounds every step to it. No bounds, no "
    "integer or discrete parameters, no equality constraint, no second-order "
    "model, no curvature estimate, and no checkpoint or resume protocol are "
    "provided.",
)


@dataclass(frozen=True, slots=True)
class CobylaResult:
    """The point a finished trust-region run returned, with its constraint values.

    Attributes:
        parameters: The returned point, shaped like the point the run started from.
        value: The objective value at that point, as a Python float.
        violations: The constraint values at that point, one per constraint and in
            the order the run was given them, so a value below zero is a violated
            constraint and its magnitude is the violation. It is the evidence a
            run leaves behind: ``residual`` and ``feasible`` below are read from
            it, so it is reported rather than discarded. Empty when the run was
            given no constraints.
        residual: The largest violation at that point, in constraint units, and
            ``0.0`` when no constraint is violated or none was given.
        feasible: Whether ``residual`` fell under ``constraint_tolerance``.
        rho: The trust-region radius the run stopped on, with ``converged`` saying
            whether it reached ``rhoend``.
        rhoend: The radius floor the run was given.
        penalty: The violation price the run ended on, after whatever doubling the
            schedule applied.
        constraint_tolerance: The feasibility tolerance the run was given.
        iterations: The number of trust-region iterations the run applied, which
            is the number of affine model sets it built and of steps it considered.
        evaluations: The number of calls the run spent on the objective and on
            every constraint together, counting a repeated point once.
        converged: Whether ``rho`` reached ``rhoend``. ``False`` means the
            iteration budget ran out first, which is a bounded run rather than a
            failed one.
        assumptions: What had to hold for the point to mean anything.
        limitations: What the point and the flag do not carry.
    """

    parameters: torch.Tensor
    value: float
    violations: torch.Tensor
    residual: float
    feasible: bool
    rho: float
    rhoend: float
    penalty: float
    constraint_tolerance: float
    iterations: int
    evaluations: int
    converged: bool
    assumptions: tuple[str, ...] = COBYLA_ASSUMPTIONS
    limitations: tuple[str, ...] = COBYLA_LIMITATIONS

    def __post_init__(self) -> None:
        if not isinstance(self.parameters, torch.Tensor):
            raise TypeError("parameters must be the tensor the returned point is")
        if self.parameters.numel() == 0:
            raise ValueError("parameters must contain at least one value")
        if not bool(torch.isfinite(self.parameters).all()):
            raise ValueError("parameters must all be finite")
        if not isinstance(self.value, float) or not math.isfinite(self.value):
            raise ValueError(f"the value must be finite, and {self.value!r} is not")
        if not isinstance(self.violations, torch.Tensor):
            raise TypeError("violations must be the tensor the constraints read")
        if self.violations.dim() != 1:
            raise ValueError(
                "violations must be one value per constraint, and "
                f"{tuple(self.violations.shape)} is not a flat list"
            )
        if not bool(torch.isfinite(self.violations).all()):
            raise ValueError("every constraint value must be finite")
        for name in ("rho", "rhoend", "penalty", "constraint_tolerance"):
            number = getattr(self, name)
            if not math.isfinite(number) or number <= 0.0:
                raise ValueError(f"{name} must be a finite positive number")
        for name in ("iterations", "evaluations"):
            number = getattr(self, name)
            if isinstance(number, bool) or not isinstance(number, int) or number < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.iterations < 1:
            raise ValueError(
                "a run builds a model and considers a step before it can stop, so "
                "a result cannot report that it applied no iteration"
            )
        if self.evaluations < self.parameters.numel() + 1:
            raise ValueError(
                "a result needs at least one evaluation per simplex point, "
                "because no affine model can have been built without them"
            )
        expected = _residual(self.violations)
        if self.residual != expected:
            raise ValueError(
                "residual must be the largest violation the constraint values "
                f"carry, and they carry {expected!r} against {self.residual!r}"
            )
        if self.feasible != (self.residual <= self.constraint_tolerance):
            raise ValueError(
                "feasible must report the residual against its own tolerance: the "
                f"residual is {self.residual!r}, the tolerance is "
                f"{self.constraint_tolerance!r}, and feasible is {self.feasible!r}"
            )
        if self.converged != (self.rho <= self.rhoend):
            raise ValueError(
                "converged must report the radius against its own floor: the "
                f"radius is {self.rho!r}, the floor is {self.rhoend!r}, and "
                f"converged is {self.converged!r}"
            )


def _largest_violation(values: Sequence[float]) -> float:
    """Return the largest violation a sequence of constraint values carries."""

    if not values:
        return 0.0
    return max(0.0, -min(values))


def _residual(violations: torch.Tensor) -> float:
    """Return the largest violation ``violations`` carries, or zero for none."""

    return _largest_violation(violations.tolist())


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


@dataclass(frozen=True, slots=True)
class _AffineModel:
    """One affine interpolation of one scalar function over one coordinate simplex.

    ``value`` is the function at the base point and ``gradient`` is the forward
    difference along each coordinate, divided by the offset that coordinate was
    moved by.  It is an interpolation and not a fit: the coordinate simplex has
    ``n + 1`` points in ``n`` dimensions, so this affine function reproduces all
    ``n + 1`` of them exactly and there is no residual left to report.
    """

    value: float
    gradient: tuple[float, ...]

    def at(self, step: Sequence[float]) -> float:
        """Return the model's value after ``step`` from the base point."""

        return self.value + math.fsum(
            slope * offset for slope, offset in zip(self.gradient, step, strict=True)
        )


#: A reduced cost is only treated as negative, and a tableau coefficient is only
#: treated as a pivot, when it is beyond this in the relevant direction. It sits
#: far below the scale of any difference quotient a caller's functions produce.
_PIVOT_TOLERANCE = 1e-12

#: A cap on pivots. Bland's rule makes the tableau method terminate, so a cap
#: being reached is a defect in this solver rather than a slow problem.
_MAX_PIVOTS = 100_000


def _initial_tableau(
    objective: _AffineModel,
    constraints: tuple[_AffineModel, ...],
    *,
    radius: float,
    penalty: float,
) -> tuple[list[list[float]], list[int], list[float], int]:
    """Return a feasible starting tableau for the trust-region subproblem.

    The subproblem is the linear program

    .. code-block:: text

        minimize    g . d + penalty * t
        subject to  a_i . d + b_i + t >= 0   for every constraint i
                    t >= 0
                    |d_j| <= radius          for every coordinate j

    where ``g`` is the objective's model gradient, ``a_i`` and ``b_i`` are
    constraint ``i``'s model, and ``t`` is the largest violation the linearized
    constraints are allowed to have at the answer.  The step is split into two
    non-negative halves, ``d = d+ - d-``, as a free variable has to be, and the
    box on each coordinate is one more row.  ``t`` is not split: the tableau
    method holds every variable non-negative, which is exactly the constraint
    ``t >= 0`` and needs no row of its own.  Dropping it is what makes the
    starting basis easy to find.  Let ``T`` be the largest violation at the base
    point.  If it is zero the base point is feasible and the constraint slacks
    ``s_i = b_i`` are a basis; if it is positive, the most violated constraint's
    row carries ``t`` with the value ``T`` and every other constraint's row
    carries its own slack.  Either basis is canonical and non-negative, so the
    program starts feasible and needs no artificial variable and no phase one.

    Args:
        objective: The objective's affine model.
        constraints: One affine model per constraint, in the caller's order.
        radius: The trust-region radius, the bound on every coordinate.
        penalty: The price of one unit of violation. It is positive, so the
            program is bounded and a minimizer exists.

    Returns:
        The tableau rows, the basic column of each row, the effective linear cost
        row, and the right-hand-side column index.
    """

    count = len(objective.gradient)
    width = 2 * count + 1 + len(constraints) + count
    rhs = width
    rows = len(constraints) + count
    slacks = 2 * count + 1
    boxes = slacks + len(constraints)

    objective_cost = [0.0] * (width + 1)
    for coordinate, slope in enumerate(objective.gradient):
        objective_cost[coordinate] = slope
        objective_cost[count + coordinate] = -slope
    objective_cost[2 * count] = penalty

    tableau: list[list[float]] = []
    basis: list[int] = []
    violation = _largest_violation([model.value for model in constraints])
    worst = -1
    if violation > 0.0:
        worst = min(
            range(len(constraints)),
            key=lambda index: constraints[index].value,
        )
    for index, model in enumerate(constraints):
        row = [0.0] * (width + 1)
        if index == worst:
            # This row carries `t` itself, at the value of the largest violation,
            # so `t` is its basic variable and the right-hand side is that
            # violation rather than the constraint value.
            for coordinate, slope in enumerate(model.gradient):
                row[coordinate] = slope
                row[count + coordinate] = -slope
            row[2 * count] = 1.0
            row[slacks + index] = -1.0
            row[rhs] = violation
            basis.append(2 * count)
            tableau.append(row)
            continue
        # Every other constraint row is this row minus the `worst` row, which
        # cancels `t` out of it and leaves its own slack as the basic variable.
        if worst < 0:
            for coordinate, slope in enumerate(model.gradient):
                row[coordinate] = -slope
                row[count + coordinate] = slope
            row[2 * count] = -1.0
            row[slacks + index] = 1.0
            row[rhs] = model.value
        else:
            anchor = constraints[worst]
            for coordinate, slope in enumerate(model.gradient):
                difference = anchor.gradient[coordinate] - slope
                row[coordinate] = difference
                row[count + coordinate] = -difference
            row[slacks + index] = 1.0
            row[slacks + worst] = -1.0
            row[rhs] = model.value - anchor.value
        basis.append(slacks + index)
        tableau.append(row)
    for coordinate in range(count):
        row = [0.0] * (width + 1)
        row[coordinate] = 1.0
        row[count + coordinate] = 1.0
        row[boxes + coordinate] = 1.0
        row[rhs] = radius
        basis.append(boxes + coordinate)
        tableau.append(row)

    # A basic variable with a non-zero cost has to be taken out of the cost row,
    # or the tableau's reduced costs would not be the ones this basis implies.
    cost = list(objective_cost)
    for index in range(rows):
        weight = objective_cost[basis[index]]
        if weight == 0.0:
            continue
        row = tableau[index]
        cost = [value - weight * entry for value, entry in zip(cost, row, strict=True)]
    return tableau, basis, cost, rhs


def _trust_region_step(
    objective: _AffineModel,
    constraints: tuple[_AffineModel, ...],
    *,
    radius: float,
    penalty: float,
) -> list[float]:
    """Return the step that minimizes the linearized merit inside a box.

    The program is solved by the tableau method over ``2 * n + 1`` structural
    columns, ``m`` slack columns and ``n`` box columns, in Python floats read out
    of the models.  Bland's rule picks the entering column and breaks the ratio
    ties, which is what keeps the iteration from cycling.

    Args:
        objective: The objective's affine model.
        constraints: One affine model per constraint, in the caller's order.
        radius: The trust-region radius, the bound on every coordinate of the
            returned step.
        penalty: The price of one unit of violation, positive.

    Returns:
        The step as a list of ``n`` Python floats, each inside the box.

    Raises:
        RuntimeError: If the tableau method runs past its pivot cap, which Bland's
            rule makes a defect in this solver rather than a property of the
            problem it was given.
    """

    tableau, basis, cost, rhs = _initial_tableau(
        objective,
        constraints,
        radius=radius,
        penalty=penalty,
    )
    count = len(objective.gradient)
    width = rhs
    rows = len(tableau)
    for _ in range(_MAX_PIVOTS):
        entering = -1
        for column in range(width):
            if cost[column] < -_PIVOT_TOLERANCE:
                entering = column
                break
        if entering < 0:
            break
        leaving = -1
        best = 0.0
        for index in range(rows):
            coefficient = tableau[index][entering]
            if coefficient <= _PIVOT_TOLERANCE:
                continue
            ratio = tableau[index][rhs] / coefficient
            if leaving < 0 or ratio < best - _PIVOT_TOLERANCE:
                leaving, best = index, ratio
            elif ratio <= best + _PIVOT_TOLERANCE and basis[index] < basis[leaving]:
                leaving = index
        if leaving < 0:
            raise RuntimeError(
                "the trust-region subproblem was reported unbounded, which the "
                "box on every coordinate makes impossible"
            )
        pivot = tableau[leaving][entering]
        tableau[leaving] = [value / pivot for value in tableau[leaving]]
        for index in range(rows):
            if index == leaving:
                continue
            factor = tableau[index][entering]
            if factor == 0.0:
                continue
            source = tableau[leaving]
            tableau[index] = [
                value - factor * entry
                for value, entry in zip(tableau[index], source, strict=True)
            ]
        factor = cost[entering]
        if factor != 0.0:
            source = tableau[leaving]
            cost = [
                value - factor * entry
                for value, entry in zip(cost, source, strict=True)
            ]
        basis[leaving] = entering
    else:
        raise RuntimeError(
            f"the trust-region subproblem did not terminate within {_MAX_PIVOTS} "
            "pivots, which Bland's rule makes a defect in this solver"
        )

    solved = [0.0] * width
    for index in range(rows):
        solved[basis[index]] = tableau[index][rhs]
    step = [
        solved[coordinate] - solved[count + coordinate] for coordinate in range(count)
    ]
    # The box is a row of the program, so this clamp is exact for any answer the
    # tableau returns and is a guard against a rounded one leaving it.
    return [min(radius, max(-radius, value)) for value in step]


def _linearized_residual(
    models: tuple[_AffineModel, ...], step: Sequence[float]
) -> float:
    """Return the largest violation ``models`` predict after ``step``."""

    return _largest_violation([model.at(step) for model in models])


class CobylaOptimizer:
    """Minimize a scalar objective under non-negative constraints, without slopes.

    The optimizer owns nothing between calls: one :meth:`minimize` call builds a
    coordinate simplex at the starting point, runs the trust region, and returns
    the record.  It does not use autograd -- every function value is read with
    ``float()`` and any graph attached to it is ignored.

    Args:
        maxiter: The number of trust-region iterations to attempt before
            returning the point reached. It is a budget, not a convergence
            condition.
        rhobeg: The radius the run starts on. It is also the offset each
            coordinate of the model simplex is moved by and the ceiling the
            radius grows back to after a step that decreased the merit, all in the
            parameters' own units, so the caller's parameter scale is the scale of
            the model.
        rhoend: The radius floor. The run stops once the radius reaches it, which
            is reported as ``converged``. It must be strictly below ``rhobeg``,
            because a run whose floor is where it started has no scale left to
            try.
        penalty: The price of one unit of constraint violation in the merit
            function, and the coefficient of the violation variable in the
            subproblem.
        penalty_ceiling: The value above which the penalty is no longer doubled.
            Passing ``penalty`` itself freezes the penalty for the whole run.
        constraint_tolerance: The residual at or below which the returned point is
            reported as ``feasible``.

    Raises:
        ValidationError: If an option is not what its name promises, or if the
            floor is not strictly below the starting radius. Every refusal about
            the objective or a constraint happens before a function value is
            stored beside a point, so a refused run leaves nothing half-described
            behind.

    Examples:
        Minimize a two-qubit Pauli energy whose minimum this run cannot reach,
        because two constraints forbid angles past 0.5 radians:

        >>> import torch
        >>> import flagquantum as fq
        >>> from flagquantum import algorithms as fqa
        >>> def energy(values):
        ...     circuit = fq.Circuit(2, dtype=torch.complex128)
        ...     circuit = circuit.ry(0, values[0]).ry(1, values[1]).cx(0, 1)
        ...     outputs = fq.expectation(fq.Z(0) + fq.Z(1))
        ...     return -fq.run(circuit, outputs=outputs).expectation().sum()
        >>> result = fqa.CobylaOptimizer(rhobeg=0.5, rhoend=1e-8).minimize(
        ...     energy,
        ...     torch.tensor([0.4, 0.4], dtype=torch.float64),
        ...     constraints=[lambda v: v[0] - 0.5, lambda v: v[1] - 0.5],
        ... )
        >>> round(result.value, 6)
        -1.647734
        >>> [round(float(value), 6) for value in result.parameters]
        [0.5, 0.5]
        >>> result.feasible
        True
    """

    def __init__(
        self,
        *,
        maxiter: int = 200,
        rhobeg: float = 0.5,
        rhoend: float = 1e-8,
        penalty: float = 1.0,
        penalty_ceiling: float = 1e9,
        constraint_tolerance: float = 1e-8,
    ) -> None:
        if isinstance(maxiter, bool) or not isinstance(maxiter, int) or maxiter < 1:
            raise ValidationError("maxiter must be a positive integer")
        self.rhobeg = _finite_positive(rhobeg, name="rhobeg")
        self.rhoend = _finite_positive(rhoend, name="rhoend")
        if self.rhoend >= self.rhobeg:
            raise ValidationError(
                "rhoend must be strictly below rhobeg, because a trust region "
                "whose floor is where it started has no scale left to try"
            )
        self.penalty = _finite_positive(penalty, name="penalty")
        self.penalty_ceiling = _finite_positive(penalty_ceiling, name="penalty_ceiling")
        if self.penalty_ceiling < self.penalty:
            raise ValidationError(
                "penalty_ceiling must be at least penalty, because the schedule "
                "doubles the penalty from where it started and never lowers it"
            )
        self.constraint_tolerance = _finite_positive(
            constraint_tolerance, name="constraint_tolerance"
        )
        self.maxiter = maxiter
        self._evaluations = 0
        self._values: dict[str, dict[tuple[float, ...], float]] = {}

    @property
    def evaluations(self) -> int:
        """The number of function calls the last run spent.

        Read after :meth:`minimize`; it is reset at the start of every run, so it
        describes one run rather than a lifetime, and it counts the objective and
        every constraint together.
        """

        return self._evaluations

    def minimize(
        self,
        objective: Objective,
        parameters: torch.Tensor,
        *,
        constraints: Sequence[Constraint] = (),
    ) -> CobylaResult:
        """Run the trust region down to its floor or to the iteration budget.

        Args:
            objective: A callable taking one parameter tensor of shape ``(n,)``
                and returning one finite scalar. It must not modify the tensor it
                is given.
            parameters: The starting point, a finite floating-point tensor of any
                shape with at least one value. It does not have to be feasible:
                the merit function and the priced violation are what move an
                infeasible start, and the run reports the residual it ended on.
                Every function is called with the flattened view of it, and the
                result's ``parameters`` is reshaped back to the shape given here.
            constraints: The constraints, each a callable taking the same
                parameter tensor and returning one finite scalar that is
                non-negative exactly when the point is admissible. Order is
                preserved in the result's ``violations``.

        Returns:
            The finished run's record, whose ``rho`` and ``converged`` say how far
            the trust region shrank and whose ``residual`` says how much
            constraint it left violated.

        Raises:
            ValidationError: If the objective or a constraint is not callable, if
                the starting point is not a finite floating-point tensor, or if a
                function returns something other than one finite scalar or
                modifies the tensor it was given.
        """

        if not callable(objective):
            raise ValidationError("objective must be callable")
        if isinstance(constraints, (str, bytes)) or not isinstance(
            constraints, Sequence
        ):
            raise ValidationError(
                "constraints must be a sequence of callables, one per constraint"
            )
        for index, constraint in enumerate(constraints):
            if not callable(constraint):
                raise ValidationError(f"constraint {index} must be callable")
        point = self._checked_parameters(parameters)
        functions: list[tuple[str, Objective]] = [("the objective", objective)]
        functions.extend(
            (f"constraint {index}", constraint)
            for index, constraint in enumerate(constraints)
        )
        self._evaluations = 0
        self._values = {label: {} for label, _ in functions}

        radius = self.rhobeg
        penalty = self.penalty
        iterations = 0
        # The models' own base values are the functions' values at their base
        # point, so whichever models were last built at the point the run keeps
        # already carry the objective and every constraint there.
        reported = self._models(functions, point, radius)
        for _ in range(self.maxiter):
            if radius <= self.rhoend:
                break
            models = self._models(functions, point, radius)
            step = _trust_region_step(
                models[0],
                tuple(models[1:]),
                radius=radius,
                penalty=penalty,
            )
            trial = point + torch.tensor(step, dtype=point.dtype, device=point.device)
            trial_models = self._models(functions, trial, radius)
            accepted = self._merit(penalty, trial_models) < self._merit(penalty, models)
            if accepted:
                point = trial
                reported = trial_models
                if max(abs(value) for value in step) >= 0.9 * radius:
                    radius = min(2.0 * radius, self.rhobeg)
            else:
                radius *= 0.5
            # The models promised one violation; the trial point's own values are
            # the truth. Two disagreements between them say the price of a
            # violation is too low to buy feasibility, and each doubles it: the
            # truth exceeding the promise, which is where the models were
            # optimistic, and an accepted step that left the point infeasible
            # without reducing the violation at all, which is where the objective
            # was simply worth more than the constraint cost.
            reached = _largest_violation([model.value for model in trial_models[1:]])
            before = _largest_violation([model.value for model in models[1:]])
            underpriced = reached > _linearized_residual(tuple(models[1:]), step)
            stalled = accepted and before > 0.0 and reached >= before
            if underpriced or stalled:
                penalty = min(2.0 * penalty, self.penalty_ceiling)
            iterations += 1

        violations = torch.tensor(
            [model.value for model in reported[1:]], dtype=torch.float64
        )
        residual = _residual(violations)
        return CobylaResult(
            parameters=point.reshape(parameters.shape).clone(),
            value=reported[0].value,
            violations=violations,
            residual=residual,
            feasible=residual <= self.constraint_tolerance,
            rho=radius,
            rhoend=self.rhoend,
            penalty=penalty,
            constraint_tolerance=self.constraint_tolerance,
            iterations=iterations,
            evaluations=self._evaluations,
            converged=radius <= self.rhoend,
        )

    def _models(
        self,
        functions: Sequence[tuple[str, Objective]],
        point: torch.Tensor,
        radius: float,
    ) -> list[_AffineModel]:
        """Return one affine model per function over the coordinate simplex.

        The simplex is the base point and one offset per coordinate, so the model
        reproduces ``n + 1`` values exactly and its gradient is a forward
        difference. Every value is stored, so a base point or a shifted point a
        later iteration asks for again is not evaluated twice.
        """

        return [
            self._model(label, function, point, radius) for label, function in functions
        ]

    def _model(
        self,
        label: str,
        function: Objective,
        point: torch.Tensor,
        radius: float,
    ) -> _AffineModel:
        """Return the affine model of ``function`` at ``point``."""

        base = self._evaluate(label, function, point)
        gradient: list[float] = []
        for coordinate in range(point.numel()):
            shifted = point.clone()
            shifted[coordinate] += radius
            gradient.append((self._evaluate(label, function, shifted) - base) / radius)
        return _AffineModel(value=base, gradient=tuple(gradient))

    def _merit(self, penalty: float, models: Sequence[_AffineModel]) -> float:
        """Return the merit of a set of models: value plus priced violation."""

        return models[0].value + penalty * _largest_violation(
            [model.value for model in models[1:]]
        )

    def _checked_parameters(self, parameters: torch.Tensor) -> torch.Tensor:
        """Return the starting point as a finite float tensor, or refuse."""

        if not isinstance(parameters, torch.Tensor):
            raise ValidationError(
                f"parameters must be a torch.Tensor; got {type(parameters).__name__}"
            )
        if not parameters.is_floating_point():
            raise ValidationError(
                "parameters must have a floating dtype, because the model simplex "
                f"is built by adding offsets to them; got {parameters.dtype}"
            )
        if parameters.numel() == 0:
            raise ValidationError("parameters must contain at least one value")
        if not bool(torch.isfinite(parameters).all()):
            raise ValidationError("parameters must all be finite")
        return parameters.detach().reshape(-1).clone()

    def _evaluate(
        self,
        label: str,
        function: Objective,
        point: torch.Tensor,
    ) -> float:
        """Return one finite scalar function value, or refuse.

        ``point`` is a tensor this optimizer created, and the callable is handed a
        clone of it rather than the stored point, so a function that writes into
        what it is handed cannot corrupt the point it was called at. Writing into
        it is refused rather than tolerated: every model here is a difference of
        two stored values, and an in-place function makes a stored value describe
        parameters it no longer belongs to.
        """

        key = tuple(float(value) for value in point)
        stored = self._values[label]
        if key in stored:
            return stored[key]
        probe = point.clone()
        value = function(probe)
        self._evaluations += 1
        if not torch.equal(probe, point):
            raise ValidationError(
                f"{label} modified the tensor it was given; this unit stores a "
                "point and its function value together, so an in-place function "
                "makes the stored pair describe different parameters"
            )
        if not isinstance(value, torch.Tensor):
            raise ValidationError(
                f"{label} must return a torch.Tensor, because this unit builds a "
                "linear model from function values; it returned "
                f"{type(value).__name__}"
            )
        if value.numel() != 1:
            raise ValidationError(
                f"{label} must return exactly one value, because a model "
                f"interpolates one scalar per function; it returned "
                f"{value.numel()} values"
            )
        scalar = value.detach().reshape(()).to(torch.float64)
        if not bool(torch.isfinite(scalar)):
            raise ValidationError(
                f"{label} returned a non-finite value, so the model it describes "
                "cannot be built; fix the function rather than letting a "
                "not-a-number decide a step"
            )
        number = float(scalar)
        stored[key] = number
        return number

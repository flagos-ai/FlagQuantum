"""Gradient utilities for trainable FlagQuantum circuits."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from .core.ir import ensure_circuit_ir
from .core.operator_schema import OPERATOR_SCHEMAS
from .errors import CapabilityError, ValidationError

CircuitBuilder = Callable[[torch.Tensor], Any]
LossFunction = Callable[[Any], torch.Tensor]
BatchLossFunction = Callable[[tuple[Any, ...]], Sequence[torch.Tensor] | torch.Tensor]

_BATCH_PARAMETER_SHIFT_GATES = frozenset({"h", "x", "rx", "ry", "rz", "cx"})
_PARAMETER_SHIFT_GATES = frozenset({"rx", "ry", "rz"})
_PROBE_DISPLACEMENT = 1.0
_GRADIENT_METHODS = (
    "auto",
    "autograd",
    "parameter_shift",
    "finite_difference",
    "spsa",
)


@dataclass(frozen=True)
class GradientResult:
    """A detached gradient together with the choices that produced it.

    Attributes:
        gradient: Detached gradient with the shape, dtype, and device of the
            parameters it was computed for.
        method: The method that ran, which is the resolved choice when the
            request was ``method="auto"``.
        exact: Whether ``method`` is analytically exact. ``"spsa"`` and
            ``"finite_difference"`` are approximations whose error depends on
            ``step``.
        step: The displacement an approximate method used, and ``None`` for the
            exact methods.
    """

    gradient: torch.Tensor
    method: str
    exact: bool
    step: float | None


def gradient(
    program: CircuitBuilder,
    parameters: torch.Tensor,
    loss: LossFunction | None = None,
    *,
    method: str = "auto",
    step: float | None = None,
    directions: int = 1,
    generator: torch.Generator | None = None,
) -> GradientResult:
    """Differentiate one scalar loss with respect to circuit parameters.

    ``method="auto"`` measures the program instead of trusting a declaration: it
    uses reverse-mode PyTorch autograd when the loss at ``parameters`` carries a
    graph, the exact per-opcode parameter-shift rule when a circuit is available,
    and central finite differences otherwise. The method that ran, and the
    displacement an approximate method used, are reported in the result, so a
    fallback is never silent.

    Args:
        program: Maps a parameter tensor to a scalar loss, or -- when ``loss`` is
            given -- to the circuit that ``loss`` scores.
        parameters: Real, finite, non-empty parameter tensor.
        loss: Scores the object ``program`` returns. Supplying it is what makes
            ``"parameter_shift"`` available, because the exact shift rule is
            declared per opcode on the circuit.
        method: One of ``"auto"``, ``"autograd"``, ``"parameter_shift"``,
            ``"finite_difference"``, or ``"spsa"``.
        step: Displacement for ``"finite_difference"`` and ``"spsa"``, in the
            units of the parameter; ``None`` derives it from the precision the
            loss is computed in. It has no effect on the exact methods.
        directions: Number of simultaneous perturbations ``"spsa"`` averages.
        generator: Random source for the ``"spsa"`` directions.

    Returns:
        The detached gradient and the method that produced it.

    Raises:
        CapabilityError: If ``method="adjoint"`` is requested, or if the chosen
            method cannot be applied to this program.
        ValidationError: If an argument is outside its accepted range.

    The stable user-facing entry is ``fq.gradient``, which carries the example.
    """

    _validate_method(method, directions, generator)
    base = _validated_parameter_tensor(parameters)
    if step is not None and (not math.isfinite(step) or step <= 0.0):
        raise ValidationError("step must be a positive finite displacement or None")
    evaluate = _loss_evaluator(program, loss)

    probe_value: torch.Tensor | None = None
    if method == "auto":
        tracked = base.detach().clone().requires_grad_(True)
        probe_value = evaluate(tracked)
        if probe_value.requires_grad and probe_value.grad_fn is not None:
            return GradientResult(
                gradient=_backward(probe_value, tracked, base),
                method="autograd",
                exact=True,
                step=None,
            )
        method = "parameter_shift" if loss is not None else "finite_difference"

    if method == "autograd":
        return GradientResult(
            gradient=_autograd_gradient(evaluate, base),
            method="autograd",
            exact=True,
            step=None,
        )
    if method == "parameter_shift":
        if loss is None:
            raise CapabilityError(
                "method='parameter_shift' needs the circuit behind the loss: the "
                "exact shift rule is declared per opcode, so pass loss= and have "
                "program return the circuit. Without it use "
                "method='finite_difference'."
            )
        return GradientResult(
            gradient=_shifted_gradient(program, base, loss),
            method="parameter_shift",
            exact=True,
            step=None,
        )

    if step is None:
        step = _default_step(probe_value if probe_value is not None else evaluate(base))
    if method == "finite_difference":
        return GradientResult(
            gradient=_central_difference_gradient(evaluate, base, step=step),
            method="finite_difference",
            exact=False,
            step=step,
        )
    return GradientResult(
        gradient=_spsa_gradient(
            evaluate,
            base,
            step=step,
            directions=directions,
            generator=generator,
        ),
        method="spsa",
        exact=False,
        step=step,
    )


def _validate_method(
    method: str,
    directions: int,
    generator: torch.Generator | None,
) -> None:
    if method == "adjoint":
        raise CapabilityError(
            "FlagQuantum has no standalone adjoint gradient: the reversible "
            "adjoint sweep is reachable only as the backward pass behind PyTorch "
            "autograd, and no result reports whether backward used adjoint "
            "replay. Use method='autograd' and read the execution mode, or "
            "method='parameter_shift'."
        )
    if method not in _GRADIENT_METHODS:
        raise ValidationError(
            f"unknown gradient method {method!r}; expected one of "
            + ", ".join(repr(item) for item in _GRADIENT_METHODS)
        )
    if method != "spsa" and (directions != 1 or generator is not None):
        raise ValidationError(
            "directions and generator apply only to method='spsa', which samples "
            f"perturbation directions; method={method!r} does not"
        )
    if not isinstance(directions, int) or isinstance(directions, bool):
        raise TypeError("directions must be an integer")
    if directions < 1:
        raise ValidationError("directions must be at least 1")


def _validated_parameter_tensor(parameters: torch.Tensor) -> torch.Tensor:
    if not isinstance(parameters, torch.Tensor):
        raise TypeError("parameters must be a torch.Tensor")
    if parameters.numel() == 0:
        raise ValidationError("parameters must not be empty")
    if parameters.is_complex() or not parameters.is_floating_point():
        raise ValidationError("parameters must be real floating-point values")
    if not bool(torch.isfinite(parameters).all()):
        raise ValidationError("parameters must be finite")
    return parameters.detach()


def _loss_evaluator(
    program: CircuitBuilder,
    loss: LossFunction | None,
) -> LossFunction:
    """Bind one callable that turns parameters into the scalar loss."""

    if loss is None:

        def evaluate_program(values: torch.Tensor) -> torch.Tensor:
            return _scalar_loss(program(values), source="program")

        return evaluate_program

    def evaluate_scored(values: torch.Tensor) -> torch.Tensor:
        return _scalar_loss(loss(program(values)), source="loss")

    return evaluate_scored


def _scalar_loss(value: Any, *, source: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.numel() != 1:
        raise ValidationError(f"{source} must return one scalar tensor")
    if value.is_complex():
        raise ValidationError(f"{source} must return a real scalar tensor")
    return value.reshape(())


def _autograd_gradient(
    evaluate: LossFunction,
    parameters: torch.Tensor,
) -> torch.Tensor:
    tracked = parameters.detach().clone().requires_grad_(True)
    value = evaluate(tracked)
    if not value.requires_grad or value.grad_fn is None:
        raise CapabilityError(
            "method='autograd' requires the loss to be built from the parameters "
            "through PyTorch autograd; this loss carries no graph. Use "
            "method='finite_difference', or pass loss= and use "
            "method='parameter_shift'."
        )
    return _backward(value, tracked, parameters)


def _backward(
    value: torch.Tensor,
    tracked: torch.Tensor,
    parameters: torch.Tensor,
) -> torch.Tensor:
    computed = torch.autograd.grad(value, tracked, allow_unused=True)[0]
    if computed is None:
        raise CapabilityError(
            "the loss does not depend on the supplied parameters, so it has no "
            "gradient with respect to them"
        )
    return computed.detach().reshape_as(parameters)


def _shifted_gradient(
    program: CircuitBuilder,
    parameters: torch.Tensor,
    loss: LossFunction,
) -> torch.Tensor:
    """Run the exact shift rule, reporting an inapplicable program as a boundary."""

    try:
        return parameter_shift_gradient(program, parameters, loss).detach()
    except ValueError as error:
        raise CapabilityError(
            "the parameter-shift rule does not apply to this program: "
            f"{error}. Every differentiated parameter must control exactly one "
            "gate parameter whose opcode declares a derivative rule, so use "
            "method='finite_difference' or method='spsa' for a circuit that does "
            "not meet that condition."
        ) from error


def _default_step(sample: torch.Tensor) -> float:
    """Derive a difference displacement from the precision the loss carries.

    A difference quotient loses significant digits to roundoff as the step
    shrinks, and the roundoff floor is set by the dtype the program computes in,
    not by the dtype the caller's parameters happen to have. The loss's dtype is
    the measured one, and the cube root of its machine epsilon is the step that
    balances truncation against roundoff for a central difference.
    """

    return math.pow(float(torch.finfo(sample.dtype).eps), 1.0 / 3.0)


def _central_difference_gradient(
    evaluate: LossFunction,
    parameters: torch.Tensor,
    *,
    step: float,
) -> torch.Tensor:
    base = parameters.detach()
    flat = base.reshape(-1)
    terms = []
    for index in range(flat.numel()):
        plus = flat.clone()
        minus = flat.clone()
        plus[index] += step
        minus[index] -= step
        terms.append(
            (evaluate(plus.reshape_as(base)) - evaluate(minus.reshape_as(base)))
            / (2.0 * step)
        )
    return _as_parameters(torch.stack(terms), base)


def _spsa_gradient(
    evaluate: LossFunction,
    parameters: torch.Tensor,
    *,
    step: float,
    directions: int,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Average simultaneous-perturbation estimates over Rademacher directions.

    Every direction perturbs all parameters at once, so one direction costs two
    loss evaluations regardless of the parameter count. The estimate is
    statistical: it is not exact even for a noiseless program, and its variance
    falls only as the averaged direction count grows.
    """

    base = parameters.detach()
    flat = base.reshape(-1)
    size = flat.numel()
    total = torch.zeros(size, dtype=torch.float64)
    for _ in range(directions):
        drawn = torch.randint(0, 2, (size,), generator=generator, dtype=torch.int64)
        # A Rademacher draw is +-1, so its reciprocal is itself and the
        # finite-difference coefficient needs no extra division.
        direction = (2.0 * drawn.to(torch.float64) - 1.0).to(base.dtype)
        plus = flat + step * direction
        minus = flat - step * direction
        difference = evaluate(plus.reshape_as(base)) - evaluate(minus.reshape_as(base))
        total += difference.to(torch.float64) * direction.to(torch.float64)
    return _as_parameters((total / (2.0 * step * directions)).reshape(-1), base)


def _as_parameters(flat: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    return flat.reshape_as(base).to(device=base.device, dtype=base.dtype)


def parameter_shift_gradient(
    circuit_builder: CircuitBuilder,
    parameters: torch.Tensor,
    loss_fn: LossFunction,
) -> torch.Tensor:
    """Estimate circuit-parameter gradients with the parameter-shift rule.

    ``circuit_builder`` receives a tensor of parameter values and must return a
    circuit or executable object accepted by ``loss_fn``. ``loss_fn`` must return
    a scalar tensor. The method is backend-agnostic and works for native
    simulators, distributed modes, shot-based cloud execution wrappers, and
    hardware-style inference paths where autograd is not available.

    Each parameter is differentiated with the rule its gate declares in
    ``OPERATOR_SCHEMAS``. ``RX``, ``RY``, and ``RZ`` need one shifted pair;
    ``CRX``, ``CRY``, and ``CRZ`` need two, so a fixed coefficient of one half
    would overstate their derivative by a factor of ``sqrt(2)``.

    Raises:
        TypeError: If ``circuit_builder`` does not return a circuit or IR.
        ValueError: If a parameter does not control exactly one declared gate
            parameter, or its opcode declares no derivative rule.
    """

    base = parameters.detach()
    flat = base.reshape(-1)
    profile = _circuit_profile_of(circuit_builder(base))
    grads = []
    for index in range(flat.numel()):
        rule = _occurrence_shift_rule(
            profile,
            circuit_builder,
            base,
            flat,
            index,
        )
        terms = []
        for coefficient, shift in rule:
            shifted = flat.clone()
            shifted[index] += shift
            terms.append(
                coefficient * loss_fn(circuit_builder(shifted.reshape_as(base)))
            )
        # A one-pair rule is summed with a single tensor so no extra zero enters
        # the graph; every evaluated term carries the caller's loss gradient.
        grads.append(terms[0] if len(terms) == 1 else torch.stack(terms).sum())
    return (
        torch.stack(grads)
        .reshape_as(base)
        .to(device=parameters.device, dtype=parameters.dtype)
    )


def _occurrence_shift_rule(
    profile: tuple[tuple[Any, ...], tuple[Any, ...]],
    circuit_builder: CircuitBuilder,
    base: torch.Tensor,
    flat: torch.Tensor,
    index: int,
) -> tuple[tuple[float, float], ...]:
    """Find the gate parameter one flat parameter index controls, then its rule."""

    probes = []
    for displacement in (_PROBE_DISPLACEMENT, -_PROBE_DISPLACEMENT):
        probed = flat.clone()
        probed[index] += displacement
        probes.append(_circuit_profile_of(circuit_builder(probed.reshape_as(base))))
    occurrence = _changed_occurrence(profile, probes, parameter_index=index)
    opcode, name = occurrence
    schema = OPERATOR_SCHEMAS.get(opcode)
    if schema is None:
        raise ValueError(
            f"parameter {index} controls unknown opcode {opcode!r}, which "
            "declares no derivative rule"
        )
    try:
        return schema.shift_rule(name)
    except ValueError as error:
        raise ValueError(
            f"parameter {index} controls {opcode}.{name}, which cannot be "
            f"differentiated: {error}"
        ) from error


def _changed_occurrence(
    base: tuple[tuple[Any, ...], tuple[Any, ...]],
    probes: Sequence[tuple[tuple[Any, ...], tuple[Any, ...]]],
    *,
    parameter_index: int,
) -> tuple[str, str]:
    """Return the gate angle a probe pair moved by the probe displacement.

    The shift rule is applied to the gate's angle, so a user parameter has to be
    that angle. A parameter that scales its gate -- ``rx(0, theta=2 * scale)`` --
    moves the angle by twice the probe displacement, and shifting the *user*
    parameter by the rule's shift would then differentiate at the wrong point.
    That is refused instead of being answered with the angle's derivative.
    """

    base_structure, base_values = base
    if any(structure != base_structure for structure, _ in probes):
        raise ValueError(
            f"parameter {parameter_index} changes circuit structure; parameter "
            "shift requires a static circuit"
        )
    changed = [
        (base_item, probed_items)
        for base_item, *probed_items in zip(
            base_values, *(values for _, values in probes), strict=True
        )
        if any(probed != base_item for probed in probed_items)
    ]
    if len(changed) != 1:
        raise ValueError(
            f"parameter {parameter_index} must control exactly one gate "
            f"parameter; found {len(changed)}"
        )
    (opcode, name, value), probed_values = changed[0]
    displacements = [float(probed[2]) - float(value) for probed in probed_values]
    if not all(
        math.isclose(abs(displacement), _PROBE_DISPLACEMENT, rel_tol=1e-4)
        for displacement in displacements
    ):
        raise ValueError(
            f"parameter {parameter_index} does not enter {opcode}.{name} as its "
            f"angle: a probe of {_PROBE_DISPLACEMENT} moved the angle by "
            f"{displacements}; enter its gate angle directly"
        )
    return opcode, name


def batched_parameter_shift_gradient(
    circuit_builder: CircuitBuilder,
    parameters: torch.Tensor,
    batch_loss_fn: BatchLossFunction,
    *,
    shift: float = torch.pi / 2,
) -> torch.Tensor:
    """Evaluate a parameter-shift gradient through one batch request.

    This first remote-safe profile accepts H, X, RX, RY, RZ, and CX circuits.
    Each input parameter must control exactly one RX, RY, or RZ occurrence.
    ``batch_loss_fn`` receives all positive and negative shifts in parameter
    order and must return one scalar tensor per circuit.

    Args:
        circuit_builder: Builds a circuit from a parameter tensor.
        parameters: Real, non-empty parameter tensor.
        batch_loss_fn: Evaluates every shifted circuit in one batch operation.
        shift: Parameter displacement. The current profile requires pi/2.

    Returns:
        A detached gradient with the same shape, dtype, and device as
        ``parameters``.

    Raises:
        TypeError: If inputs or returned losses have unsupported types.
        ValueError: If the circuit is outside the supported shift profile.
    """

    base = _validated_parameters(parameters, shift=shift)
    flat = base.reshape(-1)
    base_structure, base_values = _circuit_profile(circuit_builder(base))

    programs: list[Any] = []
    for index in range(flat.numel()):
        plus = flat.clone()
        minus = flat.clone()
        plus[index] += shift
        minus[index] -= shift
        plus_program = circuit_builder(plus.reshape_as(base))
        minus_program = circuit_builder(minus.reshape_as(base))
        _validate_shift_pair(
            (base_structure, base_values),
            _circuit_profile(plus_program),
            _circuit_profile(minus_program),
            shift=shift,
            parameter_index=index,
        )
        programs.extend((plus_program, minus_program))

    losses = _batch_losses(batch_loss_fn(tuple(programs)), expected=len(programs))
    gradient = 0.5 * (losses[0::2] - losses[1::2])
    return gradient.reshape_as(base).to(
        device=parameters.device,
        dtype=parameters.dtype,
    )


def _validated_parameters(parameters: torch.Tensor, *, shift: float) -> torch.Tensor:
    if not isinstance(parameters, torch.Tensor):
        raise TypeError("parameters must be a torch.Tensor")
    if parameters.numel() == 0:
        raise ValueError("parameters must not be empty")
    if not (parameters.is_floating_point() and torch.isfinite(parameters).all()):
        raise ValueError("parameters must contain finite real floating-point values")
    if not math.isclose(float(shift), math.pi / 2, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("batched parameter shift currently requires shift=pi/2")
    return parameters.detach()


def _circuit_profile_of(program: Any) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Describe a program as its static structure plus its scalar parameters."""

    ir = ensure_circuit_ir(program)
    if any(item.matrix is not None for item in ir.instructions):
        raise ValueError("parameter shift does not support custom matrices")
    structure = (
        ir.n_wires,
        tuple((item.name, item.wires, tuple(item.params)) for item in ir.instructions),
    )
    values = tuple(
        (item.name, name, _scalar_parameter(value))
        for item in ir.instructions
        for name, value in item.params.items()
    )
    return structure, values


def _circuit_profile(program: Any) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    ir = ensure_circuit_ir(program)
    unsupported = sorted(
        {item.name for item in ir.instructions} - _BATCH_PARAMETER_SHIFT_GATES
    )
    if unsupported:
        raise ValueError(
            "batched parameter shift supports only H, X, RX, RY, RZ, and CX; "
            f"unsupported gates: {', '.join(unsupported)}"
        )
    if any(item.matrix is not None for item in ir.instructions):
        raise ValueError("batched parameter shift does not support custom matrices")
    return _circuit_profile_of(ir)


def _validate_shift_pair(
    base: tuple[tuple[Any, ...], tuple[Any, ...]],
    plus: tuple[tuple[Any, ...], tuple[Any, ...]],
    minus: tuple[tuple[Any, ...], tuple[Any, ...]],
    *,
    shift: float,
    parameter_index: int,
) -> None:
    base_structure, base_values = base
    plus_structure, plus_values = plus
    minus_structure, minus_values = minus
    if not base_structure == plus_structure == minus_structure:
        raise ValueError(
            f"parameter {parameter_index} changes circuit structure; "
            "batched parameter shift requires a static circuit"
        )
    changed = []
    for base_item, plus_item, minus_item in zip(
        base_values, plus_values, minus_values, strict=True
    ):
        opcode, name, base_value = base_item
        _, _, plus_value = plus_item
        _, _, minus_value = minus_item
        if not (
            math.isclose(plus_value, base_value, rel_tol=0.0, abs_tol=1e-7)
            and math.isclose(minus_value, base_value, rel_tol=0.0, abs_tol=1e-7)
        ):
            changed.append((opcode, name, base_value, plus_value, minus_value))

    if len(changed) != 1:
        raise ValueError(
            f"parameter {parameter_index} must control exactly one gate occurrence; "
            f"found {len(changed)}"
        )
    opcode, name, base_value, plus_value, minus_value = changed[0]
    if opcode not in _PARAMETER_SHIFT_GATES or name != "theta":
        raise ValueError(
            f"parameter {parameter_index} must control one RX, RY, or RZ angle"
        )
    if not (
        math.isclose(plus_value - base_value, shift, rel_tol=0.0, abs_tol=1e-6)
        and math.isclose(minus_value - base_value, -shift, rel_tol=0.0, abs_tol=1e-6)
    ):
        raise ValueError(
            f"parameter {parameter_index} must enter its gate angle directly"
        )


def _scalar_parameter(value: Any) -> float:
    tensor = (
        value.detach() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    )
    if tensor.numel() != 1 or tensor.is_complex():
        raise ValueError("batched parameter shift requires scalar real gate parameters")
    scalar = float(tensor.cpu())
    if not math.isfinite(scalar):
        raise ValueError("batched parameter shift requires finite gate parameters")
    return scalar


def _batch_losses(
    values: Sequence[torch.Tensor] | torch.Tensor,
    *,
    expected: int,
) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        if values.numel() != expected:
            raise ValueError(
                f"batch_loss_fn returned {values.numel()} losses; expected {expected}"
            )
        return values.reshape(expected).detach()
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError("batch_loss_fn must return a tensor or sequence of tensors")
    if len(values) != expected:
        raise ValueError(
            f"batch_loss_fn returned {len(values)} losses; expected {expected}"
        )
    losses = []
    for value in values:
        if not isinstance(value, torch.Tensor) or value.numel() != 1:
            raise TypeError("batch_loss_fn must return one scalar tensor per circuit")
        losses.append(value.detach().reshape(()))
    return torch.stack(losses)


__all__ = [
    "batched_parameter_shift_gradient",
    "gradient",
    "parameter_shift_gradient",
]

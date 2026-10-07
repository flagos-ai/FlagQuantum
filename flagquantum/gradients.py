"""Gradient utilities for trainable FlagQuantum circuits."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from .core._parameter_occurrence import (
    _batch_occurrence_coefficient,
    _circuit_profile_of,
    _occurrence_shift_rule,
)
from .core.finite_differences import (
    central_difference_gradient,
    default_difference_step,
)
from .core.ir import ensure_circuit_ir
from .core.operator_schema import OPERATOR_SCHEMAS, OperatorSchema
from .errors import CapabilityError, ValidationError

CircuitBuilder = Callable[[torch.Tensor], Any]
LossFunction = Callable[[Any], torch.Tensor]
BatchLossFunction = Callable[[tuple[Any, ...]], Sequence[torch.Tensor] | torch.Tensor]

# The batch profile admits these constant operations plus every gate whose
# declared parameters differentiate with one evaluation pair. The constant half
# carries no parameter, so it states a protocol scope rather than a derivative
# rule; the parameterized half is derived below.
_BATCH_PROFILE_CONSTANT_GATES = frozenset({"h", "x", "cx"})


def _single_pair_shift_opcodes(
    schemas: Mapping[str, OperatorSchema],
) -> frozenset[str]:
    """Opcodes that differentiate with one positive and one negative evaluation.

    A gate whose generator has one frequency has a two-term shift rule, which is
    what the batched protocol sends for each input parameter. This answers that
    question from the declaration rather than from a gate list, so registering a
    frequency set with more terms removes an opcode from the profile without
    editing this module.

    Args:
        schemas: Opcode-to-schema mapping, normally
            ``flagquantum.core.OPERATOR_SCHEMAS``.

    Returns:
        Every opcode that declares a frequency set and whose every declared
        parameter has a two-term shift rule.
    """

    admitted = set()
    for opcode, schema in schemas.items():
        if not schema.differentiable:
            continue
        if all(len(schema.shift_rule(name)) == 2 for name in schema.parameters):
            admitted.add(opcode)
    return frozenset(admitted)


_BATCH_PROFILE_GATES = _BATCH_PROFILE_CONSTANT_GATES | _single_pair_shift_opcodes(
    OPERATOR_SCHEMAS
)
_GRADIENT_METHODS = (
    "auto",
    "autograd",
    "parameter_shift",
    "finite_difference",
    "spsa",
)
# One refusal serves every vector-derivative entry point, because they share one
# reason: a program whose output ignores the parameters has no derivative, and
# answering with a zero Jacobian would hide a program that never read them.
_NO_DEPENDENCE = (
    "the program's output does not depend on the supplied parameters, so it has "
    "no derivative with respect to them"
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
        step = default_difference_step(
            probe_value if probe_value is not None else evaluate(base)
        )
    if method == "finite_difference":
        return GradientResult(
            gradient=_as_parameters(
                central_difference_gradient(evaluate, base, step=step), base
            ),
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


def jacobian(
    program: CircuitBuilder,
    parameters: torch.Tensor,
) -> torch.Tensor:
    """Differentiate every element of a vector-valued program.

    ``fq.gradient`` answers "how does one scalar move". A program that returns
    several values -- the probabilities of a register, one expectation per
    qubit, a batch of losses -- has no single gradient, so this returns the
    derivative of every output element with respect to every parameter element,
    shaped ``(*program_output.shape, *parameters.shape)``.

    The derivative is exact. It is taken by reverse-mode PyTorch autograd, which
    is the one route available here: the per-opcode parameter-shift rule is
    declared for a scalar loss and has no vector form, and a difference quotient
    would return an approximation whose displacement the caller could not read
    off the result. ``jacobian`` therefore takes no ``method`` argument and
    refuses a program that carries no graph rather than approximating one.

    Args:
        program: Maps a parameter tensor to a real tensor.
        parameters: Real, finite, non-empty parameter tensor.

    Returns:
        A detached tensor of shape ``(*program_output.shape, *parameters.shape)``
        in the dtype and on the device of ``parameters``.

    Raises:
        CapabilityError: If the program's output carries no autograd graph, which
            means it does not depend on ``parameters``. Use ``fq.gradient`` for a
            scalar loss, including one that needs an approximate method.
        ValidationError: If ``parameters`` is empty, complex, or non-finite, or
            if the program returns a non-tensor, an empty tensor, or a complex
            tensor. A complex output would need a Wirtinger convention that no
            FlagQuantum entry point defines; take a real observable value, such
            as an expectation, instead.
        TypeError: If ``parameters`` is not a tensor.
    """

    base = _validated_parameter_tensor(parameters)
    value, tracked = _differentiable_value(program, base)
    rows = [
        _pulled_back(
            value, tracked, _output_seed(value, index), base, retain_graph=True
        )
        for index in range(value.numel())
    ]
    return torch.stack(rows).reshape((*value.shape, *base.shape))


def jvp(
    program: CircuitBuilder,
    parameters: torch.Tensor,
    tangents: torch.Tensor,
) -> torch.Tensor:
    """Apply the derivative of a vector-valued program to a parameter direction.

    This is the forward-mode action ``J v``: how the whole output moves when the
    parameters move along ``tangents``. It costs two reverse sweeps regardless of
    how many values the program returns, which is what makes it usable where the
    full Jacobian is not -- a probabilities vector over many qubits has one row
    per basis state, and :func:`jacobian` needs one sweep per row.

    The two sweeps differentiate the program twice, so the program and the
    executor behind it must support a graph of a graph. FlagQuantum's statevector,
    MPS, and tensor-network paths do; an executor that does not fails closed with
    a ``CapabilityError`` rather than being answered by a difference quotient.

    Args:
        program: Maps a parameter tensor to a real tensor.
        parameters: Real, finite, non-empty parameter tensor.
        tangents: Real floating-point tensor shaped exactly like ``parameters``.

    Returns:
        A detached tensor shaped like the program's output, in the dtype and on
        the device of ``parameters``. Unlike ``torch.autograd.functional.jvp``
        this returns the product alone: the program's own output is
        ``program(parameters)``, so returning it here would ask every caller to
        unpack a value half of them discard.

    Raises:
        CapabilityError: If the program's output carries no graph, or if the
            program cannot be differentiated twice.
        ValidationError: If ``parameters`` or ``tangents`` is empty, complex, or
            non-finite, if ``tangents`` is not shaped like ``parameters``, or if
            the program returns a non-tensor, an empty tensor, or a complex
            tensor.
        TypeError: If ``parameters`` or ``tangents`` is not a tensor.
    """

    base = _validated_parameter_tensor(parameters)
    tangent = _validated_direction(tangents, name="tangents")
    _require_shape(tangent, base.shape, name="tangents", reference="the parameters")
    value, tracked = _differentiable_value(program, base)
    # The seed lives in the parameter dtype, not the output dtype. The second
    # sweep differentiates with respect to the cotangent, and the executor
    # carries the chain rule in the parameter dtype: a seed of the forward
    # dtype raises a dtype mismatch against wider parameters. That choice is
    # also what makes the returned product's dtype match the parameters.
    seed = torch.zeros(
        value.shape, dtype=base.dtype, device=base.device, requires_grad=True
    )
    pulled = _pulled_back(value, tracked, seed, base, create_graph=True)
    try:
        product = torch.autograd.grad(
            (pulled * tangent).sum(), seed, allow_unused=True
        )[0]
    except RuntimeError as error:
        raise _second_order_refusal(error) from error
    if product is None:
        raise CapabilityError(_NO_DEPENDENCE)
    return product.detach()


def vjp(
    program: CircuitBuilder,
    parameters: torch.Tensor,
    cotangents: torch.Tensor,
) -> torch.Tensor:
    """Apply the derivative of a vector-valued program to an output direction.

    This is the reverse-mode action ``c^T J``: how one scalar built from the
    program's output, ``<cotangents, program(parameters)>``, moves with the
    parameters. It costs one reverse sweep, independent of both the parameter
    count and the output count, so it is the shape a training step wants: a
    scalar objective reached by weighting several measured values.

    Args:
        program: Maps a parameter tensor to a real tensor.
        parameters: Real, finite, non-empty parameter tensor.
        cotangents: Real floating-point tensor shaped exactly like the program's
            output. It has to match that shape rather than merely broadcast to
            it, which is the same rule ``torch.autograd.grad`` applies; use
            ``torch.ones_like(output)`` to weight every value equally.

    Returns:
        A detached tensor shaped like ``parameters``, in the dtype and on the
        device of ``parameters``.

    Raises:
        CapabilityError: If the program's output carries no graph.
        ValidationError: If ``parameters`` or ``cotangents`` is empty, complex,
            or non-finite, if ``cotangents`` is not shaped like the program's
            output, or if the program returns a non-tensor, an empty tensor, or
            a complex tensor.
        TypeError: If ``parameters`` or ``cotangents`` is not a tensor.
    """

    base = _validated_parameter_tensor(parameters)
    cotangent = _validated_direction(cotangents, name="cotangents")
    value, tracked = _differentiable_value(program, base)
    _require_shape(
        cotangent, value.shape, name="cotangents", reference="the program output"
    )
    return _pulled_back(value, tracked, cotangent, base)


def _differentiable_value(
    program: CircuitBuilder,
    base: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the program at a tracked copy of the parameters.

    Returning the tracked tensor alongside the value is what lets every caller
    differentiate against the same forward pass.
    """

    tracked = base.detach().clone().requires_grad_(True)
    value = _vector_value(program(tracked), source="program")
    if not value.requires_grad or value.grad_fn is None:
        raise CapabilityError(
            "the program's output does not depend on the supplied parameters "
            "through PyTorch autograd, so it has no derivative with respect to "
            "them. Build the output from the parameters, or use fq.gradient for "
            "a scalar loss that needs an approximate method."
        )
    return value, tracked


def _vector_value(value: Any, *, source: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValidationError(f"{source} must return a torch.Tensor")
    if value.numel() == 0:
        raise ValidationError(f"{source} must return a non-empty tensor")
    if value.is_complex():
        raise ValidationError(
            f"{source} must return a real tensor. A complex output would need a "
            "Wirtinger convention that no FlagQuantum entry point defines; take "
            "a real observable value, such as an expectation."
        )
    return value


def _validated_direction(direction: Any, *, name: str) -> torch.Tensor:
    if not isinstance(direction, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if direction.is_complex() or not direction.is_floating_point():
        raise ValidationError(f"{name} must be real floating-point values")
    if not bool(torch.isfinite(direction).all()):
        raise ValidationError(f"{name} must be finite")
    return direction.detach()


def _require_shape(
    direction: torch.Tensor,
    expected: torch.Size,
    *,
    name: str,
    reference: str,
) -> None:
    if direction.shape != expected:
        raise ValidationError(
            f"{name} must be shaped like {reference}: expected {tuple(expected)}, "
            f"got {tuple(direction.shape)}"
        )


def _output_seed(value: torch.Tensor, index: int) -> torch.Tensor:
    """Return the one-hot cotangent that selects one output element."""

    seed = torch.zeros_like(value)
    seed.reshape(-1)[index] = 1.0
    return seed


def _second_order_refusal(error: RuntimeError) -> CapabilityError:
    """Report a program that cannot be differentiated twice as a boundary."""

    return CapabilityError(
        "the vector derivative needs the program differentiated twice, and this "
        f"program or its executor does not support a graph of a graph: {error}. "
        "Use fq.gradient for a scalar loss, or differentiate once with "
        "fq.jacobian over a program the executor can replay."
    )


def _pulled_back(
    value: torch.Tensor,
    tracked: torch.Tensor,
    seed: torch.Tensor,
    base: torch.Tensor,
    *,
    retain_graph: bool = False,
    create_graph: bool = False,
) -> torch.Tensor:
    """Differentiate ``value`` against ``tracked``, refusing a program that ignores it."""

    try:
        pulled = torch.autograd.grad(
            value,
            tracked,
            grad_outputs=seed,
            retain_graph=retain_graph,
            create_graph=create_graph,
            allow_unused=True,
        )[0]
    except RuntimeError as error:
        if create_graph:
            raise _second_order_refusal(error) from error
        raise
    if pulled is None:
        raise CapabilityError(_NO_DEPENDENCE)
    return pulled.reshape_as(base).to(device=base.device, dtype=base.dtype)


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
        ValidationError: If ``parameters`` is empty, complex, or non-finite, or
            if ``loss_fn`` does not return one real scalar tensor.
    """

    base = _validated_parameter_tensor(parameters)
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
                coefficient
                * _scalar_loss(
                    loss_fn(circuit_builder(shifted.reshape_as(base))),
                    source="loss_fn",
                )
            )
        # A one-pair rule is summed with a single tensor so no extra zero enters
        # the graph; every evaluated term carries the caller's loss gradient.
        grads.append(terms[0] if len(terms) == 1 else torch.stack(terms).sum())
    return (
        torch.stack(grads)
        .reshape_as(base)
        .to(device=parameters.device, dtype=parameters.dtype)
    )


def parameter_shift_hessian(
    circuit_builder: CircuitBuilder,
    parameters: torch.Tensor,
    loss_fn: LossFunction,
) -> torch.Tensor:
    """Estimate the second derivative of a scalar loss with the parameter-shift rule.

    The rule is the one each gate already declares in ``OPERATOR_SCHEMAS``, read
    twice. A parameterized gate whose generator has frequencies
    ``{f_1, ..., f_n}`` has an expectation value that is a trigonometric
    polynomial in those frequencies, so its first derivative is a fixed linear
    combination of the evaluations at ``base + shift_i`` and its mixed second
    derivative is a linear combination of the evaluations at
    ``base + shift_i + shift_j``. This function therefore evaluates the loss at
    those sums and reads no coefficient from anywhere but the declaration, so
    there is no second table of second-order coefficients that could drift away
    from the first.

    The matrix is symmetric because a twice-differentiable loss has equal mixed
    partials, and every cell is evaluated independently, so the two halves agree
    to floating-point summation order rather than bit for bit. Symmetry is a
    property of the construction and not evidence that the values are right; the
    values are checked elsewhere against a finite-difference route that shares no
    code with this one.

    Cost grows as the square of the parameter count, which is why this is a
    separate entry point rather than a mode of :func:`parameter_shift_gradient`:
    every cell of the matrix is its own set of displaced circuits. For ``n``
    parameters whose rules all declare one frequency, each diagonal cell needs
    three evaluations and each off-diagonal cell four, so ``4 * n**2 - n`` loss
    evaluations in total -- ``33`` for the three-parameter reference program,
    against the ``6`` a first-order gradient of the same program costs.

    The result is only as precise as the loss it is assembled from, and the loss
    precision belongs to the execution rather than to ``parameters``. An
    unqualified :func:`flagquantum.run` resolves to ``complex64``, so a Hessian
    assembled from it plateaus near ``1e-07``; the same call on the same program
    with ``precision="complex128"`` reaches ``1e-16``. Ask for the wider
    precision on the execution options -- or build the circuit with
    ``dtype=torch.complex128`` -- when the second derivatives are read past about
    seven digits.

    Args:
        circuit_builder: Builds a circuit from a parameter tensor, exactly as
            :func:`parameter_shift_gradient` receives it.
        parameters: Real, non-empty parameter tensor.
        loss_fn: Turns a circuit into one scalar tensor.

    Returns:
        A detached Hessian with shape ``(*parameters.shape, *parameters.shape)``
        and the dtype and device of ``parameters``. Cell ``[i, j]`` holds the
        second derivative with respect to the flattened parameters ``i`` and
        ``j``, so a one-dimensional ``parameters`` of length ``n`` gives the
        ``(n, n)`` matrix.

    Raises:
        TypeError: If ``circuit_builder`` does not return a circuit or IR.
        ValueError: If a parameter does not control exactly one declared gate
            parameter, or its opcode declares no derivative rule.
        ValidationError: If ``parameters`` is empty, complex, or non-finite, or
            if ``loss_fn`` does not return one real scalar tensor.
    """

    base = _validated_parameter_tensor(parameters)
    flat = base.reshape(-1)
    width = flat.numel()
    profile = _circuit_profile_of(circuit_builder(base))
    rules = [
        _occurrence_shift_rule(profile, circuit_builder, base, flat, index)
        for index in range(width)
    ]
    rows = []
    for left in range(width):
        cells = []
        for right in range(width):
            terms = [
                coefficient
                * _scalar_loss(
                    loss_fn(circuit_builder((flat + shift).reshape_as(base))),
                    source="loss_fn",
                )
                for coefficient, shift in _composed_shift_terms(
                    rules[left],
                    rules[right],
                    parameters=(left, right),
                    width=width,
                    dtype=base.dtype,
                )
            ]
            cells.append(torch.stack(terms).sum())
        rows.append(torch.stack(cells))
    return (
        torch.stack(rows)
        .reshape(*base.shape, *base.shape)
        .to(device=parameters.device, dtype=parameters.dtype)
    )


def metric_tensor(
    circuit_builder: CircuitBuilder,
    parameters: torch.Tensor,
    *,
    options: Any = None,
    step: float | None = None,
) -> torch.Tensor:
    """The Fubini-Study metric tensor of a program's state.

    Cell ``[i, j]`` is ``Re[<d_i|d_j> - <d_i|psi><psi|d_j>]`` for the state
    derivatives with respect to the flattened parameters, which is the
    Fubini-Study metric. It is not the quantum Fisher information: the two agree
    in shape and in units but differ by a factor of four, and
    ``QFIM = 4 * metric_tensor(...)``. The factor is named here rather than left
    to a reader because a silent factor of four is exactly the kind of
    discrepancy that survives every shape assertion. The convention matches
    PennyLane's ``qml.metric_tensor``, which returns the same quarter, and the
    gate checks the factor against a hand-rolled quantum Fisher information
    rather than against this docstring.

    The derivative is a central difference of the state that the execution
    delivers, and deliberately not the shift rule each gate declares.
    ``OperatorSchema.parameter_frequencies`` is a statement about a gate's
    *expectation value*: an observable's shift rule and the derivative of a
    state are different objects, and reading the declared rule as a state
    derivative is wrong by a factor of ``sqrt(2)`` on a single ``ry`` -- measured
    in ``contracts/metric-tensor-contract.toml``. There is no rescaling of the
    declared rule that repairs this in general either, because the rule's shape
    depends on the generator's spectrum: a rotation and a phase gate with the
    same declared frequency differentiate differently. Reading the declaration
    and then correcting it would be a second rule table maintained by hand, so
    this route reads no rule at all and differentiates the state it is handed.

    That choice buys a property ``parameter_shift_gradient`` cannot offer: a
    program carrying an explicit matrix still has a state, so this route serves
    it, while the declared rule has nothing to say about a matrix no opcode
    declares.

    Cost is ``1 + 2 * parameters.numel()`` state evaluations -- one base state
    plus a central pair per parameter -- and does not grow with the square of the
    parameter count the way the Hessian's does, because the state derivative is
    computed once per parameter and every matrix cell reuses that pair.

    The result carries the precision the *execution* ran in, not the precision
    of ``parameters``: a ``complex64`` state yields a ``float32`` matrix, whose
    central-difference error is bounded by the step rather than by the
    differencing, so at the ``complex64`` default this matrix is good to about
    six digits and no further. Build the circuit with
    ``dtype=torch.complex128`` when the metric is read past that.

    Args:
        circuit_builder: Builds a circuit from a parameter tensor, exactly as
            :func:`parameter_shift_gradient` receives it.
        parameters: Real, non-empty parameter tensor.
        options: Execution options, forwarded to :func:`flagquantum.run`. The
            default resolves the way an unqualified run does.
        step: Central-difference displacement. The default is derived from the
            precision the state carries.

    Returns:
        A detached Fubini-Study metric tensor with shape
        ``(*parameters.shape, *parameters.shape)``. The dtype follows the state
        the execution produced and the device follows ``parameters``.

    Raises:
        TypeError: If ``circuit_builder`` is not callable.
        ValueError: If ``step`` is not a positive finite displacement.
        ValidationError: If ``parameters`` is empty, complex, or non-finite, if
            the execution returns no state, or if the state changes shape as the
            parameters move.
        CapabilityError: If the execution returns the state as a density matrix
            rather than a statevector, or as more than one state.
    """

    if not callable(circuit_builder):
        raise TypeError("circuit_builder must be callable")
    base = _validated_parameter_tensor(parameters)
    from ._api import run as run_program

    def deliver(values: torch.Tensor) -> torch.Tensor:
        return _state_of(run_program(circuit_builder(values), options=options), options)

    reference = deliver(base)
    width = base.numel()
    if step is None:
        step = default_difference_step(reference)

    def displaced(values: torch.Tensor) -> torch.Tensor:
        state = deliver(values)
        if state.shape != reference.shape:
            raise ValidationError(
                "the program is not static: its state has shape "
                f"{tuple(state.shape)} at these parameters and "
                f"{tuple(reference.shape)} at the base point, so a derivative "
                "with respect to the parameters is not defined"
            )
        return state

    deviations = central_difference_gradient(displaced, base, step=step)
    derivatives = deviations.reshape(width, -1).to(
        device=parameters.device, dtype=deviations.dtype
    )
    reference = reference.to(device=derivatives.device, dtype=deviations.dtype)
    overlaps = derivatives @ derivatives.conj().T
    projections = derivatives @ reference.conj()
    return (overlaps - torch.outer(projections, projections.conj())).real.reshape(
        *parameters.shape, *parameters.shape
    )


def _state_of(result: Any, options: Any) -> torch.Tensor:
    """The one state a metric-tensor route differentiates, or a refusal.

    ``ExecutionResult.to_statevector`` is the representation-agnostic accessor:
    it hands back whatever the backend produced, which is a ket for a
    state-producing mode and a density matrix for a density-matrix one. The
    rank is therefore the only place the representation is visible, and a
    matrix is refused rather than reinterpreted. The pure-state formula this
    route evaluates and the mixed-state quantum Fisher information are different
    quantities -- measured at ``2.30e-01`` apart on a program with a single
    ``depolarizing(1, 0.1)`` -- so reinterpreting a matrix would not be a
    precision problem, it would be a different number.
    """

    state: torch.Tensor = result.to_statevector()
    if state.ndim == 3:
        raise CapabilityError(
            "the execution returned the state as a density matrix, so its purity "
            "is not one; the metric tensor of a mixed state is the quantum "
            "Fisher information built from the symmetric logarithmic derivative, "
            "which this route does not compute. Ask for a statevector-producing "
            "execution mode on a program with no channel"
        )
    if state.ndim == 2 and state.shape[0] != 1:
        raise CapabilityError(
            f"the execution returned {state.shape[0]} states; the metric tensor "
            "is a matrix per parameter set, so one state is required and a batch "
            "must be differentiated one member at a time"
        )
    return state.reshape(-1)


def _composed_shift_terms(
    left: Sequence[tuple[float, float]],
    right: Sequence[tuple[float, float]],
    *,
    parameters: tuple[int, int],
    width: int,
    dtype: torch.dtype,
) -> tuple[tuple[float, torch.Tensor], ...]:
    """The second-order rule two declared first-order rules compose into.

    Each declared rule is a list of ``(coefficient, shift)`` terms for one gate
    parameter, so the mixed second derivative of a pair of parameters is the
    product rule of the two: every pair of terms contributes the product of
    their coefficients to the evaluation displaced by the first term's shift on
    the first parameter plus the second term's shift on the second parameter.

    Two term pairs that land on the same displacement are summed instead of
    being evaluated twice. That merge is what keeps the diagonal of a two-term
    rule at three evaluations rather than four: the two pairs whose shifts
    cancel both land on the unshifted point.

    A displacement whose collected coefficients cancel exactly is dropped. It
    would contribute nothing to the sum, so keeping it would make the cost of
    this function depend on the coefficients rather than on the rule structure.
    """

    collected: dict[tuple[float, ...], float] = {}
    for left_coefficient, left_shift in left:
        for right_coefficient, right_shift in right:
            displacement = [0.0] * width
            displacement[parameters[0]] += left_shift
            displacement[parameters[1]] += right_shift
            # Shifts are read from the declaration and re-created by addition, so
            # two paths to the same displacement can differ in their last bits.
            # Rounding to a fixed grid merges them; the coefficients stay exact.
            key = tuple(round(value, 12) for value in displacement)
            collected[key] = (
                collected.get(key, 0.0) + left_coefficient * right_coefficient
            )
    return tuple(
        (coefficient, torch.tensor(key, dtype=dtype))
        for key, coefficient in collected.items()
        if coefficient != 0.0
    )


def batched_parameter_shift_gradient(
    circuit_builder: CircuitBuilder,
    parameters: torch.Tensor,
    batch_loss_fn: BatchLossFunction,
    *,
    shift: float = torch.pi / 2,
) -> torch.Tensor:
    """Evaluate a parameter-shift gradient through one batch request.

    The profile sends one positive and one negative evaluation per input
    parameter, so it differentiates exactly the gate parameters whose declared
    frequencies produce a two-term rule: every one-frequency gate, including
    ``U1``, ``U2``, ``U3``, ``PHASE``, ``CPHASE``, ``RXX``, ``RYY``, and
    ``RZZ``. A gate whose declared frequencies need more evaluations -- ``CRX``,
    ``CRY``, and ``CRZ`` -- is refused instead of being answered with a two-term
    number. The circuits may also carry the profile's constant gates ``H``,
    ``X``, and ``CX``.

    Args:
        circuit_builder: Builds a circuit from a parameter tensor.
        parameters: Real, non-empty parameter tensor.
        batch_loss_fn: Evaluates every shifted circuit in one batch operation.
        shift: Parameter displacement. It must equal the shift the differentiated
            gate's declared rule prescribes, which is ``pi/2`` for every opcode
            that declares one frequency.

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
    coefficients: list[float] = []
    for index in range(flat.numel()):
        plus = flat.clone()
        minus = flat.clone()
        plus[index] += shift
        minus[index] -= shift
        plus_program = circuit_builder(plus.reshape_as(base))
        minus_program = circuit_builder(minus.reshape_as(base))
        coefficients.append(
            _batch_occurrence_coefficient(
                (base_structure, base_values),
                _circuit_profile(plus_program),
                _circuit_profile(minus_program),
                shift=shift,
                parameter_index=index,
            )
        )
        programs.extend((plus_program, minus_program))

    losses = _batch_losses(batch_loss_fn(tuple(programs)), expected=len(programs))
    terms = [
        coefficient * (losses[2 * index] - losses[2 * index + 1])
        for index, coefficient in enumerate(coefficients)
    ]
    return (
        torch.stack(terms)
        .reshape_as(base)
        .to(
            device=parameters.device,
            dtype=parameters.dtype,
        )
    )


def _validated_parameters(parameters: torch.Tensor, *, shift: float) -> torch.Tensor:
    if not isinstance(parameters, torch.Tensor):
        raise TypeError("parameters must be a torch.Tensor")
    if parameters.numel() == 0:
        raise ValueError("parameters must not be empty")
    if not (parameters.is_floating_point() and torch.isfinite(parameters).all()):
        raise ValueError("parameters must contain finite real floating-point values")
    try:
        displacement = float(shift)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"shift must be a positive finite displacement, got {shift!r}"
        ) from error
    if not (math.isfinite(displacement) and displacement > 0):
        raise ValueError(f"shift must be a positive finite displacement, got {shift!r}")
    return parameters.detach()


def _circuit_profile(program: Any) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    ir = ensure_circuit_ir(program)
    rejected = sorted({item.name for item in ir.instructions} - _BATCH_PROFILE_GATES)
    if rejected:
        raise ValueError(
            "batched parameter shift sends one evaluation pair per parameter, so "
            f"it differentiates one-frequency gates: {_profile_rejection(rejected)}"
        )
    if any(item.matrix is not None for item in ir.instructions):
        raise ValueError("batched parameter shift does not support custom matrices")
    return _circuit_profile_of(ir)


def _profile_rejection(opcodes: Sequence[str]) -> str:
    """Why the batched profile cannot evaluate each rejected gate."""

    reasons = []
    for opcode in opcodes:
        schema = OPERATOR_SCHEMAS.get(opcode)
        if schema is None:
            reasons.append(f"{opcode} declares no shift rule")
        elif schema.channel:
            reasons.append(f"{opcode} is a channel")
        elif not schema.parameters:
            reasons.append(
                f"{opcode} carries no declared parameter and is not one of the "
                "profile's constant gates H, X, and CX"
            )
        else:
            declared = ", ".join(
                str(frequencies) for frequencies in schema.parameter_frequencies
            )
            pairs = max(len(schema.shift_rule(name)) for name in schema.parameters) // 2
            reasons.append(
                f"{opcode} declares frequencies {declared} and needs {pairs} "
                "evaluation pairs"
            )
    return "; ".join(reasons)


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
    "jacobian",
    "jvp",
    "metric_tensor",
    "parameter_shift_gradient",
    "parameter_shift_hessian",
    "vjp",
]

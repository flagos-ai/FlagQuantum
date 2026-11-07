"""Replace a proved clean selector lifetime with its conditional unitary block."""

from collections.abc import Callable, Mapping
from dataclasses import replace

import torch

from ...compiler._construction.clean_unitary_proof import verify_clean_unitary
from ...compiler._hybrid.model import HybridProgram
from .syntax import Expression, Function, Statement, Token, reject
from .unary_selection_proof import _calls, _reachable


def _wrapper(function: Function, helper: Function) -> int | None:
    if (
        function.permutation
        or len(function.parameters) != 1
        or len(function.statements) != 2
        or len(helper.parameters) != 2
    ):
        return None
    data = function.parameters[0]
    declaration, scope = function.statements
    selector = declaration.declaration
    if (
        data.role != "inout"
        or data.kind != "qbit[]"
        or data.width != helper.parameters[0].width
        or selector is None
        or selector.kind != "qbit"
        or selector.width != 1
        or selector.name.text == data.name.text
        or data.length is not None
        or scope.within is None
        or len(scope.within) not in {1, 2}
        or len(scope.apply) != 1
    ):
        return None
    allocate, *prepare = scope.within
    (call,) = scope.apply
    if (
        allocate.token.text != "allocate"
        or tuple(arg.token.text for arg in allocate.arguments)
        != ("1", selector.name.text)
        or any(arg.children for arg in allocate.arguments)
        or call.token.text != helper.name.text
        or tuple(arg.token.text for arg in call.arguments)
        != (data.name.text, selector.name.text)
        or any(arg.children for arg in call.arguments)
    ):
        return None
    if not prepare:
        return 0
    flip = prepare[0]
    if (
        flip.token.text == "X"
        and len(flip.arguments) == 1
        and not flip.arguments[0].children
        and flip.arguments[0].token.text == selector.name.text
    ):
        return 1
    return None


def _matrix_expression(matrix: torch.Tensor, origin: Token) -> Expression:
    def literal(value: complex) -> Expression:
        real = Expression(replace(origin, text=repr(value.real)))
        if not value.imag:
            return real
        imag = Expression(replace(origin, text=repr(value.imag)))
        return Expression(
            replace(origin, text="+"),
            (real, Expression(replace(origin, text="j"), (imag,))),
        )

    return Expression(
        replace(origin, text="["),
        tuple(
            Expression(
                replace(origin, text="["),
                tuple(literal(complex(value)) for value in row),
            )
            for row in matrix.tolist()
        ),
    )


def normalize_clean_selectors(
    definitions: dict[str, Function],
    compile_probe: Callable[[Function, Mapping[str, Function]], HybridProgram],
) -> dict[str, Function]:
    """Require a verified unitary preserving the selector on the complete basis."""
    rewritten = dict(definitions)
    direct = {name: _calls(function) for name, function in definitions.items()}
    for name, helper in definitions.items():
        annotation = helper.disable_const_checks
        if annotation is None or len(helper.parameters) != 2 or name not in rewritten:
            continue
        data, control = helper.parameters
        if (
            data.role != "inout"
            or data.kind != "qbit[]"
            or data.width is None
            or not 1 <= data.width <= 4
            or control.role != "const"
            or control.kind != "qbit"
            or control.name.text != annotation.text
        ):
            continue
        users = {
            caller
            for caller, calls in direct.items()
            if name in calls and caller in rewritten
        }
        wrappers: dict[str, int] = {}
        for caller in users:
            seed = _wrapper(rewritten[caller], helper)
            if seed is not None:
                wrappers[caller] = seed
        if not wrappers or set(wrappers) != users:
            continue
        scope = _reachable(name, direct)
        if any(
            direct[other] & scope
            for other in rewritten.keys() - scope - wrappers.keys()
        ):
            reject(annotation, "ownership", "verified selector helpers cannot escape")
        proof_defs = {
            member: replace(rewritten[member], permutation=False) for member in scope
        }
        mutable_control = replace(control, role="inout")
        proof_defs[name] = replace(
            helper,
            parameters=(data, mutable_control),
            disable_const_checks=None,
        )
        for member in scope:
            original = rewritten[member]
            if not original.permutation:
                continue
            ports = original.parameters
            width = sum(port.width or 0 for port in ports)
            if any(
                port.width is None or port.role not in {"const", "inout"}
                for port in ports
            ):
                reject(
                    original.name,
                    "ownership",
                    "selector permutation proof needs fixed borrowed ports",
                )
            operator = verify_clean_unitary(
                compile_probe(proof_defs[member], proof_defs), width
            )
            permutation = operator.abs() > 0.5
            if (
                not torch.all(permutation.sum(0) == 1)
                or not torch.all(permutation.sum(1) == 1)
                or not torch.allclose(
                    operator, permutation.to(operator.dtype), atol=1e-12, rtol=0
                )
            ):
                reject(
                    original.name,
                    "permutation",
                    "selector dependency is not a basis permutation",
                )
        width = data.width + 1
        unitary = verify_clean_unitary(
            compile_probe(proof_defs[name], proof_defs), width
        )
        dimension = 1 << data.width
        if (
            unitary[:dimension, dimension:].abs().max().item() > 1e-12
            or unitary[dimension:, :dimension].abs().max().item() > 1e-12
        ):
            reject(
                annotation,
                "ownership",
                "the borrowed selector does not return to its basis value",
            )
        for caller, seed in wrappers.items():
            function = rewritten[caller]
            start, stop = seed * dimension, (seed + 1) * dimension
            matrix = unitary[start:stop, start:stop]
            token = function.name
            statement = Statement(
                replace(token, text="unitary"),
                (
                    _matrix_expression(matrix, token),
                    Expression(function.parameters[0].name),
                ),
            )
            rewritten[caller] = replace(function, statements=(statement,))
        for member in scope:
            rewritten.pop(member)
    return rewritten


def probe_source(function: Function) -> Function:
    """Build a small output-only program calling the ordinary bound helper."""
    ports = tuple(
        replace(port, role="output") for port in reversed(function.parameters)
    )
    token = function.name
    allocates = tuple(
        Statement(replace(token, text="allocate"), (Expression(port.name),))
        for port in ports
    )
    call = Statement(
        token, tuple(Expression(port.name) for port in function.parameters)
    )
    return Function(replace(token, text="main"), ports, (*allocates, call))

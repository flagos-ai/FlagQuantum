"""Capture bounded native preparation calls into existing register effects."""

from __future__ import annotations

import ast
import math
from typing import TYPE_CHECKING

from .frontend_python import _error
from .frontend_views import locations, packed_locations
from .program_types import (
    inplace_prepare_amplitudes,
    inplace_prepare_state,
    prepare_amplitudes,
    prepare_bell_state,
    prepare_ghz_state,
)
from .register_profile import MAX_PREPARED_AMPLITUDES, register_width

if TYPE_CHECKING:
    from .frontend_program import _Body


def _number(owner: _Body, node: ast.expr) -> float:
    value = owner.phase_number(node)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        numerator = owner.phase_number(node.left)
        denominator = node.right
        if (
            numerator is not None
            and isinstance(denominator, ast.Constant)
            and type(denominator.value) is int
            and denominator.value > 0
        ):
            try:
                value = numerator / denominator.value
            except OverflowError:
                value = None
    if value is None or not math.isfinite(value):
        _error(
            owner.source,
            node,
            "program.prepare",
            "preparation requires finite static reals; fractions need positive integer denominators",
        )
    return value


def capture_preparation(owner: _Body, node: ast.Call, function: object) -> None:
    """Translate familiar preparation calls without running host Python bodies."""
    if function is prepare_bell_state or function is prepare_ghz_state:
        if len(node.args) != 2:
            _error(
                owner.source,
                node,
                "program.signature",
                "expected size/state and target",
            )
        size = node.args[0]
        if not isinstance(size, ast.Constant) or type(size.value) is not int:
            _error(
                owner.source,
                size,
                "program.prepare",
                "size/state must be a literal integer",
            )
        bell = function is prepare_bell_state
        if bell and size.value not in {1, 2} or not bell and not 1 <= size.value <= 256:
            _error(
                owner.source,
                size,
                "program.prepare",
                "Bell state must be 1 or 2; GHZ size must be in 1..256",
            )
        width = 2 if bell else size.value
        target = owner.register(node.args[1], width)
        if register_width(target) != width:
            _error(
                owner.source,
                node,
                "program.width",
                "preparation size must match the target width",
            )
        owner.emit("qreg.prepare_ghz", node, (target,))
        if bell:
            owner.emit(
                "qreg.gate",
                node,
                (target,),
                gate="z" if size.value == 1 else "x",
                index=0,
            )
        return
    if len(node.args) != 3:
        _error(
            owner.source, node, "program.signature", "expected vector, bound and target"
        )
    vector, tolerance, output = node.args
    bound_vector = (
        owner.classical.get(vector.id) if isinstance(vector, ast.Name) else None
    )
    if isinstance(vector, (ast.List, ast.Tuple)):
        vector_size = len(vector.elts)
    elif isinstance(bound_vector, tuple):
        vector_size = len(bound_vector)
    else:
        _error(
            owner.source,
            vector,
            "program.prepare",
            "use a literal real vector or a bound classical real array",
        )
    if not 2 <= vector_size <= MAX_PREPARED_AMPLITUDES or vector_size & (
        vector_size - 1
    ):
        _error(
            owner.source,
            vector,
            "program.prepare",
            "use a real vector of power-of-two length in 2..256",
        )
    if isinstance(vector, (ast.List, ast.Tuple)):
        values = tuple(_number(owner, item) for item in vector.elts)
    else:
        assert isinstance(bound_vector, tuple)
        values = bound_vector
    bound = _number(owner, tolerance)
    if bound < 0:
        _error(
            owner.source,
            tolerance,
            "program.prepare",
            "preparation bound must be nonnegative",
        )
    if function not in (prepare_amplitudes, inplace_prepare_amplitudes):
        try:
            normalized = math.isclose(math.fsum(values), 1.0, rel_tol=0, abs_tol=1e-12)
        except OverflowError:
            normalized = False
        if any(value < 0 for value in values) or not normalized:
            _error(
                owner.source,
                vector,
                "program.prepare",
                "probabilities must be nonnegative and sum to one",
            )
        values = tuple(math.sqrt(value) for value in values)
        bound = 0.0
    width = len(values).bit_length() - 1
    if function in (inplace_prepare_amplitudes, inplace_prepare_state):
        if isinstance(output, (ast.List, ast.Tuple)):
            targets = tuple(reversed(packed_locations(owner, output)))
            if len(targets) != width or len(set(targets)) != len(targets):
                _error(
                    owner.source,
                    output,
                    "program.prepare",
                    "joint preparation needs distinct bits matching the vector width",
                )
            owner.emit(
                "qreg.inplace_prepare_amplitudes",
                node,
                tuple(value for value, _ in targets),
                amplitudes=values,
                bound=bound,
                indices=tuple(index for _, index in targets),
            )
            return
        if not isinstance(output, ast.Name):
            _error(
                owner.source,
                output,
                "program.view",
                "inplace preparation requires a complete register",
            )
        targets = locations(owner, output)
        target = targets[0][0]
        if targets != tuple((target, bit) for bit in range(register_width(target))):
            _error(
                owner.source,
                output,
                "program.view",
                "inplace preparation requires a complete register",
            )
        if register_width(target) != width:
            _error(
                owner.source,
                output,
                "program.width",
                "vector width must match the target",
            )
        owner.emit(
            "qreg.inplace_prepare_amplitudes",
            node,
            (target,),
            amplitudes=values,
            bound=bound,
        )
        return
    target = owner.register(output, width)
    if register_width(target) != width:
        _error(
            owner.source, output, "program.width", "vector width must match the target"
        )
    owner.emit(
        "qreg.prepare_amplitudes", node, (target,), amplitudes=values, bound=bound
    )

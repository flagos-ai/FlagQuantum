"""Convert Qmod syntax for the authoritative Compiler Boolean factor pass."""

from collections.abc import Callable
from dataclasses import replace

from ...compiler import _phase_factors
from ...compiler._phase_factors import MAX_FACTORS as MAX_FACTORS
from ...compiler._phase_factors import Cube as Cube
from .phase import Location
from .syntax import Expression


def preserve_factors(
    expression: Expression,
    quantum_word: Callable[[Expression], tuple[Location, ...] | None],
) -> tuple[Expression, tuple[Cube, ...]]:
    return _phase_factors.preserve_factors(
        expression,
        quantum_word,
        operator_of=lambda node: node.token.text,
        operands_of=lambda node: node.children,
        replace_operands=lambda node, children: replace(node, children=children),
        make_factor=lambda node, index: Expression(
            replace(node.token, text=f"$phase_factor{index}")
        ),
    )

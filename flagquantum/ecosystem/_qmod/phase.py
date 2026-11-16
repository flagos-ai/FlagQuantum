"""Adapt Qmod syntax and diagnostics to the shared compiler phase expansion."""

from collections.abc import Callable

from ...compiler._phase_polynomial import (
    MAX_PHASE_DEGREE as MAX_PHASE_DEGREE,
)
from ...compiler._phase_polynomial import (
    MAX_PHASE_INPUT_BITS as MAX_PHASE_INPUT_BITS,
)
from ...compiler._phase_polynomial import (
    MAX_PHASE_PREDICATE_BITS as MAX_PHASE_PREDICATE_BITS,
)
from ...compiler._phase_polynomial import (
    MAX_PHASE_PRODUCTS as MAX_PHASE_PRODUCTS,
)
from ...compiler._phase_polynomial import (
    MAX_PHASE_TERMS as MAX_PHASE_TERMS,
)
from ...compiler._phase_polynomial import (
    Location as Location,
)
from ...compiler._phase_polynomial import (
    Polynomial as Polynomial,
)
from ...compiler._phase_polynomial import (
    expand_phase as _expand_phase,
)
from .syntax import Expression, reject


def expand_phase(
    expression: Expression,
    quantum_word: Callable[[Expression], tuple[Location, ...] | None],
    number: Callable[[Expression], float],
    *,
    signed_word: Callable[[Expression], bool] | None = None,
    fraction_bits: Callable[[Expression], int] | None = None,
    condition: Expression | None = None,
    disjoint_condition_target: Location | None = None,
) -> tuple[Polynomial, tuple[Location, ...]]:
    """Keep Qmod diagnostics and source coordinates at the format boundary."""
    return _expand_phase(
        expression,
        quantum_word,
        number,
        operator_of=lambda node: node.token.text,
        operands_of=lambda node: node.children,
        fail=lambda node, code, message: reject(node.token, code, message),
        signed_word=signed_word,
        fraction_bits=fraction_bits,
        condition=condition,
        disjoint_condition_target=disjoint_condition_target,
    )

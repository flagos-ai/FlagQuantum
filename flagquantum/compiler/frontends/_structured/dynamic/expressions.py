"""Measurement predicates and condition-partitioned classical values."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from flagquantum._import_compat import alias_module as _alias_module
from flagquantum.compiler.frontends._structured.model import Operation
from flagquantum.compiler.frontends._structured.runtime_inputs import _runtime_shape

if TYPE_CHECKING:
    from flagquantum.compiler.frontends._structured.dynamic.program import (
        _DynamicLowerer,
    )


@dataclass(frozen=True)
class _MeasurementRef:
    classical_bit: int


@dataclass(frozen=True)
class _ClassicalPredicate:
    clauses: tuple[tuple[tuple[int, int], ...], ...]


@dataclass(frozen=True)
class _ConditionalValue:
    cases: tuple[tuple[tuple[tuple[tuple[int, int], ...], ...], Any], ...] = field(
        compare=False
    )


_TRUE_CLAUSES: tuple[tuple[tuple[int, int], ...], ...] = ((),)
_FALSE_CLAUSES: tuple[tuple[tuple[int, int], ...], ...] = ()


def predicate_clauses(
    lowerer: _DynamicLowerer,
    operation: Operation,
    value: _MeasurementRef | _ClassicalPredicate | _ConditionalValue,
) -> tuple[tuple[tuple[int, int], ...], ...]:
    if isinstance(value, _MeasurementRef):
        return (((value.classical_bit, 1),),)
    if isinstance(value, _ClassicalPredicate):
        return value.clauses
    result = _FALSE_CLAUSES
    for case_conditions, case_value in value.cases:
        if isinstance(
            case_value,
            (_MeasurementRef, _ClassicalPredicate, _ConditionalValue),
        ):
            truth = predicate_clauses(lowerer, operation, case_value)
        elif as_predicate(lowerer, operation, case_value):
            truth = _TRUE_CLAUSES
        else:
            truth = _FALSE_CLAUSES
        result = or_clauses(
            lowerer,
            operation,
            result,
            and_clauses(lowerer, operation, case_conditions, truth),
        )
    return result


def canonicalize_clauses(
    lowerer: _DynamicLowerer,
    operation: Operation,
    clauses: Sequence[Sequence[tuple[int, int]]],
) -> tuple[tuple[tuple[int, int], ...], ...]:
    canonical: set[tuple[tuple[int, int], ...]] = set()
    for clause in clauses:
        terms: dict[int, int] = {}
        contradictory = False
        for bit, expected in clause:
            previous = terms.get(bit)
            if previous is not None and previous != expected:
                contradictory = True
                break
            terms[bit] = expected
        if not contradictory:
            normalized = tuple(sorted(terms.items()))
            if not normalized:
                return _TRUE_CLAUSES
            canonical.add(normalized)
    changed = True
    while changed:
        changed = False
        ordered = tuple(canonical)
        for index, left in enumerate(ordered):
            left_terms = dict(left)
            for right in ordered[index + 1 :]:
                right_terms = dict(right)
                if left_terms.keys() != right_terms.keys():
                    continue
                differing = tuple(
                    bit for bit in left_terms if left_terms[bit] != right_terms[bit]
                )
                if len(differing) != 1:
                    continue
                common = tuple(term for term in left if term[0] != differing[0])
                if not common:
                    return _TRUE_CLAUSES
                canonical.discard(left)
                canonical.discard(right)
                canonical.add(common)
                changed = True
                break
            if changed:
                break
    reduced = tuple(
        clause
        for clause in sorted(canonical, key=lambda item: (len(item), item))
        if not any(
            set(other).issubset(clause) for other in canonical if other != clause
        )
    )
    if len(reduced) > lowerer.max_condition_clauses:
        lowerer.fail(
            operation,
            "dynamic.condition_clause_limit",
            f"predicate exceeds {lowerer.max_condition_clauses} canonical clauses",
        )
    return reduced


def and_clauses(
    lowerer: _DynamicLowerer, operation: Operation, left: Any, right: Any
) -> Any:
    if not left or not right:
        return _FALSE_CLAUSES
    return canonicalize_clauses(
        lowerer, operation, tuple((*a, *b) for a in left for b in right)
    )


def or_clauses(
    lowerer: _DynamicLowerer, operation: Operation, left: Any, right: Any
) -> Any:
    return canonicalize_clauses(lowerer, operation, (*left, *right))


def not_clauses(lowerer: _DynamicLowerer, operation: Operation, clauses: Any) -> Any:
    if clauses == _FALSE_CLAUSES:
        return _TRUE_CLAUSES
    if clauses == _TRUE_CLAUSES:
        return _FALSE_CLAUSES
    result = _TRUE_CLAUSES
    for clause in clauses:
        complement = tuple(((bit, 1 - expected),) for bit, expected in clause)
        result = and_clauses(lowerer, operation, result, complement)
    return result


def negate_predicate(lowerer: _DynamicLowerer, operation: Operation, value: Any) -> Any:
    if isinstance(value, _MeasurementRef):
        return _ClassicalPredicate((((value.classical_bit, 0),),))
    if isinstance(value, _ClassicalPredicate):
        return _ClassicalPredicate(not_clauses(lowerer, operation, value.clauses))
    if isinstance(value, _ConditionalValue):
        return _ClassicalPredicate(
            not_clauses(
                lowerer, operation, predicate_clauses(lowerer, operation, value)
            )
        )
    return not as_predicate(lowerer, operation, value)


def conjoin_predicates(
    lowerer: _DynamicLowerer, operation: Operation, values: tuple[Any, ...]
) -> Any:
    result = _TRUE_CLAUSES
    for value in values:
        if not isinstance(
            value, (_MeasurementRef, _ClassicalPredicate, _ConditionalValue)
        ):
            if not as_predicate(lowerer, operation, value):
                return False
            continue
        result = and_clauses(
            lowerer, operation, result, predicate_clauses(lowerer, operation, value)
        )
    if result == _FALSE_CLAUSES:
        return False
    if result == _TRUE_CLAUSES:
        return True
    return _ClassicalPredicate(result)


def disjoin_predicates(
    lowerer: _DynamicLowerer, operation: Operation, values: tuple[Any, ...]
) -> Any:
    result = _FALSE_CLAUSES
    for value in values:
        if not isinstance(
            value, (_MeasurementRef, _ClassicalPredicate, _ConditionalValue)
        ):
            if as_predicate(lowerer, operation, value):
                return True
            continue
        result = or_clauses(
            lowerer, operation, result, predicate_clauses(lowerer, operation, value)
        )
    if result == _FALSE_CLAUSES:
        return False
    if result == _TRUE_CLAUSES:
        return True
    return _ClassicalPredicate(result)


def compare_predicate(
    lowerer: _DynamicLowerer,
    operation: Operation,
    predicate: str,
    values: tuple[Any, ...],
) -> Any:
    if any(isinstance(value, _ConditionalValue) for value in values):
        cases: list[tuple[Any, Any]] = []
        for case_conditions, case_values in conditional_product(
            lowerer, operation, values
        ):
            if any(
                isinstance(value, (_MeasurementRef, _ClassicalPredicate))
                for value in case_values
            ):
                result = compare_predicate(lowerer, operation, predicate, case_values)
            else:
                result = compare_values(predicate, *case_values)
            cases.append((case_conditions, result))
        return merge_conditional_values(lowerer, operation, cases)
    if predicate not in {"eq", "ne"}:
        lowerer.fail(
            operation,
            "dynamic.classical_expression",
            "measurement predicates support only == or !=",
        )
    left, right = values
    symbolic_types = (_MeasurementRef, _ClassicalPredicate, _ConditionalValue)
    left_symbolic = isinstance(left, symbolic_types)
    right_symbolic = isinstance(right, symbolic_types)
    if left_symbolic and right_symbolic:
        equal = disjoin_predicates(
            lowerer,
            operation,
            (
                conjoin_predicates(lowerer, operation, (left, right)),
                conjoin_predicates(
                    lowerer,
                    operation,
                    (
                        negate_predicate(lowerer, operation, left),
                        negate_predicate(lowerer, operation, right),
                    ),
                ),
            ),
        )
        return (
            equal if predicate == "eq" else negate_predicate(lowerer, operation, equal)
        )
    symbolic = left if left_symbolic else right
    constant = right if left_symbolic else left
    if not (left_symbolic or right_symbolic) or not isinstance(constant, bool):
        lowerer.fail(
            operation,
            "dynamic.classical_expression",
            "measurement comparison requires a measurement predicate or bool constant",
        )
    should_negate = (predicate == "eq") != constant
    return negate_predicate(lowerer, operation, symbolic) if should_negate else symbolic


def compare_values(predicate: str, left: Any, right: Any) -> Any:
    comparisons = {
        "eq": lambda: left == right,
        "ne": lambda: left != right,
        "lt": lambda: left < right,
        "le": lambda: left <= right,
        "gt": lambda: left > right,
        "ge": lambda: left >= right,
    }
    return comparisons[predicate]()


def conditional_product(
    lowerer: _DynamicLowerer, operation: Operation, values: Sequence[Any]
) -> tuple[tuple[Any, tuple[Any, ...]], ...]:
    products: tuple[tuple[Any, tuple[Any, ...]], ...] = ((_TRUE_CLAUSES, ()),)
    for value in values:
        cases = (
            value.cases
            if isinstance(value, _ConditionalValue)
            else ((_TRUE_CLAUSES, value),)
        )
        expanded: list[tuple[Any, tuple[Any, ...]]] = []
        for product_conditions, product_values in products:
            for case_conditions, case_value in cases:
                combined = and_clauses(
                    lowerer, operation, product_conditions, case_conditions
                )
                if combined:
                    expanded.append((combined, (*product_values, case_value)))
        if len(expanded) > lowerer.max_condition_clauses:
            lowerer.fail(
                operation,
                "dynamic.value_case_limit",
                f"value expansion exceeds {lowerer.max_condition_clauses} cases",
            )
        products = tuple(expanded)
    return products


def merge_conditional_values(
    lowerer: _DynamicLowerer,
    operation: Operation,
    cases: Sequence[tuple[Any, Any]],
) -> Any:
    flattened: list[tuple[Any, Any]] = []
    for outer_conditions, value in cases:
        nested = (
            value.cases
            if isinstance(value, _ConditionalValue)
            else ((_TRUE_CLAUSES, value),)
        )
        for inner_conditions, inner_value in nested:
            combined = and_clauses(
                lowerer, operation, outer_conditions, inner_conditions
            )
            if combined:
                flattened.append((combined, inner_value))
    merged: list[tuple[Any, Any]] = []
    for conditions, value in flattened:
        for index, (existing_conditions, existing_value) in enumerate(merged):
            if same_runtime_value(value, existing_value):
                merged[index] = (
                    or_clauses(lowerer, operation, existing_conditions, conditions),
                    existing_value,
                )
                break
        else:
            merged.append((conditions, value))
    if len(merged) > lowerer.max_condition_clauses:
        lowerer.fail(
            operation,
            "dynamic.value_case_limit",
            f"value expansion exceeds {lowerer.max_condition_clauses} cases",
        )
    if len(merged) == 1 and merged[0][0] == _TRUE_CLAUSES:
        return merged[0][1]
    return _ConditionalValue(tuple(merged))


def map_conditional_values(
    lowerer: _DynamicLowerer,
    operation: Operation,
    values: Sequence[Any],
    function: Callable[..., Any],
) -> Any:
    cases = tuple(
        (conditions, function(*case_values))
        for conditions, case_values in conditional_product(lowerer, operation, values)
    )
    return merge_conditional_values(lowerer, operation, cases)


def same_runtime_value(left: Any, right: Any) -> bool:
    if left is right:
        return True
    if type(left) is type(right) and isinstance(left, (bool, int, float, str)):
        return bool(left == right)
    return False


def as_predicate(lowerer: _DynamicLowerer, operation: Operation, value: Any) -> bool:
    shape = _runtime_shape(value)
    if shape not in {None, ()}:
        lowerer.fail(
            operation,
            "control.predicate_shape",
            f"runtime predicate must be scalar, got shape {shape}",
        )
    try:
        return bool(value)
    except (RuntimeError, TypeError, ValueError) as exc:
        lowerer.fail(operation, "control.predicate", f"predicate is invalid: {exc}")


# Preserve historical import identities for pickles and public introspection.
for _identity_export in (
    _MeasurementRef,
    _ClassicalPredicate,
    _ConditionalValue,
    predicate_clauses,
    canonicalize_clauses,
    and_clauses,
    or_clauses,
    not_clauses,
    negate_predicate,
    conjoin_predicates,
    disjoin_predicates,
    compare_predicate,
    compare_values,
    conditional_product,
    merge_conditional_values,
    map_conditional_values,
    same_runtime_value,
    as_predicate,
):
    _identity_export.__module__ = "flagquantum.compiler._hybrid.dynamic.expressions"
del _identity_export

_alias_module("flagquantum.compiler._hybrid.dynamic.expressions", __name__)

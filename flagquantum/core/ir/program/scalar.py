"""Validate owned scalar literals without evaluating unresolved parameters."""

from math import isfinite

from ...parameters import Parameter, ParameterExpression


def scalar_literal_error(value: object) -> str | None:
    pending = [(value, frozenset[int]())]
    while pending:
        item, ancestors = pending.pop()
        if isinstance(item, Parameter):
            if not isinstance(item.name, str) or not item.name:
                return "scalar parameter requires a nonempty name"
        elif isinstance(item, ParameterExpression):
            if id(item) in ancestors:
                return "scalar expression must be acyclic"
            if not isinstance(item.op, str) or item.op not in {
                "add",
                "sub",
                "mul",
                "neg",
            }:
                return "scalar expression uses an unsupported operation"
            arity = 1 if item.op == "neg" else 2
            if not isinstance(item.args, tuple) or len(item.args) != arity:
                return "scalar expression has invalid immutable operands"
            nested = ancestors | {id(item)}
            pending.extend((arg, nested) for arg in item.args)
        elif isinstance(item, (int, float)) and type(item) in (int, float):
            try:
                if not isfinite(item):
                    return "scalar numeric leaves must be finite"
            except OverflowError:
                return "scalar numeric leaves must be finite"
        else:
            return "scalar literal requires a number or an owned parameter expression"
    return None

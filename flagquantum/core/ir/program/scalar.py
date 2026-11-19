"""Validate owned scalar literals without evaluating unresolved parameters."""

from ...parameters import _parameter_literal_error


def scalar_literal_error(value: object) -> str | None:
    return _parameter_literal_error(value)

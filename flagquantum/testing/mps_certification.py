"""Fail-closed validation for distributed MPS numerical certification records."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ._finite import require_finite

CASE_MEASUREMENTS = (
    "boundary_directional_derivative_error",
    "max_value_error",
    "max_gradient_error",
    "max_parameter_error",
    "exact_discarded_weight",
    "approximate_discarded_weight",
    "approximate_error_budget",
    "approximate_value_error",
    "approximate_gradient_error",
)
TOLERANCE_LIMITS = (
    "directional_atol",
    "value_atol",
    "gradient_atol",
    "parameter_atol",
    "approximate_value_atol",
    "approximate_gradient_atol",
)


class MPSCertificationError(ValueError):
    """A correctness artifact does not satisfy the ISSUE-091 contract."""


def require_mps_numerical_certification(payload: Mapping[str, Any]) -> None:
    """Accept an ISSUE-091 correctness matrix only if it covers the whole grid.

    The matrix is a cross product: both dtypes against one, two, four and eight
    ranks, trained with Adam. A subset that happens to pass is not a
    certification of the other cells, so the cells are compared as a set rather
    than counted -- four passing cases could be the same corner measured twice.

    Raises:
        MPSCertificationError: if the schema is not the ISSUE-091 one, if either
            dtype or any of the required rank counts is missing from the grid,
            if any tolerance or measurement is missing or not finite, or if the
            run is too short to have reached a steady state.
    """
    if payload.get("schema") != "flagquantum.issue091.mps_correctness_matrix.v1":
        raise MPSCertificationError("unexpected ISSUE-091 schema")
    if set(payload.get("dtypes", ())) != {"complex64", "complex128"}:
        raise MPSCertificationError(
            "MPS dtype certification requires complex64/complex128"
        )
    if int(payload.get("steps", 0)) < 5:
        raise MPSCertificationError("MPS certification requires at least five steps")
    tolerances = payload.get("tolerances", {})
    cases = payload.get("cases", ())
    if not cases:
        raise MPSCertificationError("MPS certification matrix is empty")
    required_worlds = {1, 2, 4, 8}
    worlds = {int(case.get("world_size", 0)) for case in cases}
    if worlds != required_worlds:
        raise MPSCertificationError("MPS certification requires 1/2/4/8 ranks")
    if {case.get("dtype") for case in cases} != {"complex64", "complex128"}:
        raise MPSCertificationError("per-case MPS dtype coverage is incomplete")
    required_matrix = {
        (dtype, world)
        for dtype in ("complex64", "complex128")
        for world in required_worlds
    }
    covered_matrix = {
        (case.get("dtype"), int(case.get("world_size", 0)))
        for case in cases
        if case.get("optimizer") == "adam"
    }
    if not required_matrix <= covered_matrix:
        raise MPSCertificationError(
            "MPS certification requires Adam dtype/world-size Cartesian coverage"
        )
    if not {"adam", "sgd"} <= {case.get("optimizer") for case in cases}:
        raise MPSCertificationError("MPS certification requires Adam and SGD coverage")
    required_sources = payload.get("source_artifacts", ())
    if len(required_sources) != len(cases) or any(
        not item.get("sha256") or not item.get("path") for item in required_sources
    ):
        raise MPSCertificationError("MPS certification source provenance is incomplete")
    limits = {
        name: require_finite(
            tolerances.get(name),
            error_type=MPSCertificationError,
            label=f"tolerances.{name}",
        )
        for name in TOLERANCE_LIMITS
    }
    for index, case in enumerate(cases):
        if case.get("gradient_ownership") != "deterministic_all_reduce":
            raise MPSCertificationError("gradient ownership is not certified")
        if not case.get("boundary_directional_derivative_passed"):
            raise MPSCertificationError("boundary directional derivative failed")
        measured = {
            name: require_finite(
                case.get(name),
                error_type=MPSCertificationError,
                label=f"cases[{index}].{name}",
            )
            for name in CASE_MEASUREMENTS
        }
        if (
            measured["boundary_directional_derivative_error"]
            > limits["directional_atol"]
        ):
            raise MPSCertificationError(
                "boundary directional derivative exceeds tolerance"
            )
        if measured["max_value_error"] > limits["value_atol"]:
            raise MPSCertificationError("MPS value error exceeds tolerance")
        if measured["max_gradient_error"] > limits["gradient_atol"]:
            raise MPSCertificationError("MPS gradient error exceeds tolerance")
        if measured["max_parameter_error"] > limits["parameter_atol"]:
            raise MPSCertificationError(
                "MPS optimizer parameter error exceeds tolerance"
            )
        if measured["exact_discarded_weight"] != 0.0:
            raise MPSCertificationError("exact MPS execution discarded weight")
        if (
            measured["approximate_discarded_weight"]
            > measured["approximate_error_budget"]
        ):
            raise MPSCertificationError("MPS truncation budget exceeded")
        if measured["approximate_value_error"] > limits["approximate_value_atol"]:
            raise MPSCertificationError("approximate MPS value error exceeds tolerance")
        if measured["approximate_gradient_error"] > limits["approximate_gradient_atol"]:
            raise MPSCertificationError(
                "approximate MPS gradient error exceeds tolerance"
            )
        rank_errors = case.get("rank_errors", ())
        if len(rank_errors) != int(case.get("world_size", 0)):
            raise MPSCertificationError("per-rank MPS errors are incomplete")
        if not case.get("passed"):
            raise MPSCertificationError("MPS certification case failed")
    if not payload.get("passed"):
        raise MPSCertificationError("MPS correctness matrix did not pass")


__all__ = ("MPSCertificationError", "require_mps_numerical_certification")

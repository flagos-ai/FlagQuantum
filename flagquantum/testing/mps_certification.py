"""Fail-closed validation for distributed MPS numerical certification records."""

from __future__ import annotations

from typing import Any, Mapping


class MPSCertificationError(ValueError):
    """A correctness artifact does not satisfy the ISSUE-091 contract."""


def require_mps_numerical_certification(payload: Mapping[str, Any]) -> None:
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
    for case in cases:
        if case.get("gradient_ownership") != "deterministic_all_reduce":
            raise MPSCertificationError("gradient ownership is not certified")
        if not case.get("boundary_directional_derivative_passed"):
            raise MPSCertificationError("boundary directional derivative failed")
        if float(
            case.get("boundary_directional_derivative_error", float("inf"))
        ) > float(tolerances["directional_atol"]):
            raise MPSCertificationError(
                "boundary directional derivative exceeds tolerance"
            )
        if float(case.get("max_value_error", float("inf"))) > float(
            tolerances["value_atol"]
        ):
            raise MPSCertificationError("MPS value error exceeds tolerance")
        if float(case.get("max_gradient_error", float("inf"))) > float(
            tolerances["gradient_atol"]
        ):
            raise MPSCertificationError("MPS gradient error exceeds tolerance")
        if float(case.get("max_parameter_error", float("inf"))) > float(
            tolerances["parameter_atol"]
        ):
            raise MPSCertificationError(
                "MPS optimizer parameter error exceeds tolerance"
            )
        if float(case.get("exact_discarded_weight", float("inf"))) != 0.0:
            raise MPSCertificationError("exact MPS execution discarded weight")
        if float(case.get("approximate_discarded_weight", float("inf"))) > float(
            case.get("approximate_error_budget", -1.0)
        ):
            raise MPSCertificationError("MPS truncation budget exceeded")
        if float(case.get("approximate_value_error", float("inf"))) > float(
            tolerances["approximate_value_atol"]
        ):
            raise MPSCertificationError("approximate MPS value error exceeds tolerance")
        if float(case.get("approximate_gradient_error", float("inf"))) > float(
            tolerances["approximate_gradient_atol"]
        ):
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

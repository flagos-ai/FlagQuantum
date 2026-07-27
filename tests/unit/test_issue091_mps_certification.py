from __future__ import annotations

import copy

import pytest

from flagquantum.testing import (
    MPSCertificationError,
    require_mps_numerical_certification,
)

pytestmark = pytest.mark.unit


def payload():
    case = {
        "world_size": 1,
        "gradient_ownership": "deterministic_all_reduce",
        "boundary_directional_derivative_passed": True,
        "boundary_directional_derivative_error": 1e-6,
        "max_value_error": 1e-7,
        "max_gradient_error": 1e-6,
        "max_parameter_error": 1e-7,
        "exact_discarded_weight": 0.0,
        "approximate_discarded_weight": 0.01,
        "approximate_error_budget": 0.02,
        "approximate_value_error": 1e-6,
        "approximate_gradient_error": 1e-5,
        "optimizer": "adam",
        "passed": True,
    }
    return {
        "schema": "flagquantum.issue091.mps_correctness_matrix.v1",
        "dtypes": ["complex64", "complex128"],
        "steps": 5,
        "tolerances": {
            "value_atol": 1e-5,
            "gradient_atol": 2e-4,
            "parameter_atol": 2e-5,
            "directional_atol": 1e-4,
            "approximate_value_atol": 1e-4,
            "approximate_gradient_atol": 5e-4,
        },
        "cases": [
            {
                **case,
                "world_size": world,
                "dtype": dtype,
                "rank_errors": tuple({"rank": rank} for rank in range(world)),
            }
            for dtype in ("complex64", "complex128")
            for world in (1, 2, 4, 8)
        ]
        + [
            {
                **case,
                "world_size": 2,
                "dtype": "complex64",
                "optimizer": "sgd",
                "rank_errors": ({"rank": 0}, {"rank": 1}),
            }
        ],
        "passed": True,
        "source_artifacts": [
            {"path": f"case-{index}.json", "sha256": "a" * 64} for index in range(9)
        ],
    }


def test_issue091_certification_accepts_complete_matrix():
    require_mps_numerical_certification(payload())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_gradient_error", 1.0),
        ("gradient_ownership", "wrong_owner"),
        ("approximate_discarded_weight", 1.0),
        ("boundary_directional_derivative_passed", False),
    ],
)
def test_issue091_certification_rejects_injected_faults(field, value):
    invalid = copy.deepcopy(payload())
    invalid["cases"][2][field] = value
    with pytest.raises(MPSCertificationError):
        require_mps_numerical_certification(invalid)


def test_issue091_certification_rejects_dtype_fault():
    invalid = payload()
    invalid["dtypes"] = ["complex64", "float32"]
    with pytest.raises(MPSCertificationError, match="dtype"):
        require_mps_numerical_certification(invalid)


def test_issue091_rejects_numeric_directional_cartesian_and_approximate_faults():
    directional = payload()
    directional["cases"][0]["boundary_directional_derivative_error"] = 1.0
    with pytest.raises(MPSCertificationError, match="directional"):
        require_mps_numerical_certification(directional)

    cartesian = payload()
    for case in cartesian["cases"]:
        if case["world_size"] > 1:
            case["dtype"] = "complex64"
    with pytest.raises(MPSCertificationError, match="Cartesian"):
        require_mps_numerical_certification(cartesian)

    approximate = payload()
    approximate["cases"][0]["approximate_gradient_error"] = 1.0
    with pytest.raises(MPSCertificationError, match="approximate"):
        require_mps_numerical_certification(approximate)

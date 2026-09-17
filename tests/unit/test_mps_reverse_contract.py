"""Contract checks for the rank-owned MPS reverse request and static QR record.

These are the fail-closed guards that must reject an unexecutable request before
any tensor allocation, plus the shape rule that lets an untruncated two-site gate
skip its metadata broadcast. Both are pure functions, so they are covered here
rather than through a distributed run.
"""

from __future__ import annotations

import pytest

from flagquantum.core.ir import Instruction
from flagquantum.runtime.executors.mps.errors import MPSReverseContractError
from flagquantum.runtime.executors.mps.reverse import (
    _static_exact_qr_record,
    _validate_reverse_request,
)

pytestmark = pytest.mark.unit

VALID_REQUEST = {
    "gradient_policy": "exact",
    "degeneracy_tolerance": 1e-7,
    "initial_bond_dimension": 1,
    "initial_mps_tensors_provided": False,
    "initial_mps_left_canonical": False,
    "canonicalization_policy": "dirty",
    "compile_site_kernels": False,
    "gradient_owner_ranks": None,
    "world_size": 2,
    "observable": {0: "z"},
    "observable_terms": None,
    "hamiltonian_terms": None,
}


def _ry() -> Instruction:
    return Instruction("ry", (0,), {"theta": 0.1})


def _validate(**overrides: object) -> None:
    _validate_reverse_request(
        instructions=overrides.pop("instructions", (_ry(),)),
        **{**VALID_REQUEST, **overrides},
    )


def test_a_supported_request_passes_validation() -> None:
    _validate()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"gradient_policy": "cheap"}, "gradient_policy must be exact or approximate"),
        ({"degeneracy_tolerance": -1.0}, "degeneracy_tolerance must be non-negative"),
        ({"initial_bond_dimension": 0}, "initial_bond_dimension must be positive"),
        (
            {"canonicalization_policy": "sometimes"},
            "canonicalization_policy must be none, dirty or full",
        ),
        (
            {"gradient_owner_ranks": [0, 2]},
            "gradient owner rank is outside the process group",
        ),
        (
            {"observable": {0: "z"}, "observable_terms": [({0: "z"}, 1.0)]},
            "observable, observable_terms and hamiltonian_terms are mutually "
            "exclusive",
        ),
    ],
)
def test_invalid_requests_are_rejected(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _validate(**overrides)


def test_left_canonical_flag_requires_supplied_tensors() -> None:
    with pytest.raises(MPSReverseContractError, match="requires initial_mps_tensors"):
        _validate(initial_mps_left_canonical=True)


def test_compiled_two_site_rotations_require_ascending_wires() -> None:
    descending = Instruction("rzz", (1, 0), {"theta": 0.2})

    _validate(instructions=(descending,))

    with pytest.raises(MPSReverseContractError, match="ascending adjacent wire"):
        _validate(instructions=(descending,), compile_site_kernels=True)


def test_static_qr_record_derives_full_rank_split_metadata() -> None:
    record = _static_exact_qr_record(
        (1, 2, 2, 3), (1, 3, 2, 4), max_bond=None, cutoff=0.0
    )

    assert record is not None
    assert record["input_shapes"] == ((1, 2, 2, 3), (1, 3, 2, 4))
    assert record["output_shapes"] == ((1, 2, 2, 4), (1, 4, 2, 4))
    assert record["split_info"] == {
        "rank": 4,
        "original_rank": 4,
        "discarded_weight": 0.0,
    }


@pytest.mark.parametrize(
    ("left_shape", "right_shape"),
    [
        ((1, 2, 2), (1, 3, 2, 4)),
        ((1, 2, 2, 3), (1, 3, 2)),
        ((1, 2, 2, 3), (2, 3, 2, 4)),
    ],
)
def test_static_qr_record_declines_incompatible_shapes(
    left_shape: tuple[int, ...], right_shape: tuple[int, ...]
) -> None:
    assert (
        _static_exact_qr_record(left_shape, right_shape, max_bond=None, cutoff=0.0)
        is None
    )


@pytest.mark.parametrize(
    ("max_bond", "cutoff"),
    [(3, 0.0), (None, 1e-9), (8, 1e-9)],
)
def test_static_qr_record_declines_truncating_or_mismatched_bounds(
    max_bond: int | None, cutoff: float
) -> None:
    assert (
        _static_exact_qr_record(
            (1, 2, 2, 3), (1, 3, 2, 4), max_bond=max_bond, cutoff=cutoff
        )
        is None
    )


def test_static_qr_record_accepts_a_bond_limit_at_the_full_rank() -> None:
    record = _static_exact_qr_record((1, 2, 2, 3), (1, 3, 2, 4), max_bond=4, cutoff=0.0)

    assert record is not None
    assert record["split_info"]["rank"] == 4

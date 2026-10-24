"""Runtime binds native source versions independently of Core codec evolution."""

from __future__ import annotations

from unittest.mock import patch

import pytest

import flagquantum as fq
from flagquantum.core.ir.program.physical_codec import native_program_to_dict
from flagquantum.core.ising import IsingBraidProgram
from flagquantum.errors import CapabilityError
from flagquantum.runtime.execution_plan_contract import (
    ExecutionPlanContractError,
    canonical_hash,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("version", ["1.0", "1.2", "2.0", 1.1, None])
def test_planning_refuses_unadmitted_source_before_outputs(version: object) -> None:
    source = IsingBraidProgram().exchange(1)
    payload = native_program_to_dict(source.to_ir())
    payload["version"] = version
    with (
        patch(
            "flagquantum.runtime._ising_plan.native_program_to_dict",
            return_value=payload,
        ),
        patch("flagquantum.runtime._ising_plan._outputs") as outputs,
        pytest.raises(CapabilityError, match="source 1.1"),
    ):
        fq.plan(source)
    outputs.assert_not_called()


@pytest.mark.parametrize("version", ["1.0", "1.2", "2.0", 1.1, None])
def test_rehashed_plan_cannot_borrow_a_new_core_source_version(version: object) -> None:
    source = IsingBraidProgram().exchange(1)
    record = source.to_ir()
    payload = fq.plan(source).to_dict()
    native = payload["program"]
    assert isinstance(native, dict)
    native["version"] = version
    fingerprints = payload["fingerprints"]
    assert isinstance(fingerprints, dict)
    fingerprints["program"] = canonical_hash(native)
    payload["identity"] = canonical_hash(
        {key: value for key, value in payload.items() if key != "identity"}
    )
    # Simulate Core learning another source version. Runtime must keep its own
    # strict plan/source binding rather than inherit that broader acceptance.
    with (
        patch(
            "flagquantum.runtime._ising_plan.native_program_from_dict",
            return_value=record,
        ) as decode,
        patch(
            "flagquantum.core.ir.program.physical_codec.native_program_to_dict",
            return_value=native,
        ),
        pytest.raises(ExecutionPlanContractError, match="source 1.1") as error,
    ):
        fq.ExecutionPlan.from_dict(payload)
    assert error.value.reason_code == "program_fingerprint_mismatch"
    decode.assert_not_called()

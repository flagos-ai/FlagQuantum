"""Fresh plan programs retain canonical transport, mutation and refusal semantics."""

import json
from dataclasses import replace
from typing import Any

import pytest
import torch

import flagquantum as fq
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ising import IsingBraidProgram
from flagquantum.runtime.execution_plan import ExecutionPlan
from flagquantum.runtime.execution_plan_contract import (
    ExecutionPlanContractError,
    plan_execution_program,
    plan_program,
    plan_to_dict,
)

pytestmark = pytest.mark.unit


def _plan() -> ExecutionPlan:
    circuit = fq.Circuit(2, dtype=torch.complex128).h(0).cx(0, 1)
    source = replace(
        circuit.to_ir(),
        metadata={
            "z": {"z": 2, "a": [3, 1]},
            "tensor": torch.tensor([0.25, 0.75], dtype=torch.float64),
            "a": 1j,
        },
    )
    return fq.plan(source, options=fq.ExecutionOptions(mode="statevector"))


@pytest.mark.parametrize("canonical", [True, False])
def test_program_matches_canonical_transport_and_is_fresh(canonical: bool) -> None:
    plan = _plan()
    if not canonical:
        payload = plan.to_dict()
        program = payload["program"]
        assert isinstance(program, dict)
        metadata = program["metadata"]
        assert isinstance(metadata, dict)
        program["metadata"] = dict(reversed(tuple(metadata.items())))
        plan = replace(plan, _contract_payload_json=json.dumps(payload, indent=2))
    before = plan.to_json()
    expected = CircuitIR.from_dict(plan_to_dict(plan)["program"])
    actual = plan_program(plan)
    assert actual.to_json() == expected.to_json()
    assert list(actual.metadata) == list(expected.metadata)
    actual.metadata["z"]["a"] = (999,)
    actual.metadata["tensor"].zero_()
    again = plan_program(plan)
    assert again.to_json() == expected.to_json()
    assert plan.to_json() == before
    result = fq.run(plan)
    expected_state = torch.tensor([[2**-0.5, 0, 0, 2**-0.5]], dtype=result.state.dtype)
    torch.testing.assert_close(result.state, expected_state)


@pytest.mark.parametrize(
    "damage", ["program", "fingerprint", "decision", "extension", "schema", "compiler"]
)
def test_program_reader_preserves_complete_validation_diagnostics(damage: str) -> None:
    plan = _plan()
    payload: dict[str, Any] = plan.to_dict()
    if damage == "program":
        payload["program"]["instructions"][0]["opcode"] = "x"
    elif damage == "fingerprint":
        payload["fingerprints"]["program"] = "0" * 64
    elif damage == "decision":
        payload["decision"]["allow_approximate"] = True
    elif damage == "extension":
        payload["extensions"] = [{"namespace": "unrecognized"}]
    elif damage == "schema":
        payload["version"] = "unrecognized"
    else:
        payload["fingerprints"]["compiler"] = "0" * 64
    changed = replace(plan, _contract_payload_json=json.dumps(payload))
    with pytest.raises(ExecutionPlanContractError) as reference:
        plan_to_dict(changed)
    for reader in (plan_program, plan_execution_program):
        with pytest.raises(ExecutionPlanContractError) as actual:
            reader(changed)
        assert actual.value.reason_code == reference.value.reason_code
        assert str(actual.value) == str(reference.value)


def test_native_plan_validation_remains_separate_from_circuit_reading() -> None:
    plan = fq.plan(IsingBraidProgram().exchange(1))
    before = plan.to_json()
    payload = plan_to_dict(plan)
    assert payload["version"] == "1.1"
    with pytest.raises(ValueError) as reference:
        CircuitIR.from_dict(payload["program"])
    with pytest.raises(type(reference.value)) as actual:
        plan_program(plan)
    assert str(actual.value) == str(reference.value)
    assert plan.to_json() == before

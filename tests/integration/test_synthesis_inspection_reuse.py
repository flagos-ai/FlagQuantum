"""Immediate resource reuse never replaces checks at later public accesses."""

from dataclasses import replace

import pytest
import torch

import flagquantum as fq
from benchmarks.synthesis.native_workloads import NativeWorkload
from flagquantum.core.ir import CircuitIR
from flagquantum.services import _synthesis_execution as execution

pytestmark = pytest.mark.integration


def test_native_construction_inspects_each_required_boundary_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspect = execution._program_identity
    calls = 0

    def observed(plan: fq.ExecutionPlan) -> tuple[CircuitIR, str, str]:
        nonlocal calls
        calls += 1
        return inspect(plan)

    monkeypatch.setattr(execution, "_program_identity", observed)
    workload = NativeWorkload("qaoa")
    program = workload.build()
    # Initial admission and final product validation remain separate checks;
    # retrieving the just-computed costs does not trigger a third inspection.
    assert calls == 2
    result = fq.run(program)
    torch.testing.assert_close(
        result.statevector().flatten(), workload.expected(program), atol=1e-12, rtol=0
    )
    assert calls > 2


def test_inspected_resource_value_does_not_cache_later_access() -> None:
    plan = fq.plan(fq.Circuit(1, dtype=torch.complex128).h(0))
    checked, resources = execution._inspect_execution_with_resources(plan)
    assert resources == checked.resources
    object.__setattr__(plan, "state_bytes", plan.state_bytes + 16)
    with pytest.raises(execution.ExecutionContractError, match="allocation"):
        _ = checked.resources


def test_same_object_and_detached_circuit_mutations_are_rejected() -> None:
    program = NativeWorkload("arithmetic").build()
    detached = replace(program, circuit=CircuitIR.from_json(program.circuit.to_json()))
    assert detached.execution is program.execution
    detached_metadata = detached.circuit.metadata
    assert isinstance(detached_metadata, dict)
    detached_metadata["changed"] = True
    with pytest.raises(ValueError, match="circuit changed"):
        _ = detached.execution
    # A repeated access must also detect changes to the shared submitted object.
    shared_metadata = program.circuit.metadata
    assert isinstance(shared_metadata, dict)
    shared_metadata["changed"] = True
    with pytest.raises(ValueError, match="circuit changed"):
        _ = program.execution

"""Module compilation and result typing retain live runtime objects."""

from collections.abc import Mapping
from inspect import Parameter as SignatureParameter
from inspect import signature

import pytest
import torch

import flagquantum as fq
from flagquantum.core.ir import Instruction
from flagquantum.runtime.builder_compilation import (
    BuilderBindings,
    CompiledInstruction,
    SlotParameterMapping,
)
from flagquantum.runtime.executors.mps.production import (
    MPSAcceptanceGates,
    MPSProductionPlan,
)

pytestmark = pytest.mark.unit


def test_compiled_instruction_keeps_unbound_slots_and_live_rebinding() -> None:
    bindings = BuilderBindings()
    slots = SlotParameterMapping({}, {"theta": 0}, bindings)
    instruction: Instruction = CompiledInstruction(
        "ry", (0,), slots, None, {}, (0,), (None,)
    )
    assert isinstance(instruction, Instruction)
    assert instruction.params is slots
    assert all(
        parameter.default is SignatureParameter.empty
        for parameter in signature(CompiledInstruction).parameters.values()
    )
    with pytest.raises(RuntimeError, match="not bound"):
        instruction.params["theta"]
    for angle in (0.2, 0.4):
        value = torch.tensor(angle, requires_grad=True)
        bindings.bind((value,))
        bound_value = instruction.params["theta"]
        assert isinstance(bound_value, torch.Tensor)
        assert bound_value is value
        torch.autograd.backward(torch.sin(bound_value))
        torch.testing.assert_close(value.grad, value.detach().cos())
        bindings.clear()


def test_parameter_groups_keep_registration_and_live_identity() -> None:
    def build(groups: Mapping[str, torch.Tensor]) -> fq.Circuit:
        return fq.Circuit(1).gate("ry", 0, theta=groups["angle"])

    module = fq.Module(build, parameters={"angle": ()}, init={"angle": 0.2})
    groups = module.parameter_groups
    assert isinstance(groups, torch.nn.ParameterDict)
    assert groups is module.named_parameter_groups
    parameter = groups["angle"]
    assert dict(module.named_parameters())["named_parameter_groups.angle"] is parameter
    module().sum().backward()
    assert parameter.grad is not None
    replacement = torch.nn.Parameter(torch.tensor(0.4))
    groups["angle"] = replacement
    assert module.parameter_groups["angle"] is replacement
    module().sum().backward()
    assert replacement.grad is not None
    torch.testing.assert_close(
        module.state_dict()["named_parameter_groups.angle"], replacement
    )


def test_local_mps_result_retains_plan_through_tensor_operations() -> None:
    def build(parameters: torch.Tensor) -> fq.Circuit:
        return fq.Circuit(2).gate("ry", 0, theta=parameters[0]).gate("cx", (0, 1))

    module = fq.Module(build, 1, init=torch.tensor([0.2]))
    result = module.execute_production_mps(
        estimated_workload_bytes=100,
        single_gpu_capacity_bytes=1000,
        gates=MPSAcceptanceGates(False, False, False),
    )
    assert isinstance(result.plan, MPSProductionPlan)
    assert result.plan.world_size == 1
    assert not result.plan.production_promotion_allowed
    assert result.to(dtype=torch.float64).plan is result.plan
    detached = result.detach()
    assert detached.plan is result.plan
    assert detached.runtime == result.runtime
    assert result.value is not None
    assert detached.value is not None
    assert not detached.value.requires_grad
    result.value.sum().backward()
    assert next(module.parameters()).grad is not None

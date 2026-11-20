"""Behavior and compatibility of the approved automatic routing default."""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

import flagquantum as fq
from flagquantum.compiler import CouplingMap, LinkErrorRates, compile, pipeline
from flagquantum.compiler.lowering import pipeline as lowering_pipeline
from flagquantum.compiler.topology import routing_strategy
from flagquantum.compiler.topology.directed_topology import DirectedCouplingMap
from flagquantum.compiler.topology.routing import (
    LINK_ERROR_ROUTING_STRATEGIES,
    route_to_topology,
)
from flagquantum.core.ir import CircuitIR, Instruction, MeasurementNode
from flagquantum.deployment.routing_evidence import validate_deployment_routing_plan

pytestmark = pytest.mark.unit


def _program() -> CircuitIR:
    return fq.Circuit(4).h(0).cx(0, 3).cx(0, 2).ry(3, theta=0.21).cx(3, 1).to_ir()


def test_default_matches_approved_authorization() -> None:
    authorization = json.loads(
        (
            Path(__file__).parents[2]
            / "contracts/compiler-routing-default-v1-candidate.json"
        ).read_text()
    )
    assert inspect.signature(compile).parameters["routing_strategy"].default == (
        authorization["change"]["approved_default"]
    )
    assert inspect.signature(route_to_topology).parameters["strategy"].default == (
        authorization["change"]["previous_default"]
    )


@pytest.mark.parametrize(
    "device_factory",
    [
        lambda: CouplingMap.line(4),
        lambda: DirectedCouplingMap(4, ((0, 1), (1, 2), (2, 3))),
        lambda: LinkErrorRates(
            CouplingMap.line(4), {(0, 1): 0.01, (1, 2): 0.02, (2, 3): 0.001}
        ),
    ],
)
def test_default_is_explicit_auto_with_deployable_layout(
    device_factory: Callable[[], CouplingMap | DirectedCouplingMap | LinkErrorRates],
) -> None:
    source = _program()
    device = device_factory()
    result = compile(source, coupling_map=device)
    # Cold devices make cache telemetry part of the comparison rather than
    # comparing the first call's misses with the second call's hits.
    explicit = compile(source, coupling_map=device_factory(), routing_strategy="auto")
    assert result.content_hash == explicit.content_hash
    assert result.metadata["routing"]["mapping_restored"] is True
    plan = result.metadata["routing"]
    topology = device.device if isinstance(device, LinkErrorRates) else device
    assert validate_deployment_routing_plan(plan, n_qubits=4, coupling_map=topology)
    if isinstance(device, LinkErrorRates):
        selection = result.metadata["routing_strategy_selection"]
        assert set(selection["candidates"]) == set(LINK_ERROR_ROUTING_STRATEGIES)
    torch.testing.assert_close(
        fq.Circuit.from_ir(result).state(), fq.Circuit.from_ir(source).state()
    )


def test_no_topology_does_not_search(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse_search(*args: object, **kwargs: object) -> None:
        raise AssertionError("a topology-free compilation must not search routes")

    monkeypatch.setattr(lowering_pipeline, "_route_automatically", refuse_search)
    result = compile(_program())
    assert "routing_strategy_selection" not in result.metadata
    assert "routing" not in result.metadata


def test_auto_reuses_search_without_reusing_another_compilations_plan() -> None:
    programs_and_devices = (
        (_program(), CouplingMap.line(4)),
        (fq.Circuit(4).h(3).cx(3, 0).cx(2, 0).to_ir(), CouplingMap.line(4)),
        (_program(), CouplingMap(4, ((0, 1), (0, 2), (0, 3)))),
    )
    for index, (source, device) in enumerate(programs_and_devices):
        with patch.object(
            routing_strategy,
            "plan_sabre_swaps",
            wraps=routing_strategy.plan_sabre_swaps,
        ) as search:
            result = compile(source, coupling_map=device, optimize=False)
        # Every uncalibrated auto call prices SABRE exactly once, including
        # calls with changed programs or devices. A winning plan is reused
        # only within that call.
        assert search.call_count == 1
        selected = result.metadata["routing_strategy_selection"]["selected_strategy"]
        if index == 0:
            assert selected == "sabre"
        explicit = compile(
            source, coupling_map=device, routing_strategy=selected, optimize=False
        )
        assert result.instructions == explicit.instructions
        plan = result.metadata["routing"]
        assert plan["mapping_restored"] is True
        assert validate_deployment_routing_plan(plan, n_qubits=4, coupling_map=device)
        torch.testing.assert_close(
            fq.Circuit.from_ir(result).state(), fq.Circuit.from_ir(source).state()
        )


def test_calibrated_auto_does_not_search_uncalibrated_sabre() -> None:
    device = LinkErrorRates(
        CouplingMap.line(4), {(0, 1): 0.01, (1, 2): 0.02, (2, 3): 0.001}
    )
    with patch.object(
        routing_strategy,
        "plan_sabre_swaps",
        side_effect=AssertionError("uncalibrated search"),
    ):
        result = compile(_program(), coupling_map=device)
    assert set(result.metadata["routing_strategy_selection"]["candidates"]) == set(
        LINK_ERROR_ROUTING_STRATEGIES
    )


@pytest.mark.parametrize("strategy", ["auto", "restore_after_each_gate"])
def test_default_keeps_facade_router_replacement(strategy: str) -> None:
    source = _program()
    device = CouplingMap.line(4)
    with patch.object(
        pipeline, "route_to_topology", wraps=route_to_topology
    ) as replacement:
        result = compile(source, coupling_map=device, routing_strategy=strategy)
    assert replacement.call_count == 1
    assert replacement.call_args.args[1] is device
    selected = (
        result.metadata["routing_strategy_selection"]["selected_strategy"]
        if strategy == "auto"
        else strategy
    )
    assert replacement.call_args.kwargs["strategy"] == selected
    torch.testing.assert_close(
        fq.Circuit.from_ir(result).state(), fq.Circuit.from_ir(source).state()
    )


def test_explicit_previous_strategy_keeps_its_plan() -> None:
    source = _program()
    device = CouplingMap.line(4)
    result = compile(
        source,
        coupling_map=device,
        routing_strategy="restore_after_each_gate",
        optimize=False,
    )
    reference = pipeline.record_post_routing_optimization(
        route_to_topology(
            source, CouplingMap.line(4), strategy="restore_after_each_gate"
        )
    )
    assert result.instructions == reference.instructions
    assert result.metadata["routing"] == reference.metadata["routing"]
    assert "routing_strategy_selection" not in result.metadata


def test_disconnected_topology_fails_without_substitution() -> None:
    device = CouplingMap(4, ((0, 1), (2, 3)))
    with pytest.raises(ValueError):
        compile(_program(), coupling_map=device)


def test_default_routing_preserves_parameter_gradients() -> None:
    def circuit(theta: torch.Tensor) -> fq.Circuit:
        return (
            fq.Circuit(4).h(0).ry(3, theta=theta).cx(0, 3).rx(1, theta=theta).cx(3, 1)
        )

    theta = torch.tensor(0.31, requires_grad=True)
    routed_theta = theta.detach().clone().requires_grad_(True)
    expected = circuit(theta).state()
    result = compile(circuit(routed_theta), coupling_map=CouplingMap.line(4))
    actual = fq.Circuit.from_ir(result).state()
    weights = torch.arange(expected.numel(), dtype=expected.real.dtype).reshape(
        expected.shape
    )
    (expected.real * weights).sum().backward()
    (actual.real * weights).sum().backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(routed_theta.grad, theta.grad)


def test_default_routing_preserves_directives_and_measurement_wires() -> None:
    boundary = Instruction("barrier", (0, 3), metadata={"is_dynamic": True})
    source = CircuitIR(
        4,
        (Instruction("cx", (0, 3)), boundary, Instruction("cx", (3, 1))),
        measurements=(MeasurementNode("sample", (0, 3), shots=100),),
    )
    result = compile(source, coupling_map=CouplingMap.line(4))
    assert result.measurements == source.measurements
    assert sum(item.name == "barrier" for item in result.instructions) == 1
    assert result.metadata["routing"]["mapping_restored"] is True


def test_symbolic_parameters_remain_bindable_after_default_routing() -> None:
    theta = fq.Parameter("theta")
    source = fq.Circuit(4).h(0).rx(3, theta=theta).cx(0, 3).ry(1, theta=2 * theta)
    result = fq.Circuit.from_ir(compile(source, coupling_map=CouplingMap.line(4)))
    for angle in (0.0, 0.31, -0.7):
        torch.testing.assert_close(
            result.bind_parameters({"theta": angle}).state(),
            source.bind_parameters({"theta": angle}).state(),
        )

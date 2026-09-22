"""Contract tests for the contraction surface shared by tensor-network plans.

``TensorNetworkContractionPlan`` and ``TensorNetworkExpectationPlan`` contract
different networks - one returns state amplitudes, the other a direct
bra-operator-ket expectation - but they select and cost contraction paths over
the same ``nodes`` and ``output_labels``. Those operations have one definition
so they cannot drift apart. These tests protect that guarantee behaviorally: for
the same network, both plans must produce the same path, cost, and slicing
results.
"""

from __future__ import annotations

import dataclasses
import inspect
from typing import Any

import pytest
import torch

from flagquantum.simulation.tensor_network.models import (
    TensorNetworkContractionPlan,
    TensorNetworkExpectationPlan,
    TensorNetworkNode,
)

pytestmark = pytest.mark.unit

# Operations whose result depends only on ``nodes`` and ``output_labels``.
SHARED_SURFACE = (
    "greedy_path",
    "memory_greedy_path",
    "quality_greedy_path",
    "quality_multistart_path",
    "quality_reconfigured_path",
    "beam_path",
    "optimal_path",
    "contraction_profile",
    "contraction_cost",
    "slicing_plan",
    "cotengra_slicing_plan",
    "contract_slicing_plan",
    "reslice_external_plan",
)

# Deterministic argument sets for the shared operations that take no slicing plan.
PATH_CALLS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("greedy_path", {}),
    ("memory_greedy_path", {}),
    ("quality_greedy_path", {}),
    ("quality_multistart_path", {}),
    ("quality_reconfigured_path", {}),
    ("beam_path", {}),
    ("optimal_path", {}),
    ("contraction_profile", {"strategy": "greedy"}),
    ("contraction_profile", {"strategy": "memory_greedy"}),
    ("contraction_profile", {"strategy": "beam"}),
    ("contraction_cost", {"strategy": "greedy"}),
    ("contraction_cost", {"strategy": "memory_greedy"}),
    ("slicing_plan", {}),
)


def _nodes() -> tuple[TensorNetworkNode, ...]:
    return (
        TensorNetworkNode(torch.ones(2), (0,), name="left"),
        TensorNetworkNode(torch.ones(2, 2), (0, 1), name="bridge"),
        TensorNetworkNode(torch.ones(2), (1,), name="right"),
    )


def _plans() -> tuple[TensorNetworkContractionPlan, TensorNetworkExpectationPlan]:
    """Build both plan kinds over one identical network."""

    nodes = _nodes()
    return (
        TensorNetworkContractionPlan(
            n_wires=2, bsz=1, nodes=nodes, output_labels=(), path=()
        ),
        TensorNetworkExpectationPlan(
            n_wires=2,
            bsz=1,
            nodes=nodes,
            output_labels=(),
            observable_wires=(0,),
            path=(),
        ),
    )


def test_shared_surface_has_one_definition_per_operation():
    for name in SHARED_SURFACE:
        contraction = getattr(TensorNetworkContractionPlan, name)
        expectation = getattr(TensorNetworkExpectationPlan, name)

        assert contraction is expectation, (
            f"{name} is defined separately on both plans; shared contraction "
            "operations must keep exactly one definition"
        )


def test_shared_surface_signatures_are_identical():
    for name in SHARED_SURFACE:
        contraction = inspect.signature(getattr(TensorNetworkContractionPlan, name))
        expectation = inspect.signature(getattr(TensorNetworkExpectationPlan, name))

        assert contraction == expectation, f"{name} signature differs between plans"


@pytest.mark.parametrize(("name", "kwargs"), PATH_CALLS, ids=lambda value: str(value))
def test_shared_operations_agree_for_the_same_network(
    name: str, kwargs: dict[str, Any]
):
    contraction, expectation = _plans()

    assert getattr(contraction, name)(**kwargs) == getattr(expectation, name)(**kwargs)


def test_shared_slicing_operations_agree_for_the_same_network():
    contraction, expectation = _plans()

    slicing = contraction.slicing_plan()
    assert slicing == expectation.slicing_plan()

    assert torch.equal(
        contraction.contract_slicing_plan(slicing),
        expectation.contract_slicing_plan(slicing),
    )

    # Reslicing applies to an imported path. A native plan must be rejected the
    # same way by both plans rather than diverging on the precondition.
    for plan in (contraction, expectation):
        with pytest.raises(ValueError, match="external reslicing requires"):
            plan.reslice_external_plan(slicing, target_slices=4)


def test_plans_remain_frozen_records_with_their_declared_fields():
    """The shared base must not add, drop, or reorder construction fields."""

    expected = {
        TensorNetworkContractionPlan: (
            "n_wires",
            "bsz",
            "nodes",
            "output_labels",
            "path",
            "program_cache",
        ),
        TensorNetworkExpectationPlan: (
            "n_wires",
            "bsz",
            "nodes",
            "output_labels",
            "observable_wires",
            "path",
        ),
    }

    for plan, fields in expected.items():
        assert dataclasses.is_dataclass(plan)
        assert plan.__dataclass_params__.frozen is True
        assert tuple(field.name for field in dataclasses.fields(plan)) == fields


def test_plans_reject_attribute_assignment():
    contraction, expectation = _plans()

    for plan in (contraction, expectation):
        with pytest.raises(dataclasses.FrozenInstanceError):
            plan.bsz = 2

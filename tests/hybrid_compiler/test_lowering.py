from __future__ import annotations

import pytest
import torch

from flagquantum.compiler import optimize
from flagquantum.compiler._hybrid import (
    CircuitStructureCache,
    capture_source,
    specialize_and_lower,
    tensor_type,
)
from flagquantum.core.ir import CircuitIR, Instruction, MeasurementNode, ObservableNode
from flagquantum.core.parameters import Parameter

pytestmark = pytest.mark.unit

WEIGHTS = tensor_type("float64", (None, 4))
DATA = tensor_type("float64", (4,))

SOURCE = """
def cost(weights, data):
    qp.AngleEmbedding(data, wires=range(4))
    for row in weights:
        for wire, parameter in enumerate(row):
            if parameter > 0:
                qp.RX(parameter, wires=wire)
            elif parameter < 0:
                qp.RY(parameter, wires=wire)
        for wire in range(4):
            qp.CNOT(wires=[wire, jnp.mod(wire + 1, 4)])
    return qp.expval(qp.PauliZ(0) + qp.PauliZ(3))
"""


def program():
    return capture_source(SOURCE, (WEIGHTS, DATA))


@pytest.mark.parametrize(
    ("weights", "selected"),
    (
        ([0.2, 0.3, 0.4, 0.5], (("rx", 0), ("rx", 1), ("rx", 2), ("rx", 3))),
        (
            [-0.2, -0.3, -0.4, -0.5],
            (("ry", 0), ("ry", 1), ("ry", 2), ("ry", 3)),
        ),
        ([0.0, 0.0, 0.0, 0.0], ()),
        ([0.2, -0.3, 0.0, 0.5], (("rx", 0), ("ry", 1), ("rx", 3))),
    ),
)
def test_each_golden_branch_pattern_matches_hand_built_structure(
    weights: list[float], selected: tuple[tuple[str, int], ...]
) -> None:
    runtime_weights = torch.tensor([weights], dtype=torch.float64)
    data = torch.zeros(4, dtype=torch.float64)
    bound = specialize_and_lower(program(), (runtime_weights, data)).bind()
    expected = (
        *(("rx", (wire,)) for wire in range(4)),
        *((name, (wire,)) for name, wire in selected),
        ("cx", (0, 1)),
        ("cx", (1, 2)),
        ("cx", (2, 3)),
        ("cx", (3, 0)),
    )

    assert tuple((item.name, item.wires) for item in bound.instructions) == expected


def test_lowered_mixed_path_matches_hand_built_circuit_ir() -> None:
    weights = torch.tensor([[0.2, -0.3, 0.0, 0.5]], dtype=torch.float64)
    data = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    lowered = specialize_and_lower(program(), (weights, data))

    bound = lowered.bind()
    expected = CircuitIR(
        4,
        (
            *(Instruction("rx", (wire,), {"theta": data[wire]}) for wire in range(4)),
            Instruction("rx", (0,), {"theta": weights[0, 0]}),
            Instruction("ry", (1,), {"theta": weights[0, 1]}),
            Instruction("rx", (3,), {"theta": weights[0, 3]}),
            Instruction("cx", (0, 1)),
            Instruction("cx", (1, 2)),
            Instruction("cx", (2, 3)),
            Instruction("cx", (3, 0)),
        ),
        observables=(ObservableNode("z", (0,)), ObservableNode("z", (3,))),
        measurements=(
            MeasurementNode(
                "expectation_ps",
                (0,),
                metadata={
                    "fq_output_index": 0,
                    "fq_output_kind": "expectation",
                    "fq_output_name": None,
                    "fq_output_term": 0,
                    "fq_output_terms": 2,
                    "fq_coefficient": 1.0,
                    "x": (),
                    "y": (),
                    "z": (0,),
                },
            ),
            MeasurementNode(
                "expectation_ps",
                (3,),
                metadata={
                    "fq_output_index": 0,
                    "fq_output_kind": "expectation",
                    "fq_output_name": None,
                    "fq_output_term": 1,
                    "fq_output_terms": 2,
                    "fq_coefficient": 1.0,
                    "x": (),
                    "y": (),
                    "z": (3,),
                },
            ),
        ),
    )

    assert tuple((item.name, item.wires) for item in bound.instructions) == tuple(
        (item.name, item.wires) for item in expected.instructions
    )
    for actual, reference in zip(bound.instructions, expected.instructions):
        if "theta" in reference.params:
            assert torch.equal(actual.params["theta"], reference.params["theta"])
    assert bound.observables == expected.observables
    assert bound.measurements == expected.measurements


def test_template_uses_slots_and_binding_preserves_autograd_edges() -> None:
    weights = torch.tensor(
        [[0.2, -0.3, 0.0, 0.5]], dtype=torch.float64, requires_grad=True
    )
    data = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64, requires_grad=True)
    lowered = specialize_and_lower(program(), (weights, data))

    assert all(
        isinstance(instruction.params["theta"], Parameter)
        for instruction in lowered.circuit_template.instructions
        if "theta" in instruction.params
    )
    bound = lowered.bind()
    parameters = [
        instruction.params["theta"]
        for instruction in bound.instructions
        if "theta" in instruction.params
    ]
    torch.stack(parameters).sum().backward()

    assert torch.equal(data.grad, torch.ones_like(data))
    assert torch.equal(
        weights.grad,
        torch.tensor([[1.0, 1.0, 0.0, 1.0]], dtype=torch.float64),
    )


def test_existing_compiler_accepts_template_and_bound_circuit() -> None:
    weights = torch.tensor([[0.2, -0.3, 0.0, 0.5]], dtype=torch.float64)
    data = torch.zeros(4, dtype=torch.float64)
    lowered = specialize_and_lower(program(), (weights, data))

    assert isinstance(optimize(lowered.circuit_template), CircuitIR)
    assert isinstance(optimize(lowered.bind()), CircuitIR)


def test_structure_cache_reports_hit_miss_and_eviction() -> None:
    cache = CircuitStructureCache(max_entries=1)
    data = torch.zeros(4, dtype=torch.float64)
    positive = torch.ones((1, 4), dtype=torch.float64)
    positive_changed = torch.full((1, 4), 2.0, dtype=torch.float64)
    negative = -positive

    first = specialize_and_lower(program(), (positive, data), cache=cache)
    second = specialize_and_lower(program(), (positive_changed, data), cache=cache)
    third = specialize_and_lower(program(), (negative, data), cache=cache)

    assert first.cache_event == "miss"
    assert second.cache_event == "hit"
    assert second.circuit_template is first.circuit_template
    assert third.cache_event == "miss_evicted"
    assert third.evicted_cache_key == first.cache_key
    assert len(cache) == 1


def test_parameter_values_are_excluded_from_template_identity() -> None:
    data = torch.zeros(4, dtype=torch.float64)
    first = specialize_and_lower(
        program(),
        (torch.tensor([[0.1, -0.2, 0.0, 0.3]], dtype=torch.float64), data),
    )
    second = specialize_and_lower(
        program(),
        (torch.tensor([[1.1, -1.2, 0.0, 1.3]], dtype=torch.float64), data),
    )

    assert first.selected_structure_identity == second.selected_structure_identity
    assert first.circuit_template.content_hash == second.circuit_template.content_hash

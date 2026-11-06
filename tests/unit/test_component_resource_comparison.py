"""Keep static IR instruction counts and logical resource assumptions distinct."""

import json

import pytest

from flagquantum.algorithms.primitives.rotation_preparation import RotationPreparation
from flagquantum.algorithms.primitives.rotation_resource_model import (
    rotation_preparation_cost,
    sparse_rotation_preparation_cost,
)
from flagquantum.algorithms.primitives.sparse_rotation_preparation import (
    SparseRotationPreparation,
)
from flagquantum.compiler._component_trace.resource_comparison import (
    compare_component_resources,
    count_emitted_instructions,
)
from flagquantum.core.ir import CircuitIR, Instruction, MeasurementNode, ObservableNode

pytestmark = pytest.mark.unit


def test_count_emitted_conditions_once_and_keep_output_requests_separate() -> None:
    ir = CircuitIR(
        5,
        (
            Instruction(
                "measure", (0,), metadata={"classical_bit": 0, "is_dynamic": True}
            ),
            Instruction(
                "measure", (1,), metadata={"classical_bit": 1, "is_dynamic": True}
            ),
            Instruction("t", (0,)),
            Instruction("tdg", (1,), metadata={"conditions": ((0, 1),)}),
            Instruction(
                "x", (2,), metadata={"condition_clauses": (((0, 0),), ((1, 1),))}
            ),
        ),
        observables=(ObservableNode("z", (3,)),),
        measurements=(MeasurementNode("sample", (4,), shots=80),),
    )
    result = count_emitted_instructions(ir)
    assert result.unconditional == {"measure": 2, "t": 1}
    assert result.conditional == {"tdg": 1, "x": 1}
    assert result.t_count == 2
    assert result.declared_qubits == 5
    assert result.instruction_qubits == 3
    assert result.observable_requests == result.measurement_requests == 1


@pytest.mark.parametrize("sparse", (False, True))
def test_rotation_model_and_emitted_dynamic_resources(sparse: bool) -> None:
    if sparse:
        preparation = SparseRotationPreparation((2, 1), (1 + 1j, 1 - 1j), 3, 8)
        model = sparse_rotation_preparation_cost(preparation, 4)
        width = 4
        work, slots = preparation.requirements(width)
        target = tuple(range(width))
        gradient = tuple(range(width, width + 8))
        workspace = tuple(range(width + 8, width + 8 + work))
        emitted = preparation.emit(target, gradient, workspace, tuple(range(slots)))
        expected_model = (92, 11, 22)
        expected_ir = (100, 25, 28)
    else:
        dense = RotationPreparation((0.5, 0.5j, -0.5, -0.5j), 8)
        model = rotation_preparation_cost(dense)
        width = 2
        work, slots = dense.requirements(0)
        target = tuple(range(width))
        gradient = tuple(range(width, width + 8))
        workspace = tuple(range(width + 8, width + 8 + work))
        emitted = dense.emit((), target, gradient, workspace, tuple(range(slots)))
        expected_model = (96, 6, 21)
        expected_ir = (108, 27, 26)

    report = compare_component_resources(model, CircuitIR(width + 8 + work, emitted))
    assert (
        report.model.gates.total_t_count(),
        report.model.gates.measurement,
        report.model.peak_logical_qubits,
    ) == expected_model
    assert (
        report.emitted.t_count,
        report.emitted.unconditional["measure"],
        report.emitted.declared_qubits,
    ) == expected_ir
    assert report.emitted.conditional == {"cz": expected_ir[1]}
    assert sum(report.emitted.unconditional.values()) + sum(
        report.emitted.conditional.values()
    ) == len(emitted)
    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["component_model"]["default_t_equivalents"] == expected_model[0]
    assert payload["emitted_ir"]["t_count"] == expected_ir[0]

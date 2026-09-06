from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from flagquantum._compiler.bindings import (
    RuntimeBindingRef,
    SymbolicExpression,
    SymbolicParameter,
)
from flagquantum._compiler.import_models import ImportStatus
from flagquantum._compiler.importers.circuit_ir import import_circuit_ir
from flagquantum._compiler.ir.operations import FrozenAttributes
from flagquantum.core.ir import CircuitIR, Instruction, MeasurementNode, ObservableNode
from flagquantum.core.operator_schema import OPERATOR_SCHEMAS
from flagquantum.core.parameters import Parameter
from flagquantum.noise import bit_flip_channel

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "tests/fixtures/internal_ir/circuit_ir_v1/manifest.json"


def _manifest() -> dict[str, object]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _fixture(fixture_id: str) -> CircuitIR:
    fixture = next(item for item in _manifest()["fixtures"] if item["id"] == fixture_id)
    return CircuitIR.from_dict(fixture["payload"])


@pytest.mark.parametrize(
    "fixture",
    _manifest()["fixtures"],
    ids=lambda fixture: fixture["id"],
)
def test_phase0_manifest_import_status_matches_declared_profile(fixture) -> None:
    source = CircuitIR.from_dict(fixture["payload"])

    assert import_circuit_ir(source).status.value == fixture["expected_import"]


@pytest.mark.parametrize(
    "fixture_id",
    [
        "static.single_qubit_registered.v1",
        "static.two_qubit_registered.v1",
        "static.three_qubit_registered.v1",
        "static.channels_registered.v1",
        "static.parameter_and_expression.v1",
        "static.two_parameter_entangled_gradient.v1",
        "static.requests_and_ordering.v1",
        "static.custom_unitary.v1",
        "static.expectation.complex64.v1",
        "static.expectation.complex128.v1",
        "static.routing_and_emitters.v1",
    ],
)
def test_phase0_static_fixture_imports_exactly(fixture_id: str) -> None:
    source = _fixture(fixture_id)
    before = source.to_json()

    result = import_circuit_ir(source)

    assert result.status is ImportStatus.SUPPORTED_EXACT
    assert result.ok is True
    assert result.diagnostics == ()
    assert result.imported.source.circuit_ir_content_hash == source.content_hash
    assert source.to_json() == before


def test_all_35_canonical_opcodes_receive_internal_operations() -> None:
    fixtures = (
        _fixture("static.single_qubit_registered.v1"),
        _fixture("static.two_qubit_registered.v1"),
        _fixture("static.three_qubit_registered.v1"),
        _fixture("static.channels_registered.v1"),
    )
    imported_names = {
        operation.name
        for source in fixtures
        for operation in import_circuit_ir(source)
        .imported.module.body.blocks[0]
        .operations
    }

    assert imported_names == {f"quantum.{name}" for name in OPERATOR_SCHEMAS}


def test_explicit_kraus_data_is_part_of_internal_program_identity() -> None:
    def source(probability: float) -> CircuitIR:
        channel = bit_flip_channel(probability)
        return CircuitIR(
            1,
            (
                Instruction(
                    channel.name,
                    (0,),
                    matrix=channel.kraus,
                    metadata={"is_channel": True},
                ),
            ),
        )

    first = import_circuit_ir(source(0.1))
    second = import_circuit_ir(source(0.2))

    assert first.ok and second.ok
    first_operation = first.imported.module.body.blocks[0].operations[0]
    assert first_operation.attributes["kraus"]["kind"] == "kraus"
    assert first.imported.internal_program_identity != (
        second.imported.internal_program_identity
    )


def test_import_builds_deterministic_linear_wire_value_chains() -> None:
    source = CircuitIR(
        2,
        (
            Instruction("h", (0,)),
            Instruction("cx", (0, 1)),
            Instruction("x", (1,)),
        ),
    )

    first = import_circuit_ir(source).imported
    second = import_circuit_ir(source).imported
    operations = first.module.body.blocks[0].operations

    assert [str(value.id) for value in first.module.body.blocks[0].arguments] == [
        "%entry.0",
        "%entry.1",
    ]
    assert operations[0].operands[0].id.index == 0
    assert operations[1].operands[0] == operations[0].results[0]
    assert operations[2].operands[0] == operations[1].results[1]
    assert first.module == second.module
    assert first.internal_program_identity == second.internal_program_identity


def test_request_is_split_from_program_identity_and_preserves_order() -> None:
    instructions = (Instruction("h", (0,)),)
    first_source = CircuitIR(
        2,
        instructions,
        observables=(
            ObservableNode("z", (1,), metadata={"name": "z1"}),
            ObservableNode("zz", (0, 1), coefficient=-0.5),
        ),
        measurements=(
            MeasurementNode(
                "sample", (1, 0), shots=32, metadata={"seed": 17, "name": "s"}
            ),
        ),
    )
    second_source = CircuitIR(
        2,
        instructions,
        measurements=(MeasurementNode("sample", (0, 1), shots=64),),
    )

    first = import_circuit_ir(first_source).imported
    second = import_circuit_ir(second_source).imported

    assert first.internal_program_identity == second.internal_program_identity
    assert first.source.circuit_ir_content_hash != second.source.circuit_ir_content_hash
    assert [item.name for item in first.request.observables] == ["z", "zz"]
    assert first.request.measurements[0].wires == (1, 0)
    assert first.request.measurements[0].shots == 32
    assert first.request.measurements[0].seed == 17


def test_request_fixture_preserves_constraints_and_request_fields() -> None:
    imported = import_circuit_ir(_fixture("static.requests_and_ordering.v1")).imported

    assert imported.constraints.dtype == "complex128"
    assert imported.constraints.shape == (1, 4)
    assert imported.constraints.batch_size == 1
    assert imported.request.measurements[0].canonical() == {
        "kind": "sample",
        "wires": [1, 0],
        "shots": 32,
        "seed": 17,
        "name": "ordered_samples",
        "format": None,
        "postselection": None,
        "max_marginal_wires": None,
        "max_postselection_draw_multiplier": None,
    }


def test_dtype_constraint_changes_internal_program_identity() -> None:
    instructions = (Instruction("h", (0,)),)
    single = import_circuit_ir(CircuitIR(1, instructions, dtype="complex64")).imported
    double = import_circuit_ir(CircuitIR(1, instructions, dtype="complex128")).imported

    assert single.module.program_identity == double.module.program_identity
    assert single.internal_program_identity != double.internal_program_identity


def test_symbolic_parameter_and_expression_structure_is_preserved() -> None:
    theta = Parameter("theta")
    source = CircuitIR(
        1,
        (
            Instruction("rx", (0,), {"theta": theta}),
            Instruction("rz", (0,), {"theta": theta * 0.5}),
        ),
    )

    imported = import_circuit_ir(source).imported
    first, second = imported.module.body.blocks[0].operations

    assert first.attributes["theta"] == SymbolicParameter("theta")
    assert second.attributes["theta"] == SymbolicExpression(
        "mul", (SymbolicParameter("theta"), 0.5)
    )


def test_trainable_tensor_uses_external_binding_without_detach_or_value_identity() -> (
    None
):
    first_tensor = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    second_tensor = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    first = import_circuit_ir(
        CircuitIR(1, (Instruction("rx", (0,), {"theta": first_tensor}),))
    ).imported
    second = import_circuit_ir(
        CircuitIR(1, (Instruction("rx", (0,), {"theta": second_tensor}),))
    ).imported
    reference = first.module.body.blocks[0].operations[0].attributes["theta"]

    assert isinstance(reference, RuntimeBindingRef)
    assert first.bindings[reference.slot] is first_tensor
    assert first.internal_program_identity == second.internal_program_identity
    assert first_tensor.requires_grad is True


def test_static_scalar_tensor_preserves_dtype_shape_and_changes_identity() -> None:
    first = import_circuit_ir(
        CircuitIR(1, (Instruction("rx", (0,), {"theta": torch.tensor(0.2)}),))
    ).imported
    second = import_circuit_ir(
        CircuitIR(1, (Instruction("rx", (0,), {"theta": torch.tensor(0.7)}),))
    ).imported
    value = first.module.body.blocks[0].operations[0].attributes["theta"]

    assert isinstance(value, FrozenAttributes)
    assert value["dtype"] == "float32"
    assert value["shape"] == ()
    assert first.internal_program_identity != second.internal_program_identity


def test_provenance_changes_source_hash_but_not_program_identity() -> None:
    instructions = (Instruction("h", (0,)),)
    first = import_circuit_ir(
        CircuitIR(1, instructions, metadata={"source": "one"})
    ).imported
    second = import_circuit_ir(
        CircuitIR(1, instructions, metadata={"source": "two"})
    ).imported

    assert first.provenance.values["source"] == "one"
    assert first.internal_program_identity == second.internal_program_identity
    assert first.source.circuit_ir_content_hash != second.source.circuit_ir_content_hash


def test_typed_instruction_semantics_are_preserved_and_affect_identity() -> None:
    plain = import_circuit_ir(CircuitIR(1, (Instruction("h", (0,)),))).imported
    annotated = import_circuit_ir(
        CircuitIR(1, (Instruction("h", (0,), metadata={"diagonal": True}),))
    ).imported

    assert annotated.instruction_semantics[0].values["diagonal"] is True
    assert plain.internal_program_identity != annotated.internal_program_identity


def test_custom_unitary_preserves_name_arity_matrix_shape_and_dtype() -> None:
    imported = import_circuit_ir(_fixture("static.custom_unitary.v1")).imported
    first, second = imported.module.body.blocks[0].operations

    assert first.attributes["symbolic_name"] == "custom_identity"
    assert first.attributes["arity"] == 1
    assert first.attributes["matrix"]["shape"] == (2, 2)
    assert second.attributes["arity"] == 2
    assert second.attributes["matrix"]["shape"] == (4, 4)

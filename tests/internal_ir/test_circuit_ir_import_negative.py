from __future__ import annotations

import pytest
import torch

from flagquantum._compiler.diagnostics import DiagnosticCode
from flagquantum._compiler.import_models import ImportStatus
from flagquantum._compiler.importers.circuit_ir import import_circuit_ir
from flagquantum.core.ir import CircuitIR, Instruction

pytestmark = pytest.mark.unit


def _assert_unsupported(source: CircuitIR, code: DiagnosticCode) -> None:
    result = import_circuit_ir(source)

    assert result.status is ImportStatus.UNSUPPORTED_WITH_DIAGNOSTICS
    assert result.imported is None
    assert code in tuple(item.code for item in result.diagnostics)


def test_non_circuit_input_is_invalid_input() -> None:
    result = import_circuit_ir(object())

    assert result.status is ImportStatus.INVALID_INPUT
    assert result.imported is None
    assert result.diagnostics[0].code is DiagnosticCode.VALUE_TYPE_MISMATCH


def test_dynamic_instruction_is_structurally_rejected() -> None:
    source = CircuitIR(
        1,
        (
            Instruction(
                "measure",
                (0,),
                metadata={"is_dynamic": True, "classical_bit": 0},
            ),
        ),
        metadata={"num_clbits": 1},
    )

    _assert_unsupported(source, DiagnosticCode.UNKNOWN_OPERATION)


def test_valid_but_unfreezable_metadata_is_unsupported_not_invalid() -> None:
    source = CircuitIR(
        1,
        (Instruction("h", (0,), metadata={"mpo": torch.tensor([1.0])}),),
    )

    result = import_circuit_ir(source)

    assert result.status is ImportStatus.UNSUPPORTED_WITH_DIAGNOSTICS
    assert result.diagnostics


def test_registered_opcode_cannot_override_matrix_or_add_parameters() -> None:
    _assert_unsupported(
        CircuitIR(1, (Instruction("h", (0,), matrix=[[1, 0], [0, 1]]),)),
        DiagnosticCode.UNKNOWN_ATTRIBUTE,
    )


def test_invalid_explicit_kraus_channel_fails_closed() -> None:
    source = CircuitIR(
        1,
        (
            Instruction(
                "bit_flip",
                (0,),
                matrix=(torch.eye(2), torch.eye(2)),
                metadata={"is_channel": True},
            ),
        ),
    )

    result = import_circuit_ir(source)

    assert result.status is ImportStatus.UNSUPPORTED_WITH_DIAGNOSTICS
    assert result.imported is None
    assert "trace-preserving" in result.diagnostics[0].message
    _assert_unsupported(
        CircuitIR(1, (Instruction("h", (0,), {"theta": 0.2}),)),
        DiagnosticCode.UNKNOWN_ATTRIBUTE,
    )


@pytest.mark.parametrize(
    "instruction",
    [
        Instruction("h", (0,), metadata={"is_channel": True}),
        Instruction("bit_flip", (0,), metadata={"is_channel": False}),
        Instruction("h", (0,), metadata={"is_channel": "yes"}),
    ],
)
def test_channel_marker_must_be_typed_and_match_schema(instruction) -> None:
    _assert_unsupported(
        CircuitIR(1, (instruction,)),
        DiagnosticCode.ATTRIBUTE_TYPE_MISMATCH,
    )


@pytest.mark.parametrize(
    "matrix",
    [
        [[1, 0, 0], [0, 1, 0]],
        [[1, 1], [0, 1]],
        [[float("nan"), 0], [0, 1]],
    ],
)
def test_invalid_custom_matrix_fails_closed(matrix) -> None:
    source = CircuitIR(1, (Instruction("custom", (0,), matrix=matrix),))

    result = import_circuit_ir(source)

    assert result.status is ImportStatus.UNSUPPORTED_WITH_DIAGNOSTICS
    assert result.diagnostics


def test_trainable_custom_matrix_is_outside_static_profile() -> None:
    matrix = torch.eye(2, requires_grad=True)
    _assert_unsupported(
        CircuitIR(1, (Instruction("custom", (0,), matrix=matrix),)),
        DiagnosticCode.ATTRIBUTE_TYPE_MISMATCH,
    )


def test_non_scalar_gate_tensor_is_rejected() -> None:
    parameter = torch.tensor([0.2, 0.3], requires_grad=True)
    _assert_unsupported(
        CircuitIR(1, (Instruction("rx", (0,), {"theta": parameter}),)),
        DiagnosticCode.ATTRIBUTE_TYPE_MISMATCH,
    )


def test_parameter_failure_carries_instruction_location() -> None:
    parameter = torch.tensor([0.2, 0.3], requires_grad=True)
    result = import_circuit_ir(
        CircuitIR(1, (Instruction("rx", (0,), {"theta": parameter}),))
    )

    assert result.diagnostics[0].location.to_dict() == {
        "source": "CircuitIR.instructions",
        "line": 1,
        "column": 0,
    }


def test_unsupported_dtype_is_rejected() -> None:
    _assert_unsupported(
        CircuitIR(1, (Instruction("h", (0,)),), dtype="float32"),
        DiagnosticCode.ATTRIBUTE_TYPE_MISMATCH,
    )

"""Verify deferred Majorana IR at the public construction and execution boundaries."""

import json

import pytest
import torch

import flagquantum as fq
from flagquantum.core.ir import CircuitIR, IRValidationError


def test_deferred_majorana_roundtrip_and_legacy_execution() -> None:
    local = fq.Circuit(2).gate(
        "selected_majorana_fermion",
        (0, 1),
        control_width=1,
        selection_width=0,
        target_width=1,
        target_gate="Y",
    )
    circuit = fq.Circuit(2).x(1).compose(local, qubits=(1, 0))
    ir = CircuitIR.from_dict(json.loads(json.dumps(circuit.to_ir().to_dict())))
    assert ir.instructions[-1].wires == (1, 0)
    assert ir.instructions[-1].matrix is None
    expected = torch.tensor([[0, 0, 0, 1j]], dtype=torch.complex64)
    torch.testing.assert_close(fq.Circuit.from_ir(ir).state(), expected)
    torch.testing.assert_close(
        fq.run(ir, outputs=fq.probabilities()).probabilities, expected.abs().square()
    )
    assert ir.instructions[-1].matrix is None

    legacy = (
        fq.Circuit(2)
        .x(1)
        .gate(
            "selected_majorana_fermion",
            (1, 0),
            matrix=torch.block_diag(torch.eye(2), torch.tensor([[0, -1j], [1j, 0]])),
            control=(1,),
            selection=(),
            target=(0,),
            target_gate="Y",
        )
    )
    legacy_ir = CircuitIR.from_dict(json.loads(json.dumps(legacy.to_ir().to_dict())))
    torch.testing.assert_close(fq.Circuit.from_ir(legacy_ir).state(), expected)
    assert fq.compile(ir).instructions[-1].matrix is not None


def test_deferred_majorana_rejects_invalid_register_widths() -> None:
    with pytest.raises(IRValidationError, match="invalid selected_majorana_fermion"):
        fq.Circuit(2).gate(
            "selected_majorana_fermion",
            (0, 1),
            control_width=0,
            selection_width=0,
            target_width=2,
            target_gate="X",
        )

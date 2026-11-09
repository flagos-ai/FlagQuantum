"""Check physical control masks and workspace-clean grouped lowering."""

import cmath

import pytest
import torch

import flagquantum as fq
from flagquantum.compiler._construction.register_lowering import (
    schedule_register_program,
)
from flagquantum.compiler._construction.register_profile import verify_register_program
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from tests.register_program_cases import Registers

pytestmark = pytest.mark.unit


def _masked_program(mask: object, *, operation: str, width: int = 5):
    model = Registers()
    word, target = model.declare("word", width), model.declare("target", 1)
    for item in (word, target):
        model.emit("qreg.allocate", item)
    for bit in range(width):
        model.emit("qreg.gate", word, gate="h", index=bit)
    if operation == "qreg.control_x":
        args = {}
    elif operation == "qreg.control_phase":
        assert isinstance(mask, int)
        mask |= 1 << width
        args = {"angle": 0.41}
        model.emit("qreg.gate", target, gate="h", index=0)
    else:
        args = {"control_count": width, "amplitudes": (0.6, 0.8), "bound": 0.0}
    model.emit(
        operation,
        *(word for _ in range(width)),
        target,
        indices=(*range(width), 0),
        control_state=mask,
        **args,
    )
    return model.finish()


@pytest.mark.parametrize("mask", [-1, 32, True, 0.0])
def test_invalid_mask_is_refused_before_lowering(mask: object) -> None:
    with pytest.raises(HybridVerificationError, match="control state must fit"):
        verify_register_program(_masked_program(mask, operation="qreg.control_x"))


def test_mask_limit_prevents_unbounded_decomposition() -> None:
    with pytest.raises(HybridVerificationError, match="control state must fit"):
        verify_register_program(_masked_program(0, width=9, operation="qreg.control_x"))


@pytest.mark.parametrize(
    "operation",
    ["qreg.control_x", "qreg.control_phase", "qreg.inplace_prepare_amplitudes"],
)
def test_negative_controls_preserve_all_branches_and_clean_work(operation: str) -> None:
    mask = 0b10100
    plan = schedule_register_program(
        _masked_program(mask, operation=operation), (), n_qubits=9
    )
    assert plan.peak_workspace <= 1
    actual = (
        fq.run(fq.Circuit.from_ir(plan.circuit, dtype=torch.complex128))
        .statevector()
        .reshape(64, -1)
    )
    expected = torch.zeros(64, dtype=torch.complex128)
    for basis in range(32):
        if operation == "qreg.control_x":
            expected[2 * basis + int(basis == mask)] = 32**-0.5
        elif operation == "qreg.control_phase":
            expected[2 * basis] = 64**-0.5
            expected[2 * basis + 1] = 64**-0.5 * (
                cmath.exp(0.41j) if basis == mask else 1
            )
        else:
            expected[2 * basis] = 32**-0.5 * (0.6 if basis == mask else 1)
            expected[2 * basis + 1] = 32**-0.5 * (0.8 if basis == mask else 0)
    torch.testing.assert_close(actual[:, 0], expected, atol=1e-12, rtol=0)
    assert actual[:, 1:].abs().max().item() < 1e-12

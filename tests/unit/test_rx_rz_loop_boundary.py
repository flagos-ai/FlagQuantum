"""Optional loop kernels must honor the statevector tensor boundary."""

import pytest
import torch

from flagquantum.core.ir import Instruction
from flagquantum.simulation import triton_kernels
from flagquantum.simulation.statevector.operations import (
    _apply_rx_rz_loop,
    _StatevectorRXRZLoopStep,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("wire", [0, 1, 2])
def test_loop_boundary_restores_wire_layout(
    monkeypatch: pytest.MonkeyPatch, wire: int
) -> None:
    state = torch.arange(16, dtype=torch.float64).to(torch.complex128).reshape(2, 8)
    step = _StatevectorRXRZLoopStep(
        wire=wire,
        pairs=(
            (
                Instruction("rx", (wire,), {"theta": 0.2}),
                Instruction("rz", (wire,), {"theta": -0.3}),
            ),
        ),
    )

    def identity_loop(
        paired: torch.Tensor, rx: torch.Tensor, rz: torch.Tensor
    ) -> torch.Tensor:
        assert paired.shape == (2, 4, 2)
        torch.testing.assert_close(rx, torch.full((2, 1), 0.2, dtype=torch.float64))
        torch.testing.assert_close(rz, torch.full((2, 1), -0.3, dtype=torch.float64))
        return paired

    monkeypatch.setitem(triton_kernels.__dict__, "repeated_rx_rz", identity_loop)
    actual = _apply_rx_rz_loop(state, step, 3, None)
    torch.testing.assert_close(actual, state)


@pytest.mark.parametrize("output", [None, (torch.tensor(0.0),)])
def test_loop_boundary_rejects_nontensor_output(
    monkeypatch: pytest.MonkeyPatch, output: object
) -> None:
    step = _StatevectorRXRZLoopStep(
        wire=0,
        pairs=(
            (
                Instruction("rx", (0,), {"theta": 0.2}),
                Instruction("rz", (0,), {"theta": -0.3}),
            ),
        ),
    )
    monkeypatch.setitem(triton_kernels.__dict__, "repeated_rx_rz", lambda *args: output)
    with pytest.raises(TypeError, match="RX/RZ loop kernel must return a tensor"):
        _apply_rx_rz_loop(torch.ones(1, 2, dtype=torch.complex64), step, 1, None)

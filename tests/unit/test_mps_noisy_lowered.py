"""Validation at the lowered noisy MPS execution boundary."""

import pytest
import torch

from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.simulation.mps.noisy import run_local_noisy_mps_trajectory
from flagquantum.simulation.mps.state import MPSState

pytestmark = pytest.mark.unit


def test_channel_without_kraus_matrices_preserves_state() -> None:
    ir = CircuitIR(
        n_wires=1,
        instructions=(Instruction("channel", (0,), metadata={"is_channel": True}),),
    )
    state = MPSState.zero(1)
    before = state.to_statevector().clone()
    with pytest.raises(ValueError, match="requires Kraus matrices"):
        run_local_noisy_mps_trajectory(ir, state, generator=None)
    torch.testing.assert_close(state.to_statevector(), before, atol=0, rtol=0)

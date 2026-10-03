"""Optional loop kernels must honor the statevector tensor boundary."""

import pytest
import torch

from flagquantum.core.ir import Instruction
from flagquantum.kernels import triton as triton_kernels
from flagquantum.simulation.statevector import rx_rz_dispatch
from flagquantum.simulation.statevector.operations import (
    _apply_rx_rz_loop,
    _StatevectorRXRZLoopStep,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def _authorized(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand in for the CUDA-only Triton route with a fake provider.

    The layout contract below is device-independent, but the catalog correctly
    refuses a CPU tensor for a CUDA-only implementation. Authorization is
    replaced here so the boundary can be exercised with a fake kernel; the
    decision itself is asserted in ``test_rx_rz_catalog_dispatch.py`` and by
    ``test_the_fused_loop_fails_closed_on_a_cpu_tensor``.
    """

    monkeypatch.setattr(
        rx_rz_dispatch,
        "_require_rx_rz_sequence_kernel",
        lambda **_: None,
    )


@pytest.mark.parametrize("wire", [0, 1, 2])
def test_loop_boundary_restores_wire_layout(
    monkeypatch: pytest.MonkeyPatch, _authorized: None, wire: int
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
    monkeypatch: pytest.MonkeyPatch, _authorized: None, output: object
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


def test_the_fused_loop_fails_closed_on_a_cpu_tensor() -> None:
    # The fused step is emitted only when the Triton loop is enabled, so a CPU
    # state reaches this path only through a caller that decided elsewhere. The
    # refusal must come from the catalog, not from the provider import.
    step = _StatevectorRXRZLoopStep(
        wire=0,
        pairs=(
            (
                Instruction("rx", (0,), {"theta": 0.2}),
                Instruction("rz", (0,), {"theta": -0.3}),
            ),
        ),
    )

    with pytest.raises(RuntimeError, match="kernel catalog: device"):
        _apply_rx_rz_loop(torch.ones(1, 2, dtype=torch.complex64), step, 1, None)

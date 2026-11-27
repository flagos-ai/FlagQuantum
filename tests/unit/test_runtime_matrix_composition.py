"""Runtime grouping delegates matrix arithmetic to the replaceable Simulation owner."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
import torch

import flagquantum as fq
from flagquantum.runtime.executors.statevector.forward_executor import (
    execute_torch_distributed_statevector,
)
from flagquantum.simulation.statevector import operations

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("direct", ["0", "1"])
@pytest.mark.parametrize(
    "family", ["same_wire", "rotation_tiles", "rzz_rotation", "barrier"]
)
def test_runtime_fusion_uses_replaced_simulation_composition(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
    direct: str,
    family: str,
) -> None:
    monkeypatch.setenv("FQ_STATEVECTOR_LOCAL_BLOCK_FUSION", "1")
    monkeypatch.setenv("FQ_STATEVECTOR_CPU_DIRECT_LOCAL", direct)
    monkeypatch.setenv("FQ_SV_RUNTIME_MODE", "auto")
    # This internal executor initializes its rank shards to |0...0>.
    initial = torch.zeros(1, 32, dtype=dtype)
    initial[:, 0] = 1
    before = initial.clone()
    circuit = fq.Circuit(5, dtype=dtype, inputs=initial)
    if family == "same_wire":
        circuit.rx(1, 0.31).ry(1, -0.27).rz(1, 0.19)
    elif family == "rotation_tiles":
        circuit.rx(0, 0.11).ry(0, -0.21).rz(0, 0.17).rx(2, -0.37).ry(2, 0.23).rz(
            2, -0.13
        )
    elif family == "rzz_rotation":
        circuit.rzz(0, 1, 0.19).rzz(1, 2, -0.13).rx(0, 0.21).ry(0, -0.31).rx(
            2, 0.41
        ).rz(2, -0.17)
    else:
        circuit.rx(0, 0.11).ry(0, -0.21).cx(0, 2).rx(2, 0.23).ry(2, -0.37).rz(1, 0.19)
    expected = circuit.state().clone()
    groups: list[int] = []

    def compose(matrices: Sequence[torch.Tensor]) -> torch.Tensor:
        groups.append(len(matrices))
        combined = matrices[0]
        for matrix in matrices[1:]:
            combined = torch.einsum("...ij,...jk->...ik", matrix, combined)
        return combined

    monkeypatch.setattr(operations, "_compose_gate_matrices", compose)
    result = execute_torch_distributed_statevector(circuit, device="cpu", dtype=dtype)
    assert groups and all(size >= 2 for size in groups)
    tolerance = 2e-6 if dtype == torch.complex64 else 1e-12
    torch.testing.assert_close(
        result.shard_state.amplitudes, expected, atol=tolerance, rtol=tolerance
    )
    assert torch.equal(initial, before)
    assert result.summary()["world_size"] == 1
    assert not result.summary()["scalability_claim_allowed"]

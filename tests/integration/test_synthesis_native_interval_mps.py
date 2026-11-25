"""Wide native programs reuse Runtime's exact independent-interval MPS proof."""

import math
from typing import NoReturn

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.errors import PlanningError
from flagquantum.simulation.mps.state import MPSState
from flagquantum.synth import (
    CX,
    H,
    Output,
    QArray,
    QBit,
    adjoint,
    allocate,
    inplace_prepare_amplitudes,
    qft,
    qfunc,
)

pytestmark = pytest.mark.integration


@qfunc
def intervals(
    weights: list[float],
    t: Output[QArray[QBit]],
    idle: Output[QArray[QBit]],
    spin: Output[QArray[QBit]],
) -> None:
    allocate(4, t)
    allocate(71, idle)
    allocate(8, spin)
    inplace_prepare_amplitudes(weights, 0, t)
    H(spin[6])
    H(spin[7])
    with adjoint():
        qft(t)


@qfunc
def long_interval(q: Output[QArray[QBit]]) -> None:
    allocate(83, q)
    H(q[0])
    CX(q[0], q[82])


@qfunc
def small(q: Output[QBit]) -> None:
    allocate(q)
    H(q)


def forbid_dense(*args: object, **kwargs: object) -> NoReturn:
    pytest.fail("wide MPS acceptance must not materialize the full statevector")


@pytest.mark.parametrize("signed", [False, True])
def test_wide_interval_exact_amplitudes_and_no_truncation(
    signed: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(MPSState, "to_statevector", forbid_dense)
    weights = [
        ((-1) ** i if signed else 1)
        * (i + 1)
        / math.sqrt(sum(j * j for j in range(1, 17)))
        for i in range(16)
    ]
    p = synth.synthesize(
        intervals,
        classical_inputs={"weights": weights},
        max_qubits=83,
        max_workspace=0,
        memory_limit_bytes=1024**2,
    )
    assert p.execution.state_mode == "mps"
    assert p.report["execution_profile"] == "exact_independent_interval_mps"
    assert p.report["intermediate_mps_bond_bound"] == 4
    assert p.execution.state_bytes < 1024**2
    actual = fq.run(p).native()
    assert isinstance(actual, MPSState) and not actual.truncation_records
    assert actual.max_bond <= 4
    active = (*p.registers["t"], p.registers["spin"][0], p.registers["spin"][1])
    # Register readout ports are in physical significance order; these last two
    # physical spin wires are the native array's logical indices 7 and 6.
    words = []
    for word in range(64):
        bits = ["0"] * 83
        for wire, value in zip(active, f"{word:06b}", strict=True):
            bits[wire] = value
        words.append("".join(bits))
    indices = torch.arange(16, dtype=torch.float64)
    fourier = torch.exp(-2j * math.pi * torch.outer(indices, indices) / 16) / 4
    expected = torch.kron(
        fourier @ torch.tensor(weights, dtype=torch.complex128),
        torch.ones(4, dtype=torch.complex128) / 2,
    )
    observed = actual.amplitudes(words).flatten()
    torch.testing.assert_close(observed, expected, atol=1e-12, rtol=0)
    assert float(observed.abs().square().sum()) == pytest.approx(1, abs=1e-12)


def test_memory_ceiling_and_unproved_long_interval_refuse() -> None:
    weights = [1 / 4] * 16
    with pytest.raises(PlanningError, match="memory|budget"):
        synth.synthesize(
            intervals, classical_inputs={"weights": weights}, memory_limit_bytes=1024
        )
    with pytest.raises(PlanningError, match="memory|budget"):
        synth.synthesize(long_interval, memory_limit_bytes=1024**2)


def test_small_native_program_retains_statevector() -> None:
    p = synth.synthesize(small)
    assert p.execution.state_mode == "statevector"
    expected = torch.tensor([[1, 1]], dtype=torch.complex128) / math.sqrt(2)
    torch.testing.assert_close(fq.run(p).statevector(), expected, atol=1e-12, rtol=0)

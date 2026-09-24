from __future__ import annotations

import pytest

import flagquantum as fq
from flagquantum.noise import (
    DeviceNoiseProfile,
    GateDuration,
    NoiseModel,
    QubitNoiseCalibration,
    ReadoutError,
)
from flagquantum.services import (
    managed_quafu_simulator_device,
    run_managed_quafu_simulator,
)

TARGETS = (
    ("quafu:Baihua-sim", "Baihua"),
    ("quafu:Shenglian-sim", "Shenglian"),
    ("quafu:Dongling-sim", "Dongling"),
)


def _noise_model(n_qubits: int, *, device: str) -> NoiseModel:
    profile = DeviceNoiseProfile(
        qubits=tuple(
            QubitNoiseCalibration(
                wire,
                t1=50_000.0,
                t2=70_000.0,
                readout_error=ReadoutError(((0.99, 0.01), (0.02, 0.98))),
            )
            for wire in range(n_qubits)
        ),
        gate_durations=(
            GateDuration("h", 35.0),
            GateDuration("cx", 200.0),
        ),
        source=f"quafu:test:{device}",
        captured_at="2026-09-24T12:00:00+08:00",
        time_unit="ns",
    )
    return NoiseModel.from_device_profile(profile)


@pytest.mark.parametrize(("target", "device"), TARGETS)
def test_each_managed_quafu_target_executes_with_bound_calibration(
    target: str,
    device: str,
) -> None:
    model = _noise_model(2, device=device)

    execution = run_managed_quafu_simulator(
        fq.Circuit(2).h(0).cx(0, 1),
        target=target,
        calibration_device=device,
        noise_model=model,
        shots=64,
        seed=7,
        memory_limit_bytes=1024**2,
    )

    assert sum(execution.result.counts[0].values()) == 64
    assert execution.receipt.target == target
    assert execution.receipt.calibration_device == device
    assert execution.receipt.noise_model_identity == model.identity
    assert execution.receipt.execution_representation == "density_matrix"
    assert execution.receipt.approximate is False


def test_managed_quafu_simulator_falls_back_to_noisy_mps() -> None:
    n_qubits = 10
    circuit = fq.Circuit(n_qubits).h(0)
    for wire in range(n_qubits - 1):
        circuit.cx(wire, wire + 1)

    execution = run_managed_quafu_simulator(
        circuit,
        target="quafu:Baihua-sim",
        calibration_device="Baihua",
        noise_model=_noise_model(n_qubits, device="Baihua"),
        shots=64,
        seed=11,
        memory_limit_bytes=1024**2,
    )

    assert sum(execution.result.counts[0].values()) == 64
    assert execution.receipt.execution_representation == "mps"
    assert execution.receipt.approximate is True


def test_managed_quafu_simulator_rejects_mismatched_calibration() -> None:
    with pytest.raises(ValueError, match="requires 'Baihua' calibration"):
        run_managed_quafu_simulator(
            fq.Circuit(1).h(0),
            target="quafu:Baihua-sim",
            calibration_device="Dongling",
            noise_model=_noise_model(1, device="Dongling"),
            shots=10,
            memory_limit_bytes=1024**2,
        )


def test_managed_quafu_simulator_rejects_unknown_target() -> None:
    with pytest.raises(ValueError, match="unsupported managed Quafu simulator"):
        managed_quafu_simulator_device("quafu:Unknown-sim")

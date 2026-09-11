"""Convert Quafu/QuarkCircuit calibration payloads into FlagQuantum noise."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping, Sequence

from ...noise import (
    CorrelatedReadoutError,
    DeviceNoiseProfile,
    GateDuration,
    NoiseModel,
    QubitNoiseCalibration,
    depolarizing_channel,
    two_qubit_depolarizing_channel,
)
from ...noise.model import _decode_readout_error


def _captured_at(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError("Quafu calibration is missing calibration_time")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        text += "+08:00"
    return text


def quafu_noise_model_from_chip_info(
    chip_info: Mapping[str, Any],
    *,
    physical_qubits: Sequence[int],
    readout_confusion_matrices: Sequence[Sequence[Sequence[float]]] | None = None,
    correlated_readout_confusion_matrix: Sequence[Sequence[float]] | None = None,
) -> NoiseModel:
    """Build a logical-wire noise model from a QuarkCircuit ``chip_info``.

    Quafu currently reports T1/T2 in microseconds and gate lengths in seconds.
    Reported one-qubit fidelities are converted to Pauli depolarizing
    probabilities using ``p = 3/2 * (1 - average_gate_fidelity)``.  Readout
    matrices are optional because current Baihua payloads may expose zero
    placeholders rather than usable assignment fidelities.
    """

    selected = tuple(int(wire) for wire in physical_qubits)
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("physical_qubits must be non-empty and unique")
    qubits = chip_info.get("qubits_info")
    if not isinstance(qubits, Mapping):
        raise ValueError("Quafu calibration is missing qubits_info")
    if (
        readout_confusion_matrices is not None
        and correlated_readout_confusion_matrix is not None
    ):
        raise ValueError(
            "choose independent or correlated readout calibration, not both"
        )
    matrices = None
    if readout_confusion_matrices is not None:
        matrices = tuple(readout_confusion_matrices)
        if len(matrices) != len(selected):
            raise ValueError("readout matrix count must match physical_qubits")

    calibrations = []
    durations = []
    two_qubit_names = {"cx", "cnot", "cz", "swap", "iswap", "rxx", "ryy", "rzz"}
    basis_gates = tuple(
        str(name).lower()
        for name in chip_info.get("basis_gates", ())
        if str(name).lower() not in two_qubit_names
    )
    # Quafu emits its physical single-qubit primitive as ``U``; FlagQuantum
    # canonicalizes that OpenQASM alias to ``u3``.
    one_qubit_gates = tuple(dict.fromkeys((*basis_gates, "x", "y", "z", "u3")))
    depolarizing_probabilities = []
    for logical_wire, physical_wire in enumerate(selected):
        raw = qubits.get(f"Q{physical_wire}")
        if not isinstance(raw, Mapping):
            raise ValueError(f"Quafu calibration is missing Q{physical_wire}")
        t1 = float(raw.get("T1", 0.0)) * 1e-6
        t2 = float(raw.get("T2", 0.0)) * 1e-6
        if t1 <= 0 or t2 <= 0 or t2 > 2 * t1:
            raise ValueError(f"Q{physical_wire} has invalid T1/T2 calibration")
        readout = None
        if matrices is not None:
            readout = _decode_readout_error(matrices[logical_wire])
        calibrations.append(
            QubitNoiseCalibration(
                logical_wire,
                t1=t1,
                t2=t2,
                readout_error=readout,
            )
        )
        duration = float(raw.get("length", 0.0))
        for gate_name in one_qubit_gates:
            durations.append(GateDuration(gate_name, duration, wires=(logical_wire,)))
        fidelity = float(raw.get("fidelity", 0.0))
        if not 0 < fidelity <= 1:
            raise ValueError(f"Q{physical_wire} has invalid gate fidelity")
        probability = min(1.0, 1.5 * (1.0 - fidelity))
        depolarizing_probabilities.append(probability)

    couplers = chip_info.get("couplers_info", {})
    if not isinstance(couplers, Mapping):
        raise ValueError("Quafu calibration couplers_info must be a mapping")
    physical_to_logical = {
        physical_wire: logical_wire
        for logical_wire, physical_wire in enumerate(selected)
    }
    selected_couplers = []
    for raw in couplers.values():
        if not isinstance(raw, Mapping):
            continue
        physical_pair = tuple(int(wire) for wire in raw.get("qubits_index", ()))
        if len(physical_pair) != 2 or not set(physical_pair).issubset(
            physical_to_logical
        ):
            continue
        logical_pair = tuple(physical_to_logical[wire] for wire in physical_pair)
        duration = float(raw.get("length", 0.0))
        fidelity = float(raw.get("fidelity", 0.0))
        if duration <= 0:
            raise ValueError(f"Quafu coupler {physical_pair} has invalid duration")
        if not 0 < fidelity <= 1:
            raise ValueError(f"Quafu coupler {physical_pair} has invalid fidelity")
        durations.append(GateDuration("cz", duration, wires=logical_pair))
        durations.append(GateDuration("cz", duration, wires=logical_pair[::-1]))
        # Quafu exposes CZ as its physical entangler, while user OpenQASM can
        # contain CX. Provider compilation lowers CX through the same coupler,
        # so use that calibrated duration for the logical CX instruction too.
        durations.append(GateDuration("cx", duration, wires=logical_pair))
        durations.append(GateDuration("cx", duration, wires=logical_pair[::-1]))
        selected_couplers.append((logical_pair, fidelity))

    profile = DeviceNoiseProfile(
        qubits=tuple(calibrations),
        gate_durations=tuple(durations),
        source="quafu:quarkcircuit:chip_info:" + ",".join(map(str, selected)),
        captured_at=_captured_at(chip_info.get("calibration_time")),
        time_unit="s",
    )
    model = NoiseModel.from_device_profile(profile)
    if correlated_readout_confusion_matrix is not None:
        model.add_correlated_readout(
            range(len(selected)),
            CorrelatedReadoutError(
                tuple(
                    tuple(float(value) for value in row)
                    for row in correlated_readout_confusion_matrix
                )
            ),
        )
    for logical_wire, probability in enumerate(depolarizing_probabilities):
        if probability:
            model.add(
                one_qubit_gates,
                depolarizing_channel(probability),
                wires=(logical_wire,),
            )
    for logical_pair, fidelity in selected_couplers:
        probability = min(1.0, 1.25 * (1.0 - fidelity))
        if probability:
            model.add(
                "cz",
                two_qubit_depolarizing_channel(probability),
                wires=logical_pair,
            )
    return model


__all__ = ("quafu_noise_model_from_chip_info",)

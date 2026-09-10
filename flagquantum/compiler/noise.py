"""Lower noise-model rules into explicit channel instructions."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

from ..core.ir import CircuitIR, Instruction, ensure_circuit_ir
from ..noise import (
    DeviceNoiseProfile,
    KrausChannel,
    NoiseModel,
    thermal_relaxation_channel,
)


def _encode_channel_instruction(
    channel: KrausChannel,
    wires: Sequence[int],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Instruction:
    """Encode a native channel as a FlagQuantum IR instruction."""

    return Instruction(
        name=channel.name,
        wires=tuple(int(wire) for wire in wires),
        matrix=channel.kraus,
        metadata={"is_channel": True, **dict(metadata or {})},
    )


def _profile_channel(
    profile: DeviceNoiseProfile,
    *,
    wire: int,
    duration: float,
    placement: str,
    source_gate_index: int | None,
    source_gate_name: str | None,
) -> Instruction | None:
    if duration <= 0:
        return None
    calibration = profile.calibration_for(wire)
    channel = thermal_relaxation_channel(
        calibration.t1,
        calibration.t2,
        duration,
        excited_population=calibration.excited_population,
    )
    return _encode_channel_instruction(
        channel,
        (wire,),
        metadata={
            "placement": placement,
            "duration": duration,
            "time_unit": profile.time_unit,
            "device_profile_identity": profile.identity,
            "source_gate_index": source_gate_index,
            "source_gate_name": source_gate_name,
        },
    )


def lower_noise_model(circuit_or_ir: Any, noise_model: NoiseModel | None) -> CircuitIR:
    """Insert channel instructions after matching unitary instructions."""

    ir = ensure_circuit_ir(circuit_or_ir)
    if noise_model is None:
        return ir
    instructions: list[Instruction] = []
    profile = noise_model.device_profile
    wire_clock = [0.0] * ir.n_wires
    for gate_index, instruction in enumerate(ir):
        if instruction.metadata.get("is_channel"):
            instructions.append(instruction)
            continue
        duration = (
            0.0
            if profile is None
            else profile.duration_for(instruction.name, instruction.wires)
        )
        start = max((wire_clock[wire] for wire in instruction.wires), default=0.0)
        if profile is not None:
            for wire in instruction.wires:
                idle = _profile_channel(
                    profile,
                    wire=wire,
                    duration=start - wire_clock[wire],
                    placement="idle_before_gate",
                    source_gate_index=gate_index,
                    source_gate_name=instruction.name,
                )
                if idle is not None:
                    instructions.append(idle)
        instructions.append(instruction)
        if profile is not None:
            for wire in instruction.wires:
                relaxation = _profile_channel(
                    profile,
                    wire=wire,
                    duration=duration,
                    placement="during_gate_approximation",
                    source_gate_index=gate_index,
                    source_gate_name=instruction.name,
                )
                if relaxation is not None:
                    instructions.append(relaxation)
                wire_clock[wire] = start + duration
        for channel, wires in noise_model.channels_for(instruction):
            instructions.append(_encode_channel_instruction(channel, wires))
    if profile is not None:
        makespan = max(wire_clock, default=0.0)
        for wire in range(ir.n_wires):
            idle = _profile_channel(
                profile,
                wire=wire,
                duration=makespan - wire_clock[wire],
                placement="terminal_idle",
                source_gate_index=None,
                source_gate_name=None,
            )
            if idle is not None:
                instructions.append(idle)
    return replace(ir, instructions=tuple(instructions))


__all__ = ("lower_noise_model",)

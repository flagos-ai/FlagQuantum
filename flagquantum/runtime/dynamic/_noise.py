"""Runtime orchestration for the bounded local dynamic-noise profile."""

from __future__ import annotations

import torch

from ...core.ir import Instruction
from ...noise import CorrelatedReadoutError, NoiseModel
from ...simulation.statevector.dynamic_noise import (
    apply_dynamic_bit_flip,
    sample_dynamic_readout,
)
from .circuit import DynamicCircuit


def validate_dynamic_noise(
    circuit: DynamicCircuit, noise_model: NoiseModel | None
) -> None:
    if noise_model is None:
        return
    if not isinstance(noise_model, NoiseModel):
        raise TypeError("dynamic noise_model must be a NoiseModel or None")
    if noise_model.device_profile is not None:
        raise ValueError("device-profile timing noise is outside the dynamic profile")
    for rule in noise_model.rules:
        if set(rule.gate_names).intersection({"measure", "reset"}):
            raise ValueError(
                "dynamic measurement noise must use independent readout rules"
            )
        if rule.channel.name != "bit_flip" or rule.channel.n_wires != 1:
            raise ValueError(
                "dynamic execution currently supports one-wire bit-flip channels only"
            )
        if rule.wires is not None and any(
            wire < 0 or wire >= circuit.n_wires for wire in rule.wires
        ):
            raise ValueError("dynamic noise rule wire is outside the circuit")
    for readout_rule in noise_model.readout_rules:
        if isinstance(readout_rule.error, CorrelatedReadoutError):
            raise ValueError("correlated readout is outside the dynamic noise profile")
        if any(wire < 0 or wire >= circuit.n_wires for wire in readout_rule.wires):
            raise ValueError("dynamic readout wire is outside the circuit")


def apply_noise_after_instruction(
    state: torch.Tensor,
    instruction: Instruction,
    noise_model: NoiseModel | None,
    *,
    n_wires: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, int, int]:
    if noise_model is None:
        return state, 0, 0
    applications = 0
    events = 0
    for channel, wires in noise_model.channels_for(instruction):
        applications += state.shape[0]
        state, count = apply_dynamic_bit_flip(
            state,
            channel,
            wires[0],
            n_wires,
            generator=generator,
        )
        events += count
    return state, applications, events


def apply_readout_error(
    bits: torch.Tensor,
    wire: int,
    noise_model: NoiseModel | None,
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, int]:
    if noise_model is None:
        return bits, 0
    for rule in noise_model.readout_rules:
        if wire in rule.wires:
            if isinstance(rule.error, CorrelatedReadoutError):
                raise ValueError(
                    "correlated readout is outside the dynamic noise profile"
                )
            return sample_dynamic_readout(bits, rule.error, generator=generator)
    return bits, 0


__all__ = (
    "apply_noise_after_instruction",
    "apply_readout_error",
    "validate_dynamic_noise",
)

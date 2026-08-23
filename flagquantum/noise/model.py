"""Backend-neutral rules for attaching noise channels to circuit operations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Mapping

import torch

from ..core.ir import Instruction
from .channels import KrausChannel


@dataclass(frozen=True)
class NoiseRule:
    gate_names: tuple[str, ...]
    channel: KrausChannel
    wires: tuple[int, ...] | None = None


@dataclass(frozen=True)
class ReadoutError:
    """Single-qubit true-to-observed classical confusion matrix."""

    probabilities: tuple[tuple[float, float], tuple[float, float]]

    def __post_init__(self) -> None:
        matrix = torch.as_tensor(self.probabilities, dtype=torch.float64)
        if matrix.shape != (2, 2):
            raise ValueError("readout confusion matrix must have shape (2, 2)")
        if not bool(torch.isfinite(matrix).all()) or bool(torch.any(matrix < 0)):
            raise ValueError("readout probabilities must be finite and non-negative")
        if not torch.allclose(matrix.sum(dim=1), torch.ones(2, dtype=torch.float64)):
            raise ValueError("each readout confusion row must sum to 1")


@dataclass(frozen=True)
class CorrelatedReadoutError:
    """Multi-qubit true-to-observed classical confusion matrix."""

    probabilities: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        matrix = torch.as_tensor(self.probabilities, dtype=torch.float64)
        size = matrix.shape[0] if matrix.ndim == 2 else 0
        if size < 2 or matrix.shape != (size, size) or size & (size - 1):
            raise ValueError(
                "correlated readout confusion matrix must be square with power-of-two size"
            )
        if not bool(torch.isfinite(matrix).all()) or bool(torch.any(matrix < 0)):
            raise ValueError("readout probabilities must be finite and non-negative")
        if not torch.allclose(matrix.sum(dim=1), torch.ones(size, dtype=torch.float64)):
            raise ValueError("each readout confusion row must sum to 1")

    @property
    def n_wires(self) -> int:
        return (len(self.probabilities)).bit_length() - 1


@dataclass(frozen=True)
class ReadoutRule:
    wires: tuple[int, ...]
    error: ReadoutError | CorrelatedReadoutError


@dataclass
class NoiseModel:
    """Attach channels after selected circuit instructions."""

    rules: list[NoiseRule] = field(default_factory=list)
    readout_rules: list[ReadoutRule] = field(default_factory=list)
    device_profile: Any | None = None

    def __post_init__(self) -> None:
        if self.device_profile is not None:
            from .device_profile import DeviceNoiseProfile

            if not isinstance(self.device_profile, DeviceNoiseProfile):
                raise TypeError("device_profile must be a DeviceNoiseProfile")

    def to_dict(self) -> dict[str, Any]:
        """Return a stable, versioned, device-independent model specification."""

        return {
            "schema": "flagquantum.noise_model.v1",
            "rules": [
                {
                    "gate_names": list(rule.gate_names),
                    "wires": None if rule.wires is None else list(rule.wires),
                    "channel": rule.channel.to_dict(),
                }
                for rule in self.rules
            ],
            "readout_rules": [
                {
                    "wires": list(rule.wires),
                    "correlated": isinstance(rule.error, CorrelatedReadoutError),
                    "probabilities": [list(row) for row in rule.error.probabilities],
                }
                for rule in self.readout_rules
            ],
            "device_profile": (
                None if self.device_profile is None else self.device_profile.to_dict()
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NoiseModel":
        if payload.get("schema") != "flagquantum.noise_model.v1":
            raise ValueError("unsupported noise model schema")
        from .channels import KrausChannel

        model = cls()
        for encoded in payload.get("rules", ()):
            names = tuple(str(name) for name in encoded.get("gate_names", ()))
            wires = encoded.get("wires")
            model.add(
                names,
                KrausChannel.from_dict(encoded["channel"]),
                wires=None if wires is None else tuple(int(wire) for wire in wires),
            )
        for encoded in payload.get("readout_rules", ()):
            model.add_readout(
                encoded.get("wires", ()),
                (CorrelatedReadoutError if encoded.get("correlated") else ReadoutError)(
                    tuple(
                        tuple(float(value) for value in row)
                        for row in encoded["probabilities"]
                    )
                ),
            )
        if payload.get("device_profile") is not None:
            from .device_profile import DeviceNoiseProfile

            model.device_profile = DeviceNoiseProfile.from_dict(
                payload["device_profile"]
            )
        return model

    @property
    def identity(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    def add(
        self,
        gate_names: str | Iterable[str],
        channel: KrausChannel,
        *,
        wires: Iterable[int] | int | None = None,
    ) -> "NoiseModel":
        if isinstance(gate_names, str):
            names = (gate_names.lower(),)
        else:
            names = tuple(name.lower() for name in gate_names)
        if not names or any(not name for name in names):
            raise ValueError("noise rules require at least one non-empty gate name")
        if wires is None:
            target_wires = None
        elif isinstance(wires, int):
            target_wires = (wires,)
        else:
            target_wires = tuple(int(wire) for wire in wires)
        self.rules.append(NoiseRule(names, channel, target_wires))
        return self

    def add_readout(
        self,
        wires: Iterable[int] | int,
        error: ReadoutError,
    ) -> "NoiseModel":
        target_wires = (
            (int(wires),)
            if isinstance(wires, int)
            else tuple(int(wire) for wire in wires)
        )
        if not target_wires or any(wire < 0 for wire in target_wires):
            raise ValueError("readout wires must be non-empty and non-negative")
        if len(set(target_wires)) != len(target_wires):
            raise ValueError("readout wires must be unique")
        occupied = {wire for rule in self.readout_rules for wire in rule.wires}
        if occupied.intersection(target_wires):
            raise ValueError("a wire can have only one readout error rule")
        self.readout_rules.append(ReadoutRule(target_wires, error))
        return self

    def add_correlated_readout(
        self,
        wires: Iterable[int],
        error: CorrelatedReadoutError,
    ) -> "NoiseModel":
        target_wires = tuple(int(wire) for wire in wires)
        if len(target_wires) != error.n_wires:
            raise ValueError("correlated readout matrix size must match wires")
        if any(wire < 0 for wire in target_wires) or len(set(target_wires)) != len(
            target_wires
        ):
            raise ValueError("readout wires must be unique and non-negative")
        occupied = {wire for rule in self.readout_rules for wire in rule.wires}
        if occupied.intersection(target_wires):
            raise ValueError("a wire can have only one readout error rule")
        self.readout_rules.append(ReadoutRule(target_wires, error))
        return self

    @classmethod
    def from_device_profile(cls, profile: Any) -> "NoiseModel":
        from .device_profile import DeviceNoiseProfile

        if not isinstance(profile, DeviceNoiseProfile):
            raise TypeError("profile must be a DeviceNoiseProfile")
        model = cls(device_profile=profile)
        for calibration in profile.qubits:
            if calibration.readout_error is not None:
                model.add_readout(calibration.wire, calibration.readout_error)
        return model

    def apply_readout_probabilities(
        self,
        probabilities: torch.Tensor,
        *,
        n_wires: int,
    ) -> torch.Tensor:
        """Apply independent classical readout confusion after state evolution."""

        values = torch.as_tensor(probabilities)
        if values.shape[-1] != 2**n_wires:
            raise ValueError("probability dimension does not match n_wires")
        output = values.reshape(*values.shape[:-1], *((2,) * n_wires))
        for rule in self.readout_rules:
            if isinstance(rule.error, CorrelatedReadoutError):
                if any(wire >= n_wires for wire in rule.wires):
                    raise ValueError("correlated readout wire is outside the circuit")
                batch_dims = output.ndim - n_wires
                selected = [batch_dims + wire for wire in rule.wires]
                unselected = [
                    batch_dims + wire
                    for wire in range(n_wires)
                    if wire not in rule.wires
                ]
                permutation = [*range(batch_dims), *unselected, *selected]
                inverse = [permutation.index(index) for index in range(output.ndim)]
                arranged = output.permute(permutation)
                prefix = arranged.shape[: -len(rule.wires)]
                matrix = torch.as_tensor(
                    rule.error.probabilities,
                    device=output.device,
                    dtype=output.dtype,
                )
                arranged = arranged.reshape(*prefix, matrix.shape[0]) @ matrix
                output = arranged.reshape(*prefix, *((2,) * len(rule.wires))).permute(
                    inverse
                )
                continue
            for wire in rule.wires:
                if wire >= n_wires:
                    raise ValueError(f"readout wire {wire} is outside the circuit")
                matrix = torch.as_tensor(
                    rule.error.probabilities,
                    device=output.device,
                    dtype=output.dtype,
                )
                output = torch.tensordot(
                    output, matrix, dims=([output.ndim - n_wires + wire], [0])
                )
                output = torch.movedim(output, -1, output.ndim - n_wires + wire)
        return output.reshape_as(values)

    def apply_readout_expectation_z(self, expectation: torch.Tensor) -> torch.Tensor:
        """Apply independent readout confusion directly to per-wire Z means."""

        output = torch.as_tensor(expectation).clone()
        n_wires = output.shape[-1]
        for rule in self.readout_rules:
            if isinstance(rule.error, CorrelatedReadoutError):
                raise ValueError(
                    "correlated readout cannot be applied to marginal Z expectations"
                )
            for wire in rule.wires:
                if wire >= n_wires:
                    raise ValueError(f"readout wire {wire} is outside the circuit")
                matrix = torch.as_tensor(
                    rule.error.probabilities,
                    device=output.device,
                    dtype=output.dtype,
                )
                true_z = output[..., wire]
                true_zero = (1 + true_z) / 2
                true_one = (1 - true_z) / 2
                output[..., wire] = true_zero * (
                    matrix[0, 0] - matrix[0, 1]
                ) + true_one * (matrix[1, 0] - matrix[1, 1])
        return output

    def channels_for(
        self,
        instruction: Instruction,
    ) -> Iterator[tuple[KrausChannel, tuple[int, ...]]]:
        for rule in self.rules:
            if instruction.name.lower() not in rule.gate_names:
                continue
            if rule.wires is not None:
                if not set(rule.wires).issubset(instruction.wires):
                    continue
                yield rule.channel, rule.wires
            elif rule.channel.n_wires == len(instruction.wires):
                yield rule.channel, instruction.wires
            elif rule.channel.n_wires == 1:
                for wire in instruction.wires:
                    yield rule.channel, (wire,)
            else:
                raise ValueError(
                    f"Channel {rule.channel.name!r} cannot be inferred for "
                    f"instruction {instruction.name!r} on wires "
                    f"{instruction.wires}."
                )


__all__ = (
    "CorrelatedReadoutError",
    "NoiseModel",
    "NoiseRule",
    "ReadoutError",
    "ReadoutRule",
)

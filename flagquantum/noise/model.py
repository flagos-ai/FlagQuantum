"""Backend-neutral rules for attaching noise channels to circuit operations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

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


def _decode_readout_error(rows: Iterable[Iterable[float]]) -> ReadoutError:
    probabilities = tuple(tuple(float(value) for value in row) for row in rows)
    if len(probabilities) != 2 or any(len(row) != 2 for row in probabilities):
        raise ValueError("readout confusion matrix must have shape (2, 2)")
    first, second = probabilities
    return ReadoutError(((first[0], first[1]), (second[0], second[1])))


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
    def n_qubits(self) -> int:
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
            qubits = encoded.get("wires")
            model.add(
                names,
                KrausChannel.from_dict(encoded["channel"]),
                qubits=(
                    None if qubits is None else tuple(int(qubit) for qubit in qubits)
                ),
            )
        for encoded in payload.get("readout_rules", ()):
            if encoded.get("correlated"):
                error = CorrelatedReadoutError(
                    tuple(
                        tuple(float(value) for value in row)
                        for row in encoded["probabilities"]
                    )
                )
                model.add_correlated_readout(encoded.get("wires", ()), error)
            else:
                model.add_readout(
                    encoded.get("wires", ()),
                    _decode_readout_error(encoded["probabilities"]),
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
        qubits: Iterable[int] | int | None = None,
    ) -> "NoiseModel":
        names: tuple[str, ...]
        if isinstance(gate_names, str):
            names = (gate_names.lower(),)
        else:
            names = tuple(name.lower() for name in gate_names)
        if not names or any(not name for name in names):
            raise ValueError("noise rules require at least one non-empty gate name")
        target_qubits: tuple[int, ...] | None
        if qubits is None:
            target_qubits = None
        elif isinstance(qubits, int):
            target_qubits = (qubits,)
        else:
            target_qubits = tuple(int(qubit) for qubit in qubits)
        self.rules.append(NoiseRule(names, channel, target_qubits))
        return self

    def add_readout(
        self,
        qubits: Iterable[int] | int,
        error: ReadoutError,
    ) -> "NoiseModel":
        target_qubits = (
            (int(qubits),)
            if isinstance(qubits, int)
            else tuple(int(qubit) for qubit in qubits)
        )
        if not target_qubits or any(qubit < 0 for qubit in target_qubits):
            raise ValueError("readout qubits must be non-empty and non-negative")
        if len(set(target_qubits)) != len(target_qubits):
            raise ValueError("readout qubits must be unique")
        occupied = {qubit for rule in self.readout_rules for qubit in rule.wires}
        if occupied.intersection(target_qubits):
            raise ValueError("a qubit can have only one readout error rule")
        self.readout_rules.append(ReadoutRule(target_qubits, error))
        return self

    def add_correlated_readout(
        self,
        qubits: Iterable[int],
        error: CorrelatedReadoutError,
    ) -> "NoiseModel":
        target_qubits = tuple(int(qubit) for qubit in qubits)
        if len(target_qubits) != error.n_qubits:
            raise ValueError("correlated readout matrix size must match qubits")
        if any(qubit < 0 for qubit in target_qubits) or len(set(target_qubits)) != len(
            target_qubits
        ):
            raise ValueError("readout qubits must be unique and non-negative")
        occupied = {qubit for rule in self.readout_rules for qubit in rule.wires}
        if occupied.intersection(target_qubits):
            raise ValueError("a qubit can have only one readout error rule")
        self.readout_rules.append(ReadoutRule(target_qubits, error))
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
        n_qubits: int,
    ) -> torch.Tensor:
        """Apply independent classical readout confusion after state evolution."""

        values = torch.as_tensor(probabilities)
        if values.shape[-1] != 2**n_qubits:
            raise ValueError("probability dimension does not match n_qubits")
        output = values.reshape(*values.shape[:-1], *((2,) * n_qubits))
        for rule in self.readout_rules:
            if isinstance(rule.error, CorrelatedReadoutError):
                if any(qubit >= n_qubits for qubit in rule.wires):
                    raise ValueError("correlated readout qubit is outside the circuit")
                batch_dims = output.ndim - n_qubits
                selected = [batch_dims + qubit for qubit in rule.wires]
                unselected = [
                    batch_dims + qubit
                    for qubit in range(n_qubits)
                    if qubit not in rule.wires
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
            for qubit in rule.wires:
                if qubit >= n_qubits:
                    raise ValueError(f"readout qubit {qubit} is outside the circuit")
                matrix = torch.as_tensor(
                    rule.error.probabilities,
                    device=output.device,
                    dtype=output.dtype,
                )
                output = torch.tensordot(
                    output, matrix, dims=([output.ndim - n_qubits + qubit], [0])
                )
                output = torch.movedim(output, -1, output.ndim - n_qubits + qubit)
        return output.reshape_as(values)

    def apply_readout_expectation_z(self, expectation: torch.Tensor) -> torch.Tensor:
        """Apply independent readout confusion directly to per-qubit Z means."""

        output = torch.as_tensor(expectation).clone()
        n_qubits = output.shape[-1]
        for rule in self.readout_rules:
            if isinstance(rule.error, CorrelatedReadoutError):
                raise ValueError(
                    "correlated readout cannot be applied to marginal Z expectations"
                )
            for qubit in rule.wires:
                if qubit >= n_qubits:
                    raise ValueError(f"readout qubit {qubit} is outside the circuit")
                matrix = torch.as_tensor(
                    rule.error.probabilities,
                    device=output.device,
                    dtype=output.dtype,
                )
                true_z = output[..., qubit]
                true_zero = (1 + true_z) / 2
                true_one = (1 - true_z) / 2
                output[..., qubit] = true_zero * (
                    matrix[0, 0] - matrix[0, 1]
                ) + true_one * (matrix[1, 0] - matrix[1, 1])
        return output

    def apply_readout_samples(
        self,
        samples: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample classical readout confusion on computational-basis bits."""

        output = torch.as_tensor(samples).clone()
        if output.ndim < 2:
            raise ValueError("readout samples must end in a qubit dimension")
        if output.dtype != torch.int64:
            output = output.to(torch.int64)
        n_qubits = int(output.shape[-1])
        if bool(((output != 0) & (output != 1)).any()):
            raise ValueError("readout samples must contain only 0 and 1")

        for rule in self.readout_rules:
            if any(qubit >= n_qubits for qubit in rule.wires):
                raise ValueError("readout qubit is outside the circuit")
            matrix = torch.as_tensor(
                rule.error.probabilities,
                device=output.device,
                dtype=torch.float64,
            )
            if isinstance(rule.error, CorrelatedReadoutError):
                true_index = torch.zeros(
                    output.shape[:-1], dtype=torch.int64, device=output.device
                )
                for qubit in rule.wires:
                    true_index = (true_index << 1) | output[..., qubit]
                observed = torch.multinomial(
                    matrix[true_index.reshape(-1)],
                    num_samples=1,
                    replacement=True,
                    generator=generator,
                ).reshape(true_index.shape)
                for offset, qubit in enumerate(rule.wires):
                    shift = len(rule.wires) - offset - 1
                    output[..., qubit] = (observed >> shift) & 1
                continue
            for qubit in rule.wires:
                probabilities = matrix[output[..., qubit].reshape(-1)]
                output[..., qubit] = torch.multinomial(
                    probabilities,
                    num_samples=1,
                    replacement=True,
                    generator=generator,
                ).reshape(output.shape[:-1])
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
            elif rule.channel.n_qubits == len(instruction.wires):
                yield rule.channel, instruction.wires
            elif rule.channel.n_qubits == 1:
                for qubit in instruction.wires:
                    yield rule.channel, (qubit,)
            else:
                raise ValueError(
                    f"Channel {rule.channel.name!r} cannot be inferred for "
                    f"instruction {instruction.name!r} on qubits "
                    f"{instruction.wires}."
                )


__all__ = (
    "CorrelatedReadoutError",
    "NoiseModel",
    "NoiseRule",
    "ReadoutError",
    "ReadoutRule",
)

"""Built-in Kraus channel specifications."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from itertools import product
from math import exp
from typing import Any, Mapping

import torch

from ..core.runtime_config import get_runtime_config


def _as_probability(value: float | torch.Tensor, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if tensor.numel() != 1:
        raise ValueError(f"{name} must be a scalar probability.")
    if bool(torch.any(tensor < 0)) or bool(torch.any(tensor > 1)):
        raise ValueError(f"{name} must be between 0 and 1.")
    return tensor


def _complex_dtype(dtype: torch.dtype | None = None) -> torch.dtype:
    return dtype or getattr(torch, get_runtime_config().complex_dtype)


def _real_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.float64 if dtype == torch.complex128 else torch.float32


def _pauli_basis(
    *,
    dtype: torch.dtype,
    device: torch.device | str | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the one-qubit Pauli basis owned by the noise model."""

    return (
        torch.tensor(((1, 0), (0, 1)), dtype=dtype, device=device),
        torch.tensor(((0, 1), (1, 0)), dtype=dtype, device=device),
        torch.tensor(((0, -1j), (1j, 0)), dtype=dtype, device=device),
        torch.tensor(((1, 0), (0, -1)), dtype=dtype, device=device),
    )


@dataclass(frozen=True)
class KrausChannel:
    """A completely-positive trace-preserving channel represented by Kraus ops."""

    name: str
    kraus: tuple[torch.Tensor, ...]
    parameters: tuple[tuple[str, str | float], ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("channel name must be non-empty")
        if not self.kraus:
            raise ValueError("Kraus channel must contain at least one operator")
        first = self.kraus[0]
        if not isinstance(first, torch.Tensor) or first.ndim != 2:
            raise ValueError("Kraus operators must be rank-2 tensors")
        if first.shape[0] != first.shape[1]:
            raise ValueError("Kraus operators must be square")
        size = int(first.shape[0])
        if size <= 0 or size & (size - 1):
            raise ValueError("Kraus operator dimension must be a positive power of two")
        for op in self.kraus:
            if not isinstance(op, torch.Tensor) or op.shape != first.shape:
                raise ValueError("all Kraus operators must have the same square shape")
            if not bool(torch.isfinite(op).all()):
                raise ValueError("Kraus operators must contain only finite values")
        check_dtype = torch.complex128
        effect = torch.zeros((size, size), dtype=check_dtype)
        for op in self.kraus:
            checked = op.detach().to(device="cpu", dtype=check_dtype)
            effect += checked.mH @ checked
        identity = torch.eye(size, dtype=check_dtype)
        if not torch.allclose(effect, identity, atol=1e-6, rtol=1e-6):
            error = float(torch.max(torch.abs(effect - identity)).item())
            raise ValueError(
                "Kraus operators must define a trace-preserving channel; "
                f"max completeness error={error:.3e}"
            )

    @property
    def n_wires(self) -> int:
        size = int(self.kraus[0].shape[-1])
        return size.bit_length() - 1

    def to(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "KrausChannel":
        out_dtype = _complex_dtype(dtype)
        return KrausChannel(
            self.name,
            tuple(op.to(device=device, dtype=out_dtype) for op in self.kraus),
            self.parameters,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a versioned, device-independent channel specification."""

        matrices = []
        for op in self.kraus:
            values = op.detach().to(device="cpu", dtype=torch.complex128)
            matrices.append(
                [
                    [[float(value.real), float(value.imag)] for value in row]
                    for row in values
                ]
            )
        return {
            "schema": "flagquantum.kraus_channel.v1",
            "name": self.name,
            "parameters": {key: value for key, value in self.parameters},
            "kraus": matrices,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "KrausChannel":
        if payload.get("schema") != "flagquantum.kraus_channel.v1":
            raise ValueError("unsupported Kraus channel schema")
        matrices = []
        for encoded in payload.get("kraus", ()):
            matrices.append(
                torch.tensor(
                    [
                        [complex(float(value[0]), float(value[1])) for value in row]
                        for row in encoded
                    ],
                    dtype=torch.complex128,
                )
            )
        parameters = tuple(
            sorted(
                (str(key), str(value) if isinstance(value, str) else float(value))
                for key, value in dict(payload.get("parameters", {})).items()
            )
        )
        return cls(str(payload.get("name", "")), tuple(matrices), parameters)

    @property
    def identity(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


def bit_flip_channel(
    probability: float | torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> KrausChannel:
    p = _as_probability(probability, "probability").to(
        dtype=_real_dtype(_complex_dtype(dtype)), device=device
    )
    identity, x, _, _ = _pauli_basis(dtype=_complex_dtype(dtype), device=device)
    return KrausChannel(
        "bit_flip",
        (torch.sqrt(1 - p) * identity, torch.sqrt(p) * x),
        (("probability", float(p.detach().cpu().item())),),
    )


def phase_flip_channel(
    probability: float | torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> KrausChannel:
    p = _as_probability(probability, "probability").to(
        dtype=_real_dtype(_complex_dtype(dtype)), device=device
    )
    identity, _, _, z = _pauli_basis(dtype=_complex_dtype(dtype), device=device)
    return KrausChannel(
        "phase_flip",
        (torch.sqrt(1 - p) * identity, torch.sqrt(p) * z),
        (("probability", float(p.detach().cpu().item())),),
    )


def depolarizing_channel(
    probability: float | torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> KrausChannel:
    p = _as_probability(probability, "probability").to(
        dtype=_real_dtype(_complex_dtype(dtype)), device=device
    )
    out_dtype = _complex_dtype(dtype)
    scale = torch.sqrt(p / 3)
    identity, x, y, z = _pauli_basis(dtype=out_dtype, device=device)
    return KrausChannel(
        "depolarizing",
        (
            torch.sqrt(1 - p) * identity,
            scale * x,
            scale * y,
            scale * z,
        ),
        (("probability", float(p.detach().cpu().item())),),
    )


def two_qubit_depolarizing_channel(
    probability: float | torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> KrausChannel:
    """Return the uniform two-qubit Pauli channel with total error ``p``.

    The identity occurs with probability ``1-p`` and each of the 15 nonidentity
    two-qubit Paulis occurs with probability ``p/15``.  Its average fidelity is
    ``1 - 4p/5``.
    """

    p = _as_probability(probability, "probability").to(
        dtype=_real_dtype(_complex_dtype(dtype)), device=device
    )
    out_dtype = _complex_dtype(dtype)
    paulis = _pauli_basis(dtype=out_dtype, device=device)
    identity = torch.kron(paulis[0], paulis[0])
    errors = tuple(
        torch.kron(left, right)
        for left, right in product(paulis, repeat=2)
        if not (left is paulis[0] and right is paulis[0])
    )
    scale = torch.sqrt(p / 15)
    return KrausChannel(
        "two_qubit_depolarizing",
        (torch.sqrt(1 - p) * identity, *(scale * op for op in errors)),
        (("probability", float(p.detach().cpu().item())),),
    )


def amplitude_damping_channel(
    gamma: float | torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> KrausChannel:
    damping = _as_probability(gamma, "gamma").to(
        dtype=_real_dtype(_complex_dtype(dtype)),
        device=device,
    )
    out_dtype = _complex_dtype(dtype)
    zero = torch.zeros((), dtype=_real_dtype(out_dtype), device=device)
    one = torch.ones((), dtype=_real_dtype(out_dtype), device=device)
    k0 = torch.stack(
        [
            torch.stack([one, zero]),
            torch.stack([zero, torch.sqrt(1 - damping)]),
        ]
    ).to(dtype=out_dtype)
    k1 = torch.stack(
        [
            torch.stack([zero, torch.sqrt(damping)]),
            torch.stack([zero, zero]),
        ]
    ).to(dtype=out_dtype)
    return KrausChannel(
        "amplitude_damping",
        (k0, k1),
        (("gamma", float(damping.detach().cpu().item())),),
    )


def phase_damping_channel(
    gamma: float | torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> KrausChannel:
    damping = _as_probability(gamma, "gamma").to(
        dtype=_real_dtype(_complex_dtype(dtype)), device=device
    )
    out_dtype = _complex_dtype(dtype)
    k0 = torch.diag(torch.stack((torch.ones_like(damping), torch.sqrt(1 - damping))))
    k1 = torch.diag(torch.stack((torch.zeros_like(damping), torch.sqrt(damping))))
    return KrausChannel(
        "phase_damping",
        (k0.to(out_dtype), k1.to(out_dtype)),
        (("gamma", float(damping.detach().cpu().item())),),
    )


def reset_error_channel(
    probability_zero: float | torch.Tensor,
    probability_one: float | torch.Tensor = 0.0,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> KrausChannel:
    p0 = _as_probability(probability_zero, "probability_zero")
    p1 = _as_probability(probability_one, "probability_one")
    if bool(torch.any(p0 + p1 > 1)):
        raise ValueError("reset probabilities must sum to at most 1")
    real_dtype = _real_dtype(_complex_dtype(dtype))
    p0, p1 = (
        p0.to(device=device, dtype=real_dtype),
        p1.to(device=device, dtype=real_dtype),
    )
    one = torch.ones((), device=device, dtype=real_dtype)
    zero = torch.zeros((), device=device, dtype=real_dtype)
    matrices = (
        torch.sqrt(1 - p0 - p1) * torch.stack((one, zero, zero, one)).reshape(2, 2),
        torch.sqrt(p0) * torch.stack((one, zero, zero, zero)).reshape(2, 2),
        torch.sqrt(p0) * torch.stack((zero, one, zero, zero)).reshape(2, 2),
        torch.sqrt(p1) * torch.stack((zero, zero, one, zero)).reshape(2, 2),
        torch.sqrt(p1) * torch.stack((zero, zero, zero, one)).reshape(2, 2),
    )
    return KrausChannel(
        "reset_error",
        tuple(item.to(_complex_dtype(dtype)) for item in matrices),
        (
            ("probability_one", float(p1.detach().cpu().item())),
            ("probability_zero", float(p0.detach().cpu().item())),
        ),
    )


def coherent_overrotation_channel(
    angle: float,
    *,
    axis: str = "x",
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> KrausChannel:
    axis = axis.lower()
    if axis not in {"x", "y", "z"}:
        raise ValueError("axis must be 'x', 'y', or 'z'")
    out_dtype = _complex_dtype(dtype)
    real_dtype = _real_dtype(out_dtype)
    identity, x, y, z = _pauli_basis(dtype=out_dtype, device=device)
    pauli = {"x": x, "y": y, "z": z}[axis]
    value = float(angle)
    half_angle = torch.tensor(value / 2, device=device, dtype=real_dtype)
    unitary = torch.cos(half_angle) * identity
    unitary = unitary - 1j * torch.sin(half_angle) * pauli
    return KrausChannel(
        "coherent_overrotation",
        (unitary.to(out_dtype),),
        (("angle", value), ("axis", axis)),
    )


def thermal_relaxation_channel(
    t1: float,
    t2: float,
    duration: float,
    *,
    excited_population: float = 0.0,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> KrausChannel:
    if t1 <= 0 or t2 <= 0 or duration < 0:
        raise ValueError("t1 and t2 must be positive and duration non-negative")
    if t2 > 2 * t1:
        raise ValueError("physical relaxation requires t2 <= 2 * t1")
    if not 0 <= excited_population <= 1:
        raise ValueError("excited_population must be between 0 and 1")
    decay = 1 - exp(-duration / t1)
    equilibrium_ground = 1 - excited_population
    real_dtype = _real_dtype(_complex_dtype(dtype))
    one = torch.ones((), device=device, dtype=real_dtype)
    zero = torch.zeros((), device=device, dtype=real_dtype)
    g = torch.tensor(decay, device=device, dtype=real_dtype)
    ground = torch.tensor(equilibrium_ground, device=device, dtype=real_dtype)
    excited = 1 - ground
    gad = (
        torch.sqrt(ground)
        * torch.stack((one, zero, zero, torch.sqrt(1 - g))).reshape(2, 2),
        torch.sqrt(ground)
        * torch.stack((zero, torch.sqrt(g), zero, zero)).reshape(2, 2),
        torch.sqrt(excited)
        * torch.stack((torch.sqrt(1 - g), zero, zero, one)).reshape(2, 2),
        torch.sqrt(excited)
        * torch.stack((zero, zero, torch.sqrt(g), zero)).reshape(2, 2),
    )
    inverse_tphi = max(0.0, 1 / t2 - 1 / (2 * t1))
    phase_gamma = 1 - exp(-2 * duration * inverse_tphi)
    phase = phase_damping_channel(phase_gamma, dtype=dtype, device=device).kraus
    composed = tuple(p @ a.to(_complex_dtype(dtype)) for p in phase for a in gad)
    return KrausChannel(
        "thermal_relaxation",
        tuple(item.to(_complex_dtype(dtype)) for item in composed),
        (
            ("duration", float(duration)),
            ("excited_population", float(excited_population)),
            ("t1", float(t1)),
            ("t2", float(t2)),
        ),
    )


__all__ = (
    "KrausChannel",
    "amplitude_damping_channel",
    "bit_flip_channel",
    "coherent_overrotation_channel",
    "depolarizing_channel",
    "phase_damping_channel",
    "phase_flip_channel",
    "reset_error_channel",
    "thermal_relaxation_channel",
    "two_qubit_depolarizing_channel",
)

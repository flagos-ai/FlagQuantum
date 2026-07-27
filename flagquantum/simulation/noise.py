"""Native density-matrix and noise-channel support for FlagQuantum."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Iterator, Sequence

import torch

from ..circuit import _gate_matrix
from ..core.ir import CircuitIR, Instruction, ensure_circuit_ir
from ..ops.matrices import GATE_MAT_DICT, get_global_precision


def _as_probability(value: float | torch.Tensor, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if bool(torch.any(tensor < 0)) or bool(torch.any(tensor > 1)):
        raise ValueError(f"{name} must be between 0 and 1.")
    return tensor


def _complex_dtype(dtype: torch.dtype | None = None) -> torch.dtype:
    return dtype or get_global_precision()


def _real_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.float64 if dtype == torch.complex128 else torch.float32


@dataclass(frozen=True)
class KrausChannel:
    """A completely-positive trace-preserving channel represented by Kraus ops."""

    name: str
    kraus: tuple[torch.Tensor, ...]

    @property
    def n_wires(self) -> int:
        size = int(self.kraus[0].shape[-1])
        return int(torch.log2(torch.tensor(size, dtype=torch.float32)).item())

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
        )


@dataclass(frozen=True)
class NoiseRule:
    gate_names: tuple[str, ...]
    channel: KrausChannel
    wires: tuple[int, ...] | None = None


@dataclass
class NoiseModel:
    """Attach native FlagQuantum channels after selected circuit instructions."""

    rules: list[NoiseRule] = field(default_factory=list)

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
        if wires is None:
            target_wires = None
        elif isinstance(wires, int):
            target_wires = (wires,)
        else:
            target_wires = tuple(int(wire) for wire in wires)
        self.rules.append(NoiseRule(names, channel, target_wires))
        return self

    def channels_for(
        self, instruction: Instruction
    ) -> Iterator[tuple[KrausChannel, tuple[int, ...]]]:
        for rule in self.rules:
            if instruction.name.lower() not in rule.gate_names:
                continue
            if rule.wires is not None:
                yield rule.channel, rule.wires
            elif rule.channel.n_wires == len(instruction.wires):
                yield rule.channel, instruction.wires
            elif rule.channel.n_wires == 1:
                for wire in instruction.wires:
                    yield rule.channel, (wire,)
            else:
                raise ValueError(
                    f"Channel {rule.channel.name!r} cannot be inferred for "
                    f"instruction {instruction.name!r} on wires {instruction.wires}."
                )


def bit_flip_channel(
    probability: float | torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> KrausChannel:
    p = _as_probability(probability, "probability").to(
        dtype=_real_dtype(_complex_dtype(dtype))
    )
    i = GATE_MAT_DICT["i"].to(device=device, dtype=_complex_dtype(dtype))
    x = GATE_MAT_DICT["x"].to(device=device, dtype=_complex_dtype(dtype))
    return KrausChannel("bit_flip", (torch.sqrt(1 - p) * i, torch.sqrt(p) * x))


def phase_flip_channel(
    probability: float | torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> KrausChannel:
    p = _as_probability(probability, "probability").to(
        dtype=_real_dtype(_complex_dtype(dtype))
    )
    i = GATE_MAT_DICT["i"].to(device=device, dtype=_complex_dtype(dtype))
    z = GATE_MAT_DICT["z"].to(device=device, dtype=_complex_dtype(dtype))
    return KrausChannel("phase_flip", (torch.sqrt(1 - p) * i, torch.sqrt(p) * z))


def depolarizing_channel(
    probability: float | torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> KrausChannel:
    p = _as_probability(probability, "probability").to(
        dtype=_real_dtype(_complex_dtype(dtype))
    )
    out_dtype = _complex_dtype(dtype)
    scale = torch.sqrt(p / 3)
    return KrausChannel(
        "depolarizing",
        (
            torch.sqrt(1 - p) * GATE_MAT_DICT["i"].to(device=device, dtype=out_dtype),
            scale * GATE_MAT_DICT["x"].to(device=device, dtype=out_dtype),
            scale * GATE_MAT_DICT["y"].to(device=device, dtype=out_dtype),
            scale * GATE_MAT_DICT["z"].to(device=device, dtype=out_dtype),
        ),
    )


def amplitude_damping_channel(
    gamma: float | torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> KrausChannel:
    g = _as_probability(gamma, "gamma").to(
        dtype=_real_dtype(_complex_dtype(dtype)), device=device
    )
    out_dtype = _complex_dtype(dtype)
    zero = torch.zeros((), dtype=_real_dtype(out_dtype), device=device)
    one = torch.ones((), dtype=_real_dtype(out_dtype), device=device)
    k0 = torch.stack(
        [
            torch.stack([one, zero]),
            torch.stack([zero, torch.sqrt(1 - g)]),
        ]
    ).to(dtype=out_dtype)
    k1 = torch.stack(
        [
            torch.stack([zero, torch.sqrt(g)]),
            torch.stack([zero, zero]),
        ]
    ).to(dtype=out_dtype)
    return KrausChannel("amplitude_damping", (k0, k1))


def density_matrix(circuit_or_state: Any) -> torch.Tensor:
    """Return a batched density matrix from a circuit or statevector."""

    state = (
        circuit_or_state.state()
        if hasattr(circuit_or_state, "state")
        else circuit_or_state
    )
    state = torch.as_tensor(state)
    if state.ndim == 1:
        state = state.reshape(1, -1)
    return state.unsqueeze(-1) * torch.conj(state).unsqueeze(-2)


def _basis_bits(index: int, n_wires: int) -> list[int]:
    return [(index >> (n_wires - 1 - wire)) & 1 for wire in range(n_wires)]


def _bits_to_index(bits: Sequence[int]) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def expand_operator(
    matrix: torch.Tensor,
    wires: Sequence[int],
    n_wires: int,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Expand a k-wire operator into the full Hilbert space."""

    wires = tuple(int(wire) for wire in wires)
    matrix = torch.as_tensor(matrix, dtype=_complex_dtype(dtype), device=device)
    if matrix.ndim == 3:
        return torch.stack(
            [
                expand_operator(
                    item, wires, n_wires, dtype=matrix.dtype, device=matrix.device
                )
                for item in matrix
            ]
        )
    dim = 2**n_wires
    full = torch.zeros(dim, dim, dtype=matrix.dtype, device=matrix.device)
    for col in range(dim):
        bits = _basis_bits(col, n_wires)
        sub_col = _bits_to_index(bits[wire] for wire in wires)
        for sub_row in range(2 ** len(wires)):
            row_bits = list(bits)
            replacement = _basis_bits(sub_row, len(wires))
            for offset, wire in enumerate(wires):
                row_bits[wire] = replacement[offset]
            row = _bits_to_index(row_bits)
            full[row, col] = matrix[sub_row, sub_col]
    return full


def apply_unitary_density(
    rho: torch.Tensor,
    matrix: torch.Tensor,
    wires: Sequence[int],
    n_wires: int,
) -> torch.Tensor:
    """Apply a unitary matrix to a batched density matrix."""

    if rho.ndim == 2:
        rho = rho.reshape(1, *rho.shape)
    full = expand_operator(matrix, wires, n_wires, dtype=rho.dtype, device=rho.device)
    if full.ndim == 2:
        full = full.expand(rho.shape[0], -1, -1)
    return torch.bmm(torch.bmm(full, rho), torch.conj(full).transpose(-1, -2))


def apply_kraus_density(
    rho: torch.Tensor,
    kraus: KrausChannel | Sequence[torch.Tensor],
    wires: Sequence[int],
    n_wires: int,
) -> torch.Tensor:
    """Apply Kraus operators to a batched density matrix."""

    if rho.ndim == 2:
        rho = rho.reshape(1, *rho.shape)
    ops = kraus.kraus if isinstance(kraus, KrausChannel) else tuple(kraus)
    out = torch.zeros_like(rho)
    for op in ops:
        full = expand_operator(op, wires, n_wires, dtype=rho.dtype, device=rho.device)
        if full.ndim == 2:
            full = full.expand(rho.shape[0], -1, -1)
        out = out + torch.bmm(torch.bmm(full, rho), torch.conj(full).transpose(-1, -2))
    return out


def channel_instruction(channel: KrausChannel, wires: Sequence[int]) -> Instruction:
    """Encode a native channel as a FlagQuantum IR instruction."""

    return Instruction(
        name=channel.name,
        wires=tuple(int(wire) for wire in wires),
        matrix=channel.kraus,
        metadata={"is_channel": True},
    )


def lower_noise_model(circuit_or_ir: Any, noise_model: NoiseModel | None) -> CircuitIR:
    """Insert channel instructions after matching unitary instructions."""

    ir = ensure_circuit_ir(circuit_or_ir)
    if noise_model is None:
        return ir
    instructions: list[Instruction] = []
    for instruction in ir:
        instructions.append(instruction)
        for channel, wires in noise_model.channels_for(instruction):
            instructions.append(channel_instruction(channel, wires))
    return replace(ir, instructions=tuple(instructions))


def density_matrix_from_ir(
    circuit_or_ir: Any,
    *,
    bsz: int = 1,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Execute unitary and channel IR instructions as a density matrix."""

    if hasattr(circuit_or_ir, "to_ir"):
        circuit = circuit_or_ir
        ir = circuit.to_ir()
        state = circuit.initial_state()
    elif isinstance(circuit_or_ir, CircuitIR):
        ir = circuit_or_ir
        out_dtype = _complex_dtype(dtype)
        state = torch.zeros(bsz, 2**ir.n_wires, dtype=out_dtype, device=device)
        state[:, 0] = 1
    else:
        raise TypeError("density_matrix_from_ir expects a Circuit or CircuitIR.")

    rho = density_matrix(state)
    for instruction in ir:
        if instruction.metadata.get("is_channel"):
            rho = apply_kraus_density(
                rho,
                instruction.matrix,
                instruction.wires,
                ir.n_wires,
            )
            continue
        matrix = _gate_matrix(instruction, bsz=rho.shape[0], device=rho.device)
        rho = apply_unitary_density(rho, matrix, instruction.wires, ir.n_wires)
    return rho


def noisy_density_matrix(
    circuit_or_ir: Any,
    noise_model: NoiseModel | None = None,
    *,
    bsz: int = 1,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Run a circuit or IR with optional per-instruction native noise channels."""

    lowered = lower_noise_model(circuit_or_ir, noise_model)
    return density_matrix_from_ir(lowered, bsz=bsz, device=device, dtype=dtype)


def expectation_z_density(
    rho: torch.Tensor,
    wires: Iterable[int] | int | None = None,
) -> torch.Tensor:
    """Compute Z expectations from a density matrix."""

    if rho.ndim == 2:
        rho = rho.reshape(1, *rho.shape)
    n_wires = int(torch.log2(torch.tensor(rho.shape[-1], dtype=torch.float32)).item())
    if wires is None:
        target_wires = tuple(range(n_wires))
    elif isinstance(wires, int):
        target_wires = (wires,)
    else:
        target_wires = tuple(int(wire) for wire in wires)

    probs = torch.real(torch.diagonal(rho, dim1=-2, dim2=-1))
    shaped = probs.reshape((rho.shape[0],) + (2,) * n_wires)
    values = []
    for wire in target_wires:
        axes = tuple(axis for axis in range(1, n_wires + 1) if axis != wire + 1)
        marginal = shaped.sum(dim=axes) if axes else shaped
        values.append(marginal[:, 0] - marginal[:, 1])
    return torch.stack(values, dim=-1)


__all__ = [
    "KrausChannel",
    "NoiseModel",
    "NoiseRule",
    "amplitude_damping_channel",
    "apply_kraus_density",
    "apply_unitary_density",
    "bit_flip_channel",
    "channel_instruction",
    "density_matrix",
    "density_matrix_from_ir",
    "depolarizing_channel",
    "expectation_z_density",
    "expand_operator",
    "lower_noise_model",
    "noisy_density_matrix",
    "phase_flip_channel",
]

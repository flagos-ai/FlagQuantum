"""Constrained matrix-product-state TEBD execution.

This module intentionally exposes a narrow, fail-closed first contract:
single-device, second-order imaginary-time evolution for static open-chain
Hamiltonians containing real one-site and adjacent two-site Pauli terms.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any

import torch

from ...algorithms.core import Hamiltonian
from ..matrices import GATE_MAT_DICT
from .models import MPSConfig
from .state import MPSState

_SUPPORTED_INITIAL_STATES = {"+x", "+z", "-z", "neel_z", "domain_wall"}
_MAX_STEPS = 100_000
_MAX_BOND = 8_192


@dataclass(frozen=True)
class TEBDResult:
    """Auditable result from :func:`run_tebd`.

    ``state`` remains available for native observable evaluation but is omitted
    from :meth:`to_dict` so the metadata is JSON serializable without dense
    state materialization.
    """

    state: MPSState = field(repr=False, compare=False)
    program_sha256: str
    n_wires: int
    term_count: int
    steps: int
    total_time: float
    time_step: float
    max_bond: int
    cutoff: float
    initial_state: str
    energy_history: tuple[float, ...]
    pre_normalization_squared_norms: tuple[float, ...]
    post_normalization_squared_norm_errors: tuple[float, ...]
    cumulative_discarded_weight: float
    maximum_bond_dimension: int
    actual_device: str
    precision: str
    schema: str = "flagquantum.tebd_result.v1"
    backend: str = "mps_tebd"
    evolution: str = "imaginary_time"
    order: int = 2
    distribution_semantics: str = "single_device_fast_path"
    scalability_claim_allowed: bool = False
    fallback_used: bool = False

    @property
    def initial_energy(self) -> float:
        return self.energy_history[0]

    @property
    def final_energy(self) -> float:
        return self.energy_history[-1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "program_sha256": self.program_sha256,
            "backend": self.backend,
            "evolution": self.evolution,
            "order": self.order,
            "n_wires": self.n_wires,
            "term_count": self.term_count,
            "steps": self.steps,
            "total_time": self.total_time,
            "time_step": self.time_step,
            "max_bond": self.max_bond,
            "cutoff": self.cutoff,
            "initial_state": self.initial_state,
            "energy_history": list(self.energy_history),
            "initial_energy": self.initial_energy,
            "final_energy": self.final_energy,
            "pre_normalization_squared_norms": list(
                self.pre_normalization_squared_norms
            ),
            "post_normalization_squared_norm_errors": list(
                self.post_normalization_squared_norm_errors
            ),
            "cumulative_discarded_weight": self.cumulative_discarded_weight,
            "maximum_bond_dimension": self.maximum_bond_dimension,
            "actual_device": self.actual_device,
            "precision": self.precision,
            "distribution_semantics": self.distribution_semantics,
            "scalability_claim_allowed": self.scalability_claim_allowed,
            "fallback_used": self.fallback_used,
        }


def _real_scalar(value: Any, *, label: str) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{label} must be a scalar.")
        if value.requires_grad:
            raise ValueError("run_tebd does not support differentiable coefficients.")
        scalar = complex(value.detach().cpu().item())
    else:
        try:
            scalar = complex(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must be a scalar.") from exc
    if not math.isfinite(scalar.real) or not math.isfinite(scalar.imag):
        raise ValueError(f"{label} must be finite.")
    if abs(scalar.imag) > 1e-12:
        raise ValueError(f"{label} must be real for imaginary-time TEBD.")
    return float(scalar.real)


def _validate_inputs(
    hamiltonian: Hamiltonian,
    *,
    n_wires: int,
    total_time: float,
    time_step: float,
    max_bond: int,
    cutoff: float,
    dtype: torch.dtype,
    evolution: str,
    order: int,
    initial_state: str,
) -> tuple[int, float, float, float]:
    if not isinstance(hamiltonian, Hamiltonian):
        raise TypeError("hamiltonian must be a FlagQuantum Hamiltonian.")
    if isinstance(n_wires, bool) or not isinstance(n_wires, int) or n_wires < 2:
        raise ValueError("n_wires must be an integer greater than or equal to 2.")
    if hamiltonian.n_wires > n_wires:
        raise ValueError("Hamiltonian references a wire outside n_wires.")
    if evolution != "imaginary_time":
        raise ValueError("Only evolution='imaginary_time' is supported.")
    if order != 2:
        raise ValueError("Only second-order Strang TEBD is supported.")
    if initial_state not in _SUPPORTED_INITIAL_STATES:
        supported = ", ".join(sorted(_SUPPORTED_INITIAL_STATES))
        raise ValueError(f"Unsupported initial_state; choose one of: {supported}.")
    if dtype not in {torch.complex64, torch.complex128}:
        raise ValueError("dtype must be torch.complex64 or torch.complex128.")
    if isinstance(max_bond, bool) or not isinstance(max_bond, int):
        raise ValueError("max_bond must be an integer.")
    if max_bond < 1 or max_bond > _MAX_BOND:
        raise ValueError(f"max_bond must be in [1, {_MAX_BOND}].")
    cutoff_value = _real_scalar(cutoff, label="cutoff")
    if cutoff_value < 0 or cutoff_value >= 1:
        raise ValueError("cutoff must be in [0, 1).")
    total = _real_scalar(total_time, label="total_time")
    step = _real_scalar(time_step, label="time_step")
    if total <= 0 or step <= 0:
        raise ValueError("total_time and time_step must be positive.")
    ratio = total / step
    steps = int(round(ratio))
    if steps < 1 or steps > _MAX_STEPS:
        raise ValueError(f"TEBD step count must be in [1, {_MAX_STEPS}].")
    if not math.isclose(ratio, steps, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("total_time must be an integer multiple of time_step.")
    return steps, total, step, cutoff_value


def _product_state(
    n_wires: int,
    initial_state: str,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    config: MPSConfig,
) -> MPSState:
    inv_sqrt_two = 1 / math.sqrt(2)
    tensors = []
    for wire in range(n_wires):
        if initial_state == "+x":
            values = (inv_sqrt_two, inv_sqrt_two)
        elif initial_state == "+z":
            values = (1.0, 0.0)
        elif initial_state == "-z":
            values = (0.0, 1.0)
        elif initial_state == "neel_z":
            values = (1.0, 0.0) if wire % 2 == 0 else (0.0, 1.0)
        else:
            values = (1.0, 0.0) if wire < n_wires // 2 else (0.0, 1.0)
        tensor = torch.as_tensor(values, device=device, dtype=dtype).reshape(1, 1, 2, 1)
        tensors.append(tensor)
    state = MPSState(tensors, config=config)
    state.orthogonality_center = 0
    return state


def _local_layers(
    hamiltonian: Hamiltonian,
    *,
    n_wires: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> tuple[tuple[tuple[int, torch.Tensor], ...], ...]:
    one_site: dict[int, torch.Tensor] = {}
    two_site: dict[int, torch.Tensor] = {}
    for index, term in enumerate(hamiltonian.terms):
        coefficient = _real_scalar(term.coefficient, label=f"term[{index}].coefficient")
        ops = tuple(term.ops)
        if len(ops) == 0:
            continue
        if len(ops) > 2:
            raise ValueError("TEBD supports only one-site and two-site Pauli terms.")
        wires = tuple(wire for wire, _ in ops)
        if any(wire < 0 or wire >= n_wires for wire in wires):
            raise ValueError("Hamiltonian references a wire outside n_wires.")
        if len(wires) == 2 and wires[1] != wires[0] + 1:
            raise ValueError(
                "Two-site TEBD terms must act on adjacent open-chain wires."
            )
        local = torch.ones(1, 1, dtype=dtype, device=device)
        for _, name in ops:
            local = torch.kron(
                local,
                GATE_MAT_DICT[name].to(device=device, dtype=dtype),
            )
        local = coefficient * local
        target = one_site if len(wires) == 1 else two_site
        key = wires[0]
        target[key] = target.get(key, torch.zeros_like(local)) + local

    layers: list[tuple[tuple[int, torch.Tensor], ...]] = []
    if one_site:
        layers.append(tuple(sorted(one_site.items())))
    even = tuple(
        sorted((wire, matrix) for wire, matrix in two_site.items() if wire % 2 == 0)
    )
    odd = tuple(
        sorted((wire, matrix) for wire, matrix in two_site.items() if wire % 2 == 1)
    )
    if even:
        layers.append(even)
    if odd:
        layers.append(odd)
    if not layers:
        raise ValueError("Hamiltonian has no non-identity TEBD terms.")
    return tuple(layers)


def _apply_layer(
    state: MPSState,
    layer: tuple[tuple[int, torch.Tensor], ...],
    duration: float,
) -> None:
    for wire, local_hamiltonian in layer:
        gate = torch.matrix_exp(-duration * local_hamiltonian)
        if gate.shape == (2, 2):
            state.apply_one(gate, wire)
        else:
            state.apply_two(gate, wire)


def _finite_scalar(value: torch.Tensor, *, label: str) -> float:
    scalar = float(value.detach().reshape(-1)[0].cpu().item())
    if not math.isfinite(scalar):
        raise RuntimeError(f"TEBD produced non-finite {label}.")
    return scalar


def _normalization_tolerance(dtype: torch.dtype) -> float:
    return 5e-5 if dtype == torch.complex64 else 1e-10


def _program_hash(
    hamiltonian: Hamiltonian,
    **parameters: Any,
) -> str:
    terms = []
    for term in hamiltonian.terms:
        terms.append(
            {
                "coefficient": _real_scalar(term.coefficient, label="coefficient"),
                "ops": [[int(wire), name] for wire, name in term.ops],
            }
        )
    payload = {"schema": "flagquantum.tebd_program.v1", "terms": terms, **parameters}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def run_tebd(
    hamiltonian: Hamiltonian,
    *,
    n_wires: int,
    total_time: float,
    time_step: float,
    max_bond: int,
    cutoff: float = 0.0,
    initial_state: str = "+x",
    evolution: str = "imaginary_time",
    order: int = 2,
    dtype: torch.dtype = torch.complex128,
    device: torch.device | str = "cpu",
) -> TEBDResult:
    """Run constrained second-order imaginary-time TEBD on an MPS.

    The supported contract is deliberately narrow and uses no dense fallback.
    It accepts real, static Pauli Hamiltonians whose non-identity terms act on
    one site or one adjacent open-chain bond. Gradients, batching, real-time
    evolution, periodic/nonlocal terms, TDVP, and distributed execution are
    unsupported.
    """

    steps, total, dt, cutoff_value = _validate_inputs(
        hamiltonian,
        n_wires=n_wires,
        total_time=total_time,
        time_step=time_step,
        max_bond=max_bond,
        cutoff=cutoff,
        dtype=dtype,
        evolution=evolution,
        order=order,
        initial_state=initial_state,
    )
    resolved_device = torch.device(device)
    config = MPSConfig(max_bond=max_bond, cutoff=cutoff_value)
    state = _product_state(
        n_wires,
        initial_state,
        device=resolved_device,
        dtype=dtype,
        config=config,
    )
    layers = _local_layers(
        hamiltonian,
        n_wires=n_wires,
        device=resolved_device,
        dtype=dtype,
    )
    energies = [_finite_scalar(hamiltonian.expectation(state), label="energy")]
    pre_norms: list[float] = []
    post_norm_errors: list[float] = []
    maximum_bond = state.max_bond

    for _ in range(steps):
        record_start = len(state.truncation_records)
        for layer in layers[:-1]:
            _apply_layer(state, layer, 0.5 * dt)
        _apply_layer(state, layers[-1], dt)
        for layer in reversed(layers[:-1]):
            _apply_layer(state, layer, 0.5 * dt)

        new_records = state.truncation_records[record_start:]
        if any(not math.isfinite(record.discarded_weight) for record in new_records):
            raise RuntimeError(
                "TEBD truncation discarded weight is unavailable or non-finite."
            )
        pre_norm = _finite_scalar(state.state_norm(), label="squared state norm")
        if pre_norm <= 0.0:
            raise RuntimeError("TEBD produced a non-positive squared state norm.")
        pre_norms.append(pre_norm)
        state._normalize()
        post_norm = _finite_scalar(
            state.state_norm(), label="normalized squared state norm"
        )
        post_norm_error = abs(post_norm - 1.0)
        if post_norm_error > _normalization_tolerance(dtype):
            raise RuntimeError(
                "TEBD normalization error exceeds the dtype-specific tolerance: "
                f"{post_norm_error:.3e}."
            )
        post_norm_errors.append(post_norm_error)
        energies.append(_finite_scalar(hamiltonian.expectation(state), label="energy"))
        maximum_bond = max(maximum_bond, state.max_bond)

    discarded_weight = float(
        sum(record.discarded_weight for record in state.truncation_records)
    )
    if not math.isfinite(discarded_weight):
        raise RuntimeError("TEBD cumulative discarded weight is non-finite.")
    program_sha256 = _program_hash(
        hamiltonian,
        n_wires=n_wires,
        total_time=total,
        time_step=dt,
        max_bond=max_bond,
        cutoff=cutoff_value,
        initial_state=initial_state,
        evolution=evolution,
        order=order,
        dtype=str(dtype),
    )
    return TEBDResult(
        state=state,
        program_sha256=program_sha256,
        n_wires=n_wires,
        term_count=hamiltonian.n_terms,
        steps=steps,
        total_time=total,
        time_step=dt,
        max_bond=max_bond,
        cutoff=cutoff_value,
        initial_state=initial_state,
        energy_history=tuple(energies),
        pre_normalization_squared_norms=tuple(pre_norms),
        post_normalization_squared_norm_errors=tuple(post_norm_errors),
        cumulative_discarded_weight=discarded_weight,
        maximum_bond_dimension=maximum_bond,
        actual_device=str(state.device),
        precision=str(dtype).removeprefix("torch."),
    )

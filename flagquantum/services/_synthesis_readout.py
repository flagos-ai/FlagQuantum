"""Bounded named views over existing exact local CPU execution results."""

from __future__ import annotations

import math
import time
from collections.abc import Iterator, Mapping, Sequence
from types import MappingProxyType
from typing import TypedDict

import torch

from ..core.ir import MeasurementNode
from ..runtime.measurements import execute_measurements
from ..runtime.result import ExecutionResult
from ._synthesis_execution import CheckedExecution


class _ReadDiagnostics(TypedDict):
    selected_bits: int
    scanned_amplitudes: int
    possible_keys: int
    materialized_keys: int
    omitted_probability: float
    probability_mass: float
    elapsed_seconds: float


class Distribution(Mapping[tuple[int, ...], float]):
    """Read-only probabilities with separate filtering and read-cost diagnostics."""

    def __init__(
        self,
        values: Mapping[tuple[int, ...], float],
        diagnostics: _ReadDiagnostics,
    ) -> None:
        self._values = MappingProxyType(dict(values))
        self.diagnostics = MappingProxyType(dict(diagnostics))

    def __getitem__(self, key: tuple[int, ...]) -> float:
        return self._values[key]

    def __iter__(self) -> Iterator[tuple[int, ...]]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)


def probabilities(
    result: ExecutionResult,
    *,
    checked: CheckedExecution,
    registers: Sequence[str],
    max_bits: int,
    max_keys: int,
    max_amplitudes: int,
    threshold: float,
) -> Distribution:
    """Validate identity and size before requesting an existing dense marginal."""
    started = time.perf_counter()
    for name, limit in (
        ("max_bits", max_bits),
        ("max_keys", max_keys),
        ("max_amplitudes", max_amplitudes),
    ):
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError(f"{name} must be a positive Python integer")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(threshold)
        or not 0 <= threshold < 1
    ):
        raise ValueError("threshold must be finite and between zero and one")
    mapping = checked.registers
    if (
        not isinstance(registers, Sequence)
        or not registers
        or isinstance(registers, str)
        or any(not isinstance(name, str) for name in registers)
        or len(set(registers)) != len(registers)
        or any(name not in mapping for name in registers)
    ):
        raise ValueError(
            "select distinct declared register names, such as ('x', 'result')"
        )
    resources = checked.verify_result(result)
    state = result.state
    assert isinstance(state, torch.Tensor)
    selected = tuple(q for name in registers for q in mapping[name])
    bits = len(selected)
    if bits > max_bits or 1 << bits > max_keys or state.numel() > max_amplitudes:
        raise ValueError(
            f"readout budget exceeded: {bits} selected bits, {1 << bits} possible keys, "
            f"{state.numel()} amplitudes; limits are {max_bits}, {max_keys}, {max_amplitudes}. "
            "Select fewer registers or explicitly increase the appropriate read limit."
        )
    with torch.no_grad():
        mass = float(torch.linalg.vector_norm(state).square().item())
        tolerance = 1e-6 if state.dtype == torch.complex64 else 1e-10
        if not math.isfinite(mass) or abs(mass - 1) > tolerance:
            raise ValueError(
                "statevector is not normalized; named reading will not repair or renormalize it"
            )
        measured = execute_measurements(
            state.reshape(1, -1),
            (MeasurementNode("probabilities", selected),),
            n_qubits=resources.n_qubits,
        )
        marginal = measured[0].value
        if not isinstance(marginal, torch.Tensor):
            raise TypeError("the existing probability reader did not return a tensor")
        probabilities = tuple(float(p) for p in marginal.reshape(-1).tolist())
    widths = tuple(len(mapping[name]) for name in registers)
    values: dict[tuple[int, ...], float] = {}
    omitted = 0.0
    for index, probability in enumerate(probabilities):
        if probability <= threshold:
            omitted += probability
            continue
        shift = bits
        key = []
        for width in widths:
            shift -= width
            key.append((index >> shift) & ((1 << width) - 1))
        values[tuple(key)] = probability
    diagnostics: _ReadDiagnostics = {
        "selected_bits": bits,
        "scanned_amplitudes": state.numel(),
        "possible_keys": 1 << bits,
        "materialized_keys": len(values),
        "omitted_probability": omitted,
        "probability_mass": sum(probabilities),
        "elapsed_seconds": time.perf_counter() - started,
    }
    return Distribution(values, diagnostics)

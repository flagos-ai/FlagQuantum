"""Backend-neutral execution of FlagQuantum IR measurement requests."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Callable, Sequence

import torch

from ..core.ir import MeasurementNode
from ..errors import CapabilityError
from .result import MeasurementResult

_SUPPORTED_KINDS = {
    "counts",
    "counts_ps",
    "expectation_identity",
    "expectation_ps",
    "expectation_z",
    "probabilities",
    "sample",
    "sample_ps",
}
_DEFAULT_MAX_MARGINAL_WIRES = 8


def _validate_wires(wires: Sequence[int], n_wires: int) -> tuple[int, ...]:
    normalized = tuple(int(wire) for wire in wires)
    if any(wire < 0 or wire >= n_wires for wire in normalized):
        raise ValueError(
            f"measurement wires {normalized!r} are outside a {n_wires}-qubit circuit"
        )
    if len(set(normalized)) != len(normalized):
        raise ValueError("measurement wires must be unique")
    return normalized


def _statevector_target(output: Any, n_wires: int) -> Any:
    if not isinstance(output, torch.Tensor):
        return output
    if output.is_complex() and output.ndim == 2 and output.shape[-1] == 2**n_wires:
        from ..circuit import Circuit

        return Circuit(
            n_qubits=n_wires,
            bsz=int(output.shape[0]),
            device=output.device,
            dtype=output.dtype,
            inputs=output,
        )
    raise CapabilityError(
        "measurements for this tensor result are not implemented; "
        "request a statevector, MPS, or tensor-network execution mode"
    )


def _density_matrix_probabilities(
    output: torch.Tensor,
    wires: tuple[int, ...],
    *,
    n_wires: int,
    noise_model: Any | None,
) -> torch.Tensor | None:
    dimension = 2**n_wires
    if (
        not output.is_complex()
        or output.ndim != 3
        or output.shape[-2:] != (dimension, dimension)
    ):
        return None
    probabilities = torch.real(torch.diagonal(output, dim1=-2, dim2=-1))
    if noise_model is not None:
        probabilities = noise_model.apply_readout_probabilities(
            probabilities,
            n_wires=n_wires,
        )
    shaped = probabilities.reshape((output.shape[0],) + (2,) * n_wires)
    unselected_axes = tuple(wire + 1 for wire in range(n_wires) if wire not in wires)
    marginal = shaped.sum(dim=unselected_axes) if unselected_axes else shaped
    current_order = tuple(sorted(wires))
    if wires != current_order:
        permutation = (0,) + tuple(current_order.index(wire) + 1 for wire in wires)
        marginal = marginal.permute(permutation)
    return marginal.reshape(output.shape[0], 2 ** len(wires))


def _generator(target: Any, metadata: dict[str, Any]) -> torch.Generator | None:
    seed = metadata.get("seed")
    if seed is None:
        return None
    device = getattr(target, "device", "cpu")
    device_type = torch.device(device).type
    generator = torch.Generator(device=device_type)
    generator.manual_seed(int(seed))
    return generator


def _postselection(metadata: dict[str, Any], n_wires: int) -> dict[int, int]:
    raw = metadata.get("postselect", {})
    if not isinstance(raw, dict):
        raise TypeError("measurement postselect metadata must be a wire-to-bit mapping")
    conditions = {int(wire): int(bit) for wire, bit in raw.items()}
    _validate_wires(tuple(conditions), n_wires)
    if any(bit not in {0, 1} for bit in conditions.values()):
        raise ValueError("postselection bits must be 0 or 1")
    return conditions


def _pauli_axes(metadata: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    return {
        axis: tuple(int(wire) for wire in metadata.get(axis, ()))
        for axis in ("x", "y", "z")
    }


def _validate_sampling_request(
    kind: str,
    wires: tuple[int, ...],
    shots: int | None,
    metadata: dict[str, Any],
    n_wires: int,
) -> None:
    conditions = _postselection(metadata, n_wires)
    if conditions and kind not in {"sample", "counts"}:
        raise ValueError("postselection is currently supported only for sample/counts")
    if kind in {"sample", "sample_ps", "counts", "counts_ps"} and shots is None:
        raise ValueError(f"{kind} measurement requires a positive shots value")
    if kind not in {"probabilities", "sample_ps", "counts_ps"}:
        return
    limit = int(metadata.get("max_marginal_wires", _DEFAULT_MAX_MARGINAL_WIRES))
    if kind == "probabilities" and limit < 1:
        raise ValueError("max_marginal_wires must be positive")
    if len(wires) <= limit:
        return
    if kind == "probabilities":
        raise ValueError(
            f"marginal probabilities over {len(wires)} wires require "
            f"2**{len(wires)} Pauli contractions; increase "
            "max_marginal_wires explicitly to accept that cost"
        )
    raise ValueError(
        f"Pauli-basis sampling over {len(wires)} wires requires "
        f"2**{len(wires)} contractions; the supported limit is {limit}"
    )


def _validate_pauli_request(
    kind: str,
    wires: tuple[int, ...],
    metadata: dict[str, Any],
    n_wires: int,
) -> None:
    if kind not in {"expectation_ps", "sample_ps", "counts_ps"}:
        return
    axes = _pauli_axes(metadata)
    if not any(axes.values()):
        return
    axis_wires = axes["x"] + axes["y"] + axes["z"]
    _validate_wires(axis_wires, n_wires)
    if set(axis_wires) != set(wires):
        raise ValueError(f"{kind} request wires must match metadata x/y/z wires")


def validate_measurements(
    requests: Sequence[MeasurementNode],
    *,
    n_wires: int,
) -> None:
    """Validate all requests before a backend performs any execution."""

    for request in requests:
        kind = request.kind.strip().lower()
        if kind not in _SUPPORTED_KINDS:
            choices = ", ".join(sorted(_SUPPORTED_KINDS))
            raise ValueError(
                f"unsupported measurement kind {kind!r}; expected {choices}"
            )
        wires = _validate_wires(
            request.wires or tuple(range(n_wires)),
            n_wires,
        )
        metadata = dict(request.metadata)
        _validate_sampling_request(kind, wires, request.shots, metadata, n_wires)
        _validate_pauli_request(kind, wires, metadata, n_wires)


def _collect_postselected_samples(
    sampler: Callable[..., torch.Tensor],
    *,
    shots: int,
    conditions: dict[int, int],
    max_draws: int,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, list[int], int]:
    retained: list[list[torch.Tensor]] | None = None
    accepted_counts: list[int] = []
    draws = 0
    while draws < max_draws:
        draw_count = min(max(int(shots), 32), max_draws - draws)
        batch = sampler(draw_count, generator=generator, format="bits")
        if retained is None:
            retained = [[] for _ in range(int(batch.shape[0]))]
            accepted_counts = [0] * int(batch.shape[0])
        mask = torch.ones(batch.shape[:2], dtype=torch.bool, device=batch.device)
        for wire, bit in conditions.items():
            mask &= batch[..., wire] == bit
        for batch_index in range(int(batch.shape[0])):
            accepted = batch[batch_index][mask[batch_index]]
            if accepted.numel():
                retained[batch_index].append(accepted)
                accepted_counts[batch_index] += int(accepted.shape[0])
        draws += draw_count
        if all(count >= shots for count in accepted_counts):
            break
    if retained is None or not all(count >= shots for count in accepted_counts):
        raise RuntimeError(
            "postselection did not retain enough samples within "
            f"{max_draws} draws per batch; increase "
            "max_postselection_draw_multiplier"
        )
    conditioned = torch.stack(
        [torch.cat(parts, dim=0)[:shots] for parts in retained],
        dim=0,
    )
    return conditioned, accepted_counts, draws


def _sample(
    target: Any,
    *,
    shots: int,
    wires: tuple[int, ...],
    metadata: dict[str, Any],
    n_wires: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    sampler = getattr(target, "sample", None)
    if not callable(sampler):
        raise CapabilityError(
            f"{type(target).__name__} does not support computational-basis sampling"
        )
    generator = _generator(target, metadata)
    conditions = _postselection(metadata, n_wires)
    if not conditions:
        samples = sampler(shots, generator=generator, format="bits")
        selected = samples[..., list(wires)]
        return selected, {
            "draws": [int(shots)] * int(selected.shape[0]),
            "acceptance_rate": [1.0] * int(selected.shape[0]),
        }

    multiplier = int(metadata.get("max_postselection_draw_multiplier", 1024))
    if multiplier < 1:
        raise ValueError("max_postselection_draw_multiplier must be positive")
    max_draws = int(shots) * multiplier
    conditioned, accepted_counts, draws = _collect_postselected_samples(
        sampler,
        shots=shots,
        conditions=conditions,
        max_draws=max_draws,
        generator=generator,
    )
    return conditioned[..., list(wires)], {
        "draws": [draws] * len(accepted_counts),
        "accepted_before_truncation": accepted_counts,
        "acceptance_rate": [count / draws for count in accepted_counts],
        "postselection": dict(sorted(conditions.items())),
    }


def _counts_from_samples(
    samples: torch.Tensor,
    *,
    format: str,
) -> list[dict[str | int, int]]:
    if format not in {"bin", "int"}:
        raise ValueError("counts format must be 'bin' or 'int'")
    outputs: list[dict[str | int, int]] = []
    for batch in samples.detach().cpu().tolist():
        encoded: list[str | int] = []
        for bits in batch:
            bitstring = "".join(str(int(bit)) for bit in bits)
            encoded.append(int(bitstring, 2) if format == "int" else bitstring)
        outputs.append(dict(Counter(encoded)))
    return outputs


def _probabilities_from_parity_expectations(
    expectations: Sequence[torch.Tensor],
    *,
    n_wires: int,
) -> torch.Tensor:
    template = expectations[0]
    probabilities = []
    for outcome in range(1 << n_wires):
        value = torch.ones_like(template)
        for mask, term in enumerate(expectations, start=1):
            parity = sum(
                (outcome >> (n_wires - index - 1)) & 1
                for index in range(n_wires)
                if mask & (1 << index)
            )
            value = value + (-term if parity % 2 else term)
        probabilities.append(value / (1 << n_wires))
    return torch.stack(probabilities, dim=-1)


def _sample_pauli_product(
    target: Any,
    *,
    shots: int,
    wires: tuple[int, ...],
    metadata: dict[str, Any],
) -> torch.Tensor:
    expectation = getattr(target, "expectation_ps", None)
    if not callable(expectation):
        raise CapabilityError(
            f"{type(target).__name__} does not support Pauli-basis sampling"
        )
    axes_by_wire = {
        wire: axis for axis, wires in _pauli_axes(metadata).items() for wire in wires
    }
    subset_expectations: list[torch.Tensor] = []
    for mask in range(1, 1 << len(wires)):
        axes = {
            axis: tuple(
                wire
                for index, wire in enumerate(wires)
                if mask & (1 << index) and axes_by_wire[wire] == axis
            )
            for axis in ("x", "y", "z")
        }
        subset_expectations.append(expectation(**axes))
    distribution = _probabilities_from_parity_expectations(
        subset_expectations,
        n_wires=len(wires),
    )
    distribution = torch.clamp(distribution, min=0)
    distribution = distribution / distribution.sum(dim=-1, keepdim=True)
    indices = torch.multinomial(
        distribution,
        shots,
        replacement=True,
        generator=_generator(target, metadata),
    )
    shifts = torch.arange(len(wires) - 1, -1, -1, device=indices.device)
    return ((indices.unsqueeze(-1) >> shifts) & 1).to(torch.int64)


def _shot_statistics(
    samples: torch.Tensor,
    *,
    counts: list[dict[str | int, int]] | None = None,
) -> dict[str, Any]:
    shots = int(samples.shape[1])
    probabilities = samples.to(torch.float32).mean(dim=1)
    standard_error = torch.sqrt(
        torch.clamp(probabilities * (1.0 - probabilities) / shots, min=0)
    )
    result: dict[str, Any] = {
        "shots": shots,
        "bit_one_probability": probabilities.detach().cpu().tolist(),
        "bit_standard_error": standard_error.detach().cpu().tolist(),
    }
    if counts is not None:
        result["outcome_probability"] = [
            {key: value / shots for key, value in row.items()} for row in counts
        ]
        result["outcome_standard_error"] = [
            {
                key: math.sqrt((value / shots) * (1.0 - value / shots) / shots)
                for key, value in row.items()
            }
            for row in counts
        ]
    return result


def _marginal_probabilities(
    target: Any,
    wires: tuple[int, ...],
) -> torch.Tensor:
    expectation = getattr(target, "expectation_ps", None)
    if not callable(expectation):
        raise CapabilityError(
            f"{type(target).__name__} does not support marginal probabilities"
        )
    subset_expectations: list[torch.Tensor] = []
    for mask in range(1, 1 << len(wires)):
        subset = tuple(wire for index, wire in enumerate(wires) if mask & (1 << index))
        subset_expectations.append(expectation(z=subset))
    return _probabilities_from_parity_expectations(
        subset_expectations,
        n_wires=len(wires),
    )


def _execute_shot_measurement(
    target: Any,
    request: MeasurementNode,
    kind: str,
    wires: tuple[int, ...],
    metadata: dict[str, Any],
    n_wires: int,
) -> tuple[torch.Tensor | list[dict[str | int, int]], dict[str, Any]]:
    assert request.shots is not None
    if kind in {"sample_ps", "counts_ps"}:
        samples = _sample_pauli_product(
            target,
            shots=request.shots,
            wires=wires,
            metadata=metadata,
        )
        sampling_statistics: dict[str, Any] = {}
    else:
        samples, sampling_statistics = _sample(
            target,
            shots=request.shots,
            wires=wires,
            metadata=metadata,
            n_wires=n_wires,
        )
    counts = (
        _counts_from_samples(
            samples,
            format=str(metadata.get("format", "bin")),
        )
        if kind in {"counts", "counts_ps"}
        else None
    )
    value = samples if counts is None else counts
    return value, {
        **_shot_statistics(samples, counts=counts),
        **sampling_statistics,
    }


def _execute_analytic_measurement(
    output: Any,
    measurement_target: Callable[[], Any],
    kind: str,
    wires: tuple[int, ...],
    metadata: dict[str, Any],
    n_wires: int,
    noise_model: Any | None,
) -> torch.Tensor:
    if kind == "expectation_identity":
        target = measurement_target()
        method = getattr(target, "expectation_ps", None)
        if not callable(method):
            raise CapabilityError(
                f"{type(target).__name__} does not support expectations"
            )
        return torch.ones_like(method(z=(0,)))
    if kind == "expectation_z":
        target = measurement_target()
        method = getattr(target, "expectation_z", None)
        if not callable(method):
            raise CapabilityError(
                f"{type(target).__name__} does not support Z expectations"
            )
        return method(wires)
    if kind == "expectation_ps":
        target = measurement_target()
        method = getattr(target, "expectation_ps", None)
        if not callable(method):
            raise CapabilityError(
                f"{type(target).__name__} does not support Pauli expectations"
            )
        axes = _pauli_axes(metadata)
        if not any(axes.values()):
            axes["z"] = wires
        return method(**axes)

    probabilities = (
        _density_matrix_probabilities(
            output,
            wires,
            n_wires=n_wires,
            noise_model=noise_model,
        )
        if isinstance(output, torch.Tensor)
        else None
    )
    return (
        _marginal_probabilities(measurement_target(), wires)
        if probabilities is None
        else probabilities
    )


def execute_measurements(
    output: Any,
    requests: Sequence[MeasurementNode],
    *,
    n_wires: int,
    noise_model: Any | None = None,
) -> tuple[MeasurementResult, ...]:
    """Execute ordered measurement requests against one native backend result."""

    if not requests:
        return ()
    validate_measurements(requests, n_wires=n_wires)
    target: Any | None = None

    def measurement_target() -> Any:
        nonlocal target
        if target is None:
            target = _statevector_target(output, n_wires)
        return target

    results: list[MeasurementResult] = []
    for request in requests:
        kind = request.kind.strip().lower()
        metadata = dict(request.metadata)
        wires = _validate_wires(
            request.wires or tuple(range(n_wires)),
            n_wires,
        )

        value: torch.Tensor | list[dict[str | int, int]]
        statistics: dict[str, Any] = {}
        if kind in {
            "expectation_identity",
            "expectation_z",
            "expectation_ps",
            "probabilities",
        }:
            value = _execute_analytic_measurement(
                output,
                measurement_target,
                kind,
                wires,
                metadata,
                n_wires,
                noise_model,
            )
        else:
            target = measurement_target()
            value, statistics = _execute_shot_measurement(
                target,
                request,
                kind,
                wires,
                metadata,
                n_wires,
            )

        results.append(
            MeasurementResult(
                kind=kind,
                wires=wires,
                value=value,
                shots=request.shots,
                metadata=metadata,
                statistics=statistics,
            )
        )
    return tuple(results)


__all__ = ("execute_measurements", "validate_measurements")

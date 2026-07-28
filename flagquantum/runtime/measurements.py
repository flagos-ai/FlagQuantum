"""Backend-neutral execution of FlagQuantum IR measurement requests."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Sequence

import torch

from ..core.ir import MeasurementNode
from .result import MeasurementResult

_SUPPORTED_KINDS = {
    "counts",
    "expectation_ps",
    "expectation_z",
    "probabilities",
    "sample",
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
    if output.is_complex() and output.ndim >= 2 and output.shape[-1] == 2**n_wires:
        from ..circuit import Circuit

        return Circuit(
            n_qubits=n_wires,
            bsz=int(output.shape[0]),
            device=output.device,
            dtype=output.dtype,
            inputs=output,
        )
    raise NotImplementedError(
        "measurements for this tensor result are not implemented; "
        "request a statevector, MPS, or tensor-network execution mode"
    )


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
        conditions = _postselection(metadata, n_wires)
        if conditions and kind not in {"sample", "counts"}:
            raise ValueError(
                "postselection is currently supported only for sample/counts"
            )
        if kind in {"sample", "counts"} and request.shots is None:
            raise ValueError(f"{kind} measurement requires a positive shots value")
        if kind == "probabilities":
            limit = int(metadata.get("max_marginal_wires", _DEFAULT_MAX_MARGINAL_WIRES))
            if limit < 1:
                raise ValueError("max_marginal_wires must be positive")
            if len(wires) > limit:
                raise ValueError(
                    f"marginal probabilities over {len(wires)} wires require "
                    f"2**{len(wires)} Pauli contractions; increase "
                    "max_marginal_wires explicitly to accept that cost"
                )
        if kind == "expectation_ps":
            axes = {
                axis: tuple(int(wire) for wire in metadata.get(axis, ()))
                for axis in ("x", "y", "z")
            }
            if any(axes.values()):
                axis_wires = axes["x"] + axes["y"] + axes["z"]
                _validate_wires(axis_wires, n_wires)
                if set(axis_wires) != set(wires):
                    raise ValueError(
                        "expectation_ps request wires must match metadata x/y/z wires"
                    )


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
        raise NotImplementedError(
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
        raise NotImplementedError(
            f"{type(target).__name__} does not support marginal probabilities"
        )
    subset_expectations: list[torch.Tensor | None] = [None]
    template: torch.Tensor | None = None
    for mask in range(1, 1 << len(wires)):
        subset = tuple(wire for index, wire in enumerate(wires) if mask & (1 << index))
        value = expectation(z=subset)
        template = value
        subset_expectations.append(value)
    assert template is not None
    probabilities = []
    for outcome in range(1 << len(wires)):
        value = torch.ones_like(template)
        for mask, term in enumerate(subset_expectations[1:], start=1):
            parity = sum(
                (outcome >> (len(wires) - index - 1)) & 1
                for index in range(len(wires))
                if mask & (1 << index)
            )
            value = value + (-term if parity % 2 else term)
        probabilities.append(value / (1 << len(wires)))
    return torch.stack(probabilities, dim=-1)


def execute_measurements(
    output: Any,
    requests: Sequence[MeasurementNode],
    *,
    n_wires: int,
) -> tuple[MeasurementResult, ...]:
    """Execute ordered measurement requests against one native backend result."""

    if not requests:
        return ()
    validate_measurements(requests, n_wires=n_wires)
    target = _statevector_target(output, n_wires)
    results: list[MeasurementResult] = []
    for request in requests:
        kind = request.kind.strip().lower()
        metadata = dict(request.metadata)
        wires = _validate_wires(
            request.wires or tuple(range(n_wires)),
            n_wires,
        )

        if kind == "expectation_z":
            method = getattr(target, "expectation_z", None)
            if not callable(method):
                raise NotImplementedError(
                    f"{type(target).__name__} does not support Z expectations"
                )
            value = method(wires)
        elif kind == "expectation_ps":
            method = getattr(target, "expectation_ps", None)
            if not callable(method):
                raise NotImplementedError(
                    f"{type(target).__name__} does not support Pauli expectations"
                )
            axes = {
                axis: tuple(int(wire) for wire in metadata.get(axis, ()))
                for axis in ("x", "y", "z")
            }
            if not any(axes.values()):
                axes["z"] = wires
            value = method(**axes)
        elif kind == "probabilities":
            value = _marginal_probabilities(target, wires)
        else:
            assert request.shots is not None
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
                if kind == "counts"
                else None
            )
            value = samples if counts is None else counts

        results.append(
            MeasurementResult(
                kind=kind,
                wires=wires,
                value=value,
                shots=request.shots,
                metadata=metadata,
                statistics=(
                    {
                        **_shot_statistics(samples, counts=counts),
                        **sampling_statistics,
                    }
                    if kind in {"sample", "counts"}
                    else {}
                ),
            )
        )
    return tuple(results)


__all__ = ("execute_measurements", "validate_measurements")

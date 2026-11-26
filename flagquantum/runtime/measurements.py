"""Backend-neutral execution of FlagQuantum IR measurement requests."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any

import torch

from flagquantum.runtime.results import MeasurementResult

from ..core.ir import MeasurementNode
from ..errors import CapabilityError

_SUPPORTED_KINDS = {
    "counts",
    "counts_ps",
    "density_matrix",
    "expectation_identity",
    "expectation_ps",
    "expectation_z",
    "probabilities",
    "sample",
    "sample_ps",
    "vn_entropy",
}
_DEFAULT_MAX_MARGINAL_WIRES = 8
# Building a density matrix from a statevector of ``n`` qubits writes ``4 ** n``
# entries, the square of the statevector it comes from, so the step is bounded
# where a marginal over the same qubits is not. Ten qubits is 2,097,152 complex
# values -- 16 MiB at complex64 and 32 MiB at complex128 -- and each further
# qubit multiplies that by four. The limit is on that construction alone: a
# density-mode run already holds the matrix, so reducing it is one pass over
# memory the run has already paid for and is not bounded here.
_DEFAULT_MAX_DENSITY_MATRIX_QUBITS = 10


def _validate_wires(wires: Sequence[int], n_wires: int) -> tuple[int, ...]:
    normalized = tuple(int(wire) for wire in wires)
    if any(wire < 0 or wire >= n_wires for wire in normalized):
        raise ValueError(
            f"measurement qubits {normalized!r} are outside a {n_wires}-qubit circuit"
        )
    if len(set(normalized)) != len(normalized):
        raise ValueError("measurement qubits must be unique")
    return normalized


def _statevector_target(
    output: Any,
    n_wires: int,
    noise_model: Any | None = None,
) -> Any:
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
    if (
        output.is_complex()
        and output.ndim == 3
        and output.shape[-2:] == (2**n_wires, 2**n_wires)
    ):
        from flagquantum.simulation import density_matrix as _density_simulation

        return _density_simulation._DensityMatrixTarget(output, noise_model)
    raise CapabilityError(
        "measurements for this tensor result are not implemented; "
        "request a statevector, MPS, or tensor-network execution mode"
    )


def _joint_marginal_probabilities(
    output: Any,
    wires: tuple[int, ...],
    *,
    n_wires: int,
    noise_model: Any | None,
) -> torch.Tensor | None:
    """Reduce a dense result to one joint marginal, or ``None`` if it is not one."""

    from flagquantum.simulation.numerics import probabilities as _probability_numerics

    if (
        noise_model is None
        and isinstance(output, torch.Tensor)
        and output.is_complex()
        and output.ndim == 2
        and output.shape[-1] == 2**n_wires
    ):
        if wires == tuple(range(n_wires)):
            from .statevector_probability_dispatch import (
                _try_apply_cataloged_statevector_probabilities,
            )

            full = _try_apply_cataloged_statevector_probabilities(output)
            if full is not None:
                return _probability_numerics._normalize_joint_marginal(full)

        from .statevector_measurement_dispatch import (
            _try_apply_cataloged_statevector_marginal,
        )

        direct = _try_apply_cataloged_statevector_marginal(output, wires)
        if direct is not None:
            return _probability_numerics._normalize_joint_marginal(direct)

    probabilities = _probability_numerics._ideal_probabilities(output, n_wires)
    if probabilities is None:
        return None
    if noise_model is not None:
        probabilities = noise_model.apply_readout_probabilities(
            probabilities,
            n_qubits=n_wires,
        )
    return _probability_numerics._reduced_joint_marginal(
        probabilities, wires, n_wires=n_wires
    )


def _generator(target: Any, metadata: dict[str, Any]) -> torch.Generator | None:
    seed = metadata.get("seed")
    if seed is None:
        return None
    from ..simulation.tensor_network.state import TensorNetworkState

    if isinstance(target, TensorNetworkState) and target.plan.nodes:
        device = target.plan.nodes[0].tensor.device
    else:
        device = getattr(target, "device", torch.device("cpu"))
    generator = torch.Generator(device=torch.device(device))
    generator.manual_seed(int(seed))
    return generator


def _postselection(metadata: dict[str, Any], n_wires: int) -> dict[int, int]:
    raw = metadata.get("postselect", {})
    if not isinstance(raw, dict):
        raise TypeError(
            "measurement postselect metadata must be a qubit-to-bit mapping"
        )
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
            f"marginal probabilities over {len(wires)} qubits exceed the "
            f"supported limit of {limit}; increase max_marginal_wires "
            "explicitly to raise it"
        )
    raise ValueError(
        f"Pauli-basis sampling over {len(wires)} qubits exceeds the supported "
        f"limit of {limit}; a request that reads any qubit in X or Y needs one "
        "contraction per subset of those qubits, so increase max_marginal_wires "
        "explicitly to accept that cost"
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
        raise ValueError(f"{kind} request qubits must match metadata x/y/z qubits")


def validate_measurements(
    requests: Sequence[MeasurementNode],
    *,
    n_qubits: int,
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
            request.wires or tuple(range(n_qubits)),
            n_qubits,
        )
        metadata = dict(request.metadata)
        _validate_sampling_request(kind, wires, request.shots, metadata, n_qubits)
        _validate_pauli_request(kind, wires, metadata, n_qubits)


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
    noise_model: Any | None,
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
        if noise_model is not None:
            samples = noise_model.apply_readout_samples(
                samples,
                generator=generator,
            )
        selected = samples[..., list(wires)]
        statistics = {
            "draws": [int(shots)] * int(selected.shape[0]),
            "acceptance_rate": [1.0] * int(selected.shape[0]),
        }
        sampling_summary = getattr(target, "sampling_summary", None)
        if callable(sampling_summary):
            statistics.update(sampling_summary())
        return selected, statistics

    multiplier = int(metadata.get("max_postselection_draw_multiplier", 1024))
    if multiplier < 1:
        raise ValueError("max_postselection_draw_multiplier must be positive")
    max_draws = int(shots) * multiplier
    conditioned, accepted_counts, draws = _collect_postselected_samples(
        (
            sampler
            if noise_model is None
            else lambda count, **kwargs: noise_model.apply_readout_samples(
                sampler(count, **kwargs),
                generator=kwargs.get("generator"),
            )
        ),
        shots=shots,
        conditions=conditions,
        max_draws=max_draws,
        generator=generator,
    )
    statistics = {
        "draws": [draws] * len(accepted_counts),
        "accepted_before_truncation": accepted_counts,
        "acceptance_rate": [count / draws for count in accepted_counts],
        "postselection": dict(sorted(conditions.items())),
    }
    sampling_summary = getattr(target, "sampling_summary", None)
    if callable(sampling_summary):
        statistics.update(sampling_summary())
    return conditioned[..., list(wires)], statistics


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


def _sample_pauli_product(
    target: Any,
    *,
    shots: int,
    wires: tuple[int, ...],
    metadata: dict[str, Any],
    state_output: Any,
    n_wires: int,
) -> torch.Tensor:
    axes_by_wire = {
        wire: axis for axis, wires in _pauli_axes(metadata).items() for wire in wires
    }
    distribution = _pauli_basis_distribution(
        target,
        wires=wires,
        axes_by_wire=axes_by_wire,
        state_output=state_output,
        n_wires=n_wires,
    )
    from flagquantum.simulation.numerics.probabilities import (
        _normalize_pauli_sampling_distribution,
        _sample_pauli_distribution,
    )

    distribution = _normalize_pauli_sampling_distribution(distribution)
    return _sample_pauli_distribution(
        distribution,
        shots=shots,
        n_qubits=len(wires),
        generator=_generator(target, metadata),
    )


def _pauli_basis_distribution(
    target: Any,
    *,
    wires: tuple[int, ...],
    axes_by_wire: dict[int, str],
    state_output: Any,
    n_wires: int,
) -> torch.Tensor:
    """The joint distribution of a Pauli-basis readout over ``wires``.

    When every requested qubit is read in the Z basis the distribution *is* the
    computational-basis marginal, so a dense result gives it by reduction. Each
    bit then carries the same meaning on both sides: ``+1`` is an even number of
    set bits, which is a Z eigenvalue of ``+1`` and therefore the computational
    bit ``0``.

    A request that reads any qubit in X or Y is a genuinely different measurement
    and keeps the parity reconstruction, which needs one full-state contraction
    per non-empty subset of ``wires``.
    """

    from flagquantum.simulation.numerics import probabilities as _probability_numerics

    if wires and all(axes_by_wire.get(wire) == "z" for wire in wires):
        reduced = _joint_marginal_probabilities(
            state_output,
            wires,
            n_wires=n_wires,
            noise_model=None,
        )
        if reduced is not None:
            return reduced
    expectation = getattr(target, "expectation_ps", None)
    if not callable(expectation):
        raise CapabilityError(
            f"{type(target).__name__} does not support Pauli-basis sampling"
        )
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
    return _probability_numerics._probabilities_from_parity_expectations(
        subset_expectations,
        n_wires=len(wires),
    )


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
    """Recover a marginal from ``2 ** len(wires) - 1`` parity expectations.

    This is the fallback for a result that is not a dense tensor, which is where
    the MPS and tensor-network targets land, and for a target that exposes
    ``expectation_ps`` without carrying a distribution at all. It is kept rather
    than removed because the expectation surface is duck-typed and narrower than
    the dense-result surface, so deleting it would withdraw a capability instead
    of replacing an implementation.

    The cost is exponential in the number of qubits because each of those
    expectations is a separate contraction over the whole state, and the
    arithmetic is reconstructed rather than read: one term per non-empty subset,
    summed with alternating signs and divided by ``2 ** len(wires)``. Because
    that sum is evaluated in floating point it can return a small negative entry
    for an outcome the state gives probability zero, which is the other reason
    the reduction is preferred where it is available.
    """

    from flagquantum.simulation.numerics import probabilities as _probability_numerics

    expectation = getattr(target, "expectation_ps", None)
    if not callable(expectation):
        raise CapabilityError(
            f"{type(target).__name__} does not support marginal probabilities"
        )
    subset_expectations: list[torch.Tensor] = []
    for mask in range(1, 1 << len(wires)):
        subset = tuple(wire for index, wire in enumerate(wires) if mask & (1 << index))
        subset_expectations.append(expectation(z=subset))
    return _probability_numerics._probabilities_from_parity_expectations(
        subset_expectations,
        n_wires=len(wires),
    )


def _execute_shot_measurement(
    output: Any,
    target: Any,
    request: MeasurementNode,
    kind: str,
    wires: tuple[int, ...],
    metadata: dict[str, Any],
    n_wires: int,
    noise_model: Any | None,
) -> tuple[torch.Tensor | list[dict[str | int, int]], dict[str, Any]]:
    assert request.shots is not None
    if kind in {"sample_ps", "counts_ps"}:
        samples = _sample_pauli_product(
            target,
            shots=request.shots,
            wires=wires,
            metadata=metadata,
            state_output=output,
            n_wires=n_wires,
        )
        sampling_statistics: dict[str, Any] = {}
    else:
        samples, sampling_statistics = _sample(
            target,
            shots=request.shots,
            wires=wires,
            metadata=metadata,
            n_wires=n_wires,
            noise_model=noise_model,
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


def _expectation_method(
    target: Any, name: str, description: str
) -> Callable[..., torch.Tensor]:
    """Return one of a target's expectation methods, or fail as a capability error.

    The expectation surface is duck-typed, so the attribute arrives through
    ``getattr``. Keeping the ``callable`` check means an attribute that exists
    but cannot be called is reported here rather than raised as a ``TypeError``
    at the call site.
    """
    method: Callable[..., torch.Tensor] | None = getattr(target, name, None)
    if not callable(method):
        raise CapabilityError(f"{type(target).__name__} does not support {description}")
    return method


def _is_density_matrix_tensor(output: Any, n_qubits: int) -> bool:
    """Whether a native result already is the batched density matrix asked for."""

    return (
        isinstance(output, torch.Tensor)
        and output.is_complex()
        and output.ndim == 3
        and output.shape[-2:] == (2**n_qubits, 2**n_qubits)
    )


def _density_matrix_output(
    output: Any,
    qubits: tuple[int, ...],
    *,
    n_qubits: int,
    noise_model: Any | None,
) -> torch.Tensor:
    """Return the state on ``qubits`` as a matrix, tracing the rest out.

    Two routes, and the difference is what each one costs. A ``density_matrix``
    run already holds the matrix, so the request is one partial trace over
    memory the run has paid for. Every other mode holds amplitudes, so the
    matrix has to be built first at four entries per basis state -- the square
    of the state it comes from -- and that step is bounded.

    Readout confusion is refused rather than folded in. It is a rule about
    outcomes applied after measurement, so a matrix that included it would
    describe a state the engine never prepared; the same reason the error
    mitigation unit refuses to extrapolate a curve that omits it. The state a
    noisy run returns is the state noise left behind, which is what a density
    matrix means.
    """

    from flagquantum.simulation.density_matrix import (
        density_matrix,
        reduced_density_matrix,
    )

    if getattr(noise_model, "readout_rules", ()):
        raise ValueError(
            "a density-matrix output cannot include readout confusion: readout "
            "acts on measurement outcomes after the state, so a matrix carrying "
            "it would report a state the engine never prepared"
        )
    if _is_density_matrix_tensor(output, n_qubits):
        return reduced_density_matrix(output, qubits)
    if n_qubits > _DEFAULT_MAX_DENSITY_MATRIX_QUBITS:
        entries = 4**n_qubits
        raise ValueError(
            f"a {n_qubits}-qubit density-matrix output built from a statevector "
            f"writes {entries} entries, which exceeds the supported limit of "
            f"{_DEFAULT_MAX_DENSITY_MATRIX_QUBITS} qubits; run the program with "
            "options=fq.ExecutionOptions(mode='density_matrix') to ask for the "
            "smaller reduction instead"
        )
    if isinstance(output, torch.Tensor):
        if output.is_complex() and output.ndim == 2 and output.shape[-1] == 2**n_qubits:
            return reduced_density_matrix(density_matrix(output), qubits)
        raise CapabilityError(
            "a density-matrix output needs a statevector or density-matrix result; "
            f"this execution returned a {tuple(output.shape)} tensor"
        )
    if not callable(getattr(output, "state", None)):
        raise CapabilityError(
            "a density-matrix output from this execution mode is not implemented; "
            "request the statevector or density_matrix mode"
        )
    return reduced_density_matrix(density_matrix(output), qubits)


def _vn_entropy_output(
    output: Any,
    qubits: tuple[int, ...],
    *,
    n_qubits: int,
    noise_model: Any | None,
    log_base: float | None,
) -> torch.Tensor:
    """Return the von Neumann entropy of ``qubits``, in nats unless a base is named.

    Two routes, chosen by what the execution hands over rather than by the mode string the
    caller wrote, because what a mode is called is a plan-level fact and what it returned is
    the measured one.

    A matrix-product state holds this execution's state already split at every bond, so the
    named sites are contracted out of its canonical form and the resulting operator is
    diagonalised: ``4 ** len(qubits)`` entries rather than ``4 ** n``, and the full state is
    never materialised. When the selection is a prefix of the chain the canonical form answers
    with an entanglement spectrum directly -- one singular-value decomposition of a bond matrix
    instead of the reduced matrix it would otherwise build -- which is the arithmetic the MPS
    already paid for when it was constructed.

    Everything else -- a dense statevector, a density matrix, a tensor-network state -- is
    reduced with the density-matrix module's own partial trace and diagonalised. That is
    deliberately the same routine ``fq.density_matrix`` reduces with, so the two requests
    cannot disagree about what "the state on these qubits" means.

    Readout confusion is refused for the same reason the density-matrix output refuses it:
    readout is a rule about outcomes applied after the state, so an entropy computed from a
    state that had readout folded in would describe a state the engine never prepared.
    """

    from flagquantum.simulation.entropy import (
        contiguous_ascending_run,
        entropy_from_density_matrix,
        entropy_from_probabilities,
        mps_prefix_spectrum,
        mps_run_density_matrix,
    )

    if getattr(noise_model, "readout_rules", ()):
        raise ValueError(
            "a von Neumann entropy output cannot include readout confusion: readout acts on "
            "measurement outcomes after the state, so an entropy carrying it would describe a "
            "state the engine never prepared"
        )

    if callable(getattr(output, "move_orthogonality_center", None)) and (
        getattr(output, "n_qubits", None) == n_qubits
    ):
        run = contiguous_ascending_run(qubits, n_qubits)
        if run is not None:
            if run == (0, len(qubits) - 1):
                spectrum = mps_prefix_spectrum(output, len(qubits))
                if spectrum is not None:
                    return entropy_from_probabilities(spectrum, log_base=log_base)
            return entropy_from_density_matrix(
                mps_run_density_matrix(output, *run), log_base=log_base
            )

    if isinstance(output, torch.Tensor):
        from flagquantum.simulation.density_matrix import (
            density_matrix,
            reduced_density_matrix,
        )

        if output.is_complex() and output.ndim == 2 and output.shape[-1] == 2**n_qubits:
            return entropy_from_density_matrix(
                reduced_density_matrix(density_matrix(output), qubits),
                log_base=log_base,
            )
        if _is_density_matrix_tensor(output, n_qubits):
            return entropy_from_density_matrix(
                reduced_density_matrix(output, qubits), log_base=log_base
            )
        raise CapabilityError(
            "a von Neumann entropy output needs a statevector, density-matrix, MPS, or "
            f"tensor-network result; this execution returned a {tuple(output.shape)} tensor"
        )

    if callable(getattr(output, "state", None)):
        from flagquantum.simulation.density_matrix import (
            density_matrix,
            reduced_density_matrix,
        )

        return entropy_from_density_matrix(
            reduced_density_matrix(density_matrix(output), qubits),
            log_base=log_base,
        )

    raise CapabilityError(
        "a von Neumann entropy output from this execution mode is not implemented; request "
        "the statevector, density_matrix, or mps mode"
    )


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
        method = _expectation_method(
            measurement_target(), "expectation_ps", "expectations"
        )
        return torch.ones_like(method(z=(0,)))
    if kind == "expectation_z":
        method = _expectation_method(
            measurement_target(), "expectation_z", "Z expectations"
        )
        return method(wires)
    if kind == "expectation_ps":
        method = _expectation_method(
            measurement_target(), "expectation_ps", "Pauli expectations"
        )
        axes = _pauli_axes(metadata)
        if not any(axes.values()):
            axes["z"] = wires
        return method(**axes)
    if kind == "density_matrix":
        return _density_matrix_output(
            output, wires, n_qubits=n_wires, noise_model=noise_model
        )
    if kind == "vn_entropy":
        return _vn_entropy_output(
            output,
            wires,
            n_qubits=n_wires,
            noise_model=noise_model,
            log_base=metadata.get("fq_log_base"),
        )

    if kind == "probabilities" and noise_model is None:
        from ..simulation.mps.bounded_probabilities import (
            _bounded_full_width_probabilities,
        )

        bounded = _bounded_full_width_probabilities(output, wires)
        if bounded is not None:
            return bounded

    probabilities = _joint_marginal_probabilities(
        output,
        wires,
        n_wires=n_wires,
        noise_model=noise_model,
    )
    if probabilities is not None:
        return probabilities
    return _marginal_probabilities(measurement_target(), wires)


def execute_measurements(
    output: Any,
    requests: Sequence[MeasurementNode],
    *,
    n_qubits: int,
    noise_model: Any | None = None,
) -> tuple[MeasurementResult, ...]:
    """Execute ordered measurement requests against one native backend result."""

    if not requests:
        return ()
    validate_measurements(requests, n_qubits=n_qubits)
    target: Any | None = None

    def measurement_target() -> Any:
        nonlocal target
        if target is None:
            target = _statevector_target(output, n_qubits, noise_model)
        return target

    results: list[MeasurementResult] = []
    for request in requests:
        kind = request.kind.strip().lower()
        metadata = dict(request.metadata)
        wires = _validate_wires(
            request.wires or tuple(range(n_qubits)),
            n_qubits,
        )

        value: torch.Tensor | list[dict[str | int, int]]
        statistics: dict[str, Any] = {}
        if kind in {
            "density_matrix",
            "expectation_identity",
            "expectation_z",
            "expectation_ps",
            "probabilities",
            "vn_entropy",
        }:
            value = _execute_analytic_measurement(
                output,
                measurement_target,
                kind,
                wires,
                metadata,
                n_qubits,
                noise_model,
            )
        else:
            target = measurement_target()
            value, statistics = _execute_shot_measurement(
                output,
                target,
                request,
                kind,
                wires,
                metadata,
                n_qubits,
                noise_model,
            )

        results.append(
            MeasurementResult(
                kind=kind,
                qubits=wires,
                value=value,
                shots=request.shots,
                metadata=metadata,
                statistics=statistics,
            )
        )
    return tuple(results)


__all__ = ("execute_measurements", "validate_measurements")

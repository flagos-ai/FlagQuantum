"""Device-preserving terminal instruments over explicit matrix coordinates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypedDict

import torch

from ..core.ir import _encode_value
from ..errors import CapabilityError, ValidationError
from ..observables.boson import DEFAULT_DENSE_MATRIX_BYTES
from ..operators import SuperOperator
from ..simulation.lindblad_request import _require_density_matrices
from .measurements import _draw_probability_indices
from .result import MeasurementResult


class _BranchRecord(TypedDict):
    label: str
    kraus: list[object]


class _InstrumentRecord(TypedDict):
    schema: str
    version: str
    dimension: int
    dtype: str
    branches: list[_BranchRecord]
    requested_seed: int | None
    effective_seed: int
    torch_version: str


class _TerminalMetadata(TypedDict):
    outcome_labels: list[str]
    probabilities: list[float] | list[list[float]]
    probability_correction: float
    terminal_instrument: _InstrumentRecord


def _require_matrix(matrix: torch.Tensor, *, name: str) -> None:
    """Check the original tensor before any conversion or numeric operation."""

    if not isinstance(matrix, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if matrix.device.type not in {"cpu", "flagos"}:
        raise CapabilityError(f"{name} must be on CPU for terminal sampling")
    if matrix.requires_grad:
        raise CapabilityError(f"{name} must not require gradients for sampling")
    if matrix.layout != torch.strided:
        raise ValidationError(f"{name} must have strided storage")
    if matrix.dtype not in (torch.complex64, torch.complex128):
        raise ValidationError(f"{name} must have complex64 or complex128 dtype")


def _read_branches(
    instrument: Mapping[str, Sequence[torch.Tensor]],
    *,
    dimension: int,
    dtype: torch.dtype,
    device: torch.device,
) -> list[tuple[str, tuple[torch.Tensor, ...]]]:
    """Retain declared order and validate tensor shapes without cloning them."""

    if not isinstance(instrument, Mapping):
        raise TypeError("instrument must be a mapping of labels to Kraus sequences")
    if not instrument:
        raise ValidationError("instrument must have at least one outcome")
    branches: list[tuple[str, tuple[torch.Tensor, ...]]] = []
    labels: set[str] = set()
    for label, operators in instrument.items():
        if not isinstance(label, str) or not label.strip() or label in labels:
            raise ValidationError("instrument labels must be unique nonempty strings")
        labels.add(label)
        if not isinstance(operators, Sequence):
            raise TypeError(f"instrument[{label!r}] must be a finite Kraus sequence")
        if not operators:
            raise ValidationError(f"instrument[{label!r}] must not be empty")
        matrices = tuple(operators)
        for matrix in matrices:
            _require_matrix(matrix, name=f"instrument[{label!r}]")
            if (
                matrix.shape != (dimension, dimension)
                or matrix.dtype != dtype
                or matrix.device != device
            ):
                raise ValidationError(
                    f"instrument[{label!r}] matrices must match the state shape and dtype"
                )
        branches.append((label, matrices))
    return branches


def _instrument_probabilities(
    density_matrix: torch.Tensor,
    branches: list[tuple[str, tuple[torch.Tensor, ...]]],
    tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Check physical inputs and retain original weights beside draw probabilities."""

    matrices = (density_matrix, *(k for _, ops in branches for k in ops))
    if any(not bool(torch.isfinite(matrix).all()) for matrix in matrices):
        raise ValidationError("density_matrix and all Kraus matrices must be finite")
    try:
        _require_density_matrices(density_matrix, tolerance=tolerance)
    except ValidationError as exc:
        raise ValidationError(f"density_matrix: {exc}") from exc
    identity = torch.eye(
        density_matrix.shape[-1],
        dtype=density_matrix.dtype,
        device=density_matrix.device,
    )
    effect = torch.zeros_like(identity)
    for _, operators in branches:
        for matrix in operators:
            effect = effect + matrix.mH @ matrix
    if not torch.allclose(effect, identity, atol=tolerance, rtol=0):
        raise ValidationError(
            "instrument must be complete: sum K^dagger K must equal I"
        )
    weights = torch.stack(
        [
            torch.diagonal(
                SuperOperator.from_kraus(ops).apply(density_matrix), dim1=-2, dim2=-1
            )
            .sum(-1)
            .real
            for _, ops in branches
        ],
        dim=-1,
    )
    if (
        not bool(torch.isfinite(weights).all())
        or bool((weights < -tolerance).any())
        or bool((weights.sum(-1) - 1).abs().gt(tolerance).any())
    ):
        raise ValidationError(
            "instrument branch probabilities must be nonnegative and sum to one"
        )
    probabilities = weights.clamp(min=0)
    probabilities = probabilities / probabilities.sum(-1, keepdim=True)
    return weights, probabilities


def sample_terminal(
    density_matrix: torch.Tensor,
    instrument: Mapping[str, Sequence[torch.Tensor]],
    *,
    shots: int,
    seed: int | None = None,
) -> MeasurementResult:
    """Sample ordered Kraus outcomes on the terminal density matrix device.

    Args:
        density_matrix: Complex CPU or FlagOS state, shaped (D,D) or (B,D,D).
        instrument: Ordered outcome labels and their nonempty Kraus sequences.
            All matrices share the state's dimension and dtype. Their combined
            effect must be the identity; missing outcomes are not postselection.
        shots: Positive Python integer number of samples per state.
        seed: Optional Python integer in [0, 2**64-1]. A private generator on the state device
            records its effective seed without changing global Torch RNG state.

    Returns:
        The existing MeasurementResult with int64 outcome indices and captured
        instrument metadata. A rank-three input retains its batch axis at B=1.
        No conditional state, loss truth, feedback or detector event is inferred.

    Raises:
        TypeError: A state, mapping, Kraus sequence or matrix has the wrong type.
        ValidationError: Shapes, dtypes, labels, shots, seed or numeric values
            violate the approved density/instrument contract.
        CapabilityError: An input is on an unsupported device, requires gradients, or the sum
            of input, probability and sample numeric bytes exceeds the existing
            dense-matrix ceiling. This is not a bound on total process RSS.
    """

    _require_matrix(density_matrix, name="density_matrix")
    shape = density_matrix.shape
    if (
        density_matrix.ndim not in (2, 3)
        or shape[-1] < 1
        or shape[-2] != shape[-1]
        or (density_matrix.ndim == 3 and shape[0] < 1)
    ):
        raise ValidationError(
            "density_matrix must have shape (D,D) or (B,D,D), D,B >= 1"
        )
    if type(shots) is not int or shots < 1:
        raise ValidationError("shots must be a positive Python integer")
    if seed is not None and (type(seed) is not int or not 0 <= seed < 2**64):
        raise ValidationError("seed must be None or a Python integer in [0, 2**64-1]")
    dimension = int(shape[-1])
    batch = int(shape[0]) if density_matrix.ndim == 3 else 1
    branches = _read_branches(
        instrument,
        dimension=dimension,
        dtype=density_matrix.dtype,
        device=density_matrix.device,
    )
    numeric_bytes = density_matrix.numel() * density_matrix.element_size()
    numeric_bytes += sum(
        k.numel() * k.element_size() for _, ops in branches for k in ops
    )
    real_bytes = 4 if density_matrix.dtype == torch.complex64 else 8
    numeric_bytes += batch * len(branches) * real_bytes + batch * shots * 8
    if numeric_bytes > DEFAULT_DENSE_MATRIX_BYTES:
        raise CapabilityError(
            f"terminal sampling needs {numeric_bytes} numeric bytes, exceeding the "
            f"{DEFAULT_DENSE_MATRIX_BYTES}-byte ceiling; reduce dimensions or shots"
        )
    tolerance = 1e-5 if density_matrix.dtype == torch.complex64 else 1e-10
    weights, probabilities = _instrument_probabilities(
        density_matrix, branches, tolerance
    )
    records: list[_BranchRecord] = [
        {"label": label, "kraus": [_encode_value(k) for k in ops]}
        for label, ops in branches
    ]
    generator = torch.Generator(device=density_matrix.device)
    effective_seed = generator.seed() if seed is None else seed
    generator.manual_seed(effective_seed)
    metadata: _TerminalMetadata = {
        "outcome_labels": [label for label, _ in branches],
        "probabilities": weights.detach().cpu().tolist(),
        "probability_correction": float((probabilities - weights).abs().max()),
        "terminal_instrument": {
            "schema": "flagquantum.terminal_instrument",
            "version": "1.0",
            "dimension": dimension,
            "dtype": str(density_matrix.dtype).removeprefix("torch."),
            "branches": records,
            "requested_seed": seed,
            "effective_seed": effective_seed,
            "torch_version": str(torch.__version__),
        },
    }
    return MeasurementResult(
        kind="sample",
        qubits=(),
        value=_draw_probability_indices(probabilities, shots, generator=generator),
        shots=shots,
        metadata=metadata,
    )

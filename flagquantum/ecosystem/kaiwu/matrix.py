"""Validated Ising-matrix helpers at the Kaiwu ecosystem boundary.

Kaiwu solvers evaluate a symmetric matrix with the convention
``E(s) = -s.T @ matrix @ s + bias`` for spins in ``{-1, +1}``.  Keeping that
sign and double-counting convention explicit here prevents adapters and
applications from silently inventing incompatible interpretations.

The integer preparation routine in this module is a FlagQuantum policy.  It is
not claimed to reproduce Kaiwu's ``PrecisionReducer``.  Equivalence must be
checked against each supported SDK version before a remote adapter relies on
it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ...algorithms import Hamiltonian


class KaiwuInteropError(ValueError):
    """Base exception for invalid data at the Kaiwu ecosystem boundary."""


class KaiwuMatrixValidationError(KaiwuInteropError):
    """Raised when an Ising matrix violates the boundary contract."""


class KaiwuPrecisionError(KaiwuInteropError):
    """Raised when an integer precision policy cannot be applied safely."""


@dataclass(frozen=True)
class IntegerPrecisionReport:
    """Auditable result of explicit symmetric integer scaling.

    Tensor values are detached CPU copies and should be treated as read-only.
    ``scale_factor`` maps floating-point coefficients to integer coefficients;
    ``dequantized`` applies its reciprocal for error inspection.
    """

    quantized: torch.Tensor
    dequantized: torch.Tensor
    normalized_dtype: str
    normalized_min: float
    normalized_max: float
    symmetry_normalization: str
    rounding_policy: str
    scale_factor: float
    target_min: int
    target_max: int
    max_abs_error: float
    mean_abs_error: float


@dataclass(frozen=True)
class QuboIsingEncoding:
    """Kaiwu Ising representation of ``x.T @ Q @ x + offset``.

    The final row and column belong to an auxiliary spin.  Decode returned
    solutions with :func:`decode_qubo_spins`; dropping the auxiliary column
    without gauge correction is incorrect.
    """

    matrix: torch.Tensor
    bias: float


@dataclass(frozen=True)
class HamiltonianIsingEncoding:
    """Kaiwu representation of a FlagQuantum Z-basis Hamiltonian.

    The final row and column belong to a gauge auxiliary spin. Decode returned
    solutions with :func:`decode_hamiltonian_spins` to recover the logical
    Pauli-Z eigenvalues in the original Hamiltonian's qubit order.
    """

    matrix: torch.Tensor
    bias: float
    n_qubits: int


MatrixLike = torch.Tensor | Sequence[Sequence[float]]
SpinLike = torch.Tensor | Sequence[float] | Sequence[Sequence[float]]
_MAX_EXACT_FLOAT64_INTEGER = 1 << 53


def _finite_real_scalar(value: object, *, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise KaiwuMatrixValidationError(f"{description} must be a real number")
    numeric = float(value)
    if not isfinite(numeric):
        raise KaiwuMatrixValidationError(f"{description} must be finite")
    return numeric


def _as_tensor(value: object, *, description: str) -> torch.Tensor:
    try:
        if isinstance(value, torch.Tensor):
            tensor = value.detach()
        else:
            inferred = torch.as_tensor(value)
            if inferred.dtype == torch.bool or inferred.is_complex():
                raise KaiwuMatrixValidationError(
                    f"{description} must contain real numeric values"
                )
            tensor = torch.as_tensor(value, dtype=torch.float64)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise KaiwuMatrixValidationError(
            f"{description} must be a rectangular real numeric array"
        ) from exc
    if tensor.dtype == torch.bool or tensor.is_complex():
        raise KaiwuMatrixValidationError(
            f"{description} must contain real numeric values"
        )
    return tensor


def canonicalize_ising_matrix(
    matrix: MatrixLike,
    *,
    symmetry_tolerance: float = 0.0,
) -> torch.Tensor:
    """Validate and return a detached CPU ``float64`` Ising matrix.

    Diagonal terms are retained.  For spins in ``{-1, +1}`` they contribute a
    constant energy offset, and discarding them would change Kaiwu-compatible
    energy values even though it would not change the minimizer.
    """

    tolerance = _finite_real_scalar(
        symmetry_tolerance, description="symmetry_tolerance"
    )
    if tolerance < 0:
        raise KaiwuMatrixValidationError(
            "symmetry_tolerance must be a finite non-negative number"
        )

    tensor = _as_tensor(matrix, description="Ising matrix")
    if tensor.ndim != 2:
        raise KaiwuMatrixValidationError("Ising matrix must have rank 2")
    if tensor.shape[0] == 0 or tensor.shape[0] != tensor.shape[1]:
        raise KaiwuMatrixValidationError("Ising matrix must be non-empty and square")

    canonical = tensor.to(device="cpu", dtype=torch.float64).clone()
    if not bool(torch.isfinite(canonical).all()):
        raise KaiwuMatrixValidationError("Ising matrix must contain only finite values")
    if not torch.allclose(
        canonical,
        canonical.transpose(0, 1),
        rtol=0.0,
        atol=tolerance,
    ):
        raise KaiwuMatrixValidationError(
            "Ising matrix must be symmetric within symmetry_tolerance"
        )
    normalized = canonical * 0.5 + canonical.transpose(0, 1) * 0.5
    if not bool(torch.isfinite(normalized).all()):
        raise KaiwuMatrixValidationError("Ising symmetry normalization overflowed")
    return normalized


def encode_qubo_as_ising(
    matrix: MatrixLike,
    *,
    offset: float = 0.0,
    symmetry_tolerance: float = 0.0,
) -> QuboIsingEncoding:
    """Encode a symmetric QUBO matrix for Kaiwu using one auxiliary spin.

    The source objective is ``x.T @ matrix @ x + offset`` for binary values
    ``x``.  With an encoded spin vector ``y = [s, t]``, decoding uses
    ``x = (s * t + 1) / 2`` and the returned representation satisfies
    ``-y.T @ encoding.matrix @ y + encoding.bias == source objective``.
    """

    qubo = canonicalize_ising_matrix(
        matrix,
        symmetry_tolerance=symmetry_tolerance,
    )
    finite_offset = _finite_real_scalar(offset, description="QUBO offset")

    width = qubo.shape[0]
    encoded = torch.zeros((width + 1, width + 1), dtype=torch.float64)
    encoded[:width, :width] = -qubo / 4.0
    encoded.diagonal()[:width].zero_()
    auxiliary_couplings = -qubo.sum(dim=1) / 4.0
    encoded[:width, width] = auxiliary_couplings
    encoded[width, :width] = auxiliary_couplings
    bias = finite_offset + float((qubo.sum() + qubo.diagonal().sum()).item()) / 4.0
    if not bool(torch.isfinite(encoded).all()) or not isfinite(bias):
        raise KaiwuMatrixValidationError("QUBO-to-Ising conversion overflowed")
    return QuboIsingEncoding(matrix=encoded, bias=bias)


def _hamiltonian_coefficient(value: object) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1 or value.requires_grad or value.is_complex():
            raise KaiwuMatrixValidationError(
                "Hamiltonian coefficients must be constant real scalars"
            )
        value = value.detach().cpu().item()
    if isinstance(value, bool) or not isinstance(value, Real):
        raise KaiwuMatrixValidationError(
            "Hamiltonian coefficients must be constant real scalars"
        )
    coefficient = float(value)
    if not isfinite(coefficient):
        raise KaiwuMatrixValidationError("Hamiltonian coefficients must be finite")
    return coefficient


def encode_hamiltonian_as_ising(
    hamiltonian: Hamiltonian,
) -> HamiltonianIsingEncoding:
    """Lower a FlagQuantum ``I``/``Z``/``ZZ`` Hamiltonian for Kaiwu.

    One final auxiliary spin represents single-qubit ``Z`` terms without
    choosing a gauge. With returned spins ``y = [s * t, t]``, the encoded
    energy equals the source Hamiltonian energy on logical spins ``s``.

    Args:
        hamiltonian: A FlagQuantum Hamiltonian containing only constant,
            single-qubit ``Z``, and two-qubit ``ZZ`` terms.

    Returns:
        The symmetric Kaiwu matrix, constant bias, and logical qubit count.

    Raises:
        KaiwuMatrixValidationError: If the input is not a FlagQuantum
            Hamiltonian, has no logical qubits, or contains unsupported or
            non-constant coefficients.
    """

    from ...algorithms import Hamiltonian

    if not isinstance(hamiltonian, Hamiltonian):
        raise KaiwuMatrixValidationError(
            "hamiltonian must be a FlagQuantum Hamiltonian"
        )

    declared_qubits = tuple(
        qubit for term in hamiltonian.terms for qubit in term.qubits
    )
    if any(qubit < 0 for qubit in declared_qubits):
        raise KaiwuMatrixValidationError(
            "Hamiltonian qubit indices must be non-negative"
        )
    n_qubits = max(
        hamiltonian.n_qubits,
        1 + max(declared_qubits, default=-1),
    )
    if n_qubits < 1:
        raise KaiwuMatrixValidationError(
            "Hamiltonian must declare at least one logical qubit"
        )

    matrix = torch.zeros((n_qubits + 1, n_qubits + 1), dtype=torch.float64)
    auxiliary = n_qubits
    bias = 0.0
    for term in hamiltonian.terms:
        coefficient = _hamiltonian_coefficient(term.coefficient)
        operations = term.ops
        if not operations:
            bias += coefficient
            continue
        if len(operations) == 1 and operations[0][1] == "z":
            qubit = operations[0][0]
            matrix[qubit, auxiliary] -= coefficient / 2.0
            matrix[auxiliary, qubit] -= coefficient / 2.0
            continue
        if len(operations) == 2 and all(name == "z" for _, name in operations):
            first, second = (qubit for qubit, _ in operations)
            matrix[first, second] -= coefficient / 2.0
            matrix[second, first] -= coefficient / 2.0
            continue
        raise KaiwuMatrixValidationError(
            "Kaiwu Hamiltonian lowering supports only I, Z, and ZZ terms"
        )

    if not isfinite(bias):
        raise KaiwuMatrixValidationError("Hamiltonian constant bias overflowed")
    return HamiltonianIsingEncoding(
        matrix=canonicalize_ising_matrix(matrix),
        bias=bias,
        n_qubits=n_qubits,
    )


def _decode_auxiliary_spins(auxiliary_spins: SpinLike) -> tuple[torch.Tensor, bool]:
    spins = _as_tensor(auxiliary_spins, description="auxiliary spins").to(
        device="cpu", dtype=torch.float64
    )
    was_vector = spins.ndim == 1
    if was_vector:
        spins = spins.unsqueeze(0)
    elif spins.ndim != 2:
        raise KaiwuMatrixValidationError("auxiliary spins must have rank 1 or 2")
    if spins.shape[1] < 2:
        raise KaiwuMatrixValidationError(
            "auxiliary spins must contain at least one problem spin and one auxiliary spin"
        )
    if not bool(torch.isfinite(spins).all()):
        raise KaiwuMatrixValidationError(
            "auxiliary spins must contain only finite values"
        )
    if not bool(((spins == -1) | (spins == 1)).all()):
        raise KaiwuMatrixValidationError("auxiliary spins must contain only -1 or +1")

    return spins[:, :-1] * spins[:, -1:], was_vector


def decode_qubo_spins(auxiliary_spins: SpinLike) -> torch.Tensor:
    """Decode Kaiwu spins from :func:`encode_qubo_as_ising` to binary values."""

    effective_spins, was_vector = _decode_auxiliary_spins(auxiliary_spins)
    binary = ((effective_spins + 1) / 2).to(dtype=torch.int64)
    return binary[0] if was_vector else binary


def decode_hamiltonian_spins(auxiliary_spins: SpinLike) -> torch.Tensor:
    """Decode auxiliary Kaiwu spins to logical Pauli-Z eigenvalues."""

    logical, was_vector = _decode_auxiliary_spins(auxiliary_spins)
    logical = logical.to(dtype=torch.int8)
    return logical[0] if was_vector else logical


def ising_energy(
    matrix: MatrixLike,
    spins: SpinLike,
    *,
    bias: float = 0.0,
) -> torch.Tensor:
    """Evaluate Kaiwu-compatible Ising energies.

    A one-dimensional spin vector produces a scalar tensor.  A two-dimensional
    batch produces one energy per row.
    """

    canonical = canonicalize_ising_matrix(matrix)
    finite_bias = _finite_real_scalar(bias, description="bias")

    spin_tensor = _as_tensor(spins, description="spins").to(
        device="cpu", dtype=torch.float64
    )
    was_vector = spin_tensor.ndim == 1
    if was_vector:
        spin_tensor = spin_tensor.unsqueeze(0)
    elif spin_tensor.ndim != 2:
        raise KaiwuMatrixValidationError("spins must have rank 1 or 2")
    if spin_tensor.shape[1] != canonical.shape[0]:
        raise KaiwuMatrixValidationError(
            "spin width must equal the Ising matrix dimension"
        )
    if not bool(torch.isfinite(spin_tensor).all()):
        raise KaiwuMatrixValidationError("spins must contain only finite values")
    if not bool(((spin_tensor == -1) | (spin_tensor == 1)).all()):
        raise KaiwuMatrixValidationError("spins must contain only -1 or +1")

    energies = (
        -torch.einsum("bi,ij,bj->b", spin_tensor, canonical, spin_tensor) + finite_bias
    )
    if not bool(torch.isfinite(energies).all()):
        raise KaiwuMatrixValidationError("Ising energy evaluation overflowed")
    return energies[0] if was_vector else energies


def prepare_integer_precision(
    matrix: MatrixLike,
    *,
    target_min: int = -127,
    target_max: int = 127,
) -> IntegerPrecisionReport:
    """Scale a matrix into an explicit symmetric signed integer range.

    Scaling uses the largest symmetric magnitude supported by the requested
    range and ``torch.round`` (round-half-to-even).  Callers must opt in; no
    conversion path applies lossy precision reduction implicitly.
    """

    canonical = canonicalize_ising_matrix(matrix)
    if isinstance(target_min, bool) or not isinstance(target_min, int):
        raise KaiwuPrecisionError("target_min must be an integer")
    if isinstance(target_max, bool) or not isinstance(target_max, int):
        raise KaiwuPrecisionError("target_max must be an integer")
    if target_min >= 0 or target_max <= 0:
        raise KaiwuPrecisionError("integer target range must straddle zero")

    symmetric_limit = min(abs(target_min), target_max)
    if symmetric_limit <= 0:
        raise KaiwuPrecisionError("integer target range has no symmetric capacity")
    if symmetric_limit > _MAX_EXACT_FLOAT64_INTEGER:
        raise KaiwuPrecisionError(
            "integer target range exceeds exact float64 evidence capacity"
        )

    max_abs = float(canonical.abs().max().item())
    scale_factor = 1.0 if max_abs == 0.0 else symmetric_limit / max_abs
    scaled = canonical * scale_factor
    if not bool(torch.isfinite(scaled).all()):
        raise KaiwuPrecisionError("scaled coefficients are not finite")

    quantized = torch.round(scaled).to(dtype=torch.int64)
    observed_min = int(quantized.min().item())
    observed_max = int(quantized.max().item())
    if observed_min < target_min or observed_max > target_max:
        raise KaiwuPrecisionError("quantized coefficients exceed the target range")
    if not torch.equal(quantized, quantized.transpose(0, 1)):
        raise KaiwuPrecisionError("integer preparation did not preserve symmetry")

    dequantized = quantized.to(dtype=torch.float64) / scale_factor
    absolute_error = (canonical - dequantized).abs()
    if not bool(torch.isfinite(dequantized).all()) or not bool(
        torch.isfinite(absolute_error).all()
    ):
        raise KaiwuPrecisionError("integer precision evidence overflowed")
    return IntegerPrecisionReport(
        quantized=quantized.clone(),
        dequantized=dequantized.clone(),
        normalized_dtype=str(canonical.dtype),
        normalized_min=float(canonical.min().item()),
        normalized_max=float(canonical.max().item()),
        symmetry_normalization="arithmetic_mean",
        rounding_policy="round_half_to_even",
        scale_factor=scale_factor,
        target_min=target_min,
        target_max=target_max,
        max_abs_error=float(absolute_error.max().item()),
        mean_abs_error=float(absolute_error.mean().item()),
    )

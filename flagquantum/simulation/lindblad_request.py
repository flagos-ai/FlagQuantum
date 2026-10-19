"""Validation and normalization of a continuous-time Lindblad evolution request.

Planning and execution consume the same validated description of a request: a
time grid, an initial density matrix, a Hamiltonian, the collapse operators and
the requested observables. This module owns translating every accepted input
form of those into the representation the planner and the integrator read, and
it owns the refusal an invalid request receives. It forms no generator and
steps no state, so the plan and result types and the two public entry points
stay in :mod:`~.lindblad` and reach these helpers through that module.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from ..core.ir.state_space import ProductStateSpace
from ..core.parameters import value_to_tensor
from ..errors import ValidationError
from ..observables import Observable
from ..observables.boson import DEFAULT_DENSE_MATRIX_BYTES, BosonOperator
from .density_matrix import expand_operator
from .lindblad_integrators import KRYLOV_SCHEMES, SOLVE_RESTART
from .matrix_free_hamiltonian import PauliSum, PauliSumTerm


class EvolutionValidationError(ValidationError):
    """A machine-readable refusal of an invalid evolution request."""

    def __init__(self, code: str, message: str, *, field: str) -> None:
        super().__init__(message)
        self.code = code
        self.field = field

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "field": self.field, "message": str(self)}


@dataclass(frozen=True, slots=True)
class CollapseOperator:
    """A physical Lindblad collapse operator and its inverse-time rate."""

    operator: object
    rate: float
    qubits: tuple[int, ...] = ()


def _error(code: str, message: str, field: str) -> EvolutionValidationError:
    return EvolutionValidationError(code, message, field=field)


def _resolve_state_space(
    n_qubits: int | None, dimensions: Sequence[int] | None
) -> ProductStateSpace:
    if n_qubits is not None and (
        isinstance(n_qubits, bool) or not isinstance(n_qubits, int) or n_qubits < 1
    ):
        raise _error(
            "invalid_n_qubits", "n_qubits must be a positive integer", "n_qubits"
        )
    if dimensions is None:
        if n_qubits is None:
            raise _error(
                "invalid_n_qubits", "n_qubits must be a positive integer", "n_qubits"
            )
        return ProductStateSpace((2,) * n_qubits)
    try:
        space = ProductStateSpace(dimensions)
    except (TypeError, ValueError) as exc:
        raise _error("invalid_dimensions", str(exc), "dimensions") from exc
    if n_qubits is not None and space.dimensions != (2,) * n_qubits:
        raise _error(
            "conflicting_state_space",
            "dimensions must equal (2,) * n_qubits when both are supplied",
            "dimensions",
        )
    return space


def _require_qubit_space(qubit_space: bool, field: str) -> None:
    if not qubit_space:
        raise _error(
            "qubit_only_input",
            f"qubit-only {field} input cannot act on a non-qubit state space; "
            "provide full-space numeric tensors",
            field,
        )


def _as_complex_matrix(
    value: Any,
    *,
    name: str,
    dim: int,
    dtype: torch.dtype,
    device: torch.device,
    dimensions: Sequence[int] | None = None,
) -> torch.Tensor:
    if isinstance(value, BosonOperator):
        if dimensions is None:
            raise _error(
                "boson_dimensions_required",
                "bosonic operators require explicit dimensions; mode k uses dimensions[k]",
                name,
            )
        # Refuse before to_matrix builds even the per-mode ladder matrices.
        if dim * dim * torch.complex128.itemsize > DEFAULT_DENSE_MATRIX_BYTES:
            raise _error(
                "invalid_boson_operator",
                f"bosonic dense materialization exceeds the existing "
                f"{DEFAULT_DENSE_MATRIX_BYTES}-byte ceiling; truncate the space",
                name,
            )
        try:
            value = value.to_matrix(dict(enumerate(dimensions)))
        except (TypeError, ValueError) as exc:
            raise _error("invalid_boson_operator", str(exc), name) from exc
    try:
        matrix = torch.as_tensor(value, dtype=dtype, device=device)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise _error(
            "invalid_matrix", f"{name} must be a numeric matrix", name
        ) from exc
    if matrix.ndim != 2 or tuple(matrix.shape) != (dim, dim):
        raise _error(
            "invalid_shape",
            f"{name} must have shape ({dim}, {dim})",
            name,
        )
    if not bool(torch.isfinite(matrix).all().item()):
        raise _error("non_finite_value", f"{name} must be finite", name)
    return matrix


def _solve_tolerance(value: float | None, *, dtype: torch.dtype, method: str) -> float:
    """Return the Krylov defect target for a request.

    An explicit scheme performs no Krylov step, so a caller-supplied tolerance
    would describe nothing; it is refused rather than ignored. The default
    follows the working precision so that the Krylov step is not the accuracy
    bottleneck.
    """

    if value is None:
        return 1e-5 if dtype == torch.complex64 else 1e-12
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _error(
            "invalid_solve_tolerance",
            "solve_tolerance must be a number",
            "solve_tolerance",
        )
    resolved = float(value)
    if not (0.0 < resolved < 1.0):
        raise _error(
            "invalid_solve_tolerance",
            "solve_tolerance must lie strictly between 0 and 1",
            "solve_tolerance",
        )
    if method not in KRYLOV_SCHEMES:
        raise _error(
            "invalid_solve_tolerance",
            "solve_tolerance applies only to the schemes that take a Krylov step "
            + ", ".join(repr(name) for name in KRYLOV_SCHEMES)
            + f", not {method!r}",
            "solve_tolerance",
        )
    return resolved


def _solve_max_iterations(dimension: int) -> int:
    """Bound a matrix-free Krylov step by the system it acts on.

    A restarted Krylov solve can need more cycles than it has basis vectors, so
    the cap is the larger of several restart windows and twice the dimension of
    the ``4**n`` state it is inverting. The exponential scheme takes at most one
    window, so this bound never restricts it.
    """

    return max(2 * SOLVE_RESTART, 2 * dimension * dimension)


def _normalize_times(
    times: Any, *, device: torch.device, real_dtype: torch.dtype
) -> torch.Tensor:
    try:
        result = torch.as_tensor(times, dtype=real_dtype, device=device)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise _error("invalid_time_grid", "times must be numeric", "times") from exc
    if result.ndim != 1 or result.numel() == 0:
        raise _error(
            "invalid_time_grid",
            "times must be a non-empty one-dimensional grid",
            "times",
        )
    if not bool(torch.isfinite(result).all().item()):
        raise _error("non_finite_value", "times must be finite", "times")
    if result.numel() > 1 and not bool((result[1:] > result[:-1]).all().item()):
        raise _error(
            "non_monotone_time_grid",
            "times must be strictly increasing; duplicates are not allowed",
            "times",
        )
    return result


def _normalize_initial_state(
    initial_state: Any,
    *,
    dim: int,
    dtype: torch.dtype,
    device: torch.device,
    tolerance: float,
    qubit_space: bool = True,
) -> torch.Tensor:
    """Validate an initial state and return it as one density matrix or a batch.

    A leading batch axis is admitted on both input forms, so ``(batch, dim)``
    statevectors and ``(batch, dim, dim)`` density matrices advance together
    under one Hamiltonian. ``(dim, dim)`` keeps its established reading as a
    single density matrix, which is why the statevector form is recognized by
    its trailing axis.
    """

    if isinstance(initial_state, str):
        _require_qubit_space(qubit_space, "initial_state")
        if len(initial_state) != dim.bit_length() - 1 or set(initial_state) - {
            "0",
            "1",
        }:
            raise _error(
                "invalid_initial_state",
                "initial_state bitstring must contain exactly n_qubits binary digits",
                "initial_state",
            )
        state = torch.zeros(dim, dtype=dtype, device=device)
        state[int(initial_state, 2)] = 1
        return torch.outer(state, torch.conj(state))
    try:
        state = torch.as_tensor(initial_state, dtype=dtype, device=device)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise _error(
            "invalid_initial_state",
            "initial_state must be a statevector or density matrix",
            "initial_state",
        ) from exc
    if not bool(torch.isfinite(state).all().item()):
        raise _error(
            "non_finite_value", "initial_state must be finite", "initial_state"
        )
    if state.ndim == 1 and state.shape[0] == dim:
        return _statevectors_to_density(state, dim=dim, tolerance=tolerance)
    if state.ndim == 2 and state.shape != (dim, dim) and state.shape[-1] == dim:
        return _statevectors_to_density(state, dim=dim, tolerance=tolerance)
    if state.ndim == 2 and tuple(state.shape) == (dim, dim):
        return _require_density_matrices(state, tolerance=tolerance)
    if state.ndim == 3 and tuple(state.shape[1:]) == (dim, dim):
        return _require_density_matrices(state, tolerance=tolerance)
    raise _error(
        "invalid_shape",
        f"initial_state must have shape ({dim},), ({dim}, {dim}), "
        f"(batch, {dim}), or (batch, {dim}, {dim})",
        "initial_state",
    )


def _statevectors_to_density(
    state: torch.Tensor, *, dim: int, tolerance: float
) -> torch.Tensor:
    """Return the outer product of each row, refusing an unnormalized one."""

    norms = torch.sum(torch.abs(state) ** 2, dim=-1)
    if float(torch.max(torch.abs(norms - 1.0)).item()) > tolerance:
        raise _error(
            "unnormalized_initial_state",
            "initial_state statevector must have unit norm",
            "initial_state",
        )
    if state.ndim == 1:
        return torch.outer(state, torch.conj(state))
    return state.unsqueeze(-1) * torch.conj(state).unsqueeze(-2)


def _require_density_matrices(
    matrices: torch.Tensor, *, tolerance: float
) -> torch.Tensor:
    """Refuse a batch that is not Hermitian, unit trace, and positive."""

    if not torch.allclose(
        matrices, torch.conj(matrices).transpose(-2, -1), atol=tolerance, rtol=0.0
    ):
        raise _error(
            "non_hermitian_initial_state",
            "initial_state density matrix must be Hermitian",
            "initial_state",
        )
    traces = torch.diagonal(matrices, dim1=-2, dim2=-1).sum(dim=-1).real
    if float(torch.max(torch.abs(traces - 1.0)).item()) > tolerance:
        raise _error(
            "unnormalized_initial_state",
            "initial_state density matrix must have unit trace",
            "initial_state",
        )
    if float(torch.min(torch.linalg.eigvalsh(matrices)).item()) < -tolerance:
        raise _error(
            "non_positive_initial_state",
            "initial_state density matrix must be positive semidefinite",
            "initial_state",
        )
    return matrices


def _pauli_operator(
    pauli: str,
    wires: Sequence[int],
    *,
    n_wires: int,
    dtype: torch.dtype,
    device: torch.device,
    field: str,
) -> torch.Tensor:
    symbols = pauli.strip().upper()
    if not symbols or set(symbols) - {"I", "X", "Y", "Z"}:
        raise _error("unknown_operator", f"unknown Pauli operator {pauli!r}", field)
    if len(symbols) != len(wires):
        raise _error(
            "invalid_operator",
            "the Pauli string length must match the number of wires",
            field,
        )
    matrices = {
        "I": torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=dtype, device=device),
        "X": torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=dtype, device=device),
        "Y": torch.tensor([[0.0, -1j], [1j, 0.0]], dtype=dtype, device=device),
        "Z": torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=dtype, device=device),
    }
    local = matrices[symbols[0]]
    for symbol in symbols[1:]:
        local = torch.kron(local, matrices[symbol])
    try:
        return expand_operator(local, wires, n_wires, dtype=dtype, device=device)
    except (TypeError, ValueError) as exc:
        raise _error("invalid_operator", str(exc), field) from exc


def _descriptor_pauli(
    descriptor: Mapping[str, Any], *, field: str
) -> tuple[str, tuple[int, ...]]:
    pauli = descriptor.get("pauli", descriptor.get("operator"))
    if not isinstance(pauli, str):
        raise _error(
            "invalid_operator", "operator descriptors require a Pauli string", field
        )
    raw_qubits = descriptor.get("qubits")
    qubits = tuple(range(len(pauli))) if raw_qubits is None else tuple(raw_qubits)
    try:
        return pauli, tuple(int(qubit) for qubit in qubits)
    except (TypeError, ValueError) as exc:
        raise _error(
            "invalid_operator", "operator qubits must be integers", field
        ) from exc


def _descriptor_matrix(
    descriptor: Mapping[str, Any],
    *,
    n_wires: int,
    dtype: torch.dtype,
    device: torch.device,
    field: str,
) -> torch.Tensor:
    pauli, qubits = _descriptor_pauli(descriptor, field=field)
    return _pauli_operator(
        pauli,
        qubits,
        n_wires=n_wires,
        dtype=dtype,
        device=device,
        field=field,
    )


def _pauli_term_wires(
    wires: Sequence[int], *, n_wires: int, field: str
) -> tuple[int, ...]:
    """Refuse a qubit the Hilbert space does not have, or one named twice.

    A repeated qubit cannot describe a Pauli string: the two symbols would act
    on the same qubit, and the dense route resolved that by silently letting
    the second placement overwrite the first, which produced a non-Hermitian
    operator for ``XX`` on ``(0, 0)`` and a Hermitian one for ``ZZ``. Neither
    is the requested Hamiltonian, so the request is refused instead.
    """

    normalized = tuple(int(wire) for wire in wires)
    if any(wire < 0 or wire >= n_wires for wire in normalized):
        raise _error("invalid_operator", "wire index out of range", field)
    if len(set(normalized)) != len(normalized):
        raise _error(
            "invalid_operator",
            "a Pauli term must not name the same wire twice",
            field,
        )
    return normalized


def _pauli_term(
    coefficient: Any,
    pauli: str,
    wires: Sequence[int],
    *,
    n_wires: int,
    field: str,
) -> PauliSumTerm:
    """Describe one Pauli-string term in the matrix-free representation."""

    symbols = pauli.strip().upper()
    if not symbols or set(symbols) - {"I", "X", "Y", "Z"}:
        raise _error("unknown_operator", f"unknown Pauli operator {pauli!r}", field)
    if len(symbols) != len(wires):
        raise _error(
            "invalid_operator",
            "the Pauli string length must match the number of wires",
            field,
        )
    normalized = _pauli_term_wires(wires, n_wires=n_wires, field=field)
    mask = 0
    signs: list[int] = []
    for symbol, wire in zip(symbols, normalized, strict=True):
        bit = n_wires - 1 - wire
        if symbol in {"X", "Y"}:
            mask |= 1 << bit
        if symbol in {"Z", "Y"}:
            signs.append(bit)
    return PauliSumTerm(coefficient, mask, tuple(signs))


def _hamiltonian_terms(
    hamiltonian: Any, *, n_wires: int
) -> tuple[PauliSumTerm, ...] | None:
    """Describe a Pauli-sum Hamiltonian, or ``None`` for an opaque matrix.

    Both Pauli input forms reach the same representation: the observable
    algebra already carries one term per Pauli string, and a descriptor
    sequence names one Pauli string per mapping. The representation itself is
    accepted as input so a caller holding one can hand it back without the
    terms being written out as descriptors and read in again.
    """

    if isinstance(hamiltonian, PauliSum):
        return hamiltonian.terms
    if isinstance(hamiltonian, Observable):
        return _observable_terms(hamiltonian, n_wires=n_wires, field="hamiltonian")
    if (
        isinstance(hamiltonian, Sequence)
        and not isinstance(hamiltonian, (str, bytes, torch.Tensor))
        and hamiltonian
        and isinstance(hamiltonian[0], Mapping)
    ):
        terms = []
        for term in hamiltonian:
            if not isinstance(term, Mapping):
                raise _error(
                    "invalid_hamiltonian",
                    "Hamiltonian terms must all be operator mappings",
                    "hamiltonian",
                )
            try:
                coefficient = float(term.get("coefficient", 1.0))
            except (TypeError, ValueError) as exc:
                raise _error(
                    "invalid_hamiltonian",
                    "Hamiltonian coefficients must be finite real numbers",
                    "hamiltonian",
                ) from exc
            if not bool(torch.isfinite(torch.tensor(coefficient)).item()):
                raise _error(
                    "invalid_hamiltonian",
                    "Hamiltonian coefficients must be finite real numbers",
                    "hamiltonian",
                )
            pauli, term_qubits = _descriptor_pauli(term, field="hamiltonian")
            terms.append(
                _pauli_term(
                    coefficient,
                    pauli,
                    term_qubits,
                    n_wires=n_wires,
                    field="hamiltonian",
                )
            )
        return tuple(terms)
    return None


def _require_hermitian_terms(
    terms: Sequence[PauliSumTerm], *, tolerance: float
) -> None:
    """Refuse a Pauli sum that is not Hermitian.

    Adding the coefficients of one Pauli string needs no matrix, and every
    Pauli string maps to exactly one matrix, so a Pauli-sum Hamiltonian is
    Hermitian exactly when the total coefficient of every distinct string is
    real. Deciding that arithmetically also keeps every Y term acceptable:
    the Pauli strings form a basis, so their coefficients are what a dense
    ``H - H^dagger`` would be read against. The bound is the same ``atol``
    applied to that comparison, whose largest entry for one string is twice
    the imaginary part of its coefficient, which the measured boundary of the
    dense comparison confirms (1e-10 accepted, 2e-10 refused at complex128).
    """

    totals: dict[tuple[int, tuple[int, ...]], complex] = {}
    for term in terms:
        key = (term.mask, term.signs)
        totals[key] = totals.get(key, 0j) + term.coefficient
    for total in totals.values():
        if 2.0 * abs(total.imag) > tolerance:
            raise _error(
                "non_hermitian_hamiltonian",
                "hamiltonian must be Hermitian",
                "hamiltonian",
            )


def _dense_operator_terms(
    terms: Sequence[Any],
    *,
    dim: int,
    dtype: torch.dtype,
    device: torch.device,
    dimensions: Sequence[int] | None,
) -> torch.Tensor:
    """Assemble static full-space terms while preserving coefficient graphs."""

    total: torch.Tensor | None = None
    for index, term in enumerate(terms):
        if (
            not isinstance(term, Mapping)
            or "operator" not in term
            or any(key in term for key in ("pauli", "qubit", "qubits"))
        ):
            raise _error(
                "invalid_hamiltonian",
                "full-space Hamiltonian terms require operator mappings; "
                "do not mix them with Pauli descriptors or qubit selectors",
                "hamiltonian",
            )
        value = term.get("coefficient", 1.0)
        field = f"hamiltonian[{index}].coefficient"
        if isinstance(value, bool) or (
            isinstance(value, torch.Tensor) and value.dtype == torch.bool
        ):
            raise _error(
                "invalid_coefficient", "coefficients must be numeric scalars", field
            )
        try:
            coefficient = value_to_tensor(value, dtype=dtype, device=device)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise _error("invalid_coefficient", str(exc), field) from exc
        if coefficient.ndim != 0 or not bool(torch.isfinite(coefficient).all().item()):
            raise _error(
                "invalid_coefficient",
                "coefficients must be finite numeric scalars",
                field,
            )
        matrix = _as_complex_matrix(
            term["operator"],
            name=f"hamiltonian[{index}].operator",
            dim=dim,
            dtype=dtype,
            device=device,
            dimensions=dimensions,
        )
        scaled = coefficient * matrix
        total = scaled if total is None else total + scaled
    if total is None:
        raise _error(
            "invalid_hamiltonian", "Hamiltonian terms must not be empty", "hamiltonian"
        )
    return total


def _normalize_hamiltonian(
    hamiltonian: Any,
    *,
    n_wires: int,
    dim: int,
    dtype: torch.dtype,
    device: torch.device,
    tolerance: float,
    qubit_space: bool = True,
    dimensions: Sequence[int] | None = None,
) -> PauliSum | torch.Tensor:
    """Validate a Hamiltonian and return what evolution consumes.

    A Pauli sum stays matrix-free; full-space matrices and static operator
    terms use the dense path. Hermiticity is decided here so planning, execution,
    and plan serialization cannot disagree about whether a request was accepted.
    Boson operators use their existing dense conversion at explicit mode dimensions.
    """

    if (
        isinstance(hamiltonian, Sequence)
        and not isinstance(hamiltonian, (str, bytes, torch.Tensor))
        and hamiltonian
        and isinstance(hamiltonian[0], Mapping)
        and not isinstance(hamiltonian[0].get("operator", ""), str)
    ):
        hamiltonian = _dense_operator_terms(
            hamiltonian, dim=dim, dtype=dtype, device=device, dimensions=dimensions
        )
    terms = _hamiltonian_terms(hamiltonian, n_wires=n_wires)
    if terms is not None:
        _require_qubit_space(qubit_space, "hamiltonian")
        _require_hermitian_terms(terms, tolerance=tolerance)
        return PauliSum(
            terms,
            n_qubits=n_wires,
            dimension=dim,
            dtype=dtype,
            device=device,
        )
    matrix = _as_complex_matrix(
        hamiltonian,
        name="hamiltonian",
        dim=dim,
        dtype=dtype,
        device=device,
        dimensions=dimensions,
    )
    if not torch.allclose(matrix, torch.conj(matrix).T, atol=tolerance, rtol=0.0):
        raise _error(
            "non_hermitian_hamiltonian",
            "hamiltonian must be Hermitian",
            "hamiltonian",
        )
    return matrix


def _observable_terms(
    observable: Observable, *, n_wires: int, field: str
) -> tuple[PauliSumTerm, ...]:
    """Describe the observable algebra's terms in the matrix-free form."""

    terms = []
    for term in observable.terms:
        if not term.factors:
            terms.append(PauliSumTerm(term.coefficient))
            continue
        terms.append(
            _pauli_term(
                term.coefficient,
                "".join(axis for _, axis in term.factors),
                tuple(wire for wire, _ in term.factors),
                n_wires=n_wires,
                field=field,
            )
        )
    return tuple(terms)


def _observable_matrix(
    observable: Observable,
    *,
    n_wires: int,
    dtype: torch.dtype,
    device: torch.device,
    field: str,
) -> torch.Tensor:
    return PauliSum(
        _observable_terms(observable, n_wires=n_wires, field=field),
        n_qubits=n_wires,
        dimension=2**n_wires,
        dtype=dtype,
        device=device,
    ).dense()


def _named_operator(
    name: str,
    wires: Sequence[int],
    *,
    n_wires: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    normalized = name.strip().lower().replace("-", "_")
    if normalized not in {"amplitude_damping", "sigma_minus"}:
        raise _error(
            "unknown_collapse_operator",
            f"unknown collapse operator {name!r}",
            "collapse_operators",
        )
    if len(wires) != 1:
        raise _error(
            "invalid_collapse_operator",
            f"{normalized} requires exactly one wire",
            "collapse_operators",
        )
    sigma_minus = torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=dtype, device=device)
    try:
        return expand_operator(sigma_minus, wires, n_wires, dtype=dtype, device=device)
    except (TypeError, ValueError) as exc:
        raise _error(
            "invalid_collapse_operator", str(exc), "collapse_operators"
        ) from exc


def _normalize_collapse_terms(
    collapse_operators: Sequence[Any] | None,
    *,
    n_wires: int,
    dim: int,
    dtype: torch.dtype,
    device: torch.device,
    qubit_space: bool = True,
    dimensions: Sequence[int] | None = None,
) -> tuple[tuple[torch.Tensor, float], ...]:
    result = []
    for index, item in enumerate(collapse_operators or ()):
        operator: Any
        rate: Any
        raw_qubits: Any
        if isinstance(item, CollapseOperator):
            if item.qubits:
                _require_qubit_space(qubit_space, "collapse_operators")
            operator = item.operator
            rate = item.rate
            raw_qubits = item.qubits
        elif isinstance(item, Mapping):
            if "qubit" in item or "qubits" in item:
                _require_qubit_space(qubit_space, "collapse_operators")
            operator = item.get("operator", item.get("kind"))
            rate = item.get("rate")
            raw_qubits = item.get("qubits", item.get("qubit", (0,)))
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            if len(item) != 2:
                raise _error(
                    "invalid_collapse_operator",
                    "collapse operator pairs must contain (operator, rate)",
                    "collapse_operators",
                )
            operator, rate = item
            raw_qubits = (0,)
        else:
            raise _error(
                "invalid_collapse_operator",
                f"collapse_operators[{index}] must provide operator and rate",
                "collapse_operators",
            )
        try:
            numeric_rate = float(rate)
        except (TypeError, ValueError) as exc:
            raise _error(
                "invalid_rate",
                "collapse rates must be finite numbers",
                "collapse_operators",
            ) from exc
        if not torch.isfinite(torch.tensor(numeric_rate)):
            raise _error(
                "invalid_rate",
                "collapse rates must be finite numbers",
                "collapse_operators",
            )
        if numeric_rate < 0.0:
            raise _error(
                "negative_rate",
                "collapse rates must be non-negative",
                "collapse_operators",
            )
        if isinstance(raw_qubits, int):
            qubits: tuple[int, ...] = (raw_qubits,)
        else:
            try:
                qubits = tuple(int(qubit) for qubit in raw_qubits)
            except (TypeError, ValueError) as exc:
                raise _error(
                    "invalid_collapse_operator",
                    "collapse operator qubits must be integer indices",
                    "collapse_operators",
                ) from exc
        if isinstance(operator, str):
            _require_qubit_space(qubit_space, "collapse_operators")
            full = _named_operator(
                operator,
                qubits,
                n_wires=n_wires,
                dtype=dtype,
                device=device,
            )
        else:
            full = _as_complex_matrix(
                operator,
                name="collapse_operators",
                dim=dim,
                dtype=dtype,
                device=device,
                dimensions=dimensions,
            )
        result.append((full, numeric_rate))
    return tuple(result)


def _normalize_collapse_operators(
    collapse_operators: Sequence[Any] | None,
    *,
    n_wires: int,
    dim: int,
    dtype: torch.dtype,
    device: torch.device,
    qubit_space: bool = True,
    dimensions: Sequence[int] | None = None,
) -> tuple[torch.Tensor, ...]:
    return tuple(
        (rate**0.5) * operator
        for operator, rate in _normalize_collapse_terms(
            collapse_operators,
            n_wires=n_wires,
            dim=dim,
            dtype=dtype,
            device=device,
            qubit_space=qubit_space,
            dimensions=dimensions,
        )
    )


def _normalize_observables(
    observables: Mapping[str, Any] | Sequence[Any] | None,
    *,
    dim: int,
    dtype: torch.dtype,
    device: torch.device,
    qubit_space: bool = True,
    dimensions: Sequence[int] | None = None,
) -> dict[str, torch.Tensor]:
    if observables is None:
        return {}
    items: Iterable[tuple[Any, Any]]
    if isinstance(observables, Mapping):
        named_mapping = True
        items = observables.items()
    else:
        named_mapping = False
        items = enumerate(observables)
    result = {}
    for name, value in items:
        if isinstance(value, (Observable, Mapping)):
            _require_qubit_space(qubit_space, "observables")
        if isinstance(value, Observable):
            label = str(name) if named_mapping else f"observable_{name}"
            matrix = _observable_matrix(
                value,
                n_wires=dim.bit_length() - 1,
                dtype=dtype,
                device=device,
                field="observables",
            )
        elif isinstance(value, Mapping):
            default_label = str(name) if named_mapping else f"observable_{name}"
            label = str(value.get("name", default_label))
            matrix = _descriptor_matrix(
                value,
                n_wires=dim.bit_length() - 1,
                dtype=dtype,
                device=device,
                field="observables",
            )
        else:
            label = str(name)
            matrix = _as_complex_matrix(
                value,
                name=f"observables[{label!r}]",
                dim=dim,
                dtype=dtype,
                device=device,
                dimensions=dimensions,
            )
        if label in result:
            raise _error(
                "duplicate_observable", f"duplicate observable {label!r}", "observables"
            )
        if not torch.allclose(matrix, torch.conj(matrix).T, atol=1e-6, rtol=0.0):
            raise _error(
                "non_hermitian_observable",
                f"observable {label!r} must be Hermitian",
                "observables",
            )
        result[label] = matrix
    return result


def _resolve_batch_size(value: Any, observed: int) -> int:
    """Return the declared batch, refusing one the initial state contradicts.

    ``None`` means the caller did not declare a batch, so the one the initial
    state carries is used. A declared batch that disagrees with the state is
    refused rather than trusted, because the two are the same fact stated
    twice and silently preferring either one would let a caller believe a
    batch size that never ran.
    """

    if value is None:
        return observed
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise _error(
            "invalid_batch_size", "batch_size must be a positive integer", "batch_size"
        )
    if value != observed:
        raise _error(
            "batch_mismatch",
            "batch_size must match the number of members in initial_state: "
            f"batch_size={value} but initial_state carries {observed}",
            "batch_size",
        )
    return value


def _require_gradients_flag(value: Any) -> bool:
    """Refuse a ``require_gradients`` that is not a boolean."""

    if type(value) is not bool:
        raise _error(
            "invalid_require_gradients",
            "require_gradients must be a boolean",
            "require_gradients",
        )
    return value

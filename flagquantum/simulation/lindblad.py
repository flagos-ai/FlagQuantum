"""Deterministic continuous-time Lindblad evolution on dense states."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from ..errors import ExecutionError, ValidationError
from ..observables import Observable
from .density_matrix import expand_operator
from .lindblad_generator import Liouvillian
from .lindblad_integrators import (
    DEFAULT_INTEGRATOR,
    INTEGRATOR_NAMES,
    KRYLOV_SCHEMES,
    SOLVE_RESTART,
    advance_interval,
    integrator_order,
    substeps_per_interval,
)
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
    wires: tuple[int, ...] = ()


def amplitude_damping(rate: float, wire: int) -> CollapseOperator:
    """Return ``sqrt(rate) * |0><1|`` on one wire.

    ``rate`` is a physical inverse-time rate, never a channel probability.
    """

    return CollapseOperator("amplitude_damping", rate, (wire,))


@dataclass(frozen=True)
class EvolutionResult:
    """A time-resolved density-matrix evolution result.

    ``populations`` has shape ``(len(times), 2**n_wires)``.  Observable values
    have one row per requested time.  Density matrices are retained only when
    ``return_density_matrices=True`` was requested.
    """

    times: torch.Tensor
    populations: torch.Tensor
    observables: dict[str, torch.Tensor]
    density_matrices: torch.Tensor | None
    maximum_trace_drift: float
    population_bounded: bool
    method: str
    order: int | None
    step_size: float
    precision: str
    population_tolerance: float
    trace_tolerance: float
    plan: object | None = None

    @property
    def probabilities(self) -> torch.Tensor:
        """Return the time-resolved computational-basis probabilities."""

        return self.populations

    @property
    def expectations(self) -> tuple[torch.Tensor, ...]:
        """Return requested expectation trajectories in output order."""

        return tuple(self.observables.values())

    def expectation(self, selector: int | str | None = None) -> torch.Tensor:
        """Return one expectation trajectory by output index or name."""

        outputs = tuple(self.observables.items())
        if not outputs:
            raise ExecutionError("evolution result does not contain an expectation")
        if selector is None:
            if len(outputs) != 1:
                raise ExecutionError(
                    "evolution result contains multiple expectations; select one by "
                    "output index or name"
                )
            return outputs[0][1]
        if type(selector) is int:
            try:
                return outputs[selector][1]
            except IndexError as exc:
                raise ExecutionError(
                    f"expectation index {selector} is outside the result"
                ) from exc
        if not isinstance(selector, str) or not selector:
            raise TypeError(
                "expectation selector must be an integer or non-empty string"
            )
        try:
            return self.observables[selector]
        except KeyError as exc:
            raise ExecutionError(
                f"evolution result has no expectation {selector!r}"
            ) from exc

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation, including provenance."""

        return {
            "times": self.times.detach().cpu().tolist(),
            "populations": self.populations.detach().cpu().tolist(),
            "observables": {
                name: values.detach().cpu().tolist()
                for name, values in self.observables.items()
            },
            "density_matrices": (
                None
                if self.density_matrices is None
                else _complex_tensor_to_json(self.density_matrices)
            ),
            "maximum_trace_drift": self.maximum_trace_drift,
            "population_bounded": self.population_bounded,
            "numerics": {
                "method": self.method,
                "order": self.order,
                "step_size": self.step_size,
                "precision": self.precision,
                "population_tolerance": self.population_tolerance,
                "trace_tolerance": self.trace_tolerance,
            },
            "plan_identity": getattr(self.plan, "identity", None),
        }


@dataclass(frozen=True)
class EvolutionPlan:
    """Validated resource and numerical-method plan for an evolution request."""

    n_wires: int
    n_times: int
    dimension: int
    trajectory_bytes: int
    method: str
    order: int | None
    step_size: float
    device: str
    precision: str
    solve_tolerance: float | None = None
    batch_size: int = 1
    require_gradients: bool = False

    @property
    def n_qubits(self) -> int:
        """Return the public qubit count; ``n_wires`` remains serialized internally."""

        return self.n_wires

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_wires": self.n_wires,
            "n_times": self.n_times,
            "dimension": self.dimension,
            "trajectory_bytes": self.trajectory_bytes,
            "batch_size": self.batch_size,
            "numerics": {
                "method": self.method,
                "order": self.order,
                "step_size": self.step_size,
                "device": self.device,
                "precision": self.precision,
                "solve_tolerance": self.solve_tolerance,
                "require_gradients": self.require_gradients,
            },
        }


def _complex_tensor_to_json(value: torch.Tensor) -> list[Any]:
    raw = value.detach().cpu()
    return torch.stack((raw.real, raw.imag), dim=-1).tolist()


def _error(code: str, message: str, field: str) -> EvolutionValidationError:
    return EvolutionValidationError(code, message, field=field)


def _as_complex_matrix(
    value: Any,
    *,
    name: str,
    dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
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
) -> torch.Tensor:
    """Validate an initial state and return it as one density matrix or a batch.

    A leading batch axis is admitted on both input forms, so ``(batch, dim)``
    statevectors and ``(batch, dim, dim)`` density matrices advance together
    under one Hamiltonian. ``(dim, dim)`` keeps its established reading as a
    single density matrix, which is why the statevector form is recognized by
    its trailing axis.
    """

    if isinstance(initial_state, str):
        if len(initial_state) != dim.bit_length() - 1 or set(initial_state) - {
            "0",
            "1",
        }:
            raise _error(
                "invalid_initial_state",
                "initial_state bitstring must contain exactly n_wires binary digits",
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
    raw_wires = descriptor.get("wires")
    wires = tuple(range(len(pauli))) if raw_wires is None else tuple(raw_wires)
    try:
        return pauli, tuple(int(wire) for wire in wires)
    except (TypeError, ValueError) as exc:
        raise _error(
            "invalid_operator", "operator wires must be integers", field
        ) from exc


def _descriptor_matrix(
    descriptor: Mapping[str, Any],
    *,
    n_wires: int,
    dtype: torch.dtype,
    device: torch.device,
    field: str,
) -> torch.Tensor:
    pauli, wires = _descriptor_pauli(descriptor, field=field)
    return _pauli_operator(
        pauli,
        wires,
        n_wires=n_wires,
        dtype=dtype,
        device=device,
        field=field,
    )


def _pauli_term_wires(
    wires: Sequence[int], *, n_wires: int, field: str
) -> tuple[int, ...]:
    """Refuse a wire the Hilbert space does not have, or one named twice.

    A repeated wire cannot describe a Pauli string: the two symbols would act
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
            pauli, wires = _descriptor_pauli(term, field="hamiltonian")
            terms.append(
                _pauli_term(
                    coefficient, pauli, wires, n_wires=n_wires, field="hamiltonian"
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


def _normalize_hamiltonian(
    hamiltonian: Any,
    *,
    n_wires: int,
    dim: int,
    dtype: torch.dtype,
    device: torch.device,
    tolerance: float,
) -> PauliSum | torch.Tensor:
    """Validate a Hamiltonian and return what evolution consumes.

    A Pauli sum stays matrix-free; only an opaque dense input is kept as a
    matrix. Hermiticity is decided here so that planning, execution and plan
    serialization cannot disagree about whether a request was accepted.
    """

    terms = _hamiltonian_terms(hamiltonian, n_wires=n_wires)
    if terms is not None:
        _require_hermitian_terms(terms, tolerance=tolerance)
        return PauliSum(
            terms,
            n_wires=n_wires,
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
        n_wires=n_wires,
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
) -> tuple[tuple[torch.Tensor, float], ...]:
    result = []
    for index, item in enumerate(collapse_operators or ()):
        operator: Any
        rate: Any
        raw_wires: Any
        if isinstance(item, CollapseOperator):
            operator = item.operator
            rate = item.rate
            raw_wires = item.wires
        elif isinstance(item, Mapping):
            operator = item.get("operator", item.get("kind"))
            rate = item.get("rate")
            raw_wires = item.get("wires", item.get("wire", (0,)))
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            if len(item) != 2:
                raise _error(
                    "invalid_collapse_operator",
                    "collapse operator pairs must contain (operator, rate)",
                    "collapse_operators",
                )
            operator, rate = item
            raw_wires = (0,)
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
        if isinstance(raw_wires, int):
            wires: tuple[int, ...] = (raw_wires,)
        else:
            try:
                wires = tuple(int(wire) for wire in raw_wires)
            except (TypeError, ValueError) as exc:
                raise _error(
                    "invalid_collapse_operator",
                    "collapse operator wires must be integer indices",
                    "collapse_operators",
                ) from exc
        if isinstance(operator, str):
            full = _named_operator(
                operator,
                wires,
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
) -> tuple[torch.Tensor, ...]:
    return tuple(
        (rate**0.5) * operator
        for operator, rate in _normalize_collapse_terms(
            collapse_operators,
            n_wires=n_wires,
            dim=dim,
            dtype=dtype,
            device=device,
        )
    )


def _normalize_observables(
    observables: Mapping[str, Any] | Sequence[Any] | None,
    *,
    dim: int,
    dtype: torch.dtype,
    device: torch.device,
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


def plan_density_matrix_evolution(
    hamiltonian: Any,
    initial_state: Any,
    n_wires: int,
    times: Any,
    collapse_operators: Sequence[Any] | None = None,
    observables: Mapping[str, Any] | Sequence[Any] | None = None,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.complex128,
    return_density_matrices: bool = False,
    method: str = DEFAULT_INTEGRATOR,
    solve_tolerance: float | None = None,
    batch_size: int | None = None,
    require_gradients: bool = False,
) -> EvolutionPlan:
    """Validate an evolution request and report its deterministic local plan."""

    if isinstance(n_wires, bool) or not isinstance(n_wires, int) or n_wires < 1:
        raise _error("invalid_n_wires", "n_wires must be a positive integer", "n_wires")
    if dtype not in {torch.complex64, torch.complex128}:
        raise _error(
            "unsupported_dtype",
            "dtype must be torch.complex64 or torch.complex128",
            "dtype",
        )
    if method not in INTEGRATOR_NAMES:
        raise _error(
            "unsupported_method",
            "method must be one of "
            + ", ".join(repr(name) for name in INTEGRATOR_NAMES)
            + f", not {method!r}",
            "method",
        )
    resolved_require_gradients = _require_gradients_flag(require_gradients)
    resolved_solve_tolerance = _solve_tolerance(
        solve_tolerance, dtype=dtype, method=method
    )
    resolved_device = torch.device(device)
    if resolved_device.type != "cpu":
        raise _error(
            "unsupported_device",
            "continuous-time evolution currently supports CPU only",
            "device",
        )
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    tolerance = 1e-5 if dtype == torch.complex64 else 1e-10
    dim = 2**n_wires
    time_grid = _normalize_times(times, device=resolved_device, real_dtype=real_dtype)
    _normalize_hamiltonian(
        hamiltonian,
        n_wires=n_wires,
        dim=dim,
        dtype=dtype,
        device=resolved_device,
        tolerance=tolerance,
    )
    initial_matrices = _normalize_initial_state(
        initial_state,
        dim=dim,
        dtype=dtype,
        device=resolved_device,
        tolerance=tolerance,
    )
    resolved_batch_size = _resolve_batch_size(
        batch_size, 1 if initial_matrices.ndim == 2 else int(initial_matrices.shape[0])
    )
    _normalize_collapse_operators(
        collapse_operators,
        n_wires=n_wires,
        dim=dim,
        dtype=dtype,
        device=resolved_device,
    )
    _normalize_observables(observables, dim=dim, dtype=dtype, device=resolved_device)
    step_size = (
        0.0
        if time_grid.numel() == 1
        else float(
            (
                torch.max(time_grid[1:] - time_grid[:-1])
                / substeps_per_interval(method)
            ).item()
        )
    )
    element_bytes = 8 if dtype == torch.complex64 else 16
    return EvolutionPlan(
        n_wires=n_wires,
        n_times=int(time_grid.numel()),
        dimension=dim,
        trajectory_bytes=(
            int(time_grid.numel()) * resolved_batch_size * dim * dim * element_bytes
        ),
        batch_size=resolved_batch_size,
        method=method,
        order=integrator_order(method),
        step_size=step_size,
        device=str(resolved_device),
        precision="complex64" if dtype == torch.complex64 else "complex128",
        solve_tolerance=(
            resolved_solve_tolerance if method in KRYLOV_SCHEMES else None
        ),
        require_gradients=resolved_require_gradients,
    )


def evolve_density_matrix(
    hamiltonian: Any,
    initial_state: Any,
    n_wires: int,
    times: Any,
    collapse_operators: Sequence[Any] | None = None,
    observables: Mapping[str, Any] | Sequence[Any] | None = None,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.complex128,
    return_density_matrices: bool = False,
    method: str = DEFAULT_INTEGRATOR,
    solve_tolerance: float | None = None,
    batch_size: int | None = None,
    require_gradients: bool = False,
) -> EvolutionResult:
    """Evolve a state on an explicit time grid with the Lindblad equation.

    Collapse entries are either ``(operator, rate)`` pairs or mappings with
    ``operator`` (a dense matrix, ``"amplitude_damping"``, or
    ``"sigma_minus"``), ``rate``, and optional ``wire``/``wires``.  Rates,
    Hamiltonian coefficients, and time values must use the same inverse-time
    unit.  The named amplitude-damping operator is ``sqrt(rate) * |0><1|``;
    rates are never interpreted as channel probabilities.

    ``method="runge-kutta"`` takes two explicit fourth-order Runge--Kutta steps
    between every pair of output times.  ``method="crank-nicolson"`` takes two
    second-order trapezoidal steps that are solved matrix-free, so a stiff
    generator stays stable at a step where the explicit scheme would diverge;
    ``solve_tolerance`` sets the relative residual that solve must reach.
    ``method="krylov-exponential"`` advances each output interval by one
    matrix-free action of the generator's exponential, which has no step-size
    error for a time-independent generator, and reports ``order=None`` because
    its accuracy is set by ``solve_tolerance`` rather than by an order.

    An ``initial_state`` with a leading batch axis evolves every member under
    the same Hamiltonian, so one backward pass covers the whole batch. The
    trajectory then reports one population row per time and member. Every step
    is written with differentiable ``torch`` operations, so the returned
    populations stay connected to a differentiable ``initial_state`` and
    Hamiltonian. ``require_gradients`` asserts that connection and raises
    instead of returning a detached trajectory that still looks correct.

    Args:
        batch_size: The declared batch axis. ``None`` uses the batch the
            ``initial_state`` itself carries; a declared batch that disagrees
            with it is refused.
        require_gradients: Refuse to return a trajectory that carries no
            gradient graph. A plan deserialized from JSON cannot supply live
            tensors, so it fails here instead of silently detaching.

    Examples:
        >>> import torch
        >>> from flagquantum.simulation import amplitude_damping, evolve_density_matrix
        >>> times = torch.linspace(0.0, 1.0, 5)
        >>> result = evolve_density_matrix(
        ...     0.0 * torch.eye(2), [[0.0, 0.0], [0.0, 1.0]], 1, times,
        ...     [amplitude_damping(rate=2.0, wire=0)], method="crank-nicolson",
        ... )
        >>> result.method, result.order
        ('crank-nicolson', 2)
    """

    if isinstance(n_wires, bool) or not isinstance(n_wires, int) or n_wires < 1:
        raise _error("invalid_n_wires", "n_wires must be a positive integer", "n_wires")
    if dtype not in {torch.complex64, torch.complex128}:
        raise _error(
            "unsupported_dtype",
            "dtype must be torch.complex64 or torch.complex128",
            "dtype",
        )
    if method not in INTEGRATOR_NAMES:
        raise _error(
            "unsupported_method",
            "method must be one of "
            + ", ".join(repr(name) for name in INTEGRATOR_NAMES)
            + f", not {method!r}",
            "method",
        )
    resolved_require_gradients = _require_gradients_flag(require_gradients)
    resolved_solve_tolerance = _solve_tolerance(
        solve_tolerance, dtype=dtype, method=method
    )
    resolved_device = torch.device(device)
    if resolved_device.type != "cpu":
        raise _error(
            "unsupported_device",
            "continuous-time evolution currently supports CPU only",
            "device",
        )
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    validation_tolerance = 1e-5 if dtype == torch.complex64 else 1e-10
    population_tolerance = 2e-5 if dtype == torch.complex64 else 1e-9
    trace_tolerance = 2e-5 if dtype == torch.complex64 else 1e-9
    dim = 2**n_wires

    time_grid = _normalize_times(times, device=resolved_device, real_dtype=real_dtype)
    hamiltonian_representation = _normalize_hamiltonian(
        hamiltonian,
        n_wires=n_wires,
        dim=dim,
        dtype=dtype,
        device=resolved_device,
        tolerance=validation_tolerance,
    )
    rho = _normalize_initial_state(
        initial_state,
        dim=dim,
        dtype=dtype,
        device=resolved_device,
        tolerance=validation_tolerance,
    )
    # A declared batch must agree with the batch the initial state carries. The
    # resolved value is not needed afterwards, because every later stage reads
    # the axis off ``rho``; the refusal is the point of the call.
    _resolve_batch_size(batch_size, 1 if rho.ndim == 2 else int(rho.shape[0]))
    collapse = _normalize_collapse_operators(
        collapse_operators,
        n_wires=n_wires,
        dim=dim,
        dtype=dtype,
        device=resolved_device,
    )
    observable_matrices = _normalize_observables(
        observables, dim=dim, dtype=dtype, device=resolved_device
    )
    generator = Liouvillian(hamiltonian_representation, collapse, hilbert_dimension=dim)

    states = [rho]
    residuals = []
    for start, stop in zip(time_grid[:-1], time_grid[1:], strict=True):
        rho, residual = advance_interval(
            generator.derivative,
            rho,
            stop - start,
            method=method,
            solve_tolerance=resolved_solve_tolerance,
            solve_max_iterations=_solve_max_iterations(dim),
        )
        residuals.append(residual)
        states.append(rho)
    if residuals and max(residuals) > resolved_solve_tolerance:
        raise _error(
            "integration_not_converged",
            "the Krylov step did not reach the requested relative defect: "
            f"{max(residuals):.3e} > {resolved_solve_tolerance:.3e}",
            "solve_tolerance",
        )
    trajectory = torch.stack(states)
    # The requirement is on the returned trajectory, not on either input: a
    # caller may parameterize the Hamiltonian and hold a fixed initial state,
    # or the other way round. Only the trajectory knows whether the graph
    # survived, and a detached trajectory is indistinguishable by value, so
    # this is the one place the requirement can be decided.
    if resolved_require_gradients and not trajectory.requires_grad:
        raise _error(
            "gradients_not_available",
            "require_gradients was requested but the trajectory carries no "
            "gradient graph; a plan restores its request from JSON, which holds "
            "detached tensors",
            "require_gradients",
        )
    populations = torch.real(torch.diagonal(trajectory, dim1=-2, dim2=-1))
    traces = torch.diagonal(trajectory, dim1=-2, dim2=-1).sum(dim=-1)
    maximum_trace_drift = float(torch.max(torch.abs(traces - 1.0)).item())
    population_bounded = bool(
        (
            (populations >= -population_tolerance)
            & (populations <= 1.0 + population_tolerance)
        )
        .all()
        .item()
    )
    observable_values = {
        name: torch.real(
            torch.einsum(
                "tij,ji->t" if trajectory.ndim == 3 else "tbij,ji->tb",
                trajectory,
                matrix,
            )
        )
        for name, matrix in observable_matrices.items()
    }
    step_size = (
        0.0
        if time_grid.numel() == 1
        else float(
            (
                torch.max(time_grid[1:] - time_grid[:-1])
                / substeps_per_interval(method)
            ).item()
        )
    )
    return EvolutionResult(
        times=time_grid,
        populations=populations,
        observables=observable_values,
        density_matrices=trajectory if return_density_matrices else None,
        maximum_trace_drift=maximum_trace_drift,
        population_bounded=population_bounded,
        method=method,
        order=integrator_order(method),
        step_size=step_size,
        precision="complex64" if dtype == torch.complex64 else "complex128",
        population_tolerance=population_tolerance,
        trace_tolerance=trace_tolerance,
    )


__all__ = (
    "CollapseOperator",
    "EvolutionPlan",
    "EvolutionResult",
    "EvolutionValidationError",
    "amplitude_damping",
    "evolve_density_matrix",
    "plan_density_matrix_evolution",
)

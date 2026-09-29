"""Deterministic continuous-time Lindblad evolution on dense states."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from ..errors import ExecutionError, ValidationError
from ..observables import Observable
from .density_matrix import expand_operator


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
    order: int
    step_size: float
    precision: str
    population_tolerance: float
    trace_tolerance: float

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
        }


@dataclass(frozen=True)
class EvolutionPlan:
    """Validated resource and numerical-method plan for an evolution request."""

    n_wires: int
    n_times: int
    dimension: int
    trajectory_bytes: int
    method: str
    order: int
    step_size: float
    device: str
    precision: str

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
            "numerics": {
                "method": self.method,
                "order": self.order,
                "step_size": self.step_size,
                "device": self.device,
                "precision": self.precision,
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
        norm = torch.sum(torch.abs(state) ** 2)
        if abs(float(norm.item()) - 1.0) > tolerance:
            raise _error(
                "unnormalized_initial_state",
                "initial_state statevector must have unit norm",
                "initial_state",
            )
        return torch.outer(state, torch.conj(state))
    if state.ndim != 2 or tuple(state.shape) != (dim, dim):
        raise _error(
            "invalid_shape",
            f"initial_state must have shape ({dim},) or ({dim}, {dim})",
            "initial_state",
        )
    if not torch.allclose(state, torch.conj(state).T, atol=tolerance, rtol=0.0):
        raise _error(
            "non_hermitian_initial_state",
            "initial_state density matrix must be Hermitian",
            "initial_state",
        )
    trace = torch.trace(state).real
    if abs(float(trace.item()) - 1.0) > tolerance:
        raise _error(
            "unnormalized_initial_state",
            "initial_state density matrix must have unit trace",
            "initial_state",
        )
    eigenvalues = torch.linalg.eigvalsh(state)
    if float(torch.min(eigenvalues).item()) < -tolerance:
        raise _error(
            "non_positive_initial_state",
            "initial_state density matrix must be positive semidefinite",
            "initial_state",
        )
    return state


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


def _descriptor_matrix(
    descriptor: Mapping[str, Any],
    *,
    n_wires: int,
    dtype: torch.dtype,
    device: torch.device,
    field: str,
) -> torch.Tensor:
    pauli = descriptor.get("pauli", descriptor.get("operator"))
    if not isinstance(pauli, str):
        raise _error(
            "invalid_operator", "operator descriptors require a Pauli string", field
        )
    raw_wires = descriptor.get("wires")
    wires = tuple(range(len(pauli))) if raw_wires is None else tuple(raw_wires)
    try:
        normalized_wires = tuple(int(wire) for wire in wires)
    except (TypeError, ValueError) as exc:
        raise _error(
            "invalid_operator", "operator wires must be integers", field
        ) from exc
    return _pauli_operator(
        pauli,
        normalized_wires,
        n_wires=n_wires,
        dtype=dtype,
        device=device,
        field=field,
    )


def _normalize_hamiltonian(
    hamiltonian: Any,
    *,
    n_wires: int,
    dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(hamiltonian, Observable):
        return _observable_matrix(
            hamiltonian,
            n_wires=n_wires,
            dtype=dtype,
            device=device,
            field="hamiltonian",
        )
    if (
        isinstance(hamiltonian, Sequence)
        and not isinstance(hamiltonian, (str, bytes, torch.Tensor))
        and hamiltonian
        and isinstance(hamiltonian[0], Mapping)
    ):
        matrix = torch.zeros((dim, dim), dtype=dtype, device=device)
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
            matrix = matrix + coefficient * _descriptor_matrix(
                term,
                n_wires=n_wires,
                dtype=dtype,
                device=device,
                field="hamiltonian",
            )
        return matrix
    return _as_complex_matrix(
        hamiltonian,
        name="hamiltonian",
        dim=dim,
        dtype=dtype,
        device=device,
    )


def _observable_matrix(
    observable: Observable,
    *,
    n_wires: int,
    dtype: torch.dtype,
    device: torch.device,
    field: str,
) -> torch.Tensor:
    dim = 2**n_wires
    matrix = torch.zeros((dim, dim), dtype=dtype, device=device)
    identity = torch.eye(dim, dtype=dtype, device=device)
    for term in observable.terms:
        if not term.factors:
            matrix = matrix + term.coefficient * identity
            continue
        wires = tuple(wire for wire, _ in term.factors)
        pauli = "".join(axis for _, axis in term.factors)
        matrix = matrix + term.coefficient * _pauli_operator(
            pauli,
            wires,
            n_wires=n_wires,
            dtype=dtype,
            device=device,
            field=field,
        )
    return matrix


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


def _normalize_collapse_operators(
    collapse_operators: Sequence[Any] | None,
    *,
    n_wires: int,
    dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
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
        result.append((numeric_rate**0.5) * full)
    return tuple(result)


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
    h_matrix = _normalize_hamiltonian(
        hamiltonian,
        n_wires=n_wires,
        dim=dim,
        dtype=dtype,
        device=resolved_device,
    )
    if not torch.allclose(h_matrix, torch.conj(h_matrix).T, atol=tolerance, rtol=0.0):
        raise _error(
            "non_hermitian_hamiltonian",
            "hamiltonian must be Hermitian",
            "hamiltonian",
        )
    _normalize_initial_state(
        initial_state,
        dim=dim,
        dtype=dtype,
        device=resolved_device,
        tolerance=tolerance,
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
        else float((torch.max(time_grid[1:] - time_grid[:-1]) / 2).item())
    )
    element_bytes = 8 if dtype == torch.complex64 else 16
    return EvolutionPlan(
        n_wires=n_wires,
        n_times=int(time_grid.numel()),
        dimension=dim,
        trajectory_bytes=int(time_grid.numel()) * dim * dim * element_bytes,
        method="runge-kutta",
        order=4,
        step_size=step_size,
        device=str(resolved_device),
        precision="complex64" if dtype == torch.complex64 else "complex128",
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
) -> EvolutionResult:
    """Evolve a state on an explicit time grid with the Lindblad equation.

    Collapse entries are either ``(operator, rate)`` pairs or mappings with
    ``operator`` (a dense matrix, ``"amplitude_damping"``, or
    ``"sigma_minus"``), ``rate``, and optional ``wire``/``wires``.  Rates,
    Hamiltonian coefficients, and time values must use the same inverse-time
    unit.  The named amplitude-damping operator is ``sqrt(rate) * |0><1|``;
    rates are never interpreted as channel probabilities.

    Two classical fourth-order Runge--Kutta steps are taken between every pair
    of output times.  Consequently the explicit output grid controls the
    integration step size and scales with the physical problem.
    """

    if isinstance(n_wires, bool) or not isinstance(n_wires, int) or n_wires < 1:
        raise _error("invalid_n_wires", "n_wires must be a positive integer", "n_wires")
    if dtype not in {torch.complex64, torch.complex128}:
        raise _error(
            "unsupported_dtype",
            "dtype must be torch.complex64 or torch.complex128",
            "dtype",
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
    h_matrix = _normalize_hamiltonian(
        hamiltonian,
        n_wires=n_wires,
        dim=dim,
        dtype=dtype,
        device=resolved_device,
    )
    if not torch.allclose(
        h_matrix, torch.conj(h_matrix).T, atol=validation_tolerance, rtol=0.0
    ):
        raise _error(
            "non_hermitian_hamiltonian",
            "hamiltonian must be Hermitian",
            "hamiltonian",
        )
    rho = _normalize_initial_state(
        initial_state,
        dim=dim,
        dtype=dtype,
        device=resolved_device,
        tolerance=validation_tolerance,
    )
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
    collapse_products = tuple(torch.conj(op).T @ op for op in collapse)

    def derivative(state: torch.Tensor) -> torch.Tensor:
        value = -1j * (h_matrix @ state - state @ h_matrix)
        for operator, product in zip(collapse, collapse_products, strict=True):
            value = value + operator @ state @ torch.conj(operator).T
            value = value - 0.5 * (product @ state + state @ product)
        return value

    states = [rho]
    for start, stop in zip(time_grid[:-1], time_grid[1:], strict=True):
        step = ((stop - start) / 2).to(dtype=real_dtype)
        for _ in range(2):
            k1 = derivative(rho)
            k2 = derivative(rho + step * k1 / 2)
            k3 = derivative(rho + step * k2 / 2)
            k4 = derivative(rho + step * k3)
            rho = rho + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        states.append(rho)
    trajectory = torch.stack(states)
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
        name: torch.real(torch.einsum("tij,ji->t", trajectory, matrix))
        for name, matrix in observable_matrices.items()
    }
    step_size = (
        0.0
        if time_grid.numel() == 1
        else float((torch.max(time_grid[1:] - time_grid[:-1]) / 2).item())
    )
    return EvolutionResult(
        times=time_grid,
        populations=populations,
        observables=observable_values,
        density_matrices=trajectory if return_density_matrices else None,
        maximum_trace_drift=maximum_trace_drift,
        population_bounded=population_bounded,
        method="runge-kutta",
        order=4,
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

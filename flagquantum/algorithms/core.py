"""Native algorithm utilities for FlagQuantum.

The functions in this module are deliberately small building blocks: they keep
Hamiltonians, ansatz circuits, and losses inside FlagQuantum's own circuit,
IR, MPS, and PyTorch runtime instead of depending on external chemistry,
optimization, or graph packages.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch

from ..circuit import Circuit, _apply_matrix
from ..ops.matrices import GATE_MAT_DICT, get_global_precision
from ..simulation.mps import MPSState
from .optimization import (
    HybridOptimizationResult,
    OptimizationStage,
    optimize_hybrid,
)

_PAULI_NAMES = {"i", "x", "y", "z"}


def _as_wire_tuple(wires: Iterable[int] | int | None) -> tuple[int, ...]:
    if wires is None:
        return ()
    if isinstance(wires, int):
        return (wires,)
    return tuple(int(wire) for wire in wires)


def _normalize_pauli(
    pauli: str | Mapping[int, str],
    wires: Iterable[int] | int | None,
) -> tuple[tuple[int, str], ...]:
    if isinstance(pauli, Mapping):
        items = tuple((int(wire), str(name).lower()) for wire, name in pauli.items())
    else:
        wire_tuple = _as_wire_tuple(wires)
        if len(pauli) != len(wire_tuple):
            raise ValueError("Pauli string length must match wires length.")
        items = tuple((wire, name.lower()) for wire, name in zip(wire_tuple, pauli))

    normalized = []
    seen = set()
    for wire, name in items:
        if name not in _PAULI_NAMES:
            raise ValueError("Pauli operators must be one of I, X, Y, or Z.")
        if wire in seen:
            raise ValueError("A Hamiltonian term cannot repeat a wire.")
        seen.add(wire)
        if name != "i":
            normalized.append((wire, name))
    return tuple(sorted(normalized))


def _coefficient_tensor(
    coefficient: float | complex | torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    return torch.as_tensor(coefficient, dtype=dtype, device=device)


def _infer_n_wires_from_state(state: torch.Tensor) -> int:
    dim = state.shape[-1]
    n_wires = int(torch.log2(torch.tensor(dim, dtype=torch.float64)).item())
    if 2**n_wires != dim:
        raise ValueError("State dimension must be a power of two.")
    return n_wires


def _expectation_from_statevector(
    state: torch.Tensor,
    term: "HamiltonianTerm",
    n_wires: int,
) -> torch.Tensor:
    batched = state.reshape(1, -1) if state.ndim == 1 else state
    transformed = batched
    for wire, name in term.ops:
        matrix = GATE_MAT_DICT[name].to(device=batched.device, dtype=batched.dtype)
        transformed = _apply_matrix(transformed, matrix, (wire,), n_wires)
    value = (torch.conj(batched) * transformed).sum(dim=-1)
    return torch.real(value)


def _pauli_operator(
    term: "HamiltonianTerm",
    n_wires: int,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    op = torch.ones(1, 1, dtype=dtype, device=device)
    ops = {wire: name for wire, name in term.ops}
    for wire in range(n_wires):
        matrix = GATE_MAT_DICT[ops.get(wire, "i")].to(device=device, dtype=dtype)
        op = torch.kron(op, matrix)
    return op


def _expectation_from_density(
    rho: torch.Tensor,
    term: "HamiltonianTerm",
    n_wires: int,
) -> torch.Tensor:
    batch = rho.reshape(1, *rho.shape) if rho.ndim == 2 else rho
    op = _pauli_operator(term, n_wires, dtype=batch.dtype, device=batch.device)
    values = torch.diagonal(torch.matmul(batch, op), dim1=-2, dim2=-1).sum(dim=-1)
    return torch.real(values)


@dataclass(frozen=True)
class HamiltonianTerm:
    """One weighted Pauli product term."""

    coefficient: float | complex | torch.Tensor
    pauli: str | Mapping[int, str]
    wires: tuple[int, ...] = ()

    def __init__(
        self,
        coefficient: float | complex | torch.Tensor,
        pauli: str | Mapping[int, str],
        wires: Iterable[int] | int | None = None,
    ) -> None:
        object.__setattr__(self, "coefficient", coefficient)
        object.__setattr__(self, "pauli", pauli)
        object.__setattr__(self, "wires", _as_wire_tuple(wires))
        object.__setattr__(self, "ops", _normalize_pauli(pauli, wires))

    @property
    def max_wire(self) -> int:
        if not self.ops:
            return -1
        return max(wire for wire, _ in self.ops)

    def expectation(self, target: Circuit | MPSState | torch.Tensor) -> torch.Tensor:
        """Evaluate this term on a circuit, MPS state, statevector, or density matrix."""

        if isinstance(target, Circuit):
            base = (
                torch.ones(
                    target.bsz, dtype=torch.float32, device=target.state().device
                )
                if not self.ops
                else target.expectation_ps(
                    x=[wire for wire, name in self.ops if name == "x"],
                    y=[wire for wire, name in self.ops if name == "y"],
                    z=[wire for wire, name in self.ops if name == "z"],
                )
            )
            coeff = _coefficient_tensor(
                self.coefficient,
                dtype=base.dtype,
                device=base.device,
            )
            return torch.real(coeff * base)

        if isinstance(target, MPSState):
            base = (
                torch.ones(
                    target.bsz, dtype=torch.float32, device=target.tensors[0].device
                )
                if not self.ops
                else target.expectation_ps(
                    x=[wire for wire, name in self.ops if name == "x"],
                    y=[wire for wire, name in self.ops if name == "y"],
                    z=[wire for wire, name in self.ops if name == "z"],
                )
            )
            coeff = _coefficient_tensor(
                self.coefficient, dtype=base.dtype, device=base.device
            )
            return torch.real(coeff * base)

        tensor = target
        if tensor.ndim >= 2 and tensor.shape[-1] == tensor.shape[-2]:
            n_wires = _infer_n_wires_from_state(tensor)
            base = (
                torch.ones(
                    tensor.shape[0] if tensor.ndim == 3 else 1,
                    dtype=torch.float32,
                    device=tensor.device,
                )
                if not self.ops
                else _expectation_from_density(tensor, self, n_wires)
            )
        else:
            n_wires = _infer_n_wires_from_state(tensor)
            base = (
                torch.ones(
                    tensor.shape[0] if tensor.ndim == 2 else 1,
                    dtype=torch.float32,
                    device=tensor.device,
                )
                if not self.ops
                else _expectation_from_statevector(tensor, self, n_wires)
            )
        coeff = _coefficient_tensor(
            self.coefficient, dtype=base.dtype, device=base.device
        )
        return torch.real(coeff * base)


class Hamiltonian:
    """A differentiable weighted sum of Pauli products."""

    def __init__(self, terms: Iterable[HamiltonianTerm]) -> None:
        self.terms = tuple(terms)
        if not self.terms:
            raise ValueError("Hamiltonian requires at least one term.")

    @property
    def n_terms(self) -> int:
        return len(self.terms)

    @property
    def n_wires(self) -> int:
        return 1 + max((term.max_wire for term in self.terms), default=-1)

    def expectation(self, target: Circuit | MPSState | torch.Tensor) -> torch.Tensor:
        if isinstance(target, MPSState):
            z_coefficients: dict[int, Any] = {}
            zz_coefficients: dict[int, Any] = {}
            chain_compatible = True
            for term in self.terms:
                ops = tuple(term.ops)
                if len(ops) == 1 and ops[0][1] == "z":
                    wire = int(ops[0][0])
                    z_coefficients[wire] = (
                        z_coefficients.get(wire, 0.0) + term.coefficient
                    )
                elif (
                    len(ops) == 2
                    and ops[0][1] == "z"
                    and ops[1][1] == "z"
                    and int(ops[1][0]) == int(ops[0][0]) + 1
                ):
                    wire = int(ops[0][0])
                    zz_coefficients[wire] = (
                        zz_coefficients.get(wire, 0.0) + term.coefficient
                    )
                else:
                    chain_compatible = False
                    break
            if chain_compatible:
                return target.expectation_z_zz_chain(
                    z_coefficients=z_coefficients,
                    zz_coefficients=zz_coefficients,
                )
        values = [term.expectation(target) for term in self.terms]
        total = values[0]
        for value in values[1:]:
            total = total + value
        return total

    def matrix(
        self,
        *,
        dtype: torch.dtype = torch.complex128,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """Materialize a dense operator for small-system diagnostics."""

        if self.n_wires > 12:
            raise ValueError("dense Hamiltonian diagnostics are limited to 12 wires")
        dimension = 1 << self.n_wires
        result = torch.zeros(dimension, dimension, dtype=dtype, device=device)
        for term in self.terms:
            coefficient = torch.as_tensor(term.coefficient, dtype=dtype, device=device)
            result = result + coefficient * _pauli_operator(
                term, self.n_wires, dtype=dtype, device=device
            )
        return result

    def ground_energy(self) -> torch.Tensor:
        """Return the exact small-system ground energy."""

        return torch.linalg.eigvalsh(self.matrix()).min().real


@dataclass(frozen=True)
class VQEResult:
    """Result returned by the native VQE optimizer."""

    parameters: torch.Tensor
    energy: torch.Tensor
    history: tuple[float, ...]

    @property
    def n_steps(self) -> int:
        return len(self.history)


@dataclass(frozen=True)
class AdaptVQEIteration:
    """One operator-selection and variational re-optimization stage."""

    selected_pool_index: int
    selected_gradient: float
    pool_gradients: tuple[float, ...]
    energy_before: float
    energy_after: float
    optimization_history: tuple[float, ...]
    screening_seconds: float
    optimization_seconds: float


@dataclass(frozen=True)
class AdaptVQEResult:
    """Result of a deterministic PyTorch-native ADAPT-VQE growth loop."""

    selected_pool_indices: tuple[int, ...]
    parameters: torch.Tensor
    energy: torch.Tensor
    initial_energy: float
    iterations: tuple[AdaptVQEIteration, ...]
    converged: bool

    @property
    def n_adapt_iterations(self) -> int:
        return len(self.iterations)


def pauli_term(
    coefficient: float | complex | torch.Tensor,
    pauli: str | Mapping[int, str],
    wires: Iterable[int] | int | None = None,
) -> HamiltonianTerm:
    """Create a weighted Pauli product term."""

    return HamiltonianTerm(coefficient, pauli, wires)


def transverse_field_ising(
    n_wires: int,
    *,
    coupling: float = 1.0,
    field: float = 1.0,
    periodic: bool = False,
) -> Hamiltonian:
    """Build ``-coupling * ZZ - field * X`` transverse-field Ising Hamiltonian."""

    terms: list[HamiltonianTerm] = []
    for wire in range(n_wires - 1):
        terms.append(pauli_term(-coupling, "ZZ", (wire, wire + 1)))
    if periodic and n_wires > 2:
        terms.append(pauli_term(-coupling, "ZZ", (n_wires - 1, 0)))
    for wire in range(n_wires):
        terms.append(pauli_term(-field, "X", (wire,)))
    return Hamiltonian(terms)


def zz_chain_hamiltonian(
    n_wires: int,
    *,
    coupling: float = 1.0,
    field: float = 0.0,
    periodic: bool = False,
) -> Hamiltonian:
    """Build a nearest-neighbor ``coupling * ZZ + field * Z`` Hamiltonian."""

    terms: list[HamiltonianTerm] = []
    for wire in range(n_wires - 1):
        terms.append(pauli_term(coupling, "ZZ", (wire, wire + 1)))
    if periodic and n_wires > 2:
        terms.append(pauli_term(coupling, "ZZ", (n_wires - 1, 0)))
    for wire in range(n_wires):
        if field:
            terms.append(pauli_term(field, "Z", (wire,)))
    return Hamiltonian(terms)


def hardware_efficient_parameter_count(
    n_wires: int,
    layers: int,
    rotations: Sequence[str] = ("ry", "rz"),
) -> int:
    return int(n_wires) * int(layers) * len(tuple(rotations))


def hardware_efficient_ansatz(
    n_wires: int,
    layers: int,
    parameters: torch.Tensor | Sequence[float],
    *,
    rotations: Sequence[str] = ("ry", "rz"),
    entanglement: str = "linear",
    device: torch.device | str = "cpu",
) -> Circuit:
    """Build a common differentiable layered ansatz."""

    params = torch.as_tensor(parameters, dtype=torch.float32, device=device).reshape(-1)
    expected = hardware_efficient_parameter_count(n_wires, layers, rotations)
    if params.numel() != expected:
        raise ValueError(f"Expected {expected} parameters, got {params.numel()}.")

    circuit = Circuit(n_wires, device=device, dtype=get_global_precision())
    cursor = 0
    for _ in range(layers):
        for wire in range(n_wires):
            for rotation in rotations:
                getattr(circuit, rotation.lower())(wire, theta=params[cursor])
                cursor += 1
        if entanglement == "linear":
            for wire in range(n_wires - 1):
                circuit.cx(wire, wire + 1)
        elif entanglement == "circular":
            for wire in range(n_wires - 1):
                circuit.cx(wire, wire + 1)
            if n_wires > 2:
                circuit.cx(n_wires - 1, 0)
        elif entanglement != "none":
            raise ValueError("entanglement must be 'linear', 'circular', or 'none'.")
    return circuit


def qaoa_circuit(
    n_wires: int,
    edges: Iterable[tuple[int, int] | tuple[int, int, float]],
    gammas: torch.Tensor | Sequence[float],
    betas: torch.Tensor | Sequence[float],
    *,
    device: torch.device | str = "cpu",
) -> Circuit:
    """Build a QAOA circuit for weighted ZZ cost edges."""

    gamma_values = torch.as_tensor(gammas, dtype=torch.float32, device=device).reshape(
        -1
    )
    beta_values = torch.as_tensor(betas, dtype=torch.float32, device=device).reshape(-1)
    if gamma_values.numel() != beta_values.numel():
        raise ValueError("gammas and betas must have the same number of layers.")

    edge_tuple = tuple(edges)
    circuit = Circuit(n_wires, device=device, dtype=get_global_precision())
    for wire in range(n_wires):
        circuit.h(wire)

    for gamma, beta in zip(gamma_values, beta_values):
        for edge in edge_tuple:
            if len(edge) == 2:
                src, dst = edge
                weight = 1.0
            else:
                src, dst, weight = edge
            circuit.rzz(int(src), int(dst), theta=2.0 * gamma * float(weight))
        for wire in range(n_wires):
            circuit.rx(wire, theta=2.0 * beta)
    return circuit


def vqe_loss(
    circuit_builder: Callable[[torch.Tensor], Circuit],
    parameters: torch.Tensor,
    hamiltonian: Hamiltonian,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Evaluate a differentiable VQE objective."""

    values = hamiltonian.expectation(circuit_builder(parameters))
    if reduction == "none":
        return values
    if reduction == "sum":
        return values.sum()
    if reduction == "mean":
        return values.mean()
    raise ValueError("reduction must be 'mean', 'sum', or 'none'.")


def heisenberg_chain_hamiltonian(
    n_wires: int, *, anisotropy: float = 1.0, field: float = 0.0
) -> Hamiltonian:
    """Return the open-boundary XX + YY + anisotropy ZZ Hamiltonian."""

    if n_wires < 2:
        raise ValueError("Heisenberg chains require at least two wires")
    terms: list[HamiltonianTerm] = []
    for left in range(n_wires - 1):
        terms.extend(
            (
                pauli_term(1.0, "XX", (left, left + 1)),
                pauli_term(1.0, "YY", (left, left + 1)),
                pauli_term(anisotropy, "ZZ", (left, left + 1)),
            )
        )
    if field:
        terms.extend(pauli_term(field, "Z", (wire,)) for wire in range(n_wires))
    return Hamiltonian(terms)


def heisenberg_hva_parameter_count(
    n_wires: int, depth: int, *, parameterization: str = "bond_resolved_phase"
) -> int:
    if n_wires < 2 or depth < 1:
        raise ValueError("n_wires >= 2 and depth >= 1 are required")
    if parameterization == "shared_parity":
        return 6 * depth
    if parameterization == "bond_resolved":
        return 3 * (n_wires - 1) * depth
    if parameterization == "bond_resolved_phase":
        return (3 * (n_wires - 1) + n_wires) * depth
    raise ValueError("unknown Heisenberg HVA parameterization")


def heisenberg_hva(
    n_wires: int,
    depth: int,
    parameters: torch.Tensor | Sequence[float],
    *,
    parameterization: str = "bond_resolved_phase",
    initial_state: str = "dimer_singlet",
) -> Circuit:
    """Build an open-chain HVA with a physics-informed initial state."""

    values = torch.as_tensor(parameters)
    expected = heisenberg_hva_parameter_count(
        n_wires, depth, parameterization=parameterization
    )
    if values.numel() != expected:
        raise ValueError(f"expected {expected} HVA parameters, got {values.numel()}")
    values = values.reshape(-1)
    circuit = Circuit(n_wires, device=values.device)
    if initial_state == "neel":
        for wire in range(1, n_wires, 2):
            circuit.x(wire)
    elif initial_state == "dimer_singlet":
        for left in range(0, n_wires - 1, 2):
            circuit.x(left + 1).h(left).cx(left, left + 1).z(left)
    else:
        raise ValueError("initial_state must be 'neel' or 'dimer_singlet'")

    for layer in range(depth):
        if parameterization == "shared_parity":
            offset = 6 * layer
            for axis, gate in enumerate((circuit.rxx, circuit.ryy, circuit.rzz)):
                for parity in (0, 1):
                    theta = values[offset + 2 * axis + parity]
                    for left in range(parity, n_wires - 1, 2):
                        gate(left, left + 1, theta=theta)
            continue
        per_layer = 3 * (n_wires - 1) + (
            n_wires if parameterization == "bond_resolved_phase" else 0
        )
        offset = per_layer * layer
        for axis, gate in enumerate((circuit.rxx, circuit.ryy, circuit.rzz)):
            for left in range(n_wires - 1):
                gate(
                    left,
                    left + 1,
                    theta=values[offset + axis * (n_wires - 1) + left],
                )
        if parameterization == "bond_resolved_phase":
            phase_offset = offset + 3 * (n_wires - 1)
            for wire in range(n_wires):
                circuit.rz(wire, theta=values[phase_offset + wire])
    return circuit


def run_vqe(
    circuit_builder: Callable[[torch.Tensor], Circuit],
    initial_parameters: torch.Tensor | Sequence[float],
    hamiltonian: Hamiltonian,
    *,
    steps: int = 100,
    lr: float = 0.05,
    optimizer_cls: type[torch.optim.Optimizer] = torch.optim.Adam,
) -> VQEResult:
    """Run a compact PyTorch-native VQE optimization loop."""

    parameters = torch.as_tensor(initial_parameters, dtype=torch.float32).clone()
    parameters = parameters.detach().requires_grad_(True)
    optimizer = optimizer_cls([parameters], lr=lr)
    history: list[float] = []

    for _ in range(int(steps)):
        optimizer.zero_grad()
        loss = vqe_loss(circuit_builder, parameters, hamiltonian)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))

    final_energy = vqe_loss(circuit_builder, parameters, hamiltonian).detach()
    return VQEResult(
        parameters=parameters.detach().clone(),
        energy=final_energy,
        history=tuple(history),
    )


def run_adapt_vqe(
    circuit_builder: Callable[[tuple[object, ...], torch.Tensor], Circuit],
    operator_pool: Sequence[object],
    hamiltonian: Hamiltonian | None = None,
    *,
    energy_function: Callable[[Circuit], torch.Tensor] | None = None,
    screening_function: (
        Callable[
            [tuple[object, ...], torch.Tensor, tuple[object, ...]], Sequence[float]
        ]
        | None
    ) = None,
    max_adapt_iterations: int = 8,
    optimization_steps: int = 50,
    lr: float = 0.05,
    gradient_tolerance: float = 1e-6,
    optimizer_cls: type[torch.optim.Optimizer] = torch.optim.Adam,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> AdaptVQEResult:
    """Grow and optimize an ansatz using exact candidate gradients.

    ``circuit_builder`` receives the selected operator records and one scalar
    parameter per selected operator. Candidate operators are screened by
    appending them at zero angle and differentiating the exact Hamiltonian
    expectation. The selected operator is then retained and all accumulated
    parameters are re-optimized. ``screening_function`` may provide an exact,
    workload-specific gradient evaluator; it receives the current operators,
    parameters, and the currently available pool operators in that order.
    """

    pool = tuple(operator_pool)
    if not pool:
        raise ValueError("ADAPT-VQE requires a non-empty operator pool")
    if max_adapt_iterations <= 0:
        raise ValueError("max_adapt_iterations must be positive")
    if optimization_steps <= 0:
        raise ValueError("optimization_steps must be positive")
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise ValueError("ADAPT-VQE parameters require a floating dtype")
    tolerance = float(gradient_tolerance)
    if tolerance < 0.0:
        raise ValueError("gradient_tolerance must be non-negative")
    if (hamiltonian is None) == (energy_function is None):
        raise ValueError("provide exactly one of hamiltonian or energy_function")

    selected_indices: list[int] = []
    selected_operators: list[object] = []
    parameters = torch.empty(0, dtype=dtype, device=device)

    def energy(operators: tuple[object, ...], values: torch.Tensor) -> torch.Tensor:
        circuit = circuit_builder(operators, values)
        evaluated = (
            hamiltonian.expectation(circuit)
            if hamiltonian is not None
            else energy_function(circuit)
        )
        return evaluated.sum()

    initial_energy = float(energy((), parameters).detach())
    records: list[AdaptVQEIteration] = []
    converged = False
    for _ in range(int(max_adapt_iterations)):
        screening_started = perf_counter()
        available = [
            pool_index
            for pool_index in range(len(pool))
            if pool_index not in selected_indices
        ]
        if not available:
            converged = True
            break
        if screening_function is None:
            gradients: list[float] = []
            for pool_index in available:
                candidate_values = torch.cat(
                    (parameters.detach(), parameters.new_zeros(1))
                ).requires_grad_(True)
                candidate_energy = energy(
                    (*selected_operators, pool[pool_index]), candidate_values
                )
                candidate_gradient = torch.autograd.grad(
                    candidate_energy, candidate_values
                )[0][-1]
                gradients.append(float(candidate_gradient.detach()))
        else:
            evaluated = tuple(
                screening_function(
                    tuple(selected_operators),
                    parameters.detach(),
                    tuple(pool[index] for index in available),
                )
            )
            if len(evaluated) != len(available):
                raise ValueError(
                    "screening_function must return one gradient per candidate"
                )
            gradients = [float(gradient) for gradient in evaluated]
        best_position = max(
            range(len(available)), key=lambda index: abs(gradients[index])
        )
        best_index = available[best_position]
        best_gradient = gradients[best_position]
        if abs(best_gradient) <= tolerance:
            converged = True
            break
        screening_seconds = perf_counter() - screening_started

        energy_before = float(energy(tuple(selected_operators), parameters).detach())
        selected_indices.append(best_index)
        selected_operators.append(pool[best_index])
        parameters = torch.cat(
            (parameters.detach(), parameters.new_zeros(1))
        ).requires_grad_(True)
        optimizer = optimizer_cls([parameters], lr=lr)
        history: list[float] = []
        optimization_started = perf_counter()
        for _ in range(int(optimization_steps)):
            optimizer.zero_grad(set_to_none=True)
            loss = energy(tuple(selected_operators), parameters)
            loss.backward()
            optimizer.step()
            history.append(float(loss.detach()))
        optimization_seconds = perf_counter() - optimization_started
        energy_after = float(energy(tuple(selected_operators), parameters).detach())
        full_gradients = [0.0] * len(pool)
        for pool_index, gradient in zip(available, gradients):
            full_gradients[pool_index] = gradient
        records.append(
            AdaptVQEIteration(
                selected_pool_index=best_index,
                selected_gradient=best_gradient,
                pool_gradients=tuple(full_gradients),
                energy_before=energy_before,
                energy_after=energy_after,
                optimization_history=tuple(history),
                screening_seconds=screening_seconds,
                optimization_seconds=optimization_seconds,
            )
        )

    final_energy = energy(tuple(selected_operators), parameters).detach()
    return AdaptVQEResult(
        selected_pool_indices=tuple(selected_indices),
        parameters=parameters.detach().clone(),
        energy=final_energy,
        initial_energy=initial_energy,
        iterations=tuple(records),
        converged=converged,
    )


def run_hybrid_vqe(
    circuit_builder: Callable[[torch.Tensor], Circuit],
    initial_parameters: torch.Tensor | Sequence[float],
    hamiltonian: Hamiltonian,
    *,
    stages: Sequence[OptimizationStage],
) -> HybridOptimizationResult:
    """Run a staged VQE schedule using classical and quantum-aware methods.

    The parameter group is named ``"quantum"``.  Use :func:`optimize_hybrid`
    directly for models with both classical and quantum parameter groups.
    QNG evaluates the exact local statevector metric and is therefore intended
    for small-system convergence work, not distributed scalability claims.
    """

    initial = torch.as_tensor(initial_parameters)
    if not (initial.is_floating_point() or initial.is_complex()):
        initial = initial.to(torch.get_default_dtype())

    def objective(groups: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return vqe_loss(circuit_builder, groups["quantum"], hamiltonian)

    def state_function(groups: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return circuit_builder(groups["quantum"]).state().reshape(-1)

    return optimize_hybrid(
        objective,
        {"quantum": initial},
        stages,
        state_function=state_function,
    )


@dataclass(frozen=True)
class LayerwiseVQEResult:
    """VQE results produced while growing an ansatz depth by depth."""

    depths: tuple[int, ...]
    stages: tuple[HybridOptimizationResult, ...]

    @property
    def parameters(self) -> torch.Tensor:
        return self.stages[-1].parameters["quantum"]

    @property
    def energy(self) -> torch.Tensor:
        return self.stages[-1].energy


def run_layerwise_vqe(
    circuit_builder: Callable[[int, torch.Tensor], Circuit],
    parameter_count: Callable[[int], int],
    initial_parameters: torch.Tensor | Sequence[float],
    hamiltonian: Hamiltonian,
    *,
    depths: Sequence[int],
    stages: Sequence[OptimizationStage],
) -> LayerwiseVQEResult:
    """Grow a variational circuit while preserving optimized earlier layers.

    New coordinates are initialized to zero, so Pauli-rotation layers begin as
    identity operations. This continuation path avoids restarting every deeper
    ansatz from an unrelated random point.
    """

    normalized_depths = tuple(int(depth) for depth in depths)
    if not normalized_depths or any(depth <= 0 for depth in normalized_depths):
        raise ValueError("depths must contain positive integers")
    if tuple(sorted(set(normalized_depths))) != normalized_depths:
        raise ValueError("depths must be strictly increasing")
    parameters = torch.as_tensor(initial_parameters).reshape(-1)
    if not (parameters.is_floating_point() or parameters.is_complex()):
        parameters = parameters.to(torch.get_default_dtype())
    first_count = int(parameter_count(normalized_depths[0]))
    if parameters.numel() != first_count:
        raise ValueError(
            f"initial parameters contain {parameters.numel()} values; expected {first_count}"
        )

    results: list[HybridOptimizationResult] = []
    for depth in normalized_depths:
        required = int(parameter_count(depth))
        if required < parameters.numel():
            raise ValueError("parameter_count must not shrink as depth increases")
        if required > parameters.numel():
            parameters = torch.cat(
                (parameters, parameters.new_zeros(required - parameters.numel()))
            )
        result = run_hybrid_vqe(
            lambda values, selected_depth=depth: circuit_builder(
                selected_depth, values
            ),
            parameters,
            hamiltonian,
            stages=stages,
        )
        results.append(result)
        parameters = result.parameters["quantum"]
    return LayerwiseVQEResult(depths=normalized_depths, stages=tuple(results))


def qaoa_loss(
    n_wires: int,
    edges: Iterable[tuple[int, int] | tuple[int, int, float]],
    gammas: torch.Tensor | Sequence[float],
    betas: torch.Tensor | Sequence[float],
    hamiltonian: Hamiltonian,
    *,
    reduction: str = "mean",
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Evaluate a differentiable QAOA objective."""

    circuit = qaoa_circuit(n_wires, edges, gammas, betas, device=device)
    values = hamiltonian.expectation(circuit)
    if reduction == "none":
        return values
    if reduction == "sum":
        return values.sum()
    if reduction == "mean":
        return values.mean()
    raise ValueError("reduction must be 'mean', 'sum', or 'none'.")


__all__ = [
    "AdaptVQEIteration",
    "AdaptVQEResult",
    "Hamiltonian",
    "HamiltonianTerm",
    "LayerwiseVQEResult",
    "VQEResult",
    "hardware_efficient_ansatz",
    "hardware_efficient_parameter_count",
    "heisenberg_chain_hamiltonian",
    "heisenberg_hva",
    "heisenberg_hva_parameter_count",
    "pauli_term",
    "qaoa_loss",
    "qaoa_circuit",
    "run_vqe",
    "run_adapt_vqe",
    "run_hybrid_vqe",
    "run_layerwise_vqe",
    "transverse_field_ising",
    "vqe_loss",
    "zz_chain_hamiltonian",
]

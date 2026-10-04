"""Native algorithm utilities for FlagQuantum.

The functions in this module are deliberately small building blocks: they keep
Hamiltonians, ansatz circuits, and losses inside FlagQuantum's own circuit,
IR, MPS, and PyTorch runtime instead of depending on external chemistry,
optimization, or graph packages.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from time import perf_counter
from typing import Literal, Protocol

import torch

from ..circuit import Circuit
from ..core.runtime_config import get_runtime_config
from ..simulation.mps.state import MPSState
from ..simulation.pauli import (
    infer_n_wires_from_dense_state,
    pauli_product_density_expectation,
    pauli_product_operator,
    pauli_product_statevector_expectation,
)
from .optimization import (
    HybridOptimizationResult,
    OptimizationStage,
    optimize_hybrid,
)

_PAULI_NAMES = {"i", "x", "y", "z"}


def _as_qubit_tuple(qubits: Iterable[int] | int | None) -> tuple[int, ...]:
    if qubits is None:
        return ()
    if isinstance(qubits, int):
        return (qubits,)
    return tuple(int(qubit) for qubit in qubits)


def _normalize_pauli(
    pauli: str | Mapping[int, str],
    qubits: Iterable[int] | int | None,
) -> tuple[tuple[int, str], ...]:
    if isinstance(pauli, Mapping):
        items = tuple((int(qubit), str(name).lower()) for qubit, name in pauli.items())
    else:
        qubit_tuple = _as_qubit_tuple(qubits)
        if len(pauli) != len(qubit_tuple):
            raise ValueError("Pauli string length must match qubits length.")
        items = tuple(
            (qubit, name.lower())
            for qubit, name in zip(qubit_tuple, pauli, strict=True)
        )

    normalized = []
    seen = set()
    for qubit, name in items:
        if name not in _PAULI_NAMES:
            raise ValueError("Pauli operators must be one of I, X, Y, or Z.")
        if qubit in seen:
            raise ValueError("A Hamiltonian term cannot repeat a qubit.")
        seen.add(qubit)
        if name != "i":
            normalized.append((qubit, name))
    return tuple(sorted(normalized))


def _coefficient_tensor(
    coefficient: float | complex | torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    if isinstance(coefficient, complex) or (
        isinstance(coefficient, torch.Tensor) and coefficient.is_complex()
    ):
        dtype = torch.promote_types(dtype, torch.complex64)
    return torch.as_tensor(coefficient, dtype=dtype, device=device)


@dataclass(frozen=True)
class HamiltonianTerm:
    """One weighted Pauli product term."""

    coefficient: float | complex | torch.Tensor
    pauli: str | Mapping[int, str]
    qubits: tuple[int, ...] = ()

    def __init__(
        self,
        coefficient: float | complex | torch.Tensor,
        pauli: str | Mapping[int, str],
        qubits: Iterable[int] | int | None = None,
    ) -> None:
        self.ops: tuple[tuple[int, str], ...]
        object.__setattr__(self, "coefficient", coefficient)
        object.__setattr__(self, "pauli", pauli)
        object.__setattr__(self, "qubits", _as_qubit_tuple(qubits))
        object.__setattr__(self, "ops", _normalize_pauli(pauli, self.qubits))

    @property
    def max_qubit(self) -> int:
        if not self.ops:
            return -1
        return max(qubit for qubit, _ in self.ops)

    def expectation(self, target: Circuit | MPSState | torch.Tensor) -> torch.Tensor:
        """Evaluate this term on a circuit, MPS state, statevector, or density matrix."""

        if isinstance(target, Circuit):
            base = (
                target.state().real.new_ones(target.bsz)
                if not self.ops
                else target.expectation_ps(
                    x=[qubit for qubit, name in self.ops if name == "x"],
                    y=[qubit for qubit, name in self.ops if name == "y"],
                    z=[qubit for qubit, name in self.ops if name == "z"],
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
                target.tensors[0].real.new_ones(target.bsz)
                if not self.ops
                else target.expectation_ps(
                    x=[qubit for qubit, name in self.ops if name == "x"],
                    y=[qubit for qubit, name in self.ops if name == "y"],
                    z=[qubit for qubit, name in self.ops if name == "z"],
                )
            )
            coeff = _coefficient_tensor(
                self.coefficient, dtype=base.dtype, device=base.device
            )
            return torch.real(coeff * base)

        tensor = target
        if tensor.ndim >= 2 and tensor.shape[-1] == tensor.shape[-2]:
            n_qubits = infer_n_wires_from_dense_state(tensor)
            base = (
                tensor.real.new_ones(
                    tensor.shape[0] if tensor.ndim == 3 else 1,
                )
                if not self.ops
                else pauli_product_density_expectation(tensor, self.ops, n_qubits)
            )
        else:
            n_qubits = infer_n_wires_from_dense_state(tensor)
            base = (
                tensor.real.new_ones(
                    tensor.shape[0] if tensor.ndim == 2 else 1,
                )
                if not self.ops
                else pauli_product_statevector_expectation(tensor, self.ops, n_qubits)
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
    def n_qubits(self) -> int:
        return 1 + max((term.max_qubit for term in self.terms), default=-1)

    def expectation(
        self,
        target: Circuit | MPSState | torch.Tensor,
        *,
        differentiation: Literal["autograd", "adjoint"] = "autograd",
    ) -> torch.Tensor:
        if differentiation not in {"autograd", "adjoint"}:
            raise ValueError("differentiation must be 'autograd' or 'adjoint'")
        if differentiation == "adjoint":
            if not isinstance(target, Circuit):
                raise TypeError("adjoint differentiation requires a Circuit target")
            if target.bsz != 1:
                raise ValueError("adjoint differentiation currently requires bsz=1")
            observable_terms: list[tuple[float, tuple[int, ...]]] = []
            constant = 0.0
            for term in self.terms:
                coefficient = term.coefficient
                if isinstance(coefficient, torch.Tensor):
                    if coefficient.numel() != 1 or coefficient.requires_grad:
                        raise ValueError(
                            "adjoint differentiation requires constant scalar coefficients"
                        )
                    coefficient = coefficient.detach().item()
                coefficient_value = complex(coefficient)
                if coefficient_value.imag != 0.0:
                    raise ValueError(
                        "adjoint differentiation requires real Hamiltonian coefficients"
                    )
                qubits = tuple(qubit for qubit, name in term.ops if name == "z")
                if len(qubits) != len(term.ops) or len(qubits) > 2:
                    raise ValueError(
                        "adjoint differentiation currently supports only Z and ZZ terms"
                    )
                if not qubits:
                    constant += coefficient_value.real
                else:
                    observable_terms.append((coefficient_value.real, qubits))
            if not observable_terms:
                raise ValueError(
                    "adjoint differentiation requires at least one Z or ZZ term"
                )
            from ..runtime.executors.statevector.reverse import (
                execute_torch_distributed_statevector_reverse,
            )

            result = execute_torch_distributed_statevector_reverse(
                target,
                observable_terms=observable_terms,
                device=target.device,
            )
            return result.value + result.value.new_tensor(constant)
        if isinstance(target, MPSState):
            z_coefficients: dict[int, float | complex | torch.Tensor] = {}
            zz_coefficients: dict[int, float | complex | torch.Tensor] = {}
            chain_compatible = True
            for term in self.terms:
                ops = term.ops
                if len(ops) == 1 and ops[0][1] == "z":
                    qubit = ops[0][0]
                    z_coefficients[qubit] = (
                        z_coefficients.get(qubit, 0.0) + term.coefficient
                    )
                elif (
                    len(ops) == 2
                    and ops[0][1] == "z"
                    and ops[1][1] == "z"
                    and ops[1][0] == ops[0][0] + 1
                ):
                    qubit = ops[0][0]
                    zz_coefficients[qubit] = (
                        zz_coefficients.get(qubit, 0.0) + term.coefficient
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

        if self.n_qubits > 12:
            raise ValueError("dense Hamiltonian diagnostics are limited to 12 qubits")
        dimension = 1 << self.n_qubits
        result = torch.zeros(dimension, dimension, dtype=dtype, device=device)
        for term in self.terms:
            coefficient = torch.as_tensor(term.coefficient, dtype=dtype, device=device)
            result = result + coefficient * pauli_product_operator(
                term.ops, self.n_qubits, dtype=dtype, device=device
            )
        return result

    def ground_energy(self) -> torch.Tensor:
        """Return the exact small-system ground energy."""

        eigenvalues: torch.Tensor = torch.linalg.eigvalsh(self.matrix())
        return eigenvalues.min()


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
    qubits: Iterable[int] | int | None = None,
) -> HamiltonianTerm:
    """Create a weighted Pauli product term."""

    return HamiltonianTerm(coefficient, pauli, qubits)


def transverse_field_ising(
    n_qubits: int,
    *,
    coupling: float = 1.0,
    field: float = 1.0,
    periodic: bool = False,
) -> Hamiltonian:
    """Build ``-coupling * ZZ - field * X`` transverse-field Ising Hamiltonian."""

    terms: list[HamiltonianTerm] = []
    for qubit in range(n_qubits - 1):
        terms.append(pauli_term(-coupling, "ZZ", (qubit, qubit + 1)))
    if periodic and n_qubits > 2:
        terms.append(pauli_term(-coupling, "ZZ", (n_qubits - 1, 0)))
    for qubit in range(n_qubits):
        terms.append(pauli_term(-field, "X", (qubit,)))
    return Hamiltonian(terms)


def zz_chain_hamiltonian(
    n_qubits: int,
    *,
    coupling: float = 1.0,
    field: float = 0.0,
    periodic: bool = False,
) -> Hamiltonian:
    """Build a nearest-neighbor ``coupling * ZZ + field * Z`` Hamiltonian."""

    terms: list[HamiltonianTerm] = []
    for qubit in range(n_qubits - 1):
        terms.append(pauli_term(coupling, "ZZ", (qubit, qubit + 1)))
    if periodic and n_qubits > 2:
        terms.append(pauli_term(coupling, "ZZ", (n_qubits - 1, 0)))
    for qubit in range(n_qubits):
        if field:
            terms.append(pauli_term(field, "Z", (qubit,)))
    return Hamiltonian(terms)


def hardware_efficient_parameter_count(
    n_qubits: int,
    layers: int,
    rotations: Sequence[str] = ("ry", "rz"),
) -> int:
    return int(n_qubits) * int(layers) * len(tuple(rotations))


def hardware_efficient_ansatz(
    n_qubits: int,
    layers: int,
    parameters: torch.Tensor | Sequence[float],
    *,
    rotations: Sequence[str] = ("ry", "rz"),
    entanglement: str = "linear",
    device: torch.device | str = "cpu",
) -> Circuit:
    """Build a common differentiable layered ansatz."""

    params = torch.as_tensor(parameters, dtype=torch.float32, device=device).reshape(-1)
    expected = hardware_efficient_parameter_count(n_qubits, layers, rotations)
    if params.numel() != expected:
        raise ValueError(f"Expected {expected} parameters, got {params.numel()}.")

    circuit = Circuit(
        n_qubits,
        device=device,
        dtype=getattr(torch, get_runtime_config().complex_dtype),
    )
    cursor = 0
    for _ in range(layers):
        for qubit in range(n_qubits):
            for rotation in rotations:
                getattr(circuit, rotation.lower())(qubit, theta=params[cursor])
                cursor += 1
        if entanglement == "linear":
            for qubit in range(n_qubits - 1):
                circuit.gate("cx", (qubit, qubit + 1))
        elif entanglement == "circular":
            for qubit in range(n_qubits - 1):
                circuit.gate("cx", (qubit, qubit + 1))
            if n_qubits > 2:
                circuit.gate("cx", (n_qubits - 1, 0))
        elif entanglement != "none":
            raise ValueError("entanglement must be 'linear', 'circular', or 'none'.")
    return circuit


def qaoa_circuit(
    n_qubits: int,
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
    circuit = Circuit(
        n_qubits,
        device=device,
        dtype=getattr(torch, get_runtime_config().complex_dtype),
    )
    for qubit in range(n_qubits):
        circuit.gate("h", qubit)

    for gamma, beta in zip(gamma_values, beta_values, strict=True):
        for edge in edge_tuple:
            if len(edge) == 2:
                src, dst = edge
                weight = 1.0
            else:
                src, dst, weight = edge
            circuit.gate("rzz", (int(src), int(dst)), theta=2.0 * gamma * float(weight))
        for qubit in range(n_qubits):
            circuit.gate("rx", qubit, theta=2.0 * beta)
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
    n_qubits: int, *, anisotropy: float = 1.0, field: float = 0.0
) -> Hamiltonian:
    """Return the open-boundary XX + YY + anisotropy ZZ Hamiltonian."""

    if n_qubits < 2:
        raise ValueError("Heisenberg chains require at least two qubits")
    terms: list[HamiltonianTerm] = []
    for left in range(n_qubits - 1):
        terms.extend(
            (
                pauli_term(1.0, "XX", (left, left + 1)),
                pauli_term(1.0, "YY", (left, left + 1)),
                pauli_term(anisotropy, "ZZ", (left, left + 1)),
            )
        )
    if field:
        terms.extend(pauli_term(field, "Z", (qubit,)) for qubit in range(n_qubits))
    return Hamiltonian(terms)


def heisenberg_hva_parameter_count(
    n_qubits: int, depth: int, *, parameterization: str = "bond_resolved_phase"
) -> int:
    if n_qubits < 2 or depth < 1:
        raise ValueError("n_qubits >= 2 and depth >= 1 are required")
    if parameterization == "shared_parity":
        return 6 * depth
    if parameterization == "bond_resolved":
        return 3 * (n_qubits - 1) * depth
    if parameterization == "bond_resolved_phase":
        return (3 * (n_qubits - 1) + n_qubits) * depth
    raise ValueError("unknown Heisenberg HVA parameterization")


def heisenberg_hva(
    n_qubits: int,
    depth: int,
    parameters: torch.Tensor | Sequence[float],
    *,
    parameterization: str = "bond_resolved_phase",
    initial_state: str = "dimer_singlet",
) -> Circuit:
    """Build an open-chain HVA with a physics-informed initial state."""

    values = torch.as_tensor(parameters)
    expected = heisenberg_hva_parameter_count(
        n_qubits, depth, parameterization=parameterization
    )
    if values.numel() != expected:
        raise ValueError(f"expected {expected} HVA parameters, got {values.numel()}")
    values = values.reshape(-1)
    circuit = Circuit(n_qubits, device=values.device)
    if initial_state == "neel":
        for qubit in range(1, n_qubits, 2):
            circuit.gate("x", qubit)
    elif initial_state == "dimer_singlet":
        for left in range(0, n_qubits - 1, 2):
            circuit.gate("x", left + 1)
            circuit.gate("h", left)
            circuit.gate("cx", (left, left + 1))
            circuit.gate("z", left)
    else:
        raise ValueError("initial_state must be 'neel' or 'dimer_singlet'")

    for layer in range(depth):
        if parameterization == "shared_parity":
            offset = 6 * layer
            for axis, gate_name in enumerate(("rxx", "ryy", "rzz")):
                for parity in (0, 1):
                    theta = values[offset + 2 * axis + parity]
                    for left in range(parity, n_qubits - 1, 2):
                        circuit.gate(gate_name, (left, left + 1), theta=theta)
            continue
        per_layer = 3 * (n_qubits - 1) + (
            n_qubits if parameterization == "bond_resolved_phase" else 0
        )
        offset = per_layer * layer
        for axis, gate_name in enumerate(("rxx", "ryy", "rzz")):
            for left in range(n_qubits - 1):
                circuit.gate(
                    gate_name,
                    (left, left + 1),
                    theta=values[offset + axis * (n_qubits - 1) + left],
                )
        if parameterization == "bond_resolved_phase":
            phase_offset = offset + 3 * (n_qubits - 1)
            for qubit in range(n_qubits):
                circuit.gate("rz", qubit, theta=values[phase_offset + qubit])
    return circuit


class OptimizerFactory(Protocol):
    """Construct an optimizer from trainable tensors and a learning rate."""

    def __call__(
        self, params: Iterable[torch.Tensor], *, lr: float
    ) -> torch.optim.Optimizer: ...


def run_vqe(
    circuit_builder: Callable[[torch.Tensor], Circuit],
    initial_parameters: torch.Tensor | Sequence[float],
    hamiltonian: Hamiltonian,
    *,
    steps: int = 100,
    lr: float = 0.05,
    optimizer_factory: OptimizerFactory = torch.optim.Adam,
) -> VQEResult:
    """Run a compact PyTorch-native VQE optimization loop."""

    parameters = torch.as_tensor(initial_parameters, dtype=torch.float32).clone()
    parameters = parameters.detach().requires_grad_(True)
    optimizer = optimizer_factory([parameters], lr=lr)
    history: list[float] = []

    for _ in range(int(steps)):
        optimizer.zero_grad()
        loss = vqe_loss(circuit_builder, parameters, hamiltonian)
        torch.autograd.backward(loss)
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
    optimizer_factory: OptimizerFactory = torch.optim.Adam,
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
    if energy_function is not None:
        evaluate_energy = energy_function
    else:
        assert hamiltonian is not None
        evaluate_energy = hamiltonian.expectation

    selected_indices: list[int] = []
    selected_operators: list[object] = []
    parameters = torch.empty(0, dtype=dtype, device=device)

    def energy(operators: tuple[object, ...], values: torch.Tensor) -> torch.Tensor:
        circuit = circuit_builder(operators, values)
        return evaluate_energy(circuit).sum()

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
        optimizer = optimizer_factory([parameters], lr=lr)
        history: list[float] = []
        optimization_started = perf_counter()
        for _ in range(int(optimization_steps)):
            optimizer.zero_grad(set_to_none=True)
            loss = energy(tuple(selected_operators), parameters)
            torch.autograd.backward(loss)
            optimizer.step()
            history.append(float(loss.detach()))
        optimization_seconds = perf_counter() - optimization_started
        energy_after = float(energy(tuple(selected_operators), parameters).detach())
        full_gradients = [0.0] * len(pool)
        for pool_index, gradient in zip(available, gradients, strict=True):
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
            partial(circuit_builder, depth),
            parameters,
            hamiltonian,
            stages=stages,
        )
        results.append(result)
        parameters = result.parameters["quantum"]
    return LayerwiseVQEResult(depths=normalized_depths, stages=tuple(results))


def qaoa_loss(
    n_qubits: int,
    edges: Iterable[tuple[int, int] | tuple[int, int, float]],
    gammas: torch.Tensor | Sequence[float],
    betas: torch.Tensor | Sequence[float],
    hamiltonian: Hamiltonian,
    *,
    reduction: str = "mean",
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Evaluate a differentiable QAOA objective."""

    circuit = qaoa_circuit(n_qubits, edges, gammas, betas, device=device)
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
    "OptimizerFactory",
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

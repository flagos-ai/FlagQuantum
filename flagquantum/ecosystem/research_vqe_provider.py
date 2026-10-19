"""Shared FlagQuantum-owned mechanics for local statevector VQE providers."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import torch

from flagquantum.algorithms import Hamiltonian, pauli_term
from flagquantum.circuit import Circuit
from flagquantum.services.ground_state import (
    ComplexAmplitude,
    ConvergencePoint,
    ExecutionProvenance,
    GroundStateResources,
    GroundStateVQERequest,
    GroundStateVQEResult,
    PauliExpectationEvidence,
    StatevectorEvidence,
    validate_ground_state_vqe_request,
)

StatevectorRunner = Callable[[Circuit], torch.Tensor]
_PARAMETER_SHIFT = math.pi / 2.0


@dataclass(frozen=True, slots=True)
class ResearchVQEProviderIdentity:
    """Provider-controlled identity included in the scientific result."""

    provider: str
    package: str
    package_version: str
    tool: str
    tool_version: str


def _canonical_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _hamiltonian(request: GroundStateVQERequest) -> Hamiltonian:
    return Hamiltonian(
        pauli_term(
            term.coefficient,
            term.pauli,
            range(request.hamiltonian.n_qubits),
        )
        for term in request.hamiltonian.terms
    )


def _circuit(theta: float) -> Circuit:
    return (
        Circuit(2, device="cpu", dtype=torch.complex64)
        .gate("x", 1)
        .gate("ry", 0, theta=theta)
        .gate("cx", (0, 1))
    )


def _state(runner: StatevectorRunner, theta: float) -> torch.Tensor:
    state = runner(_circuit(theta)).detach().reshape(-1).to(device="cpu")
    if state.numel() != 4:
        raise RuntimeError("provider returned an invalid two-qubit statevector")
    return state


def _energy(hamiltonian: Hamiltonian, state: torch.Tensor) -> float:
    return float(hamiltonian.expectation(state).reshape(-1)[0])


def _provenance(
    identity: ResearchVQEProviderIdentity, payload: dict[str, Any]
) -> ExecutionProvenance:
    values = asdict(identity)
    return ExecutionProvenance(
        **values,
        execution_digest=(
            "sha256:" + _canonical_digest({"identity": values, "evidence": payload})
        ),
    )


def run_statevector_research_vqe(
    request: GroundStateVQERequest,
    *,
    runner: StatevectorRunner,
    identity: ResearchVQEProviderIdentity,
) -> GroundStateVQEResult:
    """Run the P0 ansatz with exact parameter-shift gradients and Adam.

    The runner accepts only a FlagQuantum circuit and returns only a
    FlagQuantum-ordered tensor. External SDK objects remain in provider modules.
    """

    reason = validate_ground_state_vqe_request(request)
    if reason is not None:
        raise ValueError(reason.message)

    hamiltonian = _hamiltonian(request)
    parameters = torch.tensor(request.ansatz.initial_parameters, dtype=torch.float32)
    optimizer = torch.optim.Adam([parameters], lr=request.optimizer.learning_rate)
    history: list[float] = []
    backend_executions = 0

    for _ in range(request.optimizer.max_evaluations - 1):
        optimizer.zero_grad()
        theta = float(parameters.detach()[0])
        current = _energy(hamiltonian, _state(runner, theta))
        plus = _energy(hamiltonian, _state(runner, theta + _PARAMETER_SHIFT))
        minus = _energy(hamiltonian, _state(runner, theta - _PARAMETER_SHIFT))
        backend_executions += 3
        parameters.grad = torch.tensor([(plus - minus) / 2.0])
        optimizer.step()
        history.append(current)

    final_state = _state(runner, float(parameters.detach()[0]))
    backend_executions += 1
    energy = _energy(hamiltonian, final_state)
    parameter_values = tuple(float(value) for value in parameters.detach())
    trace = tuple(
        ConvergencePoint(evaluation=index, energy=value)
        for index, value in enumerate((*history, energy), start=1)
    )
    term_evidence = []
    for specification, native_term in zip(
        request.hamiltonian.terms, hamiltonian.terms, strict=True
    ):
        contribution = float(native_term.expectation(final_state).reshape(-1)[0])
        unweighted = pauli_term(
            1.0,
            specification.pauli,
            range(request.hamiltonian.n_qubits),
        )
        expectation = float(unweighted.expectation(final_state).reshape(-1)[0])
        term_evidence.append(
            PauliExpectationEvidence(
                pauli=specification.pauli,
                coefficient=float(specification.coefficient),
                expectation=expectation,
                contribution=contribution,
            )
        )
    statevector = StatevectorEvidence(
        basis_states=tuple(format(index, "02b") for index in range(4)),
        amplitudes=tuple(
            ComplexAmplitude(real=float(value.real), imag=float(value.imag))
            for value in final_state
        ),
        norm=float(torch.linalg.vector_norm(final_state)),
    )
    resources = GroundStateResources(
        n_qubits=2,
        hamiltonian_terms=len(request.hamiltonian.terms),
        ansatz_gates=3,
        trainable_parameters=1,
        optimizer_steps=request.optimizer.max_evaluations - 1,
        objective_evaluations=request.optimizer.max_evaluations,
        backend_executions=backend_executions,
    )
    evidence = {
        "request": request.to_dict(),
        "status": "completed",
        "energy": energy,
        "parameters": parameter_values,
        "convergence_trace": tuple(asdict(point) for point in trace),
        "term_evidence": tuple(asdict(item) for item in term_evidence),
        "statevector": asdict(statevector),
        "resources": asdict(resources),
    }
    return GroundStateVQEResult(
        status="completed",
        request=request,
        energy=energy,
        parameters=parameter_values,
        convergence_trace=trace,
        term_evidence=tuple(term_evidence),
        statevector=statevector,
        resources=resources,
        provenance=_provenance(identity, evidence),
    )


__all__ = (
    "ResearchVQEProviderIdentity",
    "StatevectorRunner",
    "run_statevector_research_vqe",
)

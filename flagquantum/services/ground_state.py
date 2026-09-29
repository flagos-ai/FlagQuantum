"""Auditable provider-owned variational ground-state execution.

The first contract intentionally supports one small, replayable workload: a
real two-qubit Pauli sum in the weight-one sector, optimized with FlagQuantum's
native statevector VQE path. Unsupported requests return a structured result;
they are never rerouted to another implementation.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from ..algorithms import Hamiltonian, pauli_term, run_vqe
from ..version import __version__

_REQUEST_SCHEMA = "flagquantum.ground_state_vqe_request.v1"
_RESULT_SCHEMA = "flagquantum.ground_state_vqe_result.v1"
_REFERENCE_SCHEMA = "flagquantum.exact_sector_reference.v1"
_TOOL_NAME = "flagquantum.services.run_ground_state_vqe"
_TOOL_VERSION = "1.0.0"
_BASIS_CONVENTION = "wire_0_is_leftmost"
_ANSATZ_NAME = "two_qubit_single_excitation"
_OPTIMIZER_NAME = "adam"


@dataclass(frozen=True, slots=True)
class PauliSumTerm:
    """One real coefficient and full-width Pauli word."""

    coefficient: float
    pauli: str


@dataclass(frozen=True, slots=True)
class PauliSum:
    """Versioned Pauli-sum Hamiltonian supplied to the provider tool."""

    n_qubits: int
    terms: tuple[PauliSumTerm, ...]
    coefficient_unit: str = "hartree"
    basis_convention: str = _BASIS_CONVENTION
    schema: str = "flagquantum.pauli_sum.v1"


@dataclass(frozen=True, slots=True)
class GroundStateAnsatz:
    """Explicit ansatz and starting point for a ground-state request."""

    name: str = _ANSATZ_NAME
    initial_basis_state: str = "01"
    initial_parameters: tuple[float, ...] = (0.0,)


@dataclass(frozen=True, slots=True)
class GroundStateOptimizer:
    """Bounded deterministic optimizer configuration."""

    name: str = _OPTIMIZER_NAME
    max_evaluations: int = 121
    learning_rate: float = 0.1


@dataclass(frozen=True, slots=True)
class GroundStateVQERequest:
    """Serializable request accepted by the provider-owned VQE workflow."""

    hamiltonian: PauliSum
    fixed_hamming_weight: int | None
    ansatz: GroundStateAnsatz = field(default_factory=GroundStateAnsatz)
    optimizer: GroundStateOptimizer = field(default_factory=GroundStateOptimizer)
    schema: str = _REQUEST_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class UnsupportedCapability:
    """Machine-readable reason that a request was refused without fallback."""

    code: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ConvergencePoint:
    """One independently ordered objective evaluation."""

    evaluation: int
    energy: float


@dataclass(frozen=True, slots=True)
class PauliExpectationEvidence:
    """Expectation and weighted contribution for one supplied Pauli term."""

    pauli: str
    coefficient: float
    expectation: float
    contribution: float


@dataclass(frozen=True, slots=True)
class ComplexAmplitude:
    """JSON-safe statevector amplitude."""

    real: float
    imag: float


@dataclass(frozen=True, slots=True)
class StatevectorEvidence:
    """Final state in the declared computational-basis order."""

    basis_states: tuple[str, ...]
    amplitudes: tuple[ComplexAmplitude, ...]
    norm: float


@dataclass(frozen=True, slots=True)
class GroundStateResources:
    """Resources consumed by the bounded local execution."""

    n_qubits: int
    hamiltonian_terms: int
    ansatz_gates: int
    trainable_parameters: int
    optimizer_steps: int
    objective_evaluations: int
    execution_mode: str = "local_statevector"


@dataclass(frozen=True, slots=True)
class ExecutionProvenance:
    """Provider-controlled identity for one result."""

    provider: str
    package: str
    package_version: str
    tool: str
    tool_version: str
    execution_digest: str


@dataclass(frozen=True, slots=True)
class GroundStateVQEResult:
    """Completed or explicitly unsupported provider VQE result."""

    status: Literal["completed", "unsupported"]
    request: GroundStateVQERequest
    provenance: ExecutionProvenance
    energy: float | None = None
    parameters: tuple[float, ...] = ()
    convergence_trace: tuple[ConvergencePoint, ...] = ()
    term_evidence: tuple[PauliExpectationEvidence, ...] = ()
    statevector: StatevectorEvidence | None = None
    resources: GroundStateResources | None = None
    unsupported: UnsupportedCapability | None = None
    schema: str = _RESULT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ExactSectorReference:
    """Independent dense diagonalization reference for one fixed sector."""

    status: Literal["completed", "unsupported"]
    hamiltonian: PauliSum
    fixed_hamming_weight: int
    provenance: ExecutionProvenance
    energy: float | None = None
    sector_basis_states: tuple[str, ...] = ()
    unsupported: UnsupportedCapability | None = None
    schema: str = _REFERENCE_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _canonical_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _provenance(tool: str, payload: dict[str, Any]) -> ExecutionProvenance:
    identity = {
        "provider": "FlagQuantum",
        "package": "flagquantum",
        "package_version": __version__,
        "tool": tool,
        "tool_version": _TOOL_VERSION,
    }
    return ExecutionProvenance(
        **identity,
        execution_digest=f"sha256:{_canonical_digest({'identity': identity, 'evidence': payload})}",
    )


def ground_state_vqe_capability() -> dict[str, Any]:
    """Describe the exact released scope that an edge adapter may route to."""

    return {
        "available": True,
        "provider": "FlagQuantum",
        "package": "flagquantum",
        "package_version": __version__,
        "tool": _TOOL_NAME,
        "tool_version": _TOOL_VERSION,
        "request_schema": _REQUEST_SCHEMA,
        "result_schema": _RESULT_SCHEMA,
        "supported": {
            "n_qubits": (2,),
            "basis_conventions": (_BASIS_CONVENTION,),
            "coefficient_units": ("hartree",),
            "fixed_hamming_weights": (1,),
            "ansatzes": (_ANSATZ_NAME,),
            "optimizers": (_OPTIMIZER_NAME,),
            "execution_modes": ("local_statevector",),
        },
        "fallback": False,
    }


def _unsupported_reason(
    hamiltonian: PauliSum,
    *,
    fixed_hamming_weight: int | None,
    ansatz: GroundStateAnsatz | None = None,
    optimizer: GroundStateOptimizer | None = None,
) -> UnsupportedCapability | None:
    if hamiltonian.schema != "flagquantum.pauli_sum.v1":
        return UnsupportedCapability(
            "UNSUPPORTED_HAMILTONIAN_SCHEMA",
            "only flagquantum.pauli_sum.v1 is supported",
            {"received": hamiltonian.schema},
        )
    if (
        not isinstance(hamiltonian.n_qubits, int)
        or isinstance(hamiltonian.n_qubits, bool)
        or hamiltonian.n_qubits != 2
    ):
        return UnsupportedCapability(
            "UNSUPPORTED_QUBIT_COUNT",
            "the initial provider VQE contract supports exactly two qubits",
            {"received": hamiltonian.n_qubits, "supported": 2},
        )
    if hamiltonian.basis_convention != _BASIS_CONVENTION:
        return UnsupportedCapability(
            "UNSUPPORTED_BASIS_CONVENTION",
            f"only {_BASIS_CONVENTION!r} is supported",
            {"received": hamiltonian.basis_convention},
        )
    if hamiltonian.coefficient_unit != "hartree":
        return UnsupportedCapability(
            "UNSUPPORTED_COEFFICIENT_UNIT",
            "the initial provider VQE contract supports Hartree coefficients",
            {"received": hamiltonian.coefficient_unit},
        )
    if not hamiltonian.terms:
        return UnsupportedCapability(
            "UNSUPPORTED_EMPTY_HAMILTONIAN",
            "at least one Pauli term is required",
        )
    for index, term in enumerate(hamiltonian.terms):
        if (
            not isinstance(term.coefficient, (int, float))
            or isinstance(term.coefficient, bool)
            or not math.isfinite(float(term.coefficient))
            or not isinstance(term.pauli, str)
            or len(term.pauli) != hamiltonian.n_qubits
            or any(name not in "IXYZ" for name in term.pauli)
        ):
            return UnsupportedCapability(
                "UNSUPPORTED_PAULI_TERM",
                "terms require finite real coefficients and full-width I/X/Y/Z words",
                {"term_index": index},
            )
    if (
        not isinstance(fixed_hamming_weight, int)
        or isinstance(fixed_hamming_weight, bool)
        or fixed_hamming_weight != 1
    ):
        return UnsupportedCapability(
            "UNSUPPORTED_SECTOR",
            "the initial provider VQE contract supports fixed Hamming weight 1",
            {"received": fixed_hamming_weight},
        )
    if ansatz is not None:
        if ansatz.name != _ANSATZ_NAME or ansatz.initial_basis_state != "01":
            return UnsupportedCapability(
                "UNSUPPORTED_ANSATZ",
                "only the two-qubit single-excitation ansatz starting from |01> is supported",
                {
                    "name": ansatz.name,
                    "initial_basis_state": ansatz.initial_basis_state,
                },
            )
        if (
            len(ansatz.initial_parameters) != 1
            or not isinstance(ansatz.initial_parameters[0], (int, float))
            or isinstance(ansatz.initial_parameters[0], bool)
            or not math.isfinite(float(ansatz.initial_parameters[0]))
        ):
            return UnsupportedCapability(
                "UNSUPPORTED_ANSATZ_PARAMETERS",
                "the supported ansatz requires one finite initial parameter",
            )
    if optimizer is not None:
        if optimizer.name != _OPTIMIZER_NAME:
            return UnsupportedCapability(
                "UNSUPPORTED_OPTIMIZER",
                "only the bounded Adam optimizer is supported",
                {"received": optimizer.name},
            )
        if (
            not isinstance(optimizer.max_evaluations, int)
            or isinstance(optimizer.max_evaluations, bool)
            or not 2 <= optimizer.max_evaluations <= 10_000
            or not isinstance(optimizer.learning_rate, (int, float))
            or isinstance(optimizer.learning_rate, bool)
            or not math.isfinite(float(optimizer.learning_rate))
            or optimizer.learning_rate <= 0
        ):
            return UnsupportedCapability(
                "UNSUPPORTED_OPTIMIZER_BUDGET",
                "max_evaluations must be in [2, 10000] and learning_rate must be finite and positive",
                {
                    "max_evaluations": optimizer.max_evaluations,
                    "learning_rate": optimizer.learning_rate,
                },
            )
    return None


def _native_hamiltonian(specification: PauliSum) -> Hamiltonian:
    return Hamiltonian(
        pauli_term(term.coefficient, term.pauli, range(specification.n_qubits))
        for term in specification.terms
    )


def _unsupported_vqe_result(
    request: GroundStateVQERequest, reason: UnsupportedCapability
) -> GroundStateVQEResult:
    payload = {
        "request": request.to_dict(),
        "status": "unsupported",
        "unsupported": asdict(reason),
    }
    return GroundStateVQEResult(
        status="unsupported",
        request=request,
        unsupported=reason,
        provenance=_provenance(_TOOL_NAME, payload),
    )


def exact_sector_reference(
    hamiltonian: PauliSum, *, fixed_hamming_weight: int
) -> ExactSectorReference:
    """Compute a separately identified exact reference in one Hamming sector."""

    reason = _unsupported_reason(
        hamiltonian,
        fixed_hamming_weight=fixed_hamming_weight,
    )
    if reason is not None:
        payload = {
            "hamiltonian": asdict(hamiltonian),
            "fixed_hamming_weight": fixed_hamming_weight,
            "status": "unsupported",
            "unsupported": asdict(reason),
        }
        return ExactSectorReference(
            status="unsupported",
            hamiltonian=hamiltonian,
            fixed_hamming_weight=fixed_hamming_weight,
            unsupported=reason,
            provenance=_provenance(
                "flagquantum.services.exact_sector_reference", payload
            ),
        )

    import torch

    basis_states = tuple(
        format(index, f"0{hamiltonian.n_qubits}b")
        for index in range(1 << hamiltonian.n_qubits)
        if index.bit_count() == fixed_hamming_weight
    )
    basis_indices = torch.tensor(
        [int(state, 2) for state in basis_states], dtype=torch.int64
    )
    matrix = _native_hamiltonian(hamiltonian).matrix(
        dtype=torch.complex128, device="cpu"
    )
    sector_matrix = matrix.index_select(0, basis_indices).index_select(1, basis_indices)
    energy = float(torch.linalg.eigvalsh(sector_matrix).min())
    payload = {
        "hamiltonian": asdict(hamiltonian),
        "fixed_hamming_weight": fixed_hamming_weight,
        "status": "completed",
        "energy": energy,
        "sector_basis_states": basis_states,
    }
    return ExactSectorReference(
        status="completed",
        hamiltonian=hamiltonian,
        fixed_hamming_weight=fixed_hamming_weight,
        energy=energy,
        sector_basis_states=basis_states,
        provenance=_provenance("flagquantum.services.exact_sector_reference", payload),
    )


def run_ground_state_vqe(request: GroundStateVQERequest) -> GroundStateVQEResult:
    """Run the bounded FlagQuantum-owned VQE contract without provider fallback.

    Use :func:`exact_sector_reference` separately when a classical sector
    reference is required. This keeps variational simulation evidence distinct
    from dense diagonalization evidence.
    """

    if request.schema != _REQUEST_SCHEMA:
        return _unsupported_vqe_result(
            request,
            UnsupportedCapability(
                "UNSUPPORTED_REQUEST_SCHEMA",
                f"only {_REQUEST_SCHEMA} is supported",
                {"received": request.schema},
            ),
        )
    reason = _unsupported_reason(
        request.hamiltonian,
        fixed_hamming_weight=request.fixed_hamming_weight,
        ansatz=request.ansatz,
        optimizer=request.optimizer,
    )
    if reason is not None:
        return _unsupported_vqe_result(request, reason)

    import torch

    native_hamiltonian = _native_hamiltonian(request.hamiltonian)

    def build_ansatz(parameters: torch.Tensor) -> Any:
        from ..circuit import Circuit

        return (
            Circuit(2, device="cpu", dtype=torch.complex64)
            .gate("x", 1)
            .gate("ry", 0, theta=parameters[0])
            .gate("cx", (0, 1))
        )

    native_result = run_vqe(
        build_ansatz,
        request.ansatz.initial_parameters,
        native_hamiltonian,
        steps=request.optimizer.max_evaluations - 1,
        lr=request.optimizer.learning_rate,
    )
    parameters = tuple(float(value) for value in native_result.parameters)
    final_circuit = build_ansatz(native_result.parameters)
    state = final_circuit.state().detach().reshape(-1).to("cpu")
    energy = float(native_result.energy.reshape(-1)[0])
    trace = tuple(
        ConvergencePoint(evaluation=index, energy=value)
        for index, value in enumerate((*native_result.history, energy), start=1)
    )
    term_evidence = []
    for specification, native_term in zip(
        request.hamiltonian.terms, native_hamiltonian.terms, strict=True
    ):
        contribution = float(native_term.expectation(state).reshape(-1)[0])
        unweighted = pauli_term(
            1.0,
            specification.pauli,
            range(request.hamiltonian.n_qubits),
        )
        expectation = float(unweighted.expectation(state).reshape(-1)[0])
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
            for value in state
        ),
        norm=float(torch.linalg.vector_norm(state)),
    )
    resources = GroundStateResources(
        n_qubits=2,
        hamiltonian_terms=len(request.hamiltonian.terms),
        ansatz_gates=3,
        trainable_parameters=1,
        optimizer_steps=request.optimizer.max_evaluations - 1,
        objective_evaluations=request.optimizer.max_evaluations,
    )
    evidence = {
        "request": request.to_dict(),
        "status": "completed",
        "energy": energy,
        "parameters": parameters,
        "convergence_trace": tuple(asdict(point) for point in trace),
        "term_evidence": tuple(asdict(item) for item in term_evidence),
        "statevector": asdict(statevector),
        "resources": asdict(resources),
    }
    return GroundStateVQEResult(
        status="completed",
        request=request,
        energy=energy,
        parameters=parameters,
        convergence_trace=trace,
        term_evidence=tuple(term_evidence),
        statevector=statevector,
        resources=resources,
        provenance=_provenance(_TOOL_NAME, evidence),
    )


__all__ = (
    "ComplexAmplitude",
    "ConvergencePoint",
    "ExactSectorReference",
    "ExecutionProvenance",
    "GroundStateAnsatz",
    "GroundStateOptimizer",
    "GroundStateResources",
    "GroundStateVQERequest",
    "GroundStateVQEResult",
    "PauliExpectationEvidence",
    "PauliSum",
    "PauliSumTerm",
    "StatevectorEvidence",
    "UnsupportedCapability",
    "exact_sector_reference",
    "ground_state_vqe_capability",
    "run_ground_state_vqe",
)

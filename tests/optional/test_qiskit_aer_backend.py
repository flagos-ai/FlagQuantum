"""Real-dependency tests for the explicit Qiskit Aer backend plugin."""

from __future__ import annotations

import importlib.util
import subprocess
import sys

import pytest
import torch

import flagquantum as fq
from flagquantum.core import CircuitIR, Instruction
from flagquantum.ecosystem.extensions import (
    CapabilityRequest,
    ExtensionConfig,
    ExtensionRegistry,
    discover_extensions,
)
from flagquantum.ecosystem.extensions.admission import (
    admit_backend_extension,
    withdraw_backend,
)
from flagquantum.ecosystem.qiskit import QiskitAerBackend, run
from flagquantum.ecosystem.qiskit.research_provider import (
    QiskitAerResearchVQEProvider,
)
from flagquantum.services import (
    GroundStateOptimizer,
    GroundStateVQERequest,
    PauliSum,
    PauliSumTerm,
)

pytestmark = [
    pytest.mark.qiskit,
    pytest.mark.skipif(
        importlib.util.find_spec("qiskit_aer") is None,
        reason="Qiskit Aer optional dependency is not installed",
    ),
]


def test_import_is_lazy() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from flagquantum.ecosystem.qiskit import run; "
            "assert callable(run); "
            "assert 'qiskit' not in sys.modules; "
            "assert 'qiskit_aer' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128))
def test_statevector_matches_native_wire_order(dtype: torch.dtype) -> None:
    circuit = fq.Circuit(3, dtype=dtype).h(0).ry(1, 0.37).cx(0, 2).cz(2, 1)

    result = run(circuit, options=fq.ExecutionOptions(seed=13))

    expected = circuit.state(refresh=True)
    assert result.state is not None
    assert result.state.dtype == dtype
    assert torch.allclose(result.state, expected, rtol=1e-5, atol=1e-6)
    assert result.runtime["backend"] == "qiskit_aer"
    assert result.runtime["fallback_used"] is False
    assert result.compatibility["external_backend_explicit"] is True
    assert result.provenance["flagquantum_ir_hash"] == circuit.to_ir().content_hash


def test_research_provider_executes_real_aer_statevectors() -> None:
    provider = QiskitAerResearchVQEProvider()
    request = GroundStateVQERequest(
        hamiltonian=PauliSum(
            n_qubits=2,
            terms=(PauliSumTerm(1.0, "ZI"),),
        ),
        fixed_hamming_weight=1,
        optimizer=GroundStateOptimizer(max_evaluations=2),
    )

    assert provider.availability().available
    result = provider.run_vqe(request)

    assert result.status == "completed"
    assert result.resources is not None
    assert result.resources.objective_evaluations == 2
    assert result.resources.backend_executions == 4
    assert result.provenance.provider == "Qiskit Aer"


def test_seeded_samples_and_counts_use_requested_wire_order() -> None:
    circuit = fq.Circuit(3).x(0).x(2)

    samples = run(
        circuit,
        outputs=fq.samples(qubits=(2, 1, 0)),
        options=fq.ExecutionOptions(shots=8, seed=7),
    )
    counts = run(
        circuit,
        outputs=fq.counts(qubits=(2, 1, 0)),
        options=fq.ExecutionOptions(shots=8, seed=7),
    )

    assert samples.samples is not None
    assert samples.samples.shape == (1, 8, 3)
    assert samples.samples.unique(dim=1).tolist() == [[[1, 0, 1]]]
    assert counts.counts == [{"101": 8}]


def test_extension_lifecycle_executes_owned_program_and_result() -> None:
    backend = QiskitAerBackend()
    handle = (
        ExtensionRegistry()
        .with_extension(backend)
        .negotiate(
            "backend",
            "qiskit_aer",
            CapabilityRequest(
                required=frozenset({"cpu", "statevector"}),
                dtype="complex64",
                device_type="cpu",
            ),
        )
    )
    handle.start(ExtensionConfig({"threads": 1, "seed": 11}))
    try:
        result = handle.invoke("execute", fq.Circuit(1).h(0))
    finally:
        handle.close()

    assert result.state is not None
    assert result.state.shape == (1, 2)
    assert backend._active is False


def test_unsupported_requests_fail_before_aer_execution() -> None:
    with pytest.raises(TypeError, match=r"requires fq\.samples"):
        run(fq.Circuit(1), shots=10)
    with pytest.raises(ValueError, match="require shots"):
        run(fq.Circuit(1), outputs=fq.counts())
    with pytest.raises(ValueError, match="must be unique"):
        run(fq.Circuit(2), outputs=fq.samples(qubits=(0, 0)), shots=4)
    with pytest.raises(RuntimeError, match=r"supports fq\.samples or fq\.counts"):
        run(fq.Circuit(2), outputs=fq.probabilities())
    with pytest.raises(RuntimeError, match="dynamic circuit"):
        run(
            CircuitIR(
                1,
                (Instruction("x", (0,), metadata={"conditions": ((0, 1),)}),),
            )
        )
    with pytest.raises(RuntimeError, match="does not support autograd"):
        run(fq.Circuit(1).rx(0, torch.tensor(0.2, requires_grad=True)))
    response = QiskitAerBackend().negotiate(CapabilityRequest(require_gradients=True))
    assert response.accepted is False
    assert response.blockers == ("Qiskit Aer bridge does not support gradients",)


def test_the_declared_entry_point_serves_fq_run_and_agrees_with_the_builtin() -> None:
    """The installed declaration is what admits, and Aer agrees with the default.

    Every other test here reaches the plugin directly. This one starts from the
    ``flagquantum.extensions`` group instead, which is the only path a third
    party's simulator takes: discovery reads the installed metadata, admission
    checks and starts what it found, and ``fq.run`` reaches it as an admitted
    route rather than through the module-level bridge function. The statevector
    it returns is then compared against a default ``fq.run``, because a route
    that executes is not the same thing as a route that computes the right
    answer, and the row that claims admission states both.
    """

    declared = discover_extensions("backend").entries
    assert ("backend", "qiskit_aer") in declared, declared.keys()
    handle, capabilities = admit_backend_extension(
        declared[("backend", "qiskit_aer")],
        config=ExtensionConfig({"seed": 13, "threads": 1}),
    )
    try:
        circuit = (
            fq.Circuit(3, dtype=torch.complex128).h(0).ry(1, 0.37).cx(0, 2).cz(2, 1)
        )
        result = fq.run(
            circuit,
            options=fq.ExecutionOptions(backend=capabilities.name, mode="statevector"),
        )
        reference = fq.run(circuit)
        assert result.runtime["execution_path"] == "admitted_extension_backend"
        assert torch.allclose(result.state, reference.state, rtol=1e-6, atol=1e-8)
    finally:
        handle.close()
        withdraw_backend(capabilities.name)

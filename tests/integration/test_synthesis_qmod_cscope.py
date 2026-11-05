"""Verify complete native QAOA/VQE workflows, including their saved outputs."""

import ast
import cmath
import hashlib
import json
import re
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.ecosystem import InteropConversionError
from flagquantum.ecosystem._qmod.source import load_qmod
from flagquantum.services._qmod_workflow import run_qaoa_workflow

pytestmark = pytest.mark.integration
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod"
MANIFEST = json.loads((FIXTURES / "cscope_sources.json").read_text())
SOURCES = {item["file"]: item for item in MANIFEST["fixtures"]}
BUDGET = 512 * 1024**2


def source_text(name: str) -> str:
    raw = (FIXTURES / name).read_bytes()
    record = SOURCES[name]
    assert hashlib.sha256(raw).hexdigest() == record["sha256"]
    assert (
        hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == record["git_blob_sha1"]
    )
    assert (
        hashlib.sha256((FIXTURES / MANIFEST["license_file"]).read_bytes()).hexdigest()
        == MANIFEST["license_sha256"]
    )
    return raw.decode()


def original_terms(text: str) -> tuple[tuple[str, float], ...]:
    terms = re.findall(
        r"PauliTerm\s*\{\s*pauli=\[(.*?)\],\s*coefficient=([^}]+)\}", text, re.S
    )
    return tuple(
        (
            "".join(re.findall(r"Pauli::([IZ])", word)),
            ast.literal_eval(coefficient.strip()),
        )
        for word, coefficient in terms
    )


def reference_state(
    terms: tuple[tuple[str, float], ...], parameters: tuple[float, ...]
) -> torch.Tensor:
    width = len(terms[0][0])
    circuit = fq.Circuit(width, dtype=torch.complex128)
    for wire in range(width):
        circuit.h(wire)
    phase = 1 + 0j
    for gamma, beta in zip(parameters[::2], parameters[1::2], strict=True):
        for word, coefficient in terms:
            if gamma * coefficient == 0:
                continue
            support = [wire for wire, axis in enumerate(reversed(word)) if axis == "Z"]
            if not support:
                phase *= cmath.exp(-1j * gamma * coefficient)
                continue
            *controls, pivot = support
            for wire in controls:
                circuit.cx(wire, pivot)
            circuit.rz(pivot, 2 * gamma * coefficient)
            for wire in reversed(controls):
                circuit.cx(wire, pivot)
        for wire in range(width):
            circuit.rx(wire, beta)
    return fq.run(circuit).statevector().reshape(-1) * phase


def reference_cost(
    terms: tuple[tuple[str, float], ...],
    state: torch.Tensor,
    *,
    alpha: float,
    maximize: bool,
) -> tuple[float, float, float]:
    # Form the spectral values by broadcasting parity factors per term, rather
    # than performing the Walsh transform used by the workflow.
    width = len(terms[0][0])
    energies = torch.zeros((2,) * width, dtype=torch.float64)
    sign = torch.tensor([1.0, -1.0], dtype=torch.float64)
    for word, coefficient in terms:
        partial = torch.tensor(coefficient, dtype=torch.float64).reshape((1,) * width)
        for wire, axis in enumerate(reversed(word)):
            if axis == "Z":
                shape = [1] * width
                shape[wire] = 2
                partial = partial * sign.reshape(shape)
        energies.add_(partial)
    values = energies.flatten()
    probabilities = state.abs().square_()
    probabilities /= probabilities.sum()
    mean = torch.dot(values, probabilities).item()
    variance = torch.dot(values.square(), probabilities).item() - mean**2
    if alpha == 1:
        return mean, mean, variance
    ordered, permutation = values.sort(descending=maximize)
    weights = probabilities[permutation]
    remaining = alpha - (weights.cumsum(0) - weights)
    contribution = torch.minimum(remaining.clamp_min_(0), weights)
    return torch.dot(contribution, ordered).item() / alpha, mean, variance


@pytest.mark.parametrize("name", SOURCES)
def test_original_workflow_state_objective_and_saved_outputs(name: str) -> None:
    path = FIXTURES / name
    text = source_text(name)
    terms = original_terms(text)
    imported = load_qmod(path)
    workflow = imported.workflow
    assert workflow is not None
    settings = workflow.settings
    result = run_qaoa_workflow(workflow, memory_limit_bytes=BUDGET)
    vqe = result.vqe
    assert 1 <= len(vqe.evaluations) <= settings.max_iteration
    assert vqe.evaluations[0].parameters == settings.initial_point
    assert result.saved["vqe_result"] is vqe
    assert result.saved["hamiltonian"] == terms
    energies = [evaluation.metrics.energy for evaluation in vqe.evaluations]
    optimum = max if settings.maximize else min
    assert vqe.best.metrics.energy == optimum(energies)

    reference = reference_state(terms, vqe.best.parameters)
    torch.testing.assert_close(vqe.statevector, reference, atol=1e-11, rtol=0)
    cvar, mean, variance = reference_cost(
        terms, reference, alpha=settings.alpha_cvar, maximize=settings.maximize
    )
    assert vqe.best.metrics.energy == pytest.approx(cvar, rel=1e-11, abs=1e-10)
    assert vqe.best.metrics.mean == pytest.approx(mean, rel=1e-11, abs=1e-10)
    assert vqe.best.metrics.variance == pytest.approx(variance, rel=1e-10, abs=1e-9)


def test_explicit_bindings_and_source_initial_point_agree_with_circuit() -> None:
    name = "qaoa_in_qaoa.qmod"
    path = FIXTURES / name
    terms = original_terms(source_text(name))
    for bindings in (None, {"params_list": (0.12, -0.29, 0.53, 0.83)}):
        params = (0.0, 1.0, 1.0, 0.0) if bindings is None else bindings["params_list"]
        program = synth.synthesize(path, classical_inputs=bindings)
        actual = fq.run(program).statevector().reshape(-1)
        torch.testing.assert_close(
            actual, reference_state(terms, params), atol=1e-12, rtol=0
        )


@pytest.mark.parametrize(
    "edit",
    [
        lambda text: text.replace("Optimizer.COBYLA", "Optimizer.SPSA"),
        lambda text: text.replace("initial_point=[", "initial_point=run([", 1),
        lambda text: text.replace("alpha_cvar=1.0", "alpha_cvar=0.0"),
        lambda text: text.replace(
            '"hamiltonian": hamiltonian', '"hamiltonian": open("file")'
        ),
        lambda text: text.replace("qaoa_penalty(target.len", "qaoa_penalty(1"),
        lambda text: text.replace("Pauli::I", "Pauli::X", 1),
        lambda text: text.replace("save({", "print('side effect')\nsave({"),
    ],
)
def test_workflow_refuses_unproved_classical_or_quantum_effects(edit) -> None:
    text = source_text("qaoa_in_qaoa.qmod")
    with pytest.raises(InteropConversionError):
        synth.synthesize(edit(text), source_format="qmod")


def test_workflow_refuses_insufficient_state_budget() -> None:
    imported = load_qmod(FIXTURES / "qaoa_in_qaoa.qmod")
    assert imported.workflow is not None
    with pytest.raises(ValueError, match="bytes for its state"):
        run_qaoa_workflow(imported.workflow, memory_limit_bytes=1)

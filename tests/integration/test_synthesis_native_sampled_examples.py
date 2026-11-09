"""Retained Pauli sources, native data and independent Circuits agree exactly."""

import hashlib
import importlib
import json
from pathlib import Path
from typing import Literal

import pytest

from examples.synthesis.compare_states import compare_states
from examples.synthesis.pauli_evolution import load_terms, reference_sequence
from flagquantum import synth
from flagquantum.compiler._construction.frontend_program import capture_program
from tools.synthesis_catalog import load_catalog
from tools.validate_qmod_large_pauli import original_terms

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
NAMES = (
    "hamiltonian_simulation_guide",
    "hamiltonian_simulation_guide_qdrift",
    "qdrift",
    "example_exponentiation",
    "hamiltonian_evolution",
)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("objective", ("width", "depth"))
def test_complete_original_native_and_independent_circuit(
    name: str, objective: Literal["width", "depth"]
) -> None:
    case = next(c for c in load_catalog()["cases"] if c["id"].endswith(f"/{name}.qmod"))
    module = ".".join(Path(case["directory"]).parts)
    native = importlib.import_module(module + ".main")
    reference = importlib.import_module(module + ".comparisons.circuit")
    for time in ((0.37, -0.81) if name == "hamiltonian_simulation_guide" else (None,)):
        bindings = {} if time is None else {"t": time}
        for entry in (native.main, ROOT / case["original"]["path"]):
            program = synth.synthesize(
                entry, classical_inputs=bindings, objective=objective, max_workspace=0
            )
            registers = tuple(program.registers)
            expected = (
                reference.reference_circuit()
                if time is None
                else reference.reference_circuit(time)
            )
            compare_states(program, expected, registers)
            assert program.report["workspace_qubits"] == 0


@pytest.mark.parametrize("name", ("example_exponentiation", "hamiltonian_evolution"))
def test_all_large_terms_indices_and_sampled_payloads_are_retained(name: str) -> None:
    case = next(c for c in load_catalog()["cases"] if c["id"].endswith(f"/{name}.qmod"))
    directory = ROOT / case["directory"]
    width, terms = load_terms(directory / "hamiltonian.json")
    source_terms = original_terms((ROOT / case["original"]["path"]).read_text())
    assert width == 12 and len(terms) == len(source_terms) == 551
    assert terms == tuple((term.coefficient, term.support) for term in source_terms)
    provenance = json.loads((directory / "provenance.json").read_text())
    assert (
        hashlib.sha256((directory / "hamiltonian.json").read_bytes()).hexdigest()
        == provenance["hamiltonian_data_sha256"]
    )
    native = importlib.import_module(".".join(Path(case["directory"]).parts) + ".main")
    program, _ = capture_program(native.main)
    rotations = [
        op
        for op in program.body.blocks[0].operations
        if op.name == "qreg.pauli_rotation"
    ]
    count = 2000 if name == "hamiltonian_evolution" else None
    sequence = reference_sequence(terms, 1.0, count)
    assert len(sequence) == len(rotations) == (count or 551)
    for operation, (index, angle) in zip(rotations, sequence, strict=True):
        assert operation.attributes["support"] == tuple(sorted(terms[index][1]))
        assert operation.attributes["angle"] == angle
    if count:
        indices = tuple(index for index, _ in sequence)
        assert (
            hashlib.sha256(
                json.dumps(indices, separators=(",", ":")).encode()
            ).hexdigest()
            == "e5e359ee6b71f74c13ee5cebe72575f3bed29ffbe2b931dafc78f1aff542e988"
        )

"""Native folding contact helpers preserve the entire frozen quantum model."""

from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from typing import Literal, cast

import pytest
import torch

import flagquantum as fq
from examples.synthesis.applications.chemistry.protein_folding.protein_folding_with_qaoa.protein_folding import (
    main as folding,
)
from examples.synthesis.applications.chemistry.protein_folding.protein_folding_with_qaoa.protein_folding.comparisons import (
    circuit as reference,
)
from examples.synthesis.compare_states import compare_states
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate
from tools.validate_qmod_records import RecordCost

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    ROOT
    / "tests/fixtures/classiq_qmod/corpus/original/applications/chemistry/protein_folding/protein_folding_with_qaoa/protein_folding.qmod"
)


def test_all_128_classical_energies_match_frozen_arithmetic() -> None:
    # The existing independent tool has no annotations; type its arithmetic boundary.
    oracle_factory = cast(
        Callable[[str], Callable[[tuple[int, ...]], Fraction]], RecordCost
    )
    oracle = oracle_factory(SOURCE.read_text())
    for basis in range(128):
        bits = tuple((basis >> bit) & 1 for bit in range(7))
        assert oracle(bits) == reference.cost(bits)
    # The activity bit changes contact energies, including the retained offset.
    assert any(
        reference.cost(tuple((basis >> bit) & 1 for bit in range(6)) + (0,))
        != reference.cost(tuple((basis >> bit) & 1 for bit in range(6)) + (1,))
        for basis in range(64)
    )


@pytest.mark.parametrize("binding", [0, 1])
@pytest.mark.parametrize("objective", ["width", "depth"])
def test_complete_folding_original_and_native(
    binding: int, objective: Literal["width", "depth"]
) -> None:
    params = reference.BINDINGS[binding]
    native = synth.synthesize(
        folding.main,
        classical_inputs={"params": params},
        objective=objective,
        max_workspace=0,
    )
    source = synth.synthesize(
        SOURCE, classical_inputs={"params": params}, objective=objective
    )
    circuit = reference.reference_circuit(params)
    compare_states(native, circuit, ("fold", "active"))
    compare_states(source, circuit, reference.SOURCE_FIELDS)
    resources = native.report["resources"]
    assert isinstance(resources, ResourceEstimate)
    assert resources.n_qubits == 7
    assert native.report["workspace_qubits"] == 0


def test_zero_angles_and_binding_validation() -> None:
    program = folding.build_program((0.0,) * 10)
    circuit = fq.Circuit(7, dtype=torch.complex128)
    for wire in range(7):
        circuit.h(wire)
    compare_states(program, circuit, ("fold", "active"))
    with pytest.raises(ValueError, match="parameters"):
        folding.build_program((0.1,))
    with pytest.raises(ValueError, match="finite"):
        folding.build_program((float("nan"),) * 10)


@pytest.mark.parametrize(
    "budget", ["max_qubits", "max_operations", "max_depth", "memory_limit_bytes"]
)
def test_folding_program_respects_physical_budgets(budget: str) -> None:
    with pytest.raises(ValueError):
        synth.synthesize(
            folding.main,
            classical_inputs={"params": reference.BINDINGS[0]},
            max_qubits=6 if budget == "max_qubits" else None,
            max_operations=1 if budget == "max_operations" else None,
            max_depth=1 if budget == "max_depth" else None,
            memory_limit_bytes=1 if budget == "memory_limit_bytes" else None,
        )

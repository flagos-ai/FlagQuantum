"""Both retained molecular ansatz paths preserve the signed Y rotation."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest

import flagquantum as fq
from examples.synthesis.applications.chemistry.molecular_energy_curve import (
    main as curve,
)
from examples.synthesis.applications.chemistry.molecular_energy_curve.comparisons import (
    circuit as curve_reference,
)
from examples.synthesis.applications.chemistry.molecule_eigensolver import (
    main as eigensolver,
)
from examples.synthesis.applications.chemistry.molecule_eigensolver.comparisons import (
    circuit as eigensolver_reference,
)
from examples.synthesis.compare_states import compare_states
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate
from flagquantum.compiler._construction.frontend_python import _QuantumFunction

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Case:
    name: str
    source: str
    definition: _QuantumFunction
    reference: Callable[[tuple[float, ...]], fq.Circuit]
    bindings: tuple[tuple[float, ...], ...]


CASES = (
    Case(
        "curve",
        "applications/chemistry/molecular_energy_curve/molecular_energy_curve.qmod",
        curve.main,
        curve_reference.reference_circuit,
        curve_reference.BINDINGS,
    ),
    Case(
        "eigensolver",
        "applications/chemistry/molecule_eigensolver/molecule_eigensolver.qmod",
        eigensolver.main,
        eigensolver_reference.reference_circuit,
        eigensolver_reference.BINDINGS,
    ),
)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("dataset", [0, 1])
@pytest.mark.parametrize("objective", ["width", "depth"])
def test_complete_original_and_native_match_signed_ry_reference(
    case: Case, dataset: int, objective: Literal["width", "depth"]
) -> None:
    params = case.bindings[dataset]
    assert params[0] != 0
    reference = case.reference(params)
    for entry in (
        case.definition,
        ROOT / "tests/fixtures/classiq_qmod/corpus/original" / case.source,
    ):
        program = synth.synthesize(
            entry,
            classical_inputs={"params": params},
            objective=objective,
            max_workspace=0,
        )
        compare_states(program, reference, ("state",))
        resource = program.report["resources"]
        assert isinstance(resource, ResourceEstimate)
        assert resource.n_qubits == 1
        assert program.report["workspace_qubits"] == 0


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_one_instruction_budget_and_insufficient_memory(case: Case) -> None:
    params = case.bindings[0]
    program = synth.synthesize(
        case.definition, classical_inputs={"params": params}, max_operations=1
    )
    resource = program.report["resources"]
    assert isinstance(resource, ResourceEstimate)
    assert resource.n_operations == 1
    compare_states(program, case.reference(params), ("state",))
    with pytest.raises(ValueError):
        synth.synthesize(
            case.definition, classical_inputs={"params": params}, memory_limit_bytes=1
        )


@pytest.mark.parametrize("build", [curve.build_program, eigensolver.build_program])
def test_user_entry_rejects_missing_or_extra_angles(
    build: Callable[[tuple[float, ...]], synth.Program],
) -> None:
    for params in ((), (0.37, -0.81)):
        with pytest.raises(ValueError):
            build(params)

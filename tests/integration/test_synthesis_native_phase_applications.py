"""Retained native phase applications preserve all cost data and logical fields."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest
import torch

import flagquantum as fq
from examples.synthesis.applications.cybersecurity.patching_management.patch_min_vertex_cover import (
    main as patch,
)
from examples.synthesis.applications.cybersecurity.patching_management.patch_min_vertex_cover.comparisons import (
    circuit as patch_reference,
)
from examples.synthesis.compare_states import compare_states
from examples.synthesis.tutorials.basics.optimization.learning_optimization import (
    main as learning,
)
from examples.synthesis.tutorials.basics.optimization.learning_optimization.comparisons import (
    circuit as learning_reference,
)
from examples.synthesis.tutorials.basics.quantum_programming.qmod_tutorial_part1 import (
    main as interference,
)
from examples.synthesis.tutorials.basics.quantum_programming.qmod_tutorial_part1.comparisons.circuit import (
    reference_circuit as interference_reference,
)
from flagquantum import synth
from flagquantum.compiler import estimate_resources
from flagquantum.compiler._construction.frontend_python import _QuantumFunction

pytestmark = pytest.mark.integration
CORPUS = Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod/corpus/original"


@dataclass(frozen=True)
class Case:
    path: str
    entry: _QuantumFunction
    reference: Callable[[tuple[float, ...]], fq.Circuit]
    bindings: tuple[tuple[float, ...], ...]
    registers: tuple[str, ...]
    source_registers: tuple[str, ...]


CASES = (
    Case(
        "applications/cybersecurity/patching_management/patch_min_vertex_cover.qmod",
        patch.main,
        patch_reference.reference_circuit,
        patch_reference.BINDINGS,
        ("v",),
        ("v.x",),
    ),
    Case(
        "tutorials/basic_tutorials/optimization/learning_optimization.qmod",
        learning.main,
        learning_reference.reference_circuit,
        learning_reference.BINDINGS,
        ("x0", "x1", "slack"),
        ("v.x[0]", "v.x[1]", "v.constraint_slack_var"),
    ),
)


@pytest.mark.parametrize("case", CASES, ids=lambda case: Path(case.path).stem)
@pytest.mark.parametrize("dataset", [0, 1])
@pytest.mark.parametrize("objective", ["width", "depth"])
def test_complete_phase_applications_match_original_and_independent_cost_circuit(
    case: Case, dataset: int, objective: Literal["width", "depth"]
) -> None:
    params = case.bindings[dataset]
    native = synth.synthesize(
        case.entry,
        classical_inputs={"params": params},
        objective=objective,
        max_workspace=0,
    )
    source = synth.synthesize(
        CORPUS / case.path, classical_inputs={"params": params}, objective=objective
    )
    expected = case.reference(params)
    for program, names in ((native, case.registers), (source, case.source_registers)):
        compare_states(program, expected, names)
        assert program.report["workspace_qubits"] == 0
        assert estimate_resources(program.circuit).n_qubits == 8


@pytest.mark.parametrize("objective", ["width", "depth"])
def test_fixed_four_bit_interference_program_preserves_both_source_phase_flips(
    objective: Literal["width", "depth"],
) -> None:
    native = synth.synthesize(interference.main, objective=objective, max_workspace=0)
    source = synth.synthesize(
        CORPUS
        / "tutorials/basic_tutorials/the_classiq_tutorial/Qmod_tutorial_part1.qmod",
        objective=objective,
    )
    expected = interference_reference()
    for program in (native, source):
        compare_states(program, expected, ("x",))
    assert native.report["workspace_qubits"] == 0
    assert source.report["workspace_qubits"] == 1
    assert estimate_resources(native.circuit).n_qubits == 4
    assert estimate_resources(source.circuit).n_qubits == 5


def test_constraint_slack_weights_and_offset_preserve_absolute_phase() -> None:
    assert learning_reference.cost((0,) * 8) == -80
    assert learning_reference.cost((0, 0, 0, 1, 0, 0, 0, 0)) == -500
    params = (0.17, 0.0)
    program = learning.build_program(params)
    values = torch.tensor(
        [
            learning_reference.cost(tuple((basis >> i) & 1 for i in range(8)))
            for basis in range(256)
        ],
        dtype=torch.float64,
    )
    actual = (
        fq.run(program)
        .statevector()
        .reshape((2,) * 8)
        .permute(
            tuple(
                wire
                for name in ("x0", "x1", "slack")
                for wire in program.registers[name]
            )
        )
        .flatten()
    )
    torch.testing.assert_close(
        actual, torch.exp(1j * params[0] * values) / 16, atol=1e-12, rtol=0
    )


def test_patching_keeps_all_eighteen_uncovered_edges() -> None:
    assert patch_reference.cost((0,) * 8) == -180
    assert patch_reference.cost((1,) * 8) == -8
    # One selected isolated-in-this-assignment vertex still has all other costs.
    assert patch_reference.cost((0, 0, 0, 0, 1, 0, 0, 0)) == -161

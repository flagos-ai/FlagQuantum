"""Original-width Grover programs agree with independent exact Circuit ladders."""

from pathlib import Path
from typing import Annotated

import pytest
import torch

import flagquantum as fq
from examples.synthesis.applications.cybersecurity.whitebox_fuzzing import main as white
from examples.synthesis.applications.cybersecurity.whitebox_fuzzing.comparisons import (
    circuit as white_ref,
)
from examples.synthesis.tutorials.basics.grover_graph_coloring import (
    main as graph,
)
from examples.synthesis.tutorials.basics.grover_graph_coloring.comparisons import (
    circuit as graph_ref,
)
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate
from flagquantum.synth import (
    Output,
    QNum,
    allocate,
    control,
    hadamard_transform,
    phase,
    qfunc,
)

pytestmark = pytest.mark.integration
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod/corpus/original"


def logical_state(program: synth.Program, names: tuple[str, ...]) -> torch.Tensor:
    wires = tuple(bit for name in names for bit in program.registers[name])
    resources = program.report["resources"]
    assert isinstance(resources, ResourceEstimate)
    scratch = tuple(bit for bit in range(resources.n_qubits) if bit not in wires)
    state = (
        fq.run(program)
        .statevector()
        .reshape((2,) * resources.n_qubits)
        .permute(*wires, *scratch)
        .reshape(1 << len(wires), -1)
    )
    assert float(state[:, 1:].abs().square().sum()) < 1e-22
    assert isinstance(state, torch.Tensor)
    return state[:, 0]


@pytest.mark.parametrize("kind", ["white", "graph"])
@pytest.mark.parametrize("r", [0, 1, 2])
def test_native_independent_complete_states(kind: str, r: int) -> None:
    module = white if kind == "white" else graph
    reference = white_ref if kind == "white" else graph_ref
    names = ("x", "y") if kind == "white" else ("a", "b", "c", "d", "e")
    width = sum(reference.WIDTHS)
    state = fq.run(reference.reference_circuit(r)).statevector().reshape(1 << width, -1)
    assert float(state[:, 1:].abs().square().sum()) < 1e-22
    torch.testing.assert_close(
        logical_state(module.build_program(r), names), state[:, 0], atol=1e-12, rtol=0
    )


@pytest.mark.parametrize("kind", ["white", "graph"])
def test_original_full_iteration_count_matches_native_and_independent(
    kind: str,
) -> None:
    widths: tuple[int, ...]
    names: tuple[str, ...]
    original_names: tuple[str, ...]
    if kind == "white":
        source = "applications/cybersecurity/whitebox_fuzzing/whitebox_fuzzing.qmod"
        names = ("x", "y")
        original_names = ("vs.x", "vs.y")
        program, widths, circuit = (
            white.build_program(),
            white_ref.WIDTHS,
            white_ref.reference_circuit(10),
        )
    else:
        source = (
            "tutorials/basic_tutorials/grover_graph_coloring/grover_graph_coloring.qmod"
        )
        names = ("a", "b", "c", "d", "e")
        original_names = tuple("vars." + name for name in names)
        program, widths, circuit = (
            graph.build_program(),
            graph_ref.WIDTHS,
            graph_ref.reference_circuit(1),
        )
    expected = fq.run(circuit).statevector().reshape(1 << sum(widths), -1)
    assert float(expected[:, 1:].abs().square().sum()) < 1e-22
    torch.testing.assert_close(
        logical_state(program, names), expected[:, 0], atol=1e-12, rtol=0
    )
    original = synth.synthesize(FIXTURES / source)
    torch.testing.assert_close(
        logical_state(original, original_names), expected[:, 0], atol=1e-12, rtol=0
    )


@qfunc
def coherent_phase(
    angle: float, x: Output[Annotated[QNum, 6]], y: Output[Annotated[QNum, 6]]
) -> None:
    allocate(x)
    allocate(y)
    hadamard_transform(x)
    hadamard_transform(y)
    control(((x + y) < 9) & (((x % 4) * (y % 2)) == 1), lambda: phase(1, angle))


@pytest.mark.parametrize("angle", [0.0, 0.37, -1.1])
def test_arbitrary_coherent_phase_without_alignment(angle: float) -> None:
    program = synth.synthesize(
        coherent_phase,
        classical_inputs={"angle": angle},
        max_workspace=8,
        memory_limit_bytes=64 * 1024**2,
    )
    expected = torch.full((4096,), 1 / 64, dtype=torch.complex128)
    expected[list(white_ref.marked())] *= torch.exp(
        torch.tensor(1j * angle, dtype=torch.complex128)
    )
    torch.testing.assert_close(
        logical_state(program, ("x", "y")), expected, atol=1e-12, rtol=0
    )


@qfunc
def uninitialized_phase(
    x: Output[Annotated[QNum, 6]], y: Output[Annotated[QNum, 6]]
) -> None:
    allocate(x)
    control(((x + y) < 9) & (((x % 4) * (y % 2)) == 1), lambda: phase(1, 0.0))


def test_zero_angle_does_not_hide_uninitialized_reads() -> None:
    from flagquantum.compiler._hybrid.verifier import HybridVerificationError

    with pytest.raises(HybridVerificationError):
        synth.synthesize(uninitialized_phase)


@pytest.mark.parametrize(
    "constraint", ["qubits", "workspace", "operations", "depth", "memory"]
)
def test_actual_budgets_refuse(constraint: str) -> None:
    from flagquantum.errors import PlanningError

    with pytest.raises((synth.SynthesisError, PlanningError)):
        if constraint == "qubits":
            synth.synthesize(white.main, classical_inputs={"r": 1}, max_qubits=12)
        elif constraint == "workspace":
            synth.synthesize(white.main, classical_inputs={"r": 1}, max_workspace=0)
        elif constraint == "operations":
            synth.synthesize(white.main, classical_inputs={"r": 1}, max_operations=1)
        elif constraint == "depth":
            synth.synthesize(white.main, classical_inputs={"r": 1}, max_depth=1)
        else:
            synth.synthesize(
                white.main, classical_inputs={"r": 1}, memory_limit_bytes=16
            )


@qfunc
def always_phase(
    angle: float, x: Output[Annotated[QNum, 6]], y: Output[Annotated[QNum, 6]]
) -> None:
    allocate(x)
    allocate(y)
    hadamard_transform(x)
    hadamard_transform(y)
    control((x >= 0) & (x == x) & (y == y), lambda: phase(1, angle))


@qfunc
def never_phase(
    angle: float, x: Output[Annotated[QNum, 6]], y: Output[Annotated[QNum, 6]]
) -> None:
    allocate(x)
    allocate(y)
    hadamard_transform(x)
    hadamard_transform(y)
    control((x >= 0) & (x != x) & (y == y), lambda: phase(1, angle))


@pytest.mark.parametrize("always", [False, True])
def test_constant_predicate_preserves_absolute_phase(always: bool) -> None:
    program = synth.synthesize(
        always_phase if always else never_phase, classical_inputs={"angle": 0.41}
    )
    expected = torch.full((4096,), 1 / 64, dtype=torch.complex128)
    if always:
        expected *= torch.exp(torch.tensor(0.41j, dtype=torch.complex128))
    torch.testing.assert_close(
        logical_state(program, ("x", "y")), expected, atol=1e-12, rtol=0
    )


def test_marked_assignment_counts_and_boundaries() -> None:
    assert len(white_ref.marked()) == 6
    assert len(graph_ref.marked()) == 16
    assert 1 * 64 + 1 in white_ref.marked()
    assert 5 * 64 + 3 in white_ref.marked()
    assert 5 * 64 + 5 not in white_ref.marked()


@qfunc
def alias_main(x: Output[Annotated[QNum, 6]]) -> None:
    allocate(x)
    hadamard_transform(x)
    white.mark(x, x)


def test_repeated_mutable_helper_actual_refuses_before_lowering() -> None:
    from flagquantum.compiler._hybrid.verifier import HybridVerificationError

    with pytest.raises(HybridVerificationError, match="disjoint writable"):
        synth.synthesize(alias_main)

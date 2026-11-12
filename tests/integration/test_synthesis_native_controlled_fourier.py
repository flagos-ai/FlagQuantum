"""Coherent Fourier control shares lowering and preserves absolute amplitudes."""

import math
from pathlib import Path
from typing import Literal

import pytest
import torch

import flagquantum as fq
from examples.synthesis.algorithms.quantum_primitives.hadamard_test.comparisons.circuit import (
    reference_circuit,
)
from examples.synthesis.algorithms.quantum_primitives.hadamard_test.main import (
    build_program,
)
from examples.synthesis.compare_states import compare_states
from examples.synthesis.functions.qft.comparisons.circuit import fourier_matrix
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate
from flagquantum.compiler._construction.frontend_python import _QuantumFunction
from tests.team.compiler.test_construction_python_program import program_definition

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "source",
    [
        "algorithms/quantum_primitives/hadamard_test/hadamard_test.qmod",
        "tutorials/basic_tutorials/quantum_primitives/hadamard_test_tutorial/hadamard_test_tutorial.qmod",
    ],
)
def test_complete_original_native_and_independent_hadamard_test(source: str) -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "fixtures/classiq_qmod/corpus/original"
        / source
    )
    original = synth.synthesize(path, max_workspace=0)
    native = build_program()
    reference = reference_circuit()
    compare_states(native, reference, ("expectation_value", "psi"))
    expected = fq.run(reference).statevector()
    torch.testing.assert_close(
        fq.run(original).statevector(), expected, atol=1e-12, rtol=0
    )
    for program in (native, original):
        result = fq.run(program)
        assert dict(
            program.probabilities(result, registers=("expectation_value",))
        ) == pytest.approx({(0,): 5 / 8, (1,): 3 / 8})
        assert program.report["workspace_qubits"] == 0
        resource = program.report["resources"]
        assert isinstance(resource, ResourceEstimate)
        assert resource.n_qubits == 5
    # The target is entangled, not clean scratch: selecting readout does not reset it.
    assert float(expected.reshape(2, 16)[:, 1:].abs().square().sum()) > 0.1
    assert set(original.registers) == {"expectation_value"}
    assert set(native.registers) == {"expectation_value", "psi"}


def definition(body: str, width: int = 3) -> _QuantumFunction:
    return program_definition(
        "from flagquantum.synth import *\n"
        "@qfunc\ndef main(c: Output[QBit], q: Output[QArray[QBit]]):\n"
        f"    allocate(c)\n    allocate({width},q)\n"
        + "\n".join("    " + line for line in body.splitlines())
        + "\n"
    )


def controlled_matrix(width: int) -> torch.Tensor:
    size = 1 << width
    matrix = torch.zeros((2 * size, 2 * size), dtype=torch.complex128)
    matrix[:size, :size] = torch.eye(size, dtype=torch.complex128)
    matrix[size:, size:] = fourier_matrix(width)
    return matrix


@pytest.mark.parametrize("control", [0, 1])
@pytest.mark.parametrize(
    ("width", "basis"), [(w, b) for w in range(1, 4) for b in range(1 << w)]
)
def test_every_small_column_in_both_control_branches(
    width: int, basis: int, control: int
) -> None:
    prep = ["X(c)"] if control else []
    prep.extend(f"X(q[{bit}])" for bit in range(width) if basis & (1 << bit))
    program = synth.synthesize(
        definition("\n".join([*prep, "control(c,lambda: qft(q))"]), width),
        max_workspace=0,
    )
    reference = fq.Circuit(width + 1, dtype=torch.complex128)
    if control:
        reference.x(0)
    for bit in range(width):
        if basis & (1 << bit):
            reference.x(width - bit)
    reference.any(*range(width + 1), matrix=controlled_matrix(width))
    compare_states(program, reference, ("c", "q"))
    assert program.report["workspace_qubits"] == 0


@pytest.mark.parametrize("objective", ["width", "depth"])
def test_coherent_selector_interference_retains_absolute_global_phase(
    objective: Literal["width", "depth"],
) -> None:
    program = synth.synthesize(
        definition(
            "H(c)\nhadamard_transform(q)\nRX(0.73,q[0])\nRZ(-0.49,q[1])\n"
            "phase(1,0.31)\ncontrol(c,lambda: qft(q))\nH(c)"
        ),
        objective=objective,
        max_workspace=0,
    )
    reference = fq.Circuit(4, dtype=torch.complex128)
    for wire in range(4):
        reference.h(wire)
    reference.rx(3, theta=0.73).rz(2, theta=-0.49)
    reference.any(
        0,
        matrix=torch.eye(2, dtype=torch.complex128)
        * complex(math.cos(0.31), math.sin(0.31)),
    )
    reference.any(*range(4), matrix=controlled_matrix(3)).h(0)
    compare_states(program, reference, ("c", "q"))


def test_comparison_predicate_is_uncomputed_after_disjoint_transform() -> None:
    entry = program_definition(
        "from flagquantum.synth import *\n"
        "@qfunc\ndef main(c:Output[QNum[2]], q:Output[QArray[QBit]]):\n"
        "    allocate(c)\n    allocate(2,q)\n    hadamard_transform(c)\n"
        "    X(q[0])\n    control(c>=2,lambda: qft(q))\n"
    )
    program = synth.synthesize(entry)
    blocks = [
        fourier_matrix(2) if c >= 2 else torch.eye(4, dtype=torch.complex128)
        for c in range(4)
    ]
    reference = fq.Circuit(4, dtype=torch.complex128).h(0).h(1).x(3)
    matrix = torch.zeros((16, 16), dtype=torch.complex128)
    for index, block in enumerate(blocks):
        matrix[4 * index : 4 * index + 4, 4 * index : 4 * index + 4] = block
    reference.any(*range(4), matrix=matrix)
    compare_states(program, reference, ("c", "q"))
    workspace = program.report["workspace_qubits"]
    assert isinstance(workspace, int) and workspace > 0


@pytest.mark.parametrize(
    "body",
    [
        "control(c,lambda: qft(c))",
        "control(q>=2,lambda: qft(q))",
        "control(c,lambda: qft(q[:]))",
        "control(c,lambda: qft(q,q))",
        "control(c,lambda: qft())",
        "control(c,lambda: qft(target=q))",
        "control(c,lambda: control(c,lambda: qft(q)))",
    ],
)
def test_aliases_and_unsupported_callback_shapes_are_refused(body: str) -> None:
    with pytest.raises(ValueError):
        synth.synthesize(definition(body))


@pytest.mark.parametrize("target", ["const", "uninitialized"])
def test_readonly_and_uninitialized_targets_are_refused(target: str) -> None:
    if target == "const":
        source = (
            "@qfunc\ndef helper(c:Const[QBit],q:Const[QNum[2]]):\n"
            "    control(c,lambda: qft(q))\n"
            "@qfunc\ndef main(c:Output[QBit],q:Output[QNum[2]]):\n"
            "    allocate(c)\n    allocate(q)\n    helper(c,q)\n"
        )
    else:
        source = (
            "@qfunc\ndef main(c:Output[QBit],q:Output[QNum[2]]):\n"
            "    allocate(c)\n    control(c,lambda: qft(q))\n"
        )
    with pytest.raises(ValueError):
        synth.synthesize(
            program_definition("from flagquantum.synth import *\n" + source)
        )


@pytest.mark.parametrize(
    ("qubits", "operations", "memory"),
    [(3, None, None), (None, 1, None), (None, None, 1)],
)
def test_actual_joint_state_and_circuit_budgets_are_enforced(
    qubits: int | None,
    operations: int | None,
    memory: int | None,
) -> None:
    with pytest.raises(ValueError):
        synth.synthesize(
            definition("H(c)\ncontrol(c,lambda: qft(q))"),
            max_qubits=qubits,
            max_operations=operations,
            memory_limit_bytes=memory,
        )

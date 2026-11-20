"""Array-valued source declarations select their exact live matrix leaves."""

from typing import Any

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR
from flagquantum.ecosystem.cirq._register_graph import import_register_graph

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
np = pytest.importorskip("numpy")


@pytest.mark.parametrize("width", (1, 2, 3))
def test_complete_original_seeded_matrix_family_imports_and_appends(width: int) -> None:
    from qualtran import BloqBuilder, QAny, Signature
    from qualtran.bloqs.for_testing.matrix_gate import MatrixGate
    from qualtran.resource_counting import QubitCount, get_cost_value

    rng = np.random.RandomState(42)
    matrices = [MatrixGate.random(width, random_state=rng) for _ in range(5)]
    builder, inputs = BloqBuilder.from_signature(Signature.build(q=width))
    current = inputs["q"]
    for matrix in matrices:
        current = builder.add(matrix, q=current)
    graph = builder.finalize(q=current)
    program = import_register_graph(graph)
    assert program.functions[0].registers is not None
    assert program.functions[0].registers[0].bitsize == QAny(width).num_qubits
    # Reverse the physical word to expose accidental matrix-axis reordering.
    wires = tuple(range(width - 1, -1, -1))
    bound = bind_register_program(program, (wires,))
    assert bound.outputs == (wires,) and not bound.events
    assert len(bound.circuit.instructions) == 5
    for operation, source in zip(bound.circuit.instructions, matrices, strict=True):
        assert operation.name == "any" and operation.wires == wires
        torch.testing.assert_close(
            operation.matrix, torch.as_tensor(np.array(source.matrix)), atol=0, rtol=0
        )
    assert get_cost_value(graph, QubitCount()) == width
    size = 1 << width
    expected = torch.eye(size, dtype=torch.complex128)
    for matrix in matrices:
        expected = torch.as_tensor(np.array(matrix.matrix)) @ expected
    # Every basis column and an entangled external spectator share this check.
    rows = torch.arange(1, (1 << (width + 1)) + 1, dtype=torch.float64)
    joint = torch.complex(rows, 2 - rows)
    joint /= joint.norm()
    states = [*torch.eye(size, dtype=torch.complex128), joint]
    for state in states:
        spectator = state.numel() == 2 * size
        axes = (*reversed(range(width)), *((width,) if spectator else ()))
        logical = state.reshape((2,) * (width + int(spectator)))
        physical = logical.permute(axes).flatten()
        target = (expected @ state.reshape(size, -1)).reshape(logical.shape)
        target = target.permute(axes).flatten()
        circuit = Circuit(width + int(spectator), inputs=physical, dtype=physical.dtype)
        circuit.append(FieldOperation(circuit.n_qubits, bound.circuit.instructions))
        torch.testing.assert_close(
            circuit.state().flatten(), target, atol=2e-14, rtol=0
        )
        inverse = bind_register_program(
            import_register_graph(graph.adjoint()), (wires,)
        )
        circuit.append(FieldOperation(circuit.n_qubits, inverse.circuit.instructions))
        torch.testing.assert_close(
            circuit.state().flatten(), physical, atol=2e-14, rtol=0
        )
    assert (
        CircuitIR.from_json(bound.circuit.to_json()).to_json()
        == bound.circuit.to_json()
    )


def test_distinct_array_declarations_and_repeated_equal_matrix_share_correct_leaves() -> (
    None
):
    from qualtran import BloqBuilder, Signature
    from qualtran.bloqs.for_testing.matrix_gate import MatrixGate

    first = MatrixGate(1, ((0, 1j), (1j, 0)))
    second = MatrixGate(1, ((1j, 0), (0, -1j)))
    builder, inputs = BloqBuilder.from_signature(Signature.build(q=1))
    q: Any = inputs["q"]
    for gate in (first, second, MatrixGate(1, first.matrix)):
        q = builder.add(gate, q=q)
    bound = bind_register_program(import_register_graph(builder.finalize(q=q)), ((0,),))
    for operation, gate in zip(
        bound.circuit.instructions, (first, second, first), strict=True
    ):
        torch.testing.assert_close(
            operation.matrix, torch.as_tensor(np.array(gate.matrix))
        )

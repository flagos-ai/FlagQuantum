"""Compare the complete source recursive control recipes at their local leaves."""

from hashlib import sha256
from importlib.metadata import version
from itertools import product
from pathlib import Path

from tests._reference_dependencies import unavailable_reference

try:
    import numpy as np
except ModuleNotFoundError as _reference_import_error:
    unavailable_reference(_reference_import_error)
import pytest
import torch

from flagquantum.compiler.optimization.controlled_unitary_synthesis import (
    Matrix,
    borrowed_controlled_x_ir,
    controlled_unitary_ir,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.ecosystem.cirq._feedback_import import import_cirq_feedback
from flagquantum.simulation.unitary import get_unitary

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def reference_versions() -> None:
    import kickmix

    try:
        from cirq.transformers.analytical_decompositions import (
            controlled_gate_decomposition,
        )
    except ModuleNotFoundError as _reference_import_error:
        unavailable_reference(_reference_import_error)

    raw = Path(controlled_gate_decomposition.__file__).read_bytes()
    assert (
        sha256(raw).hexdigest()
        == "4c5e4d7b0faa0b28f55a65ea786cb869f5684a9d7a1a9698d56bc0a6ccbf0ce8"
    )
    assert version("qualtran") == "0.7.0"
    assert version("cirq-core") == "1.7.0"
    assert getattr(kickmix, "__version__") == "0.1" and hasattr(kickmix, "Simulator")


def assert_source_stream(actual: CircuitIR, source: CircuitIR) -> None:
    assert len(actual.instructions) == len(source.instructions)
    for native, original in zip(actual.instructions, source.instructions, strict=True):
        assert native.name == original.name and native.wires == original.wires
        assert native.metadata == original.metadata == {}
        assert native.params.keys() == original.params.keys()
        for key in native.params:
            assert native.params[key] == pytest.approx(original.params[key], abs=4e-12)
        if original.matrix is None:
            assert native.matrix is None
        else:
            torch.testing.assert_close(
                torch.as_tensor(native.matrix, dtype=torch.complex128),
                torch.as_tensor(original.matrix, dtype=torch.complex128),
                atol=4e-12,
                rtol=0,
            )


@pytest.mark.parametrize(
    "count,kind", tuple(product(range(6), ("h", "yhalf", "su", "scalar", "generic")))
)
def test_recursive_rotation_original_stream_and_full_operator(
    count: int, kind: str
) -> None:
    try:
        import cirq
    except ModuleNotFoundError as _reference_import_error:
        unavailable_reference(_reference_import_error)
    try:
        from cirq.transformers.analytical_decompositions.controlled_gate_decomposition import (
            decompose_multi_controlled_rotation,
        )
    except ModuleNotFoundError as _reference_import_error:
        unavailable_reference(_reference_import_error)

    matrix = {
        "h": cirq.unitary(cirq.H),
        "yhalf": cirq.unitary(cirq.Y**0.5),
        "su": cirq.unitary(cirq.rz(0.37)) @ cirq.unitary(cirq.ry(-0.51)),
        "scalar": np.eye(2, dtype=complex) * np.exp(0.31j),
        "generic": np.exp(0.23j)
        * cirq.unitary(cirq.rx(0.31))
        @ cirq.unitary(cirq.ry(-0.49)),
    }[kind]
    qubits = cirq.LineQubit.range(count + 1)
    source_ops = decompose_multi_controlled_rotation(matrix, qubits[:-1], qubits[-1])
    original = import_cirq_feedback(
        cirq.Circuit(source_ops, strategy=cirq.InsertStrategy.NEW), qubit_order=qubits
    ).ir
    actual = CircuitIR(
        count + 1, controlled_unitary_ir(matrix, tuple(range(count)), count)
    )
    assert_source_stream(actual, original)
    expected = cirq.unitary(cirq.MatrixGate(matrix).controlled(count))
    for restored in (actual, CircuitIR.from_json(actual.to_json())):
        torch.testing.assert_close(
            get_unitary(restored, dtype=torch.complex128),
            torch.as_tensor(expected, dtype=torch.complex128),
            atol=2e-11,
            rtol=0,
        )


@pytest.mark.parametrize("count,borrowed", tuple(product(range(6), (0, 1, 3))))
def test_borrowed_x_source_recipe_restores_arbitrary_borrowed_bits(
    count: int, borrowed: int
) -> None:
    try:
        import cirq
    except ModuleNotFoundError as _reference_import_error:
        unavailable_reference(_reference_import_error)
    try:
        from cirq.transformers.analytical_decompositions.controlled_gate_decomposition import (
            decompose_multi_controlled_x,
        )
    except ModuleNotFoundError as _reference_import_error:
        unavailable_reference(_reference_import_error)

    qubits = cirq.LineQubit.range(count + 1 + borrowed)
    ops = decompose_multi_controlled_x(
        qubits[:count], qubits[count], qubits[count + 1 :]
    )
    original = import_cirq_feedback(
        cirq.Circuit(ops, strategy=cirq.InsertStrategy.NEW), qubit_order=qubits
    ).ir
    actual = CircuitIR(
        len(qubits),
        borrowed_controlled_x_ir(
            tuple(range(count)), count, tuple(range(count + 1, len(qubits)))
        ),
    )
    assert_source_stream(actual, original)
    # Compare complete operators at small widths; larger borrowed branches keep
    # exact source streams and use basis permutation checks without dense matrices.
    for column in range(1 << len(qubits)):
        bits = tuple((column >> (len(qubits) - 1 - i)) & 1 for i in range(len(qubits)))
        expected = column ^ ((1 << borrowed) if all(bits[:count]) else 0)
        # The full matrix is unnecessary for checking its one nonzero column.
        initial = torch.zeros(1 << len(qubits), dtype=torch.complex128)
        initial[column] = 1
        from examples.algorithms.binary_field_arithmetic import FieldOperation
        from flagquantum import Circuit

        result = (
            Circuit(len(qubits), inputs=initial, dtype=torch.complex128)
            .append(FieldOperation(len(qubits), actual.instructions))
            .state()
            .flatten()
        )
        target = torch.zeros_like(result)
        target[expected] = 1
        torch.testing.assert_close(result, target, atol=2e-11, rtol=0)


@pytest.mark.parametrize(
    "matrix",
    [np.eye(3), np.zeros((2, 2)), np.full((2, 2), np.nan), [[1, 0], [0, 2]]],
)
def test_invalid_local_matrices_are_not_synthesized(
    matrix: Matrix | list[list[complex]],
) -> None:
    with pytest.raises(ValueError, match="finite 2x2 unitary"):
        controlled_unitary_ir(matrix, (0,), 1)

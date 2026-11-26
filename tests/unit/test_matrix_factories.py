"""Serialized matrix leaves preserve exact entries and native execution ownership."""

from importlib import import_module
from typing import Any

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler.formats.legacy_component_library import (
    deserialize_component_library,
    serialize_component_library,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_matrix_factories import matrix_factories
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program import verify_program_record
from flagquantum.errors import SerializationError
from tests.unit.test_cirq_decomposition_file import locked_source as locked_source
from tests.unit.test_legacy_arithmetic_factories import _factories, _library

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("width", (1, 2, 3))
@pytest.mark.parametrize("seed", (7, 42, 1234))
@pytest.mark.parametrize("inverse", (False, True))
@pytest.mark.parametrize("depth", (0, 1, 2))
def test_matrix_constructor_exact_entries_full_complex_columns_and_source_free_append(
    width: int,
    seed: int,
    inverse: bool,
    depth: int,
    locked_source: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cirq

    source_module = import_module("qualtran.bloqs.for_testing.matrix_gate")
    resource = import_module("qualtran.resource_counting")
    upstream = import_module("qualtran")
    source = source_module.MatrixGate.random(width, random_state=seed)
    if inverse:
        source = source.adjoint()
    library = _library(source, depth)
    encoded = serialize_component_library(library)
    library = deserialize_component_library(encoded)
    assert serialize_component_library(library) == encoded
    wires = tuple(reversed(range(1, width + 1)))
    with monkeypatch.context() as patch:
        patch.setitem(__import__("sys").modules, "qualtran", None)
        program = resolve_component_program(
            library, 0, {**_factories(), **matrix_factories()}
        )
        assert verify_program_record(program) == ()
        bound = bind_register_program(program, (wires,), borrowed=(0,))
    (operation,) = bound.circuit.instructions
    assert operation.name == "any" and operation.wires == wires
    assert operation.matrix is not None
    torch.testing.assert_close(
        operation.matrix,
        torch.tensor(source.matrix, dtype=torch.complex128),
        atol=0,
        rtol=0,
    )
    assert bound.outputs == (wires,) and bound.circuit.n_wires == width + 1
    original = cirq.Circuit(source.on(*(cirq.LineQubit(w) for w in wires)))
    expected = torch.tensor(
        original.unitary(qubit_order=cirq.LineQubit.range(width + 1)),
        dtype=torch.complex128,
    )
    with monkeypatch.context() as patch:
        patch.setitem(__import__("sys").modules, "qualtran", None)
        patch.setitem(__import__("sys").modules, "cirq", None)
        ir = CircuitIR.from_json(bound.circuit.to_json())
        for column in range(1 << (width + 1)):
            state = torch.zeros(1 << (width + 1), dtype=torch.complex128)
            state[column] = 1
            circuit = Circuit(width + 1, inputs=state, dtype=state.dtype).append(
                FieldOperation(width + 1, ir.instructions)
            )
            torch.testing.assert_close(
                circuit.state().flatten(), expected[:, column], atol=4e-14, rtol=0
            )
    assert resource.get_cost_value(source, resource.QubitCount()) == width
    with pytest.raises(upstream.DecomposeNotImplementedError):
        resource.get_cost_value(source, resource.QECGatesCost())


def test_matrix_payload_is_copied_and_does_not_relax_core_matrix_or_scalar_contract(
    locked_source: Any,
) -> None:
    import numpy as np

    from flagquantum.core.ir.program import FunctionRecord

    factory = next(iter(matrix_factories().values()))
    matrix = np.array([[0, 1j], [1j, 0]])
    record = factory("matrix", None, {"bitsize": 1, "matrix": matrix})
    assert isinstance(record, FunctionRecord)
    assert (
        verify_program_record(
            import_module("flagquantum.core.ir.program").ProgramRecord((record,))
        )
        == ()
    )
    matrix[:] = 0
    assert record.blocks[0].operations[1].matrix == ((0j, 1j), (1j, 0j))
    for args in (
        {"bitsize": 0, "matrix": np.array([[1j]])},
        {"bitsize": True, "matrix": np.eye(2)},
        {"bitsize": 1, "matrix": np.diag([1, 2]), "atol": 10},
        {"bitsize": 1, "matrix": [["1", "0"], ["0", "1"]]},
        {"bitsize": 1, "matrix": [[1, 0, 0], [0, 1]]},
        {"bitsize": 1, "matrix": np.eye(2), "atol": float("nan")},
        {"bitsize": 1, "matrix": np.eye(2), "atol": -1},
        {"bitsize": 1, "matrix": np.eye(2), "extra": 1},
    ):
        with pytest.raises(SerializationError):
            factory("invalid", None, args)
    with pytest.raises(SerializationError, match="signature"):
        factory("invalid", (), {"bitsize": 1, "matrix": np.eye(2)})
    source = import_module("qualtran.bloqs.for_testing.matrix_gate").MatrixGate(
        1, np.diag([1, 1 + 2e-8])
    )
    with pytest.raises(SerializationError, match="unitary"):
        resolve_component_program(_library(source, 0), 0, matrix_factories())
    assert source.atol == 1e-10  # Original NumPy relative tolerance is distinct.

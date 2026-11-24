"""Stored rotation-array code executes without reconstructing lost constructors."""

from dataclasses import replace
from itertools import product
from typing import Any

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler.formats.legacy_bookkeeping_factories import (
    bookkeeping_factories,
)
from flagquantum.compiler.formats.legacy_component_library import (
    deserialize_component_library,
    serialize_component_library,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    ComponentFactory,
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_exchange_factories import exchange_factories
from flagquantum.compiler.formats.legacy_fixed_gate_factories import (
    fixed_gate_factories,
)
from flagquantum.compiler.formats.legacy_qrom_factories import qrom_factories
from flagquantum.compiler.formats.legacy_rotation_factories import rotation_factories
from flagquantum.compiler.registers.component_graph import copy_component_program
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program import FunctionRecord, verify_program_record
from flagquantum.errors import SerializationError
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_multi_control_component import atomic_leaves
from tests.unit.test_multiplexed_source_binding import logical_operator
from tests.unit.test_programmable_rotation_array_file import (
    source_module as source_module,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def _factories() -> dict[str, ComponentFactory]:
    return {
        **bookkeeping_factories(),
        **fixed_gate_factories(),
        **exchange_factories(),
        **rotation_factories(),
        **qrom_factories(),
    }


@pytest.mark.parametrize("kappa,interleaver", ((1, "x"), (2, "h"), (3, "i")))
def test_multitable_stored_source_graph_full_dirty_operator_and_workzero(
    source_module: Any, kappa: int, interleaver: str
) -> None:
    cirq = pytest.importorskip("cirq")

    source = source_module.ProgrammableRotationGateArray(
        (1, 3),
        (2, 1),
        kappa=kappa,
        rotation_gate=cirq.Z,
        interleaved_unitaries=({"x": cirq.X, "h": cirq.H, "i": cirq.I}[interleaver],),
    )
    library = _library(source, 1)
    assert not library.entries[0].declaration.arguments
    data = serialize_component_library(library)
    assert serialize_component_library(deserialize_component_library(data)) == data
    record = resolve_component_program(library, 0, _factories())
    assert (
        verify_program_record(record) == () and copy_component_program(record) == record
    )
    compare_layers(record, source, atomic_leaves(record))
    logical = kappa + 2
    inputs = ((0,), tuple(range(1, kappa + 1)), (kappa + 1,))
    bound = bind_register_program(record, inputs, clean=(logical,))
    assert bound.outputs == inputs
    actual = logical_operator(bound.circuit, logical)
    expected = torch.tensor(source.tensor_contract(), dtype=torch.complex128)
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=0)
    assert CircuitIR.from_json(bound.circuit.to_json()) == bound.circuit
    assert (
        Circuit(bound.circuit.n_wires)
        .append(FieldOperation(bound.circuit.n_wires, bound.circuit.instructions))
        .to_ir()
        .instructions
        == bound.circuit.instructions
    )


def test_eight_bit_angles_all_complex_measurement_branches(source_module: Any) -> None:
    cirq = pytest.importorskip("cirq")

    source = source_module.ProgrammableRotationGateArray(
        (3, 137), kappa=2, rotation_gate=cirq.Z
    )
    record = resolve_component_program(_library(source, 1), 0, _factories())
    compare_layers(record, source, atomic_leaves(record))
    bound = bind_register_program(record, ((0,), (1, 2), (3,)), clean=(4,))
    gates = bound.circuit.instructions
    count = sum(g.name == "measure" for g in gates)
    assert count == 8
    values = torch.arange(1, 17, dtype=torch.float64)
    amplitudes = torch.complex(values, values.flip(0))
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(32, dtype=torch.complex128)
    initial[::2] = amplitudes
    expected = torch.zeros_like(initial)
    expected[::2] = (
        torch.tensor(source.tensor_contract(), dtype=initial.dtype) @ amplitudes
    )
    for history in product((0, 1), repeat=count):
        branch = forced_branch(initial, gates, history)
        torch.testing.assert_close(
            branch, expected / 2 ** (count / 2), atol=1e-11, rtol=0
        )
        assert torch.count_nonzero(branch[1::2]) == 0


def test_shallow_source_loss_and_registered_owner_precedence(
    source_module: Any,
) -> None:
    from qualtran.serialization.bloq import bloqs_from_proto, bloqs_to_proto

    source = source_module._programmable_rotation_gate_array.make()
    with pytest.raises(TypeError, match="kappa.*rotation_gate"):
        bloqs_from_proto(bloqs_to_proto(source, max_depth=0))
    with pytest.raises(SerializationError, match="No native factory"):
        resolve_component_program(_library(source, 0), 0, _factories())
    library = _library(source, 1)
    root = library.entries[0]
    registered: dict[str, ComponentFactory] = {
        root.declaration.name: lambda name, registers, arguments: FunctionRecord(
            name, (), registers=registers
        ),
    }
    record = resolve_component_program(library, 0, registered)
    assert len(record.functions) == 1 and not record.functions[0].blocks
    with pytest.raises(SerializationError, match="No native factory"):
        resolve_component_program(library, 0, {})
    # Counts alone cannot substitute for executable connections.
    without_body = replace(root, connections=())
    with pytest.raises(SerializationError, match="No native factory"):
        resolve_component_program(
            replace(library, entries=(without_body, *library.entries[1:])),
            0,
            _factories(),
        )


def test_named_stored_graph_retains_real_constructor_metadata() -> None:
    from qualtran.bloqs.basic_gates import Hadamard, OnEach

    source = OnEach(2, Hadamard())
    library = _library(source, 1)
    assert library.entries[0].declaration.arguments
    # The stored body carries its executable leaf constants independently.
    record = resolve_component_program(library, 0, _factories())
    compare_layers(record, source, atomic_leaves(record))
    bound = bind_register_program(record, ((0, 1),))
    torch.testing.assert_close(
        logical_operator(bound.circuit, 2),
        torch.tensor(source.tensor_contract(), dtype=torch.complex128),
        atol=1e-11,
        rtol=0,
    )

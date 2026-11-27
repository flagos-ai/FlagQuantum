"""Live source graphs retain scalar occurrences omitted from serialized edges."""

from __future__ import annotations

import builtins
from collections import Counter
from importlib import import_module
from importlib.metadata import version
from itertools import product
from typing import Any

import pytest
import torch

from tests._reference_dependencies import unavailable_reference

try:
    import numpy as np
except ModuleNotFoundError as error:
    unavailable_reference(error)

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler.formats.legacy_component_library import (
    deserialize_component_library,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    _resolve_component_program,
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_rotation_factories import rotation_factories
from flagquantum.compiler.formats.legacy_scalar_factories import scalar_phase_factories
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program import ProgramRecord, ValueRef, verify_program_record
from flagquantum.ecosystem.cirq._register_graph import import_register_graph
from flagquantum.errors import SerializationError
from tests.unit.measurement_checks import forced_branch

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def locked_reference() -> None:
    kickmix = import_module("kickmix")
    assert version("qualtran") == "0.7.0"
    assert getattr(kickmix, "__version__") == "0.1"
    assert hasattr(kickmix, "Simulator")


def assert_live_calls(program: ProgramRecord, source: Any) -> None:
    """End opaque source Soquets at the test boundary and compare every operand."""
    from qualtran._infra.quantum_graph import LeftDangle, Soquet

    root = program.functions[0]
    block = root.blocks[0]
    definitions = {function.name: function for function in program.functions}
    source_inputs = [
        Soquet(LeftDangle, register, index)
        for register in source.signature.lefts()
        for index in register.all_idxs()
    ]
    values: dict[object, ValueRef] = dict(
        zip(source_inputs, block.arguments, strict=True)
    )
    original_calls = tuple(source.iter_bloqsoqs())
    assert len(block.operations) == len(original_calls)
    for operation, (instance, inputs, outputs) in zip(
        block.operations, original_calls, strict=True
    ):
        assert operation.opcode == "call"
        child = definitions[operation.callee or ""]
        assert [
            (r.name, r.bitsize, r.shape, r.side.value) for r in child.registers or ()
        ] == [
            (r.name, r.bitsize, r.shape, r.side.value) for r in instance.bloq.signature
        ]
        expected_inputs = tuple(
            values[soquet]
            for register in instance.bloq.signature.lefts()
            for soquet in np.asarray(inputs[register.name], dtype=object).flat
        )
        assert operation.operands == expected_inputs
        source_outputs = tuple(
            soquet for word in outputs for soquet in np.asarray(word, dtype=object).flat
        )
        values.update(zip(source_outputs, operation.results, strict=True))
    assert block.return_values == tuple(
        values[soquet]
        for register in source.signature.rights()
        for soquet in np.asarray(source.final_soqs()[register.name], dtype=object).flat
    )
    assert verify_program_record(program) == ()


@pytest.mark.parametrize("factory", ["_su2_rotation_gate", "_hadamard", "_t_gate"])
@pytest.mark.parametrize("adjoint", [False, True])
def test_complete_original_su2_factories_live_graph_and_absolute_phase(
    factory: str, adjoint: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from qualtran.bloqs.basic_gates import su2_rotation
    from qualtran.resource_counting._bloq_counts import QECGatesCost
    from qualtran.resource_counting._costing import get_cost_value

    source = getattr(su2_rotation, factory).make()
    if adjoint:
        source = source.adjoint()
    graph = source.decompose_bloq().flatten()
    program = import_register_graph(graph)
    assert_live_calls(program, graph)
    expected_matrix = torch.as_tensor(graph.tensor_contract(), dtype=torch.complex128)
    original_import = builtins.__import__

    def native_only(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".")[0] in {"qualtran", "cirq", "kickmix"}:
            raise AssertionError(f"Owned graph retained a reference dependency: {name}")
        return original_import(name, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(builtins, "__import__", native_only)
        bound = bind_register_program(program, ((0,),), clean=(2,))
        assert bound.outputs == ((0,),) and bound.classical_outputs == ()
        for dtype in (torch.complex64, torch.complex128):
            coefficients = torch.tensor([1 + 2j, -3j, 2 - 1j, -4 + 3j], dtype=dtype)
            coefficients /= coefficients.norm()
            initial = torch.zeros(8, dtype=dtype)
            initial[::2] = coefficients
            circuit = Circuit(3, inputs=initial, dtype=dtype)
            circuit.append(FieldOperation(3, bound.circuit.instructions))
            actual = circuit.state().flatten()
            expected = torch.zeros_like(initial)
            expected[::2] = (
                expected_matrix.to(dtype) @ coefficients.reshape(2, 2)
            ).flatten()
            torch.testing.assert_close(
                actual,
                expected,
                atol=3e-6 if dtype == torch.complex64 else 2e-13,
                rtol=0,
            )
            assert torch.count_nonzero(actual[1::2]) == 0
        assert CircuitIR.from_json(bound.circuit.to_json()) == bound.circuit
    histogram = Counter(g.name for g in bound.circuit.instructions)
    assert histogram["measure"] == 0
    assert all(event.kind in {"allocate", "free"} for event in bound.events)
    # Scalar analytical cost is zero, independently of its emitted carrier gates.
    for instance in graph.bloq_instances:
        if not tuple(instance.bloq.signature):
            assert get_cost_value(instance.bloq, QECGatesCost()).asdict() == {}


def test_repeated_scalar_occurrences_and_registerless_root_preserve_multiplicity() -> (
    None
):
    from qualtran._infra.composite_bloq import BloqBuilder
    from qualtran.bloqs.basic_gates.global_phase import GlobalPhase

    builder = BloqBuilder()
    for _ in range(3):
        builder.add(GlobalPhase(exponent=0.125))
    source = builder.finalize()
    program = import_register_graph(source)
    assert_live_calls(program, source)
    assert len(program.functions[0].blocks[0].operations) == 3
    bound = bind_register_program(program, (), clean=(0,))
    actual = (
        Circuit(1, dtype=torch.complex128)
        .append(FieldOperation(1, bound.circuit.instructions))
        .state()
    )
    expected = complex(source.tensor_contract())
    torch.testing.assert_close(
        actual.flatten(),
        torch.tensor([expected, 0], dtype=actual.dtype),
        atol=2e-13,
        rtol=0,
    )
    assert Counter(e.kind for e in bound.events) == {"allocate": 3, "free": 3}


def test_scalar_after_measured_arithmetic_preserves_every_complex_branch() -> None:
    from qualtran._infra.composite_bloq import BloqBuilder
    from qualtran._infra.data_types import QUInt
    from qualtran.bloqs.arithmetic.addition import Add
    from qualtran.bloqs.basic_gates.global_phase import GlobalPhase

    adder = Add(QUInt(2))
    builder, inputs = BloqBuilder.from_signature(adder.signature)
    outputs = builder.add_d(adder, **inputs)
    builder.add(GlobalPhase(exponent=0.375))
    source = builder.finalize(**outputs).flatten()
    program = import_register_graph(source)
    assert_live_calls(program, source)
    bound = bind_register_program(program, ((0, 1), (2, 3)), clean=(5, 6, 7, 8))
    initial = torch.zeros(512, dtype=torch.complex128)
    logical = torch.complex(
        torch.arange(1, 33).double(), torch.arange(32, 0, -1).double()
    )
    logical /= logical.norm()
    initial[::16] = logical
    original = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    expected[::16] = (original @ logical.reshape(16, 2)).flatten()
    count = sum(g.name == "measure" for g in bound.circuit.instructions)
    assert 0 < count <= 4
    for history in product((0, 1), repeat=count):
        actual = forced_branch(initial, bound.circuit.instructions, history)
        torch.testing.assert_close(
            actual, expected / 2 ** (count / 2), atol=3e-13, rtol=0
        )
        assert torch.count_nonzero(actual.reshape(32, 16)[:, 1:]) == 0
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic

    circuit = DynamicCircuit(9, inputs=initial, dtype=initial.dtype)
    circuit.append(FieldOperation(9, bound.circuit.instructions))
    executed = run_dynamic(circuit, shots=4, strategy="trajectory", seed=1084)
    for state in executed.final_states:
        torch.testing.assert_close(state.flatten(), expected, atol=3e-13, rtol=0)


def test_lossy_wire_still_refuses_order_inference_and_live_order_is_validated() -> None:
    from qualtran.bloqs.basic_gates.su2_rotation import _hadamard
    from qualtran.serialization.bloq import bloqs_to_proto

    source = _hadamard.make().decompose_bloq().flatten()
    wire = bloqs_to_proto(source, max_depth=1, pred=lambda _: False).SerializeToString()
    library = deserialize_component_library(wire)
    factories = {**rotation_factories(), **scalar_phase_factories()}
    with pytest.raises(SerializationError, match="zero-port execution child"):
        resolve_component_program(library, 0, factories)
    from flagquantum.compiler.formats.legacy_component_discovery import (
        _ordered_occurrences,
    )

    connected = tuple(_ordered_occurrences(library.entries[0].connections))
    with pytest.raises(SerializationError, match="omits a connected instance"):
        _resolve_component_program(library, 0, factories, execution_orders={0: ()})
    with pytest.raises(SerializationError, match="repeats an instance"):
        _resolve_component_program(
            library, 0, factories, execution_orders={0: (*connected, connected[0])}
        )

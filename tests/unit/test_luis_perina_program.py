"""Stored LP constructors execute native composition without retained SDK objects."""

import sys
from collections import Counter
from math import sqrt

import pytest
import torch

from tests._reference_dependencies import unavailable_reference

try:
    import kickmix
    import numpy as np
    import sympy
    from qualtran.bloqs.phase_estimation.lp_resource_state import (
        LPResourceState,
        LPRSInterimPrep,
    )
except ModuleNotFoundError as error:
    unavailable_reference(error)

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.algorithms.primitives.luis_perina_state import (
    luis_perina_state_program,
)
from flagquantum.compiler.formats.legacy_component_library import (
    deserialize_component_library,
    serialize_component_library,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program import ValueRef, verify_program_record
from flagquantum.ecosystem._phase_window_library import phase_window_factories
from flagquantum.ecosystem.cirq._register_graph import import_register_graph
from flagquantum.errors import SerializationError
from tests.unit.measurement_certificate import certify_erasure_histories
from tests.unit.measurement_checks import forced_branch
from tests.unit.sparse_register_execution import execute_sparse_register
from tests.unit.test_legacy_arithmetic_factories import _library

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("width", (1, 2, 3, 5, 7))
@pytest.mark.parametrize("interim", (False, True))
def test_complete_stored_constructors_complex_state_and_owned_outputs(
    width: int, interim: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert getattr(kickmix, "__version__") == "0.1" and hasattr(kickmix, "Simulator")
    source = LPRSInterimPrep(width, eps=2e-9) if interim else LPResourceState(width)
    expected = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
    library = _library(source, 0)
    encoded = serialize_component_library(library)
    library = deserialize_component_library(encoded)
    assert serialize_component_library(library) == encoded
    # Conversion ends before resolution, binding and the ordinary append entry.
    for name in ("qualtran", "cirq", "kickmix"):
        monkeypatch.setitem(sys.modules, name, None)
    program = resolve_component_program(library, 0, phase_window_factories())
    assert verify_program_record(program) == ()
    clean = (
        tuple(range(width + 2, 3 * width + 15))
        if interim
        else tuple(range(1, 3 * width + 13))
    )
    inputs = (tuple(range(1, width + 1)), (width + 1,)) if interim else ()
    bound = bind_register_program(
        program, inputs, clean=clean, allow_workspace_outputs=not interim
    )
    assert bound.outputs == inputs if interim else len(bound.outputs) == 1
    ir = (
        Circuit(bound.circuit.n_wires)
        .append(FieldOperation(bound.circuit.n_wires, bound.circuit.instructions))
        .to_ir()
    )
    ir = CircuitIR.from_json(ir.to_json())
    if interim:
        assert not bound.events and not any(
            g.name == "measure" for g in ir.instructions
        )
        values = torch.randn(
            1 << (width + 2),
            dtype=torch.complex128,
            generator=torch.Generator().manual_seed(104 + width),
        )
        values /= values.norm()
        # The large reserved pool is idle. Execute the actual logical operands.
        actual = forced_branch(values, ir.instructions, ())
        operator = torch.kron(torch.eye(2, dtype=values.dtype), expected)
        torch.testing.assert_close(actual, operator @ values, rtol=0, atol=3e-12)
        return
    total = ir.n_wires
    events = certify_erasure_histories(
        ir.instructions, n_wires=total, zero_wires=clean, check_clean_output=False
    )
    assert events == width + 1
    assert Counter(g.name for g in ir.instructions)["t"] + Counter(
        g.name for g in ir.instructions
    )["tdg"] == 4 * (width + 1)
    spectator = (sqrt(0.3), 1j * sqrt(0.7))
    actual_sparse = execute_sparse_register(
        ir, {0: spectator[0], 1 << (total - 1): spectator[1]}, seed=20261127 + width
    )
    target = {}
    outputs = bound.outputs[0]
    for external, coefficient in enumerate(spectator):
        for word, amplitude in enumerate(np.asarray(expected).reshape(-1)):
            address = external << (total - 1)
            address |= sum(
                ((word >> (width - bit - 1)) & 1) << (total - wire - 1)
                for bit, wire in enumerate(outputs)
            )
            target[address] = complex(coefficient * amplitude)
    assert (
        max(
            abs(actual_sparse.get(k, 0) - target.get(k, 0))
            for k in actual_sparse.keys() | target.keys()
        )
        < 3e-12
    )
    work_mask = ((1 << (total - 1)) - 1) ^ sum(
        1 << (total - wire - 1) for wire in outputs
    )
    assert sum(abs(a) ** 2 for k, a in actual_sparse.items() if k & work_mask) < 1e-27


@pytest.mark.parametrize("interim", (False, True))
def test_symbolic_source_declarations_are_retained_without_claiming_execution(
    interim: bool,
) -> None:
    width = sympy.Symbol("n", positive=True, integer=True)
    source = LPRSInterimPrep(width) if interim else LPResourceState(width)
    library = _library(source, 0)
    program = resolve_component_program(library, 0, phase_window_factories())
    assert program.functions[0].blocks == ()
    assert program.functions[0].registers is not None
    ports = library.entries[0].declaration.registers
    assert ports is not None
    # The existing source symbol wire omits assumptions; preserve its value.
    assert program.functions[0].registers[0].bitsize == ports[0].bitsize
    assert str(ports[0].bitsize) == str(width)


@pytest.mark.parametrize("width", (1, 3, 7))
def test_complete_source_root_calls_operands_and_primitive_inventories(
    width: int,
) -> None:
    source = LPResourceState(width)
    graph = source.decompose_bloq()
    program = luis_perina_state_program(width)
    root = program.functions[0]
    definitions = {f.name: f for f in program.functions}
    calls = tuple(sorted(graph.iter_bloqsoqs(), key=lambda item: item[0].i))
    operations = root.blocks[0].operations
    assert len(operations) == len(calls) == 15
    values: dict[object, ValueRef] = {}
    for operation, (instance, inputs, outputs) in zip(operations, calls, strict=True):
        child = definitions[operation.callee or ""]
        assert [
            (r.name, r.bitsize, r.shape, r.side.value) for r in child.registers or ()
        ] == [
            (r.name, r.bitsize, r.shape, r.side.value) for r in instance.bloq.signature
        ]
        assert operation.operands == tuple(
            values[soquet]
            for reg in instance.bloq.signature.lefts()
            for soquet in np.asarray(inputs[reg.name], dtype=object).flat
        )
        original_outputs = tuple(
            soquet for word in outputs for soquet in np.asarray(word, dtype=object).flat
        )
        values.update(zip(original_outputs, operation.results, strict=True))
    assert root.blocks[0].return_values == tuple(
        values[soquet]
        for reg in graph.signature.rights()
        for soquet in np.asarray(graph.final_soqs()[reg.name], dtype=object).flat
    )
    clean = tuple(range(1, 3 * width + 13))
    bound = bind_register_program(
        program, (), clean=clean, allow_workspace_outputs=True
    )
    live = bind_register_program(
        import_register_graph(source), (), clean=clean, allow_workspace_outputs=True
    )
    actual = Counter(g.name for g in bound.circuit.instructions)
    original = Counter(g.name for g in live.circuit.instructions)
    # Atomic Identity(width) has no source primitive recipe. Native expands it
    # to per-bit I; Cirq removes wide identity while retaining one-bit I.
    assert actual["i"] == 2 * (width + 2)
    assert original["i"] == (6 if width == 1 else 4)
    assert {k: v for k, v in actual.items() if k != "i"} == {
        k: v for k, v in original.items() if k != "i"
    }


def test_constructor_fields_and_signature_remain_checked() -> None:
    factories = phase_window_factories()
    prefix = "qualtran.bloqs.phase_estimation.lp_resource_state."
    for kind in ("LPRSInterimPrep", "LPResourceState"):
        with pytest.raises(SerializationError, match="constructor arguments"):
            factories[prefix + kind]("main", None, {"bitsize": 3, "extra": 1})
        with pytest.raises(SerializationError, match="register signature"):
            factories[prefix + kind]("main", (), {"bitsize": 3})
    for width in (0, -1, True):
        with pytest.raises(ValueError, match="positive width"):
            luis_perina_state_program(width)


@pytest.mark.parametrize("interim", (False, True))
def test_shallow_and_expanded_source_libraries_use_the_same_complete_owner(
    interim: bool,
) -> None:
    source = LPRSInterimPrep(3) if interim else LPResourceState(3)
    records = [
        resolve_component_program(_library(source, depth), 0, phase_window_factories())
        for depth in (0, 1)
    ]
    assert records[0] == records[1]

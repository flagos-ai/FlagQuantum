"""Typed unary lifetimes and generic adjoints match the frozen source graph."""

from collections import Counter
from dataclasses import replace
from itertools import product
from typing import Any

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.algorithms.primitives.circuit_register_emitter import (
    CircuitRegisterEmitter,
)
from flagquantum.compiler.registers.component_adjoint import adjoint_component_hierarchy
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_gate_function import bit_gate_function
from flagquantum.compiler.registers.control.conjunction_program import (
    and_component_program,
)
from flagquantum.compiler.registers.control.unary_component_calls import (
    append_bounded_unary_calls,
)
from flagquantum.compiler.registers.layout.component_workspace import (
    allocate_word_function,
    free_word_function,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.register_bindings import RegisterBinding
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_relative_controlled_swap import assert_source_dependencies
from tests.unit.test_subtraction_foreign_file import native_signature
from tests.unit.test_unary_iteration_allocating_child import (
    allocating_source as allocating_source,
)
from tests.unit.test_unary_iteration_file import source as source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
np = pytest.importorskip("numpy")


def private_leaf_program() -> ProgramRecord:
    """Compose the same child through native calls and explicit private storage."""
    bit = quantum_bit_type()
    primitives = tuple(
        bit_gate_function(
            name,
            (Register("bits", bit, (2,)),),
            CircuitIR(2, (Instruction(name, (0, 1)),)),
        )
        for name in ("cx", "cz")
    )
    builder = ComponentProgramBuilder(
        Signature.build(control=1, target=1, phase=1),
        (
            *primitives,
            allocate_word_function("allocate", bit, False),
            free_word_function("free", bit, False),
        ),
        name="leaf",
    )
    control, target, phase = (
        builder.inputs[name] for name in ("control", "target", "phase")
    )
    work = builder.call_dict("allocate", {})["reg"]
    control, work = np.asarray(
        builder.call_dict("cx", {"bits": np.array([control, work], dtype=object)})[
            "bits"
        ]
    )
    work, phase = np.asarray(
        builder.call_dict("cz", {"bits": np.array([work, phase], dtype=object)})["bits"]
    )
    work, target = np.asarray(
        builder.call_dict("cx", {"bits": np.array([work, target], dtype=object)})[
            "bits"
        ]
    )
    control, work = np.asarray(
        builder.call_dict("cx", {"bits": np.array([control, work], dtype=object)})[
            "bits"
        ]
    )
    builder.call_dict("free", {"reg": work})
    return builder.finalize({"control": control, "target": target, "phase": phase})


def typed_lookup(signature: Signature, extent: int = 3) -> ProgramRecord:
    builder = ComponentProgramBuilder(signature)
    leaf = private_leaf_program()
    target = builder.inputs["target"]
    phase = np.asarray(builder.inputs["phase"]).copy()
    selection = builder.inputs["selection"]
    assert isinstance(selection, RegisterBinding)
    control = builder.inputs.get("enable")
    assert control is None or isinstance(control, RegisterBinding)

    def emit(index: int, predicate: RegisterBinding) -> RegisterBinding:
        nonlocal target
        output = builder.call_program(
            leaf,
            {"control": predicate, "target": target, "phase": phase[index % 2]},
            entry="leaf",
        )
        target = output["target"]
        phase[index % 2] = output["phase"]
        successor = output["control"]
        assert isinstance(successor, RegisterBinding)
        return successor

    bits, control = append_bounded_unary_calls(
        builder, tuple(builder.split(selection)), extent, emit, control=control
    )
    outputs: dict[str, object] = {
        "selection": builder.join(bits, selection.register.dtype),
        "target": target,
        "phase": phase,
    }
    if control is not None:
        outputs["enable"] = control
    return builder.finalize(outputs)


def typed_inverse(record: ProgramRecord) -> ProgramRecord:
    inverse_definitions = tuple(
        function
        for values, erase in (((1, 0), True), ((1, 1), False))
        for function in and_component_program(
            values,
            name=f"__fq_unary_and_{values[0]}{values[1]}" + ("_dg" if erase else ""),
            uncompute=erase,
        ).functions
    )
    record = replace(record, functions=(*record.functions, *inverse_definitions))
    adjoints = {
        "allocate": "free",
        "free": "allocate",
        "cx": "cx",
        "cz": "cz",
        "__fq_unary_x": "__fq_unary_x",
        "__fq_unary_cx": "__fq_unary_cx",
        "__fq_unary_and_10": "__fq_unary_and_10_dg",
        "__fq_unary_and_11_dg": "__fq_unary_and_11",
    }
    # Split and Join preserve the original bounded selector descriptor.
    views = [
        function
        for function in record.functions
        if function.name.startswith("__fq_rebinding_")
    ]
    assert len(views) == 2
    adjoints.update({views[0].name: views[1].name, views[1].name: views[0].name})
    return adjoint_component_hierarchy(record, adjoints)


def source_slot_names(
    gates: tuple[Instruction, ...], work: tuple[int, ...]
) -> tuple[Instruction, ...]:
    """Check feedback lifetimes before replacing fresh SSA slots by source keys."""
    slots: dict[int, int] = {}
    latest: dict[int, int] = {}
    renamed = []
    for gate in gates:
        metadata = dict(gate.metadata)
        if gate.name == "measure":
            slot = metadata["classical_bit"]
            wire = gate.wires[0]
            slots[slot] = work.index(wire)
            latest[wire] = slot
            metadata["classical_bit"] = slots[slot]
        if "conditions" in metadata:
            for slot, _ in metadata["conditions"]:
                assert latest[work[slots[slot]]] == slot
            metadata["conditions"] = tuple(
                (slots[slot], value) for slot, value in metadata["conditions"]
            )
        renamed.append(replace(gate, metadata=metadata))
    return tuple(renamed)


@pytest.mark.parametrize("controlled", (False, True))
def test_typed_calls_and_generic_inverse_full_graph_operands_and_channels(
    allocating_source: Any, controlled: bool
) -> None:
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    original = allocating_source(controlled)
    signature = Signature(native_signature(original.signature))
    forward = typed_lookup(signature)
    records = (forward, typed_inverse(forward))
    logical = int(controlled) + 5
    work = tuple(range(logical, logical + 2 + int(controlled)))
    ports = {}
    inputs = []
    offset = 0
    for register in signature:
        start = offset
        for _ in register.all_idxs():
            width = int(register.bitsize)
            inputs.append(tuple(range(offset, offset + width)))
            offset += width
        ports[register.name] = tuple(range(start, offset))
    circuits = []
    for adjoint, record in enumerate(records):
        bound = bind_register_program(record, tuple(inputs), clean=work)
        assert bound.outputs == tuple(inputs)
        direction = original.adjoint() if adjoint else original
        # Both references use the original typed CompositeBloq graph route.
        # The direct forward generator reserves its whole work pool earlier.
        reference = reference_arithmetic_gates(
            direction,
            ports,
            work,
            dict(zip(work, range(len(work)), strict=True)),
            cirq_style=False,
            keep_pauli_leaves=True,
        )
        assert_source_dependencies(
            source_slot_names(bound.circuit.instructions, work), reference
        )
        counts = Counter(gate.name for gate in bound.circuit.instructions)
        cost = get_cost_value(direction, QECGatesCost())
        assert counts["measure"] == cost.measurement
        assert counts["t"] + counts["tdg"] == cost.total_t_count()
        assert get_cost_value(direction, QubitCount()) == logical + len(work)
        assert len(
            [event for event in bound.events if event.kind == "allocate"]
        ) == len([event for event in bound.events if event.kind == "free"])
        appended = (
            Circuit(logical + len(work))
            .append(FieldOperation(logical + len(work), bound.circuit.instructions))
            .to_ir()
        )
        assert CircuitIR.from_json(appended.to_json()) == appended
        for basis in range(1 << logical):
            state = torch.zeros(1 << (logical + len(work)), dtype=torch.complex128)
            state[basis << len(work)] = torch.tensor(0.6 + 0.8j, dtype=state.dtype)
            for history in product((0, 1), repeat=counts["measure"]):
                actual = forced_branch(state, appended.instructions, history)
                expected = forced_branch(state, reference, history)
                torch.testing.assert_close(actual, expected, atol=2e-12, rtol=0)
                assert (
                    torch.count_nonzero(actual.reshape(-1, 1 << len(work))[:, 1:]) == 0
                )
        circuits.append(appended)
    # Replace the actual foreign snapshot provider with native Program circuits
    # behind the same named-register emitter consumer.
    native_emitter = CircuitRegisterEmitter(
        tuple((reg.name, int(reg.total_bits())) for reg in signature), tuple(circuits)
    )
    foreign_emitter = import_register_emitter(original)
    assert native_emitter.requirements() == foreign_emitter.requirements()
    for adjoint in (False, True):
        emitted = native_emitter(
            ports, work, tuple(range(native_emitter.requirements()[1])), adjoint
        )
        assert emitted == circuits[int(adjoint)].instructions


@pytest.mark.parametrize(
    "lower,extent,fold,visited",
    ((1, 3, False, (1, 2)), (0, 4, True, (0, 2)), (3, 3, False, ())),
)
def test_interval_fold_and_empty_calls_preserve_coherent_inputs(
    lower: int, extent: int, fold: bool, visited: tuple[int, ...]
) -> None:
    bit = quantum_bit_type()
    signature = Signature(
        (
            Register("selection", bit, (2,)),
            Register("target", bit),
            Register("enable", bit),
        )
    )
    builder = ComponentProgramBuilder(signature)
    leaf = ProgramRecord(
        (
            bit_gate_function(
                "toggle",
                (Register("bits", bit, (2,)),),
                CircuitIR(2, (Instruction("cx", (0, 1)),)),
            ),
        )
    )
    target = builder.inputs["target"]
    calls = []

    def emit(index: int, predicate: RegisterBinding) -> RegisterBinding:
        nonlocal target
        calls.append(index)
        result = np.asarray(
            builder.call_program(
                leaf,
                {"bits": np.array([predicate, target], dtype=object)},
                entry="toggle",
            )["bits"]
        )
        target = result[1]
        successor = result[0]
        assert isinstance(successor, RegisterBinding)
        return successor

    enable = builder.inputs["enable"]
    assert isinstance(enable, RegisterBinding)
    selection, returned_enable = append_bounded_unary_calls(
        builder,
        tuple(np.asarray(builder.inputs["selection"]).flat),
        extent,
        emit,
        control=enable,
        lower=lower,
        break_early=(lambda left, right: right - left == 2) if fold else None,
    )
    assert tuple(calls) == visited
    record = builder.finalize(
        {
            "selection": np.array(selection, dtype=object),
            "target": target,
            "enable": returned_enable,
        }
    )
    bound = bind_register_program(record, ((0,), (1,), (2,), (3,)), clean=(4, 5))
    events = sum(g.name == "measure" for g in bound.circuit.instructions)
    initial = torch.zeros(64, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    for basis in range(16):
        address = basis >> 2
        if visited and not lower <= address < extent:
            continue
        amplitude = torch.tensor(complex(basis + 1, 2 - basis), dtype=initial.dtype)
        initial[basis << 2] = amplitude
        toggle = bool(visited) and bool(basis & 1)
        expected[(basis ^ (2 if toggle else 0)) << 2] = amplitude
    expected /= initial.norm()
    initial /= initial.norm()
    for history in product((0, 1), repeat=events):
        actual = forced_branch(initial, bound.circuit.instructions, history)
        torch.testing.assert_close(
            actual, expected / 2 ** (events / 2), atol=2e-12, rtol=0
        )
        assert torch.count_nonzero(actual.reshape(-1, 4)[:, 1:]) == 0


def test_invalid_interval_refuses_before_consuming_bindings() -> None:
    signature = Signature((Register("selection", quantum_bit_type(), (2,)),))
    for lower, extent in ((-1, 2), (0, 5)):
        builder = ComponentProgramBuilder(signature)
        with pytest.raises(ValueError, match="fit the selection"):
            append_bounded_unary_calls(
                builder,
                tuple(np.asarray(builder.inputs["selection"]).flat),
                extent,
                lambda *_: pytest.fail("Invalid interval invoked a leaf"),
                lower=lower,
            )
        record = builder.finalize(builder.inputs)
        assert record.functions[0].blocks[0].operations == ()


@pytest.mark.parametrize("controlled", (False, True))
@pytest.mark.parametrize("width,extent", ((1, 2), (3, 5), (3, 8)))
def test_deeper_private_child_graphs_keep_exact_source_allocation_stream(
    allocating_source: Any, controlled: bool, width: int, extent: int
) -> None:
    from attrs import frozen
    from qualtran import BQUInt
    from qualtran import Register as SourceRegister
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    @frozen
    class SizedSelect(allocating_source):  # type: ignore[misc]
        controlled: bool

        @property
        def selection_registers(self) -> tuple[Any, ...]:
            return (SourceRegister("selection", BQUInt(width, extent)),)

    original = SizedSelect(controlled)
    signature = Signature(native_signature(original.signature))
    forward = typed_lookup(signature, extent)
    offset = 0
    inputs = []
    ports = {}
    for register in signature:
        start = offset
        for _ in register.all_idxs():
            size = int(register.bitsize)
            inputs.append(tuple(range(offset, offset + size)))
            offset += size
        ports[register.name] = tuple(range(start, offset))
    work = tuple(range(offset, offset + width + int(controlled)))
    for adjoint, record in enumerate((forward, typed_inverse(forward))):
        bound = bind_register_program(record, tuple(inputs), clean=work)
        direction = original.adjoint() if adjoint else original
        reference = reference_arithmetic_gates(
            direction,
            ports,
            work,
            dict(zip(work, range(len(work)), strict=True)),
            cirq_style=False,
            keep_pauli_leaves=True,
        )
        assert_source_dependencies(
            source_slot_names(bound.circuit.instructions, work), reference
        )
        assert bound.outputs == tuple(inputs)
        counts = Counter(gate.name for gate in bound.circuit.instructions)
        model = get_cost_value(direction, QECGatesCost())
        assert counts["t"] + counts["tdg"] == model.total_t_count()
        assert counts["measure"] == model.measurement
        assert offset + len(work) == get_cost_value(direction, QubitCount())


def test_native_construction_adjoint_and_append_execute_with_sdk_imports_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    from flagquantum.core.ir.program.register_types import (
        bounded_unsigned_register_type,
    )

    original_import = builtins.__import__

    def native_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".")[0] in {"qualtran", "cirq", "kickmix"}:
            raise AssertionError(f"Native unary construction imported {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", native_import)
    bit = quantum_bit_type()
    signature = Signature(
        (
            Register("selection", bounded_unsigned_register_type(2, 3)),
            Register("target", bit),
            Register("phase", bit, (2,)),
            Register("enable", bit),
        )
    )
    forward = typed_lookup(signature)
    inverse = typed_inverse(forward)
    for record in (forward, inverse):
        bound = bind_register_program(
            record, ((0, 1), (2,), (3,), (4,), (5,)), clean=(6, 7, 8)
        )
        circuit = (
            Circuit(9).append(FieldOperation(9, bound.circuit.instructions)).to_ir()
        )
        assert len(circuit.instructions) == 52
        initial = torch.zeros(512, dtype=torch.complex128)
        initial[40] = 1
        expected = torch.zeros_like(initial)
        expected[104] = -0.5
        actual = forced_branch(initial, circuit.instructions, (0, 0))
        torch.testing.assert_close(actual, expected, atol=2e-12, rtol=0)

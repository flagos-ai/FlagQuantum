"""Unary child borrowing restores the joint dirty/spectator quantum state."""

from collections import Counter
from itertools import product
from typing import Any

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_gate_function import bit_gate_function
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
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_relative_controlled_swap import assert_source_dependencies
from tests.unit.test_subtraction_foreign_file import native_signature
from tests.unit.test_unary_component_calls import source_slot_names, typed_inverse
from tests.unit.test_unary_iteration_file import source as source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
np = pytest.importorskip("numpy")


@pytest.fixture(scope="module")
def borrowing_source(source: Any) -> Any:
    cirq = pytest.importorskip("cirq")
    from attrs import frozen
    from qualtran import BQUInt, GateWithRegisters, QBit, Register, Signature
    from qualtran.bloqs.basic_gates import CNOT
    from qualtran.resource_counting import QubitCount

    @frozen
    class BorrowedEcho(GateWithRegisters):  # type: ignore[misc]
        @property
        def signature(self) -> Any:
            return Signature.build(control=1, target=1)

        def decompose_from_registers(self, *, context: Any, **quregs: Any) -> Any:
            control, target = (quregs[name][0] for name in ("control", "target"))
            work = context.qubit_manager.qborrow(1)
            yield cirq.CNOT(work[0], target)
            yield cirq.CNOT(control, work[0])
            yield cirq.CNOT(work[0], target)
            yield cirq.CNOT(control, work[0])
            context.qubit_manager.qfree(work)

        def build_composite_bloq(
            self, bb: Any, control: Any, target: Any
        ) -> dict[str, Any]:
            from qualtran.bloqs.bookkeeping import Allocate, Free

            work = bb.add(Allocate(QBit(), dirty=True))
            work, target = bb.add(CNOT(), ctrl=work, target=target)
            control, work = bb.add(CNOT(), ctrl=control, target=work)
            work, target = bb.add(CNOT(), ctrl=work, target=target)
            control, work = bb.add(CNOT(), ctrl=control, target=work)
            bb.add(Free(QBit(), dirty=True), reg=work)
            return {"control": control, "target": target}

        def build_call_graph(self, ssa: Any) -> dict[Any, int]:
            return {CNOT(): 4}

        def my_static_costs(self, cost_key: Any) -> Any:
            return 3 if isinstance(cost_key, QubitCount) else NotImplemented

    @frozen
    class BorrowingSelect(source.UnaryIterationGate):  # type: ignore[misc]
        controlled: bool

        @property
        def control_registers(self) -> tuple[Any, ...]:
            return (Register("enable", QBit()),) if self.controlled else ()

        @property
        def selection_registers(self) -> tuple[Any, ...]:
            return (Register("selection", BQUInt(2, 3)),)

        @property
        def target_registers(self) -> tuple[Any, ...]:
            return (Register("target", QBit()),)

        def nth_operation(
            self, context: Any, control: Any, selection: int, target: Any
        ) -> Any:
            return BorrowedEcho().on_registers(control=control, target=target)

        def nth_operation_callgraph(self, **values: int) -> set[tuple[Any, int]]:
            return {(BorrowedEcho(), 1)}

    return BorrowingSelect


def borrowing_program(signature: Signature) -> ProgramRecord:
    bit = quantum_bit_type()
    leaf_builder = ComponentProgramBuilder(
        Signature.build(control=1, target=1),
        (
            allocate_word_function("allocate", bit, True),
            free_word_function("free", bit, True),
            bit_gate_function(
                "cx",
                (Register("bits", bit, (2,)),),
                CircuitIR(2, (Instruction("cx", (0, 1)),)),
            ),
        ),
        name="borrowed_echo",
    )
    values = dict(leaf_builder.inputs)
    values["work"] = leaf_builder.call_dict("allocate", {})["reg"]
    for left, right in (("work", "target"), ("control", "work")) * 2:
        gate_outputs = np.asarray(
            leaf_builder.call_dict(
                "cx", {"bits": np.array([values[left], values[right]], dtype=object)}
            )["bits"]
        )
        values[left], values[right] = gate_outputs[0], gate_outputs[1]
    leaf_builder.call_dict("free", {"reg": values.pop("work")})
    leaf = leaf_builder.finalize(values)
    builder = ComponentProgramBuilder(signature)
    target = builder.inputs["target"]
    selection = builder.inputs["selection"]
    control = builder.inputs.get("enable")
    assert isinstance(selection, RegisterBinding)
    assert control is None or isinstance(control, RegisterBinding)

    def emit(index: int, predicate: RegisterBinding) -> RegisterBinding:
        nonlocal target
        outputs = builder.call_program(
            leaf, {"control": predicate, "target": target}, entry="borrowed_echo"
        )
        target = outputs["target"]
        successor = outputs["control"]
        assert isinstance(successor, RegisterBinding)
        return successor

    bits, control = append_bounded_unary_calls(
        builder, tuple(builder.split(selection)), 3, emit, control=control
    )
    outputs: dict[str, object] = {
        "selection": builder.join(bits, selection.register.dtype),
        "target": target,
    }
    if control is not None:
        outputs["enable"] = control
    return builder.finalize(outputs)


@pytest.mark.parametrize("controlled", (False, True))
def test_dirty_child_exact_graph_and_joint_state_restoration(
    borrowing_source: Any, controlled: bool
) -> None:
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    original = borrowing_source(controlled)
    signature = Signature(native_signature(original.signature))
    forward = borrowing_program(signature)
    logical = 3 + int(controlled)
    clean_count = 1 + int(controlled)
    clean = tuple(range(logical, logical + clean_count))
    dirty = logical + clean_count
    spectator = dirty + 1
    width = spectator + 1
    inputs = []
    ports = {}
    offset = 0
    for register in signature:
        wires = tuple(range(offset, offset + int(register.bitsize)))
        inputs.append(wires)
        ports[register.name] = wires
        offset += int(register.bitsize)
    for adjoint, record in enumerate((forward, typed_inverse(forward))):
        bound = bind_register_program(
            record, tuple(inputs), clean=clean, borrowed=(dirty,)
        )
        assert bound.outputs == tuple(inputs)
        direction = original.adjoint() if adjoint else original
        reference = reference_arithmetic_gates(
            direction,
            ports,
            clean,
            dict(zip(clean, range(clean_count), strict=True)),
            cirq_style=False,
            keep_pauli_leaves=True,
            borrowed_allocations=(dirty,),
        )
        assert_source_dependencies(
            source_slot_names(bound.circuit.instructions, clean), reference
        )
        counts = Counter(g.name for g in bound.circuit.instructions)
        model = get_cost_value(direction, QECGatesCost())
        assert counts["measure"] == model.measurement == clean_count
        assert counts["t"] + counts["tdg"] == model.total_t_count()
        assert get_cost_value(direction, QubitCount()) == logical + clean_count + 1
        events = Counter((event.kind, event.dirty) for event in bound.events)
        assert events[("allocate", True)] == events[("free", True)] == 3
        assert events[("allocate", False)] == events[("free", False)]
        circuit = Circuit(width)
        circuit.append(FieldOperation(width, bound.circuit.instructions))
        gates = circuit.to_ir().instructions
        coherent = torch.zeros(1 << width, dtype=torch.complex128)
        for basis in range(1 << (logical + 2)):
            logical_basis, environment = divmod(basis, 4)
            index = (logical_basis << (clean_count + 2)) | environment
            amplitude = complex(basis + 1, 2 - basis)
            state = torch.zeros_like(coherent)
            state[index] = torch.tensor(amplitude, dtype=state.dtype)
            coherent += state
            states: tuple[torch.Tensor, ...] = (state,)
            if basis == (1 << (logical + 2)) - 1:
                states += (coherent / coherent.norm(),)
            for initial in states:
                # Echo implements the selected CX while preserving every joint
                # borrowed/spectator amplitude, including off-diagonal phases.
                expected = torch.zeros_like(initial)
                for item in range(1 << logical):
                    enabled = not controlled or bool(item >> 3)
                    # Source pruning aliases the padded address to its final
                    # retained leaf. Every leaf has the same echo in this fixture.
                    output = item ^ int(enabled)
                    start = item << (clean_count + 2)
                    end = output << (clean_count + 2)
                    expected[end : end + 4] = initial[start : start + 4]
                for history in product((0, 1), repeat=counts["measure"]):
                    actual = forced_branch(initial, gates, history)
                    upstream = forced_branch(initial, reference, history)
                    torch.testing.assert_close(actual, upstream, atol=2e-12, rtol=0)
                    torch.testing.assert_close(
                        actual, expected / 2 ** (clean_count / 2), atol=2e-12, rtol=0
                    )
                    assert (
                        torch.count_nonzero(
                            actual.reshape(-1, 1 << clean_count, 4)[:, 1:, :]
                        )
                        == 0
                    )


def test_source_cirq_conversion_loses_borrowed_declaration(
    borrowing_source: Any,
) -> None:
    from qualtran.bloqs.bookkeeping import Allocate
    from qualtran.cirq_interop._cirq_to_bloq import decompose_from_cirq_style_method

    composite = borrowing_source(False).decompose_bloq()
    leaf = next(
        instance.bloq
        for instance in composite.bloq_instances
        if type(instance.bloq).__name__ == "BorrowedEcho"
    )
    typed = leaf.decompose_bloq()
    converted = decompose_from_cirq_style_method(leaf)
    assert [
        instance.bloq.dirty
        for instance in typed.bloq_instances
        if isinstance(instance.bloq, Allocate)
    ] == [True]
    # The unchanged Cirq-to-Bloq converter infers all introduced wires as clean.
    # Exact dirty lifetime comparisons therefore use the explicit typed graph.
    assert [
        instance.bloq.dirty
        for instance in converted.bloq_instances
        if isinstance(instance.bloq, Allocate)
    ] == [False]

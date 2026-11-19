"""Complete source/native register export with live workspace output transfer."""

from typing import Any

import numpy as np
import pytest

from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_gate_function import bit_gate_function
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.register_bindings import RegisterBinding
from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.ecosystem.cirq._register_program_export import (
    export_bound_register_cirq,
)
from tests.test_cirq_register_export import source_export as source_export

pytestmark = [
    pytest.mark.unit,
    pytest.mark.ftqc_reference,
    pytest.mark.usefixtures("source_export"),
]
pytest.importorskip("numpy")
pytest.importorskip("cirq")
pytest.importorskip("qualtran")


def workspace_fixture(dirty: bool) -> tuple[ProgramRecord, Any]:
    from qualtran import BloqBuilder, QBit
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature
    from qualtran.bloqs.basic_gates import CNOT, Hadamard

    bit = RegisterType("bit", 1)
    signature = Signature((Register("data", bit, shape=(2,)),))
    cnot = bit_gate_function(
        "CNOT",
        (Register("ctrl", bit), Register("target", bit)),
        CircuitIR(2, (Instruction("cx", (0, 1)),)),
    )
    hadamard = bit_gate_function(
        "H",
        (Register("q", bit),),
        CircuitIR(1, (Instruction("h", (0,)),)),
    )
    native = ComponentProgramBuilder(signature, (cnot, hadamard))
    source, source_inputs = BloqBuilder.from_signature(
        SourceSignature((SourceRegister("data", QBit(), shape=(2,)),))
    )
    data = np.asarray(native.inputs["data"], dtype=object).copy()
    source_data = source_inputs["data"].copy()
    for _ in range(2):
        work = native.allocate(dirty=dirty)
        source_work = source.allocate(dtype=QBit(), dirty=dirty)
        for _ in range(2):
            result = native.call_dict("CNOT", {"ctrl": data[0], "target": work})
            data[0] = result["ctrl"]
            assert isinstance(result["target"], RegisterBinding)
            work = result["target"]
            source_data[0], source_work = source.add(
                CNOT(), ctrl=source_data[0], target=source_work
            )
        native.free(work, dirty=dirty)
        source.free(source_work, dirty=dirty)
    data[1] = native.call_dict("H", {"q": data[1]})["q"]
    source_data[1] = source.add(Hadamard(), q=source_data[1])
    return native.finalize({"data": data[::-1]}), source.finalize(
        data=source_data[::-1]
    )


@pytest.mark.parametrize("dirty", [False, True])
def test_reordered_shaped_ports_and_reused_workspace_match_source(dirty: bool) -> None:
    import cirq

    record, source = workspace_fixture(dirty)
    signature = Signature(record.functions[0].registers or ())
    bound = bind_register_program(
        record,
        ((0,), (1,)),
        clean=() if dirty else (2,),
        borrowed=(2,) if dirty else (),
    )
    events: list[tuple[str, tuple[Any, ...]]] = []
    work = cirq.NamedQubit("work")

    class Manager(cirq.SimpleQubitManager):
        def qalloc(self, n: int, dim: int = 2) -> list[Any]:
            assert (n, dim) == (1, 2)
            events.append(("clean", (work,)))
            return [work]

        def qborrow(self, n: int, dim: int = 2) -> list[Any]:
            assert (n, dim) == (1, 2)
            events.append(("borrow", (work,)))
            return [work]

        def qfree(self, qubits: Any) -> None:
            events.append(("free", tuple(qubits)))

    qubits = np.array([[cirq.NamedQubit("b")], [cirq.NamedQubit("a")]], dtype=object)
    expected, expected_out = source.to_cirq_circuit_and_quregs(
        qubit_manager=Manager(),
        data=qubits,
    )
    source_events = events.copy()
    events.clear()
    exported, outputs = export_bound_register_cirq(
        bound,
        signature,
        input_quregs={"data": qubits},
        qubit_manager=Manager(),
    )
    assert exported.report.lossless
    assert events == source_events
    np.testing.assert_array_equal(outputs["data"], expected_out["data"])
    for qubit in (*qubits.flat, work):
        assert tuple(
            op for op in exported.circuit.all_operations() if qubit in op.qubits
        ) == tuple(op for op in expected.all_operations() if qubit in op.qubits)
    spectator = cirq.NamedQubit("spectator")
    order = (*qubits.flat, work, spectator)
    values = np.arange(16)
    initial = (values + 1 + 1j * (values[::-1] + 2)).astype(np.complex128)
    if not dirty:
        initial.reshape(2, 2, 2, 2)[:, :, 1, :] = 0
    initial /= np.linalg.norm(initial)
    expected_state = expected.unitary(qubit_order=order) @ initial
    actual = exported.circuit.unitary(qubit_order=order) @ initial
    np.testing.assert_allclose(actual, expected_state, atol=1e-15, rtol=0)
    if not dirty:
        assert np.max(abs(actual.reshape(2, 2, 2, 2)[:, :, 1, :])) == 0


@pytest.mark.parametrize("dirty", [False, True])
def test_allocated_reversed_output_stays_owned_by_caller(dirty: bool) -> None:
    import cirq
    from qualtran import BloqBuilder, QAny, QBit, Side
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature

    signature = Signature(
        (
            Register("data", RegisterType("bit", 1)),
            Register("out", RegisterType("opaque", 2), side=RegisterSide.RIGHT),
        )
    )
    native = ComponentProgramBuilder(signature)
    output = native.allocate(2, dirty=dirty)
    output = native.join(native.split(output)[::-1])
    record = native.finalize({"data": native.inputs["data"], "out": output})
    bound = bind_register_program(
        record,
        ((0,),),
        clean=() if dirty else (1, 2),
        borrowed=(1, 2) if dirty else (),
        allow_workspace_outputs=True,
    )
    source, inputs = BloqBuilder.from_signature(
        SourceSignature(
            (
                SourceRegister("data", QBit()),
                SourceRegister("out", QAny(2), side=Side.RIGHT),
            )
        )
    )
    work = source.allocate(dtype=QAny(2), dirty=dirty)
    source_out = source.join(source.split(work)[::-1])
    composite = source.finalize(data=inputs["data"], out=source_out)
    events: list[str] = []
    work_q = list(cirq.NamedQubit.range(2, prefix="work"))

    class Manager(cirq.SimpleQubitManager):
        def qalloc(self, n: int, dim: int = 2) -> list[Any]:
            assert n == dim == 2
            events.append("clean")
            return work_q.copy()

        def qborrow(self, n: int, dim: int = 2) -> list[Any]:
            assert n == dim == 2
            events.append("borrow")
            return work_q.copy()

        def qfree(self, qubits: Any) -> None:
            events.append("free")

    data = cirq.NamedQubit("data")
    expected, source_outputs = composite.to_cirq_circuit_and_quregs(
        qubit_manager=Manager(),
        data=np.array([data]),
    )
    source_events = events.copy()
    events.clear()
    exported, outputs = export_bound_register_cirq(
        bound,
        signature,
        input_quregs={"data": data},
        qubit_manager=Manager(),
    )
    assert events == source_events == ["borrow" if dirty else "clean"]
    assert exported.circuit == expected
    np.testing.assert_array_equal(outputs["out"], source_outputs["out"])
    assert tuple(outputs["out"]) == tuple(work_q[::-1])


def test_default_names_and_invalid_ports_precede_manager_mutation() -> None:
    from dataclasses import replace

    import cirq

    record, _ = workspace_fixture(False)
    signature = Signature(record.functions[0].registers or ())
    bound = bind_register_program(record, ((0,), (1,)), clean=(2,))
    exported, outputs = export_bound_register_cirq(bound, signature)
    assert tuple(outputs["data"].flat) == (
        cirq.NamedQubit("data[1]"),
        cirq.NamedQubit("data[0]"),
    )
    assert len(tuple(exported.circuit.all_operations())) == 5
    with pytest.raises(ValueError, match="exactly"):
        export_bound_register_cirq(bound, signature, input_quregs={})
    with pytest.raises(ValueError, match="shape"):
        export_bound_register_cirq(
            bound, signature, input_quregs={"data": [cirq.LineQubit(0)]}
        )
    with pytest.raises(ValueError, match="distinct"):
        export_bound_register_cirq(
            bound, signature, input_quregs={"data": [[cirq.LineQubit(0)]] * 2}
        )
    with pytest.raises(ValueError, match="two-dimensional"):
        export_bound_register_cirq(bound, signature, input_quregs={"data": [[0], [1]]})
    with pytest.raises(ValueError, match="in-range"):
        export_bound_register_cirq(replace(bound, outputs=((0,), (0,))), signature)
    with pytest.raises(ValueError, match="widths"):
        export_bound_register_cirq(replace(bound, outputs=()), signature)
    with pytest.raises(ValueError, match="quantum"):
        export_bound_register_cirq(replace(bound, classical_outputs=((0,),)), signature)


@pytest.mark.parametrize("dirty", [False, True])
def test_preowned_left_release_and_reallocated_right_match_source(dirty: bool) -> None:
    from collections.abc import Iterable

    import cirq
    from qualtran import BloqBuilder, QAny, Side
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature

    word = RegisterType("opaque", 3)
    signature = Signature(
        (
            Register("input", word, side=RegisterSide.LEFT),
            Register("out", word, side=RegisterSide.RIGHT),
        )
    )
    builder = ComponentProgramBuilder(signature)
    input_word = builder.inputs["input"]
    assert isinstance(input_word, RegisterBinding)
    reverse = builder.join(builder.split(input_word)[::-1])
    builder.free(reverse, dirty=dirty)
    output = builder.allocate(3, dirty=dirty)
    record = builder.finalize({"out": output})
    bound = bind_register_program(
        record,
        ((0, 1, 2),),
        input_ownership=tuple((i, dirty) for i in range(3)),
        allow_workspace_outputs=True,
    )
    source, inputs = BloqBuilder.from_signature(
        SourceSignature(
            (
                SourceRegister("input", QAny(3), side=Side.LEFT),
                SourceRegister("out", QAny(3), side=Side.RIGHT),
            )
        )
    )
    source.free(source.join(source.split(inputs["input"])[::-1]), dirty=dirty)
    source_out = source.allocate(dtype=QAny(3), dirty=dirty)
    composite = source.finalize(out=source_out)
    events: list[tuple[str, tuple[Any, ...]]] = []

    class Manager(cirq.GreedyQubitManager):
        def qalloc(self, n: int, dim: int = 2) -> list[Any]:
            values = super().qalloc(n, dim)
            events.append(("allocate", tuple(values)))
            return values

        def qborrow(self, n: int, dim: int = 2) -> list[Any]:
            values = super().qborrow(n, dim)
            events[-1] = ("borrow", tuple(values))
            return values

        def qfree(self, qubits: Iterable[Any]) -> None:
            values = tuple(qubits)
            super().qfree(values)
            events.append(("free", values))

    source_manager = Manager("owned", size=3, maximize_reuse=False)
    native_manager = Manager("owned", size=3, maximize_reuse=False)
    source_qids = source_manager.qborrow(3) if dirty else source_manager.qalloc(3)
    native_qids = native_manager.qborrow(3) if dirty else native_manager.qalloc(3)
    events.clear()
    expected, expected_outputs = composite.to_cirq_circuit_and_quregs(
        qubit_manager=source_manager, input=np.array(source_qids)
    )
    source_events = events.copy()
    events.clear()
    exported, outputs = export_bound_register_cirq(
        bound,
        signature,
        input_quregs={"input": native_qids},
        qubit_manager=native_manager,
    )
    assert (
        events
        == source_events
        == [
            ("free", tuple(native_qids[::-1])),
            ("borrow" if dirty else "allocate", tuple(native_qids[::-1])),
        ]
    )
    assert exported.circuit == expected
    np.testing.assert_array_equal(outputs["out"], expected_outputs["out"])


@pytest.mark.parametrize("values", [(1, 1), (0, 1)])
def test_measured_child_reused_work_and_every_complex_branch(
    values: tuple[int, int],
) -> None:
    from itertools import product

    import cirq
    import torch
    from qualtran import BloqBuilder, QBit
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature
    from qualtran.bloqs.mcmt import And
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    import flagquantum as fq
    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.compiler.registers.control.multi_control import multi_and_ir
    from tests.unit.measurement_checks import forced_branch
    from tests.unit.test_cirq_feedback_import import source_branch

    gates = multi_and_ir((0, 1), values, (2,)) + multi_and_ir(
        (0, 1), values, (2,), classical_bits=(0,), uncompute=True
    )
    bit = RegisterType("bit", 1)
    cycle = bit_gate_function(
        "cycle",
        (Register("ctrl", bit, shape=(2,)), Register("work", bit)),
        CircuitIR(3, gates),
    )
    signature = Signature((Register("ctrl", bit, shape=(2,)),))
    builder = ComponentProgramBuilder(signature, (cycle,))
    controls = builder.inputs["ctrl"]
    source, inputs = BloqBuilder.from_signature(
        SourceSignature((SourceRegister("ctrl", QBit(), shape=(2,)),))
    )
    source_controls = inputs["ctrl"]
    for _ in range(2):
        work = builder.allocate()
        result = builder.call_dict("cycle", {"ctrl": controls, "work": work})
        controls = result["ctrl"]
        assert isinstance(result["work"], RegisterBinding)
        builder.free(result["work"])
        source_controls, target = source.add(And(*values), ctrl=source_controls)
        source_controls = source.add(
            And(*values, uncompute=True), ctrl=source_controls, target=target
        )
    record = builder.finalize({"ctrl": controls})
    composite = source.finalize(ctrl=source_controls)
    bound = bind_register_program(record, ((0,), (1,)), clean=(2,))
    qids = tuple(cirq.NamedQubit(name) for name in ("a", "b", "work", "spectator"))
    events: list[tuple[str, tuple[Any, ...]]] = []

    class Manager(cirq.SimpleQubitManager):
        def qalloc(self, n: int, dim: int = 2) -> list[Any]:
            assert (n, dim) == (1, 2)
            events.append(("allocate", (qids[2],)))
            return [qids[2]]

        def qfree(self, qubits: Any) -> None:
            events.append(("free", tuple(qubits)))

    expected, source_outputs = composite.to_cirq_circuit_and_quregs(
        qubit_manager=Manager(), ctrl=np.array(qids[:2]).reshape(2, 1)
    )
    source_events = events.copy()
    events.clear()
    exported, outputs = export_bound_register_cirq(
        bound,
        signature,
        input_quregs={"ctrl": np.array(qids[:2]).reshape(2, 1)},
        qubit_manager=Manager(),
        feedback_key_prefix="direct",
    )
    assert events == source_events
    np.testing.assert_array_equal(outputs["ctrl"], source_outputs["ctrl"])
    source_ops = tuple(
        op for child in expected.all_operations() for op in cirq.decompose_once(child)
    )
    native_ops = tuple(exported.circuit.all_operations())
    # Source names each measurement after its work wire. Native keys are indexed
    # by instruction occurrence, so reused work has distinct keys.
    source_key = next(
        cirq.measurement_key_name(op) for op in source_ops if cirq.is_measurement(op)
    )
    key = ""
    renamed = []
    native_keys = iter(
        cirq.measurement_key_name(op) for op in native_ops if cirq.is_measurement(op)
    )
    for op in source_ops:
        if cirq.is_measurement(op):
            key = next(native_keys)
        if cirq.is_measurement(op) or op.classical_controls:
            op = cirq.with_measurement_key_mapping(op, {source_key: key})
        renamed.append(op)
    assert native_ops == tuple(renamed)
    assert len(native_ops) == len(bound.circuit.instructions)
    counts = get_cost_value(composite, QECGatesCost())
    assert counts.and_bloq == 2 and counts.measurement == 2
    assert counts.clifford == 2
    assert len(native_ops) == 34 + 8 * values.count(0)
    assert sum(op.gate in (cirq.T, cirq.T**-1) for op in native_ops) == 8
    initial_tensor = np.zeros((2, 2, 2, 2), dtype=np.complex128)
    weights = np.arange(1, 9) + 1j * np.arange(8, 0, -1)
    initial_tensor[:, :, 0, :] = (weights / np.linalg.norm(weights)).reshape(2, 2, 2)
    initial = initial_tensor.reshape(-1)
    source_circuit = cirq.Circuit(source_ops)
    for history in product((0, 1), repeat=2):
        want = source_branch(source_circuit, qids, initial, history)
        actual = source_branch(exported.circuit, qids, initial, history)
        native = forced_branch(
            torch.from_numpy(initial), bound.circuit.instructions, history
        )
        np.testing.assert_allclose(actual, want, atol=2e-15, rtol=0)
        np.testing.assert_allclose(actual, native.numpy(), atol=2e-15, rtol=0)
        np.testing.assert_allclose(actual, initial / 2, atol=2e-15, rtol=0)
        assert np.max(abs(actual.reshape(2, 2, 2, 2)[:, :, 1, :])) < 2e-15
    appended = fq.Circuit(4).append(FieldOperation(3, bound.circuit.instructions))
    assert appended.to_ir().instructions == bound.circuit.instructions


def test_failed_replay_returns_allocated_work_to_manager() -> None:
    from dataclasses import replace

    import cirq

    from flagquantum.compiler.registers.register_binding import WorkspaceEvent

    record, _ = workspace_fixture(False)
    signature = Signature(record.functions[0].registers or ())
    bound = bind_register_program(record, ((0,), (1,)), clean=(2,))
    work = cirq.NamedQubit("work")
    freed: list[Any] = []

    class Manager(cirq.SimpleQubitManager):
        def qalloc(self, n: int, dim: int = 2) -> list[Any]:
            return [work]

        def qfree(self, qubits: Any) -> None:
            freed.extend(qubits)

    # A second lease at the same physical address must fail and release the first.
    event = bound.events[0]
    invalid = replace(bound, events=(event, event))
    with pytest.raises(ValueError, match="unbound"):
        export_bound_register_cirq(invalid, signature, qubit_manager=Manager())
    assert freed == [work]
    freed.clear()
    invalid = replace(bound, events=(event, replace(event, kind="retire")))
    with pytest.raises(ValueError, match="retired"):
        export_bound_register_cirq(invalid, signature, qubit_manager=Manager())
    assert freed == [work]
    freed.clear()
    invalid = replace(
        bound,
        events=(
            event,
            WorkspaceEvent(
                1, "free", (99,), False, instruction_offset=event.instruction_offset
            ),
        ),
    )
    with pytest.raises(ValueError, match="unowned"):
        export_bound_register_cirq(invalid, signature, qubit_manager=Manager())
    assert freed == [work]


def test_empty_outputs_and_zero_width_arrays_survive_export() -> None:
    import cirq
    from qualtran import BloqBuilder, QAny, Side
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature

    signature = Signature(
        (Register("input", RegisterType("opaque", 2), side=RegisterSide.LEFT),)
    )
    builder = ComponentProgramBuilder(signature)
    input_word = builder.inputs["input"]
    assert isinstance(input_word, RegisterBinding)
    builder.free(input_word)
    bound = bind_register_program(
        builder.finalize({}), ((0, 1),), input_ownership=((0, False), (1, False))
    )
    source, inputs = BloqBuilder.from_signature(
        SourceSignature((SourceRegister("input", QAny(2), side=Side.LEFT),))
    )
    source.free(inputs["input"])
    composite = source.finalize()
    events: list[tuple[Any, ...]] = []

    class Manager(cirq.SimpleQubitManager):
        def qfree(self, qubits: Any) -> None:
            events.append(tuple(qubits))

    qids = cirq.LineQubit.range(2)
    expected, expected_out = composite.to_cirq_circuit_and_quregs(
        qubit_manager=Manager(), input=np.array(qids)
    )
    exported, outputs = export_bound_register_cirq(
        bound, signature, input_quregs={"input": qids}, qubit_manager=Manager()
    )
    assert exported.circuit == expected
    assert outputs == expected_out == {}
    assert events == [tuple(qids), tuple(qids)]
    signature = Signature((Register("empty", RegisterType("opaque", 0), shape=(2,)),))
    builder = ComponentProgramBuilder(signature)
    bound = bind_register_program(
        builder.finalize({"empty": builder.inputs["empty"]}), ((), ()), clean=(0,)
    )
    exported, outputs = export_bound_register_cirq(bound, signature)
    assert outputs["empty"].shape == (2, 0)
    assert tuple(exported.circuit.all_operations()) == ()

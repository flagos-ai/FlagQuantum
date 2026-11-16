"""Differential and fail-closed tests for register placement at the Cirq boundary."""

from hashlib import sha1
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.control.multi_control import multi_and_ir
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program.register_types import opaque_register_type
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.ecosystem.cirq import CirqConversionError, export_cirq
from flagquantum.ecosystem.cirq._register_export import export_register_cirq


@pytest.fixture(scope="module")
def cirq() -> Any:
    return pytest.importorskip("cirq")


@pytest.fixture(scope="module")
def source_export() -> Any:
    import kickmix
    from qualtran import cirq_interop

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    folder = Path(cirq_interop.__file__).parent
    for name, digest in (
        ("__init__.py", "f6727d52660d008766a65672129094209ca8922d"),
        ("_bloq_to_cirq.py", "d49ab63f9f74104dc5f2856a9e1c997635e21e5b"),
        ("_bloq_to_cirq_test.py", "6b2ab24b21b3bbb926a9228ac56a6b14a735104c"),
        ("cirq_interop.ipynb", "bc9c7e175f1d6d0f8d085e6079342224d6cc503a"),
    ):
        raw = (folder / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == digest
    return cirq_interop


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("width", (1, 2, 3, 4, 5))
def test_original_swap_fixtures(cirq: Any, source_export: Any, width: int) -> None:
    from qualtran.bloqs.basic_gates import Swap

    left, right = tuple(range(width)), tuple(range(width, 2 * width))
    body = CircuitIR(
        2 * width,
        tuple(Instruction("swap", (a, b)) for a, b in zip(left, right, strict=True)),
    )
    native = Circuit(body.n_wires).compose(body)
    original = Swap(width)
    if width == 5:
        inputs = {"x": cirq.LineQubit.range(5), "y": cirq.LineQubit.range(100, 105)}
    else:
        inputs = source_export.get_named_qubits(original.signature)
    placed, out = export_register_cirq(
        native,
        Signature.build(x=width, y=width),
        {"x": left, "y": right},
        input_quregs=inputs,
    )
    qorder = [*inputs["x"], *inputs["y"]]
    expected = cirq.Circuit(cirq.decompose_once(original.on(*qorder)))
    assert list(placed.circuit.all_operations()) == list(expected.all_operations())
    assert placed.report.lossless
    assert list(out["x"]) == list(inputs["x"])
    assert list(out["y"]) == list(inputs["y"])
    np.testing.assert_allclose(
        placed.circuit.unitary(qubit_order=qorder), expected.unitary(qubit_order=qorder)
    )
    if width == 1:
        assert placed.circuit.to_text_diagram() == "x: ───×───\n      │\ny: ───×───"


@pytest.mark.ftqc_reference
def test_original_allocating_multi_and(cirq: Any, source_export: Any) -> None:
    from qualtran.bloqs.mcmt import MultiAnd

    source = MultiAnd(cvs=(1, 1, 1, 1))
    signature = Signature(
        (
            Register("ctrl", opaque_register_type(1), shape=(4,)),
            Register(
                "junk", opaque_register_type(1), shape=(2,), side=RegisterSide.RIGHT
            ),
            Register("target", opaque_register_type(1), side=RegisterSide.RIGHT),
        )
    )
    native = CircuitIR(7, multi_and_ir((0, 1, 2, 3), (1, 1, 1, 1), (4, 5, 6)))
    inputs = source_export.get_named_qubits(source.signature.lefts())
    original, orig_ports = source.decompose_bloq().to_cirq_circuit_and_quregs(
        qubit_manager=cirq.SimpleQubitManager(), **inputs
    )
    assert tuple(orig_ports) == tuple(reg.name for reg in signature.rights())
    expected_order = [q for name in orig_ports for q in orig_ports[name].flat]
    initial_state = np.zeros(128, dtype=np.complex128)
    initial_state[::8] = np.exp(1j * np.arange(16) / 3) / 4
    expected = cirq.Simulator(dtype=np.complex128).simulate(
        original, qubit_order=expected_order, initial_state=initial_state.copy()
    )
    import torch

    actual = Circuit.from_ir(
        native, inputs=torch.from_numpy(initial_state), dtype=torch.complex128
    )
    np.testing.assert_allclose(
        actual.state().reshape(-1).numpy(),
        expected.final_state_vector,
        atol=2e-12,
        rtol=0,
    )
    placed, ports = export_register_cirq(
        native,
        signature,
        {"ctrl": (0, 1, 2, 3), "junk": (4, 5), "target": (6,)},
        input_quregs=inputs,
        output_quregs={name: orig_ports[name] for name in ("junk", "target")},
    )
    source_leaves = cirq.decompose(
        original,
        keep=lambda op: op.gate in {cirq.H, cirq.S, cirq.T, cirq.T**-1, cirq.CNOT},
    )
    assert list(placed.circuit.all_operations()) == source_leaves
    assert placed.report.lossless
    assert [q for name in ports for q in ports[name].flat] == expected_order
    np.testing.assert_allclose(
        placed.circuit.unitary(qubit_order=expected_order),
        cirq.Circuit(source_leaves).unitary(qubit_order=expected_order),
        atol=2e-12,
        rtol=0,
    )


def test_register_export_preserves_inverse_phase_operands(cirq: Any) -> None:
    ir = CircuitIR(2, (Instruction("tdg", (0,)), Instruction("sdg", (1,))))
    placed, _ = export_register_cirq(
        ir, Signature.build(x=1, y=1), {"x": (0,), "y": (1,)}
    )
    x, y = cirq.NamedQubit("x"), cirq.NamedQubit("y")
    assert list(placed.circuit.all_operations()) == [cirq.T(x) ** -1, cirq.S(y) ** -1]
    assert placed.report.lossless
    with pytest.raises(CirqConversionError) as caught:
        export_cirq(ir)
    assert {issue.operation_name for issue in caught.value.report.issues} == {
        "sdg",
        "tdg",
    }


@pytest.mark.ftqc_reference
def test_original_toffoli_diagram(cirq: Any, source_export: Any) -> None:
    from qualtran.bloqs.basic_gates import Toffoli

    signature = Signature(
        (
            Register("ctrl", opaque_register_type(1), shape=(2,)),
            Register("target", opaque_register_type(1)),
        )
    )
    ir = CircuitIR(3, (Instruction("ccx", (0, 1, 2)),))
    qids = cirq.LineQubit.range(3)
    placed, _ = export_register_cirq(
        ir,
        signature,
        {"ctrl": (0, 1), "target": (2,)},
        input_quregs={"ctrl": [[qids[0]], [qids[1]]], "target": qids[2]},
    )
    expected = cirq.Circuit(Toffoli().on(*qids))
    for unicode in (False, True):
        assert placed.circuit.to_text_diagram(
            use_unicode_characters=unicode
        ) == expected.to_text_diagram(use_unicode_characters=unicode)


def test_left_and_right_ports_reserve_input_and_scratch(cirq: Any) -> None:
    signature = Signature(
        (
            Register("out", opaque_register_type(1), side=RegisterSide.RIGHT),
            Register("spent", opaque_register_type(1), side=RegisterSide.LEFT),
            Register("ctrl", opaque_register_type(1)),
        )
    )
    ir = CircuitIR(
        4, (Instruction("x", (1,)), Instruction("cx", (3, 0)), Instruction("h", (2,)))
    )
    supplied = {"spent": cirq.NamedQubit("out"), "ctrl": cirq.LineQubit(2)}
    placed, ports = export_register_cirq(
        ir, signature, {"out": (0,), "spent": (1,), "ctrl": (3,)}, input_quregs=supplied
    )
    (out,) = ports["out"]
    assert tuple(ports) == ("out", "ctrl")
    assert out not in supplied.values()
    assert set(placed.circuit.all_qubits()) == {
        out,
        *supplied.values(),
        cirq.NamedQubit("__fq_work_2"),
    }
    assert tuple(placed.circuit.all_operations())[1].qubits == (supplied["ctrl"], out)
    assert placed.report.to_dict() == export_cirq(ir).report.to_dict()

    explicit_out = cirq.LineQubit(2)
    wires = {"out": (0,), "spent": (1,), "ctrl": (3,)}
    rebound, explicit_ports = export_register_cirq(
        ir, signature, wires, output_quregs={"out": explicit_out}
    )
    assert tuple(explicit_ports["out"]) == (explicit_out,)
    assert tuple(explicit_ports["ctrl"]) == (cirq.NamedQubit("ctrl"),)
    assert tuple(rebound.circuit.all_operations())[1].qubits == (
        cirq.NamedQubit("ctrl"),
        explicit_out,
    )
    assert cirq.NamedQubit("__fq_work_2") in rebound.circuit.all_qubits()
    assert rebound.report.to_dict() == placed.report.to_dict()

    with pytest.raises(ValueError, match="distinct qubits"):
        export_register_cirq(
            ir,
            signature,
            wires,
            input_quregs=supplied,
            output_quregs={"out": supplied["ctrl"]},
        )
    with pytest.raises(ValueError, match="exactly the right-only registers"):
        export_register_cirq(ir, signature, wires, output_quregs={"ctrl": explicit_out})


@pytest.mark.parametrize(
    "failure", ("shape", "duplicate_qids", "dimension", "native", "missing", "loss")
)
def test_register_export_rejects_bad_placement(cirq: Any, failure: str) -> None:
    signature = Signature.build(x=1, y=2)
    inputs: dict[str, Any] = {"x": cirq.LineQubit(0), "y": cirq.LineQubit.range(1, 3)}
    wires = {"x": (0,), "y": (1, 2)}
    if failure == "shape":
        inputs["y"] = [inputs["y"]]
    elif failure == "duplicate_qids":
        inputs["y"][0] = inputs["x"]
    elif failure == "dimension":
        inputs["x"] = cirq.LineQid(0, dimension=3)
    elif failure == "native":
        wires["x"] = (1,)
    elif failure == "missing":
        del inputs["x"]
    ir = CircuitIR(3, (Instruction("x", (0,)),))
    if failure == "loss":
        with pytest.raises(CirqConversionError) as error:
            export_register_cirq(ir, signature, wires, input_quregs=inputs)
        assert "idle_wire_extent_not_represented" in {
            issue.code for issue in error.value.report.issues
        }
    else:
        with pytest.raises(ValueError, match="Cirq"):
            export_register_cirq(ir, signature, wires, input_quregs=inputs)


def test_export_does_not_drop_standalone_classical_feedback(cirq: Any) -> None:
    ir = CircuitIR(1, (Instruction("x", (0,), metadata={"conditions": ((0, 1),)}),))
    with pytest.raises(CirqConversionError) as caught:
        export_cirq(ir)
    assert [issue.code for issue in caught.value.report.issues] == [
        "unsupported_operation"
    ]
    lossy = export_cirq(ir, allow_lossy=True)
    assert not lossy.report.lossless
    assert list(lossy.circuit.all_operations()) == []


def test_shaped_names_and_terminal_measurement_keep_operand_order(cirq: Any) -> None:
    signature = Signature(
        (
            Register("words", opaque_register_type(2), shape=(2,)),
            Register("empty", opaque_register_type(1), shape=(0,)),
        )
    )
    ir = CircuitIR(
        4,
        (
            Instruction("h", (1,)),
            Instruction("cx", (1, 3)),
            *(
                Instruction(
                    "measure",
                    (wire,),
                    metadata={
                        "is_dynamic": True,
                        "classical_bit": i,
                        "measurement_key": "m",
                    },
                )
                for i, wire in enumerate((3, 0, 2, 1))
            ),
        ),
    )
    result, ports = export_register_cirq(
        ir, signature, {"words": (3, 1, 0, 2), "empty": ()}
    )
    words = [
        cirq.NamedQubit(name)
        for name in ("words[0][0]", "words[0][1]", "words[1][0]", "words[1][1]")
    ]
    assert ports["words"].tolist() == [words[:2], words[2:]]
    assert ports["empty"].shape == (0, 1)
    assert list(result.circuit.all_operations()) == [
        cirq.H(words[1]),
        cirq.CNOT(words[1], words[0]),
        cirq.measure(words[0], words[2], words[3], words[1], key="m"),
    ]
    assert result.report.lossless


@pytest.mark.ftqc_reference
def test_original_compute_uncompute_and_feedback_guard(
    cirq: Any, source_export: Any
) -> None:
    import torch
    from qualtran.bloqs.mcmt import And

    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic

    gates = (
        *multi_and_ir((0, 1), (1, 1), (2,)),
        *multi_and_ir((0, 1), (1, 1), (2,), uncompute=True, classical_bits=(0,)),
    )
    ir = CircuitIR(3, gates)
    source = And().as_composite_bloq()
    compute = source.tensor_contract()
    erase = And().adjoint().as_composite_bloq().tensor_contract()
    np.testing.assert_allclose(erase @ compute, np.eye(4), atol=2e-12, rtol=0)
    state = torch.zeros(8, dtype=torch.complex128)
    state[::2] = torch.tensor((1, 1j, -1, complex(0, -1)), dtype=torch.complex128) / 2
    circuit = DynamicCircuit(3, inputs=state, dtype=torch.complex128)
    circuit.compose(ir)
    result = run_dynamic(circuit, shots=8, seed=37, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states, state.expand(8, -1), atol=2e-12, rtol=0
    )
    with pytest.raises(
        CirqConversionError, match="does not preserve|unsupported_operation"
    ):
        export_register_cirq(ir, Signature.build(a=1, b=1), {"a": (0,), "b": (1,)})


@pytest.mark.ftqc_reference
def test_original_positive_integer_swap_power(cirq: Any, source_export: Any) -> None:
    from qualtran.bloqs.basic_gates import TwoBitSwap

    gate = source_export.BloqAsCirqGate(TwoBitSwap())
    qids = cirq.LineQubit.range(2)
    with pytest.raises(ValueError, match="positive integer"):
        (gate**0.5).on(*qids)
    swap = CircuitIR(2, (Instruction("swap", (0, 1)),))
    program = Circuit(2).compose(swap).compose(swap)
    actual, _ = export_register_cirq(
        program,
        Signature.build(x=1, y=1),
        {"x": (0,), "y": (1,)},
        input_quregs={"x": qids[0], "y": qids[1]},
    )
    np.testing.assert_allclose(
        actual.circuit.unitary(), cirq.unitary(gate**2), atol=2e-12, rtol=0
    )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("width", (1, 2, 3, 4))
def test_exported_gate_uses_bound_body_or_owned_matrix(
    cirq: Any, source_export: Any, width: int
) -> None:
    import torch
    from qualtran.cirq_interop._bloq_to_cirq_test import (
        SwapTest,
        SwapTestWithOnlyTensorData,
    )

    from flagquantum.ecosystem.cirq._register_gate import RegisterCirqGate

    expected = source_export.BloqAsCirqGate(SwapTestWithOnlyTensorData(width))
    signature = Signature.build(x=width, y=width)
    qids = tuple(cirq.LineQubit.range(2 * width))
    body = CircuitIR(
        2 * width,
        tuple(Instruction("swap", (i, i + width)) for i in range(width)),
    )
    gate = RegisterCirqGate(body, signature)
    np.testing.assert_allclose(cirq.unitary(gate), cirq.unitary(expected))
    assert (
        cirq.circuit_diagram_info(gate).wire_symbols
        == cirq.circuit_diagram_info(expected).wire_symbols
    )
    original = source_export.BloqAsCirqGate(SwapTest(width))
    assert tuple(
        cirq.decompose(gate.on(*qids), keep=lambda op: op.gate == cirq.SWAP)
    ) == tuple(cirq.decompose(original.on(*qids), keep=lambda op: op.gate == cirq.SWAP))

    dimension = 1 << (2 * width)
    rows = np.arange(dimension)
    columns = (rows % (1 << width)) * (1 << width) + (rows >> width)
    matrix = torch.zeros((dimension, dimension), dtype=torch.complex128)
    matrix[rows, columns] = 1
    tensor_body = CircuitIR(
        2 * width, (Instruction("matrix", tuple(range(2 * width)), matrix=matrix),)
    )
    tensor_gate = RegisterCirqGate(tensor_body, signature)
    assert cirq.decompose_once(tensor_gate.on(*qids), default=None) is None
    np.testing.assert_allclose(cirq.unitary(tensor_gate), cirq.unitary(expected))


def test_exported_gate_power_and_sided_decomposition(cirq: Any) -> None:
    from flagquantum.ecosystem.cirq._register_gate import RegisterCirqGate

    a, b = cirq.LineQubit.range(2)
    body = CircuitIR(2, (Instruction("h", (1,)), Instruction("t", (1,))))
    through = Signature.build(x=1, y=1)
    gate = RegisterCirqGate(body, through)
    assert gate**1 is gate
    for power in (-2, -1, 2, 3):
        np.testing.assert_allclose(
            cirq.unitary(gate**power),
            np.linalg.matrix_power(cirq.unitary(gate), power),
            atol=2e-12,
            rtol=0,
        )
    for invalid_power in (0, 0.5, True):
        with pytest.raises(ValueError, match="positive integer"):
            gate**invalid_power

    sided = Signature(
        (
            through.get_left("x"),
            Register("y", opaque_register_type(1), side=RegisterSide.RIGHT),
        )
    )
    directional = RegisterCirqGate(body, sided)
    assert directional._unitary_() is NotImplemented
    assert list(cirq.decompose_once(directional.on(a, b))) == [cirq.H(b), cirq.T(b)]
    assert list(cirq.decompose_once((directional**-1).on(a, b))) == [
        cirq.T(b) ** -1,
        cirq.H(b),
    ]
    with pytest.raises(ValueError, match="through registers"):
        directional**2
    with pytest.raises(ValueError, match="one declared operand"):
        RegisterCirqGate(body, Signature.build(x=1))


@pytest.mark.ftqc_reference
def test_original_uniform_preparation_flattened_workspace_extent(
    cirq: Any, source_export: Any
) -> None:
    from qualtran.bloqs.state_preparation.prepare_uniform_superposition import (
        PrepareUniformSuperposition,
    )

    from flagquantum.compiler.registers.preparation.uniform_preparation import (
        uniform_preparation_ir,
        uniform_preparation_requirements,
    )

    source = PrepareUniformSuperposition(n=3, cvs=(1,))
    flat = source.decompose_bloq().flatten()
    manager = cirq.GreedyQubitManager(prefix="anc", maximize_reuse=True)
    converted = flat.to_cirq_circuit(qubit_manager=manager)
    target_width, work_width, slot_count = uniform_preparation_requirements(3, 1)
    assert (target_width, work_width, slot_count) == (2, 4, 2)
    gates = uniform_preparation_ir(
        3, (1, 2), (3, 4, 5, 6), (0, 1), controls=(0,), control_values=(1,)
    )
    assert len(converted.all_qubits()) == len({q for g in gates for q in g.wires}) == 7
    measurements = tuple(g for g in gates if g.name == "measure")
    assert len(measurements) == 6
    assert {g.metadata["classical_bit"] for g in measurements} == {0, 1}


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("dirty", (False, True))
def test_gate_workspace_context_replays_native_leases(
    cirq: Any, source_export: Any, dirty: bool
) -> None:
    from collections.abc import Iterable
    from dataclasses import replace

    from cirq import Qid, SimpleQubitManager
    from qualtran import BloqBuilder, QBit
    from qualtran import Signature as SourceSignature
    from qualtran.bloqs.basic_gates import CNOT

    from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
    from flagquantum.compiler.registers.register_binding import bind_register_program
    from flagquantum.core.ir.program.register_bindings import RegisterBinding
    from flagquantum.ecosystem.cirq._register_gate import RegisterCirqGate
    from tests.unit.test_register_program_lowering import packed_circuit

    signature = Signature.build(data=1)
    module = packed_circuit(CircuitIR(2, (Instruction("cx", (0, 1)),)), (1, 1))
    definitions = tuple(
        replace(f, name="cnot") if f.name == "main" else f for f in module.functions
    )
    builder = ComponentProgramBuilder(signature, definitions)
    data = builder.inputs["data"]
    source, inputs = BloqBuilder.from_signature(SourceSignature.build(data=1))
    source_data = inputs["data"]
    for _ in range(2):
        work = builder.allocate(dirty=dirty)
        source_work = source.allocate(dtype=QBit(), dirty=dirty)
        for _ in range(2):
            data, next_work = builder.call_tuple("cnot", {"word0": data, "word1": work})
            assert isinstance(next_work, RegisterBinding)
            work = next_work
            source_data, source_work = source.add(
                CNOT(), ctrl=source_data, target=source_work
            )
        builder.free(work, dirty=dirty)
        source.free(source_work, dirty=dirty)
    bound = bind_register_program(
        builder.finalize({"data": data}),
        ((0,),),
        clean=() if dirty else (1,),
        borrowed=(1,) if dirty else (),
    )
    composite = source.finalize(data=source_data)
    data_q, work_q, spectator = cirq.LineQubit.range(3)
    events: list[str] = []

    class Manager(SimpleQubitManager):
        def qalloc(self, n: int, dim: int = 2) -> list[Qid]:
            assert (n, dim) == (1, 2)
            events.append("clean")
            return [work_q]

        def qborrow(self, n: int, dim: int = 2) -> list[Qid]:
            assert (n, dim) == (1, 2)
            events.append("borrow")
            return [work_q]

        def qfree(self, qubits: Iterable[Qid]) -> None:
            assert tuple(qubits) == (work_q,)
            events.append("free")

    original, _ = composite.to_cirq_circuit_and_quregs(
        data=np.array([data_q]), qubit_manager=Manager()
    )
    source_events = tuple(events)
    events.clear()
    wrapped = RegisterCirqGate(bound, signature)
    emitted = cirq.decompose_once(
        wrapped.on(data_q), context=cirq.DecompositionContext(Manager())
    )
    assert (
        tuple(emitted)
        == tuple(original.all_operations())
        == (cirq.CNOT(data_q, work_q),) * 4
    )
    assert (
        tuple(events) == source_events == ("borrow" if dirty else "clean", "free") * 2
    )
    initial = np.array([1, 2j, -3j, 4, 2 + 1j, -1, 3, -2j], dtype=np.complex128)
    if not dirty:
        initial.reshape(2, 2, 2)[:, 1, :] = 0
    initial /= np.linalg.norm(initial)
    simulated = cirq.Simulator(dtype=np.complex128).simulate(
        cirq.Circuit(emitted),
        qubit_order=(data_q, work_q, spectator),
        initial_state=initial.copy(),
    )
    np.testing.assert_allclose(
        simulated.final_state_vector, initial, rtol=0, atol=2e-12
    )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("reuse", (False, True))
def test_gate_transfers_only_final_right_output_lease(
    cirq: Any, source_export: Any, reuse: bool
) -> None:
    from collections.abc import Iterable
    from dataclasses import replace

    from cirq import GreedyQubitManager, Qid
    from qualtran.bloqs.basic_gates import IntState

    from flagquantum.compiler.registers.component_basis_vectors import (
        integer_basis_function,
    )
    from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
    from flagquantum.compiler.registers.register_binding import (
        WorkspaceEvent,
        bind_register_program,
    )
    from flagquantum.ecosystem.cirq._register_gate import RegisterCirqGate

    prepare = integer_basis_function("prepare", opaque_register_type(2), 1)
    assert prepare.registers is not None
    signature = Signature(prepare.registers)
    builder = ComponentProgramBuilder(signature, (prepare,))
    output = builder.call("prepare", {})
    bound = bind_register_program(
        builder.finalize({"val": output}),
        (),
        clean=(0, 1),
        allow_workspace_outputs=True,
    )
    assert bound.outputs == ((0, 1),)
    if reuse:
        (output_lease,) = bound.events
        # Use an explicit ordered trace so scheduling the independent output
        # preparation cannot move its allocation ahead of the private lease.
        bound = replace(
            bound,
            circuit=CircuitIR(
                2, (Instruction("x", (0,)),) * 2 + bound.circuit.instructions
            ),
            events=(
                WorkspaceEvent(0, "allocate", (0,), False, instruction_offset=0),
                WorkspaceEvent(3, "free", (0,), False, instruction_offset=2),
                replace(output_lease, position=4, instruction_offset=2),
            ),
        )
    events: list[tuple[str, tuple[Qid, ...]]] = []

    class Manager(GreedyQubitManager):
        def qalloc(self, n: int, dim: int = 2) -> list[Qid]:
            result = list(super().qalloc(n, dim))
            events.append(("allocate", tuple(result)))
            return result

        def qfree(self, qubits: Iterable[Qid]) -> None:
            values = tuple(qubits)
            if values:
                events.append(("free", values))
            super().qfree(values)

    msb, lsb = cirq.NamedQubit("result_msb"), cirq.NamedQubit("result_lsb")
    manager = Manager(prefix="temp", maximize_reuse=True)
    emitted = tuple(
        cirq.decompose_once(
            RegisterCirqGate(bound, signature).on(msb, lsb),
            context=cirq.DecompositionContext(manager),
        )
    )
    assert emitted[-1:] == (cirq.X(lsb),)
    assert events[-2][0] == "allocate"
    temporaries = events[-2][1]
    assert events[-1] == ("free", temporaries)
    order: tuple[Qid, ...]
    if reuse:
        assert [(event.kind, event.wires) for event in bound.events] == [
            ("allocate", (0,)),
            ("free", (0,)),
            ("allocate", (0, 1)),
        ]
        private = events[0][1][0]
        assert private in temporaries
        assert emitted[:-1] == (cirq.X(private),) * 2
        assert not ({msb, lsb} & set(temporaries))
        order = (msb, lsb, private)
        expected = 2
    else:
        native_events = events.copy()
        events.clear()
        original = source_export.BloqAsCirqGate(IntState(1, 2)).on(msb, lsb)
        original_ops = cirq.decompose_once(
            original,
            context=cirq.DecompositionContext(
                Manager(prefix="temp", maximize_reuse=True)
            ),
        )
        assert tuple(original_ops) == emitted
        assert [kind for kind, _ in events] == ["allocate", "allocate", "free"]
        assert events[0][1] + events[1][1] == native_events[0][1]
        assert events[-1] == native_events[-1]
        order = (msb, lsb)
        expected = 1
    result = cirq.Simulator(dtype=np.complex128).simulate(
        cirq.Circuit(emitted), qubit_order=order
    )
    want = np.zeros(1 << len(order), dtype=np.complex128)
    want[expected] = 1
    np.testing.assert_array_equal(result.final_state_vector, want)

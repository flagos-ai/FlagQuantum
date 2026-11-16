"""Differential and fail-closed tests for register placement at the Cirq boundary."""

from hashlib import sha1
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

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
    np.testing.assert_allclose(cirq.unitary(wrapped), np.eye(2), atol=2e-12, rtol=0)
    altered = replace(
        bound,
        circuit=CircuitIR(2, (Instruction("cz" if dirty else "cx", (0, 1)),)),
    )
    assert cirq.unitary(RegisterCirqGate(altered, signature), default=None) is None
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


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("reuse", (False, True))
def test_gate_releases_left_input_without_freeing_caller_qid(
    cirq: Any, source_export: Any, reuse: bool
) -> None:
    from collections.abc import Iterable
    from dataclasses import replace

    from cirq import GreedyQubitManager, Qid
    from qualtran import BloqBuilder, CompositeBloq, QBit, Side
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature
    from qualtran.bloqs.basic_gates import IntEffect, XGate
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    from flagquantum.compiler.registers.component_basis_vectors import (
        integer_basis_function,
    )
    from flagquantum.compiler.registers.register_binding import (
        WorkspaceEvent,
        bind_register_program,
    )
    from flagquantum.core.ir.program import ProgramRecord
    from flagquantum.ecosystem.cirq._register_gate import RegisterCirqGate

    source_signature = SourceSignature((SourceRegister("val", QBit(), side=Side.LEFT),))
    source_builder, source_inputs = BloqBuilder.from_signature(source_signature)
    source_builder.free(source_builder.add(XGate(), q=source_inputs["val"]))
    reference = source_builder.finalize()

    class Consume:
        signature = source_signature

        def decompose_bloq(self) -> CompositeBloq:
            return reference

    released: list[Qid] = []

    class Manager(GreedyQubitManager):
        def qfree(self, qubits: Iterable[Qid]) -> None:
            values = tuple(qubits)
            released.extend(values)
            super().qfree(values)

    consuming_q = cirq.NamedQubit("input")
    result_q = cirq.NamedQubit("output")
    spectator = cirq.NamedQubit("spectator")
    manager = Manager(prefix="temp", maximize_reuse=True)
    context = cirq.DecompositionContext(manager)
    original = source_export.BloqAsCirqGate(Consume()).on(consuming_q)
    assert tuple(cirq.decompose_once(original, context=context)) == (
        cirq.X(consuming_q),
    )
    assert released == []
    assert get_cost_value(reference, QECGatesCost()).clifford == 1
    with pytest.raises(ValueError, match="There is no Cirq equivalent"):
        cirq.decompose_once(
            source_export.BloqAsCirqGate(IntEffect(1, 1)).on(consuming_q),
            context=context,
        )

    native = integer_basis_function("main", opaque_register_type(1), 1, state=False)
    assert native.registers is not None
    signature = Signature(native.registers)
    bound = bind_register_program(
        ProgramRecord(functions=(native,)), ((0,),), input_ownership=((0, False),)
    )
    assert bound.circuit.instructions == (Instruction("x", (0,)),)
    (release,) = bound.events
    assert (release.kind, release.wires, release.instruction_offset) == (
        "free",
        (0,),
        1,
    )

    qids: tuple[Qid, ...]
    expected: tuple[object, ...]
    if reuse:
        signature = Signature(
            (
                *signature,
                Register("out", opaque_register_type(1), side=RegisterSide.RIGHT),
            )
        )
        bound = replace(
            bound,
            circuit=CircuitIR(1, bound.circuit.instructions * 2),
            outputs=((0,),),
            events=(
                release,
                WorkspaceEvent(
                    release.position + 1, "allocate", (0,), False, instruction_offset=1
                ),
            ),
        )
        qids = (consuming_q, result_q)
        expected = (cirq.X(consuming_q), cirq.X(result_q))
    else:
        qids = (consuming_q,)
        expected = (cirq.X(consuming_q),)
        missing = RegisterCirqGate(replace(bound, events=()), signature)
        with pytest.raises(ValueError, match="release every owned left-only input"):
            cirq.decompose_once(missing.on(*qids), context=context)
    emitted = tuple(
        cirq.decompose_once(
            RegisterCirqGate(bound, signature).on(*qids), context=context
        )
    )
    assert emitted == expected
    assert consuming_q not in released and result_q not in released
    assert len(released) == int(reuse)
    initial = np.zeros(2 ** (len(qids) + 1), dtype=np.complex128)
    initial[len(initial) // 2 : len(initial) // 2 + 2] = (
        1 / np.sqrt(2),
        1j / np.sqrt(2),
    )
    want = np.zeros_like(initial)
    first = 2 * int(reuse)
    want[first : first + 2] = (1 / np.sqrt(2), 1j / np.sqrt(2))
    actual = cirq.Simulator(dtype=np.complex128).simulate(
        cirq.Circuit(emitted),
        qubit_order=(*qids, spectator),
        initial_state=initial.copy(),
    )
    np.testing.assert_allclose(actual.final_state_vector, want, atol=2e-12, rtol=0)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("values", ((1, 1), (0, 1)))
def test_cirq_feedback_replays_each_and_branch(
    cirq: Any, source_export: Any, values: tuple[int, int]
) -> None:
    from itertools import product

    import torch
    from qualtran.bloqs.mcmt import And

    from flagquantum.compiler.registers.register_binding import (
        BoundRegisterProgram,
        WorkspaceEvent,
    )
    from flagquantum.ecosystem.cirq._register_gate import RegisterCirqGate
    from tests.unit.measurement_checks import forced_branch

    wires = (0, 1)
    compute = multi_and_ir(wires, values, (2,))
    erase = multi_and_ir(wires, values, (2,), classical_bits=(3,), uncompute=True)
    body = CircuitIR(3, (compute + erase) * 2)
    signature = Signature.build(a=1, b=1, scratch=1)
    qids = tuple(cirq.LineQubit.range(3))
    supplied = dict(zip(("a", "b", "scratch"), qids, strict=True))
    export, _ = export_register_cirq(
        body,
        signature,
        {"a": (0,), "b": (1,), "scratch": (2,)},
        input_quregs=supplied,
        feedback_key_prefix="feedback",
    )
    emitted = tuple(export.circuit.all_operations())
    assert export.report.lossless and len(emitted) == len(body.instructions)
    positions = [
        i for i, gate in enumerate(body.instructions) if gate.name == "measure"
    ]
    keys = tuple(f"feedback_{position}" for position in positions)
    assert tuple(cirq.measurement_key_name(emitted[i]) for i in positions) == keys
    original_compute = tuple(cirq.decompose_once(And(*values).on(*qids)))
    original_erase = tuple(cirq.decompose_once(And(*values, uncompute=True).on(*qids)))
    source_key = next(
        cirq.measurement_key_name(op)
        for op in original_erase
        if cirq.is_measurement(op)
    )
    assert emitted == tuple(
        op
        for key in keys
        for op in (
            *original_compute,
            *(
                (
                    cirq.with_measurement_key_mapping(item, {source_key: key})
                    if cirq.is_measurement(item)
                    or isinstance(item, cirq.ClassicallyControlledOperation)
                    else item
                )
                for item in original_erase
            ),
        )
    )
    with pytest.raises(CirqConversionError):
        export_cirq(body)

    # Replaying workspace events preserves the exact emitted instruction indices.
    bound = BoundRegisterProgram(
        body,
        ((0,), (1,)),
        (
            WorkspaceEvent(0, "allocate", (2,), False, instruction_offset=0),
            WorkspaceEvent(
                len(emitted) + 1, "free", (2,), False, instruction_offset=len(emitted)
            ),
        ),
    )
    wrapped = RegisterCirqGate(
        bound, Signature.build(a=1, b=1), feedback_key_prefix="feedback"
    )
    actual = tuple(cirq.decompose_once(wrapped.on(*qids[:2])))
    (scratch,) = {q for op in actual for q in op.qubits} - set(qids[:2])
    assert (
        tuple(
            op.transform_qubits(lambda q: qids[2] if q == scratch else q)
            for op in actual
        )
        == emitted
    )

    # Project each Cirq measurement explicitly to retain unnormalized amplitudes.
    order = (*qids, cirq.LineQubit(3))
    initial = np.zeros((2, 2, 2, 2), dtype=np.complex128)
    weights = np.arange(1, 9).astype(np.complex128) + 1j * np.arange(8, 0, -1)
    initial[:, :, 0, :] = (weights / np.linalg.norm(weights)).reshape(2, 2, 2)
    for history in product((0, 1), repeat=2):
        state = _cirq_feedback_branch(cirq, emitted, order, initial, history)
        native = forced_branch(
            torch.from_numpy(initial.reshape(-1)), body.instructions, history
        )
        np.testing.assert_allclose(state, native.numpy(), atol=2e-12, rtol=0)
        np.testing.assert_allclose(state, initial.reshape(-1) / 2, atol=2e-12, rtol=0)


def _cirq_feedback_branch(
    cirq: Any,
    emitted: tuple[Any, ...],
    order: tuple[Any, ...],
    initial: NDArray[np.complex128],
    history: tuple[int, ...],
) -> NDArray[np.complex128]:
    """Project each measurement and apply resets only on collapsed wires."""
    state: NDArray[np.complex128] = initial.reshape(-1).copy()
    bits = iter(history)
    shape = (2,) * len(order)
    classical = cirq.ClassicalDataDictionaryStore()
    for operation in emitted:
        if cirq.is_measurement(operation):
            bit = next(bits)
            (qubit,) = operation.qubits
            np.moveaxis(state.reshape(shape), order.index(qubit), 0)[1 - bit] = 0
            classical.record_measurement(
                cirq.measurement_key_obj(operation), (bit,), operation.qubits
            )
            continue
        if isinstance(operation, cirq.ClassicallyControlledOperation):
            if not all(
                condition.resolve(classical)
                for condition in operation.classical_controls
            ):
                continue
            operation = operation.without_classical_controls()
        if isinstance(operation.gate, cirq.ResetChannel):
            (qubit,) = operation.qubits
            zero, one = np.moveaxis(state.reshape(shape), order.index(qubit), 0)
            if np.linalg.norm(one) <= 1e-14:
                continue
            np.testing.assert_allclose(zero, 0, atol=1e-14, rtol=0)
            operation = cirq.X(qubit)
        state = np.asarray(
            cirq.Circuit(operation).unitary(qubit_order=order) @ state,
            dtype=np.complex128,
        )
    return state


def test_cirq_feedback_keeps_both_polarities_in_disjunctions(cirq: Any) -> None:
    from itertools import product

    signature = Signature.build(a=1, b=1, target=1)
    wires = {"a": (0,), "b": (1,), "target": (2,)}
    measurements = (
        Instruction("measure", (0,), metadata={"is_dynamic": True, "classical_bit": 5}),
        Instruction("measure", (1,), metadata={"is_dynamic": True, "classical_bit": 2}),
    )
    controlled = Instruction(
        "x",
        (2,),
        metadata={"condition_clauses": (((2, 0), (5, 1)), ((2, 1), (5, 0)))},
    )
    program = CircuitIR(3, (*measurements, controlled))
    exported, ports = export_register_cirq(
        program, signature, wires, feedback_key_prefix="external"
    )
    order = tuple(ports[name][0] for name in wires)
    simulator = cirq.Simulator(dtype=np.complex128)
    for a, b in product((0, 1), repeat=2):
        actual = simulator.simulate(
            exported.circuit, qubit_order=order, initial_state=4 * a + 2 * b
        )
        expected = np.zeros(8, dtype=np.complex128)
        expected[4 * a + 2 * b + (a ^ b)] = 1
        np.testing.assert_array_equal(actual.final_state_vector, expected)
    with pytest.raises(ValueError, match="read before measurement"):
        export_register_cirq(
            CircuitIR(3, (controlled, *measurements)),
            signature,
            wires,
            feedback_key_prefix="external",
        )


def test_gate_feedback_snapshots_xor_and_remaps_workspace_offsets(cirq: Any) -> None:
    from dataclasses import replace
    from itertools import product

    import torch

    from flagquantum.compiler.registers.register_binding import (
        BoundRegisterProgram,
        WorkspaceEvent,
    )
    from flagquantum.core.classical_dataflow import ClassicalXor
    from flagquantum.ecosystem.cirq._register_gate import RegisterCirqGate
    from tests.unit.measurement_checks import forced_branch

    def measure(wire: int, slot: int) -> Instruction:
        return Instruction(
            "measure", (wire,), metadata={"is_dynamic": True, "classical_bit": slot}
        )

    assignment = ClassicalXor((0, 2), 2).instruction()
    assignment = replace(
        assignment, metadata={**assignment.metadata, "conditions": ((1, 1),)}
    )
    prefix = (
        measure(0, 0),
        measure(1, 1),
        measure(2, 2),
        assignment,
        Instruction("reset", (0,), metadata={"is_dynamic": True}),
        measure(0, 0),
    )
    conditional = Instruction("x", (4,), metadata={"conditions": ((2, 1),)})
    body = CircuitIR(5, (*prefix, conditional, Instruction("cx", (4, 3)), conditional))
    bound = BoundRegisterProgram(
        body,
        ((0,), (1,), (2,), (3,)),
        (
            WorkspaceEvent(0, "allocate", (4,), False, instruction_offset=len(prefix)),
            WorkspaceEvent(
                1, "free", (4,), False, instruction_offset=len(body.instructions)
            ),
        ),
    )
    signature = Signature.build(a=1, b=1, c=1, answer=1)
    gate = RegisterCirqGate(bound, signature, feedback_key_prefix="memory")
    ports = tuple(cirq.LineQubit.range(4))
    emitted = tuple(cirq.decompose_once(gate.on(*ports)))
    assert len(emitted) == len(body.instructions) - 1
    (work,) = {q for op in emitted for q in op.qubits} - set(ports)
    assert set(cirq.measurement_key_names(cirq.Circuit(emitted))) == {
        "memory_0",
        "memory_1",
        "memory_2",
        "memory_5",
    }
    assert {str(key) for key in cirq.control_keys(emitted[-3])} == {
        "memory_0",
        "memory_1",
        "memory_2",
    }

    spectator = cirq.NamedQubit("spectator")
    order = (*ports, work, spectator)
    amplitudes = np.array([1, 2j], dtype=np.complex128) / np.sqrt(5)
    simulator = cirq.Simulator(dtype=np.complex128)
    for a, b, c in product((0, 1), repeat=3):
        initial = np.zeros(64, dtype=np.complex128)
        start = 32 * a + 16 * b + 8 * c
        initial[start : start + 2] = amplitudes
        expected = np.zeros_like(initial)
        output = c ^ (a & b)
        end = 16 * b + 8 * c + 4 * output
        expected[end : end + 2] = amplitudes
        actual = simulator.simulate(
            cirq.Circuit(emitted), qubit_order=order, initial_state=initial.copy()
        )
        native = forced_branch(
            torch.from_numpy(initial), body.instructions, (a, b, c, 0)
        )
        np.testing.assert_allclose(
            actual.final_state_vector, expected, atol=2e-12, rtol=0
        )
        np.testing.assert_allclose(
            actual.final_state_vector, native.numpy(), atol=2e-12, rtol=0
        )


def test_cirq_guarded_measurement_preserves_the_inactive_instrument(cirq: Any) -> None:
    from dataclasses import replace
    from itertools import product

    import torch

    from flagquantum.compiler.registers.register_binding import (
        BoundRegisterProgram,
        WorkspaceEvent,
    )
    from flagquantum.core.classical_dataflow import ClassicalXor
    from flagquantum.ecosystem.cirq._register_gate import RegisterCirqGate
    from tests.unit.measurement_checks import forced_branch

    effect = {"is_dynamic": True, "guarded_effect": True, "conditions": ((0, 1),)}
    body = CircuitIR(
        4,
        (
            Instruction(
                "measure", (0,), metadata={"is_dynamic": True, "classical_bit": 0}
            ),
            ClassicalXor((), 2, constant=True).instruction(),
            Instruction("measure", (1,), metadata={**effect, "classical_bit": 2}),
            Instruction("x", (2,), metadata={"conditions": ((2, 1),)}),
            Instruction("reset", (1,), metadata=effect),
        ),
    )
    signature = Signature.build(guard=1, data=1, answer=1)
    wires = {"guard": (0,), "data": (1,), "answer": (2,)}
    qids = tuple(cirq.LineQubit.range(4))
    placement = dict(zip(wires, qids[:3], strict=True))
    exported, _ = export_register_cirq(
        body,
        signature,
        wires,
        input_quregs=placement,
        feedback_key_prefix="guarded",
        guarded_measurement_work=3,
    )
    emitted = tuple(exported.circuit.all_operations())
    assert len(emitted) == 6 and exported.report.lossless
    assert emitted[1] == cirq.CNOT(qids[1], qids[3]).with_classical_controls(
        "guarded_0"
    )
    assert emitted[2:4] == (cirq.measure(qids[3], key="guarded_2"), cirq.reset(qids[3]))
    assert emitted[-1] == cirq.reset(qids[1]).with_classical_controls("guarded_0")
    assert {str(key) for key in cirq.control_keys(emitted[-2])} == {
        "guarded_0",
        "guarded_2",
    }
    with pytest.raises(CirqConversionError) as caught:
        export_register_cirq(body, signature, wires, feedback_key_prefix="guarded")
    assert any(
        "requires a clean work wire" in issue.message
        for issue in caught.value.report.issues
    )

    allocation = WorkspaceEvent(0, "allocate", (3,), False, instruction_offset=2)
    bound = BoundRegisterProgram(
        body,
        ((0,), (1,), (2,)),
        (allocation, WorkspaceEvent(1, "free", (3,), False, instruction_offset=3)),
    )
    with pytest.raises(ValueError, match="clean work lease"):
        RegisterCirqGate(
            replace(bound, events=(replace(allocation, dirty=True), *bound.events[1:])),
            signature,
            feedback_key_prefix="guarded",
            guarded_measurement_work=3,
        )
    gate = RegisterCirqGate(
        bound,
        signature,
        feedback_key_prefix="guarded",
        guarded_measurement_work=3,
    )
    wrapped = tuple(cirq.decompose_once(gate.on(*qids[:3])))
    (work,) = {q for operation in wrapped for q in operation.qubits} - set(qids[:3])
    assert (
        tuple(
            op.transform_qubits(lambda qubit: qids[3] if qubit == work else qubit)
            for op in wrapped
        )
        == emitted
    )

    spectator = cirq.NamedQubit("spectator")
    initial = np.zeros((2,) * 5, dtype=np.complex128)
    weights = np.arange(1, 9, dtype=np.complex128) + 1j * np.arange(8, 0, -1)
    initial[:, :, 0, 0, :] = (weights / np.linalg.norm(weights)).reshape(2, 2, 2)
    for history in product((0, 1), repeat=2):
        state = _cirq_feedback_branch(
            cirq, emitted, (*qids, spectator), initial, history
        )
        native = forced_branch(
            torch.from_numpy(initial.reshape(-1)), body.instructions, history
        )
        np.testing.assert_allclose(state, native.numpy(), atol=2e-12, rtol=0)
        assert not np.any(state.reshape(initial.shape)[:, :, :, 1, :])
        if history == (0, 0):
            np.testing.assert_allclose(
                state.reshape(initial.shape)[0, :, 1, 0, :],
                initial[0, :, 0, 0, :],
                atol=2e-12,
                rtol=0,
            )


@pytest.mark.ftqc_reference
def test_valid_modexp_register_export_preserves_source_output_and_feedback(
    cirq: Any, source_export: Any
) -> None:
    from collections import Counter

    from qualtran.bloqs.cryptography.rsa.rsa_mod_exp import ModExp
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    from flagquantum.algorithms.primitives.modular_exponentiation import (
        ModularExponentiation,
    )
    from flagquantum.dynamic import DynamicCircuit

    source = ModExp(2, 3, 1, 2)
    logical_source = source.decompose_bloq()
    original_ports = source_export.get_named_qubits(source.signature)
    with pytest.raises(ValueError, match="There is no Cirq equivalent"):
        logical_source.flatten().to_cirq_circuit_and_quregs(
            exponent=original_ports["exponent"],
            qubit_manager=cirq.GreedyQubitManager("source", maximize_reuse=True),
        )
    named_source = tuple(q for word in original_ports.values() for q in word.flat)
    outer = tuple(
        cirq.decompose_once(source_export.BloqAsCirqGate(source).on(*named_source))
    )
    assert tuple(op.qubits for op in outer) == (named_source[1:], named_source)

    program = ModularExponentiation(2, 3, 1, 2)
    work, slots = program.workspace_requirements()
    width = 3 + work
    ir = (
        DynamicCircuit(width)
        .append(
            program.make_on(
                exponent=(0,),
                target=(1, 2),
                workspace=tuple(range(3, width)),
                classical_bits=tuple(range(slots)),
            )
        )
        .to_ir()
    )
    assert (work, slots, width) == (13, 4, 16)
    exported, output_ports = export_register_cirq(
        ir,
        program.source_signature,
        {"exponent": (0,), "x": (1, 2)},
        input_quregs={"exponent": original_ports["exponent"]},
        output_quregs={"x": original_ports["x"]},
        feedback_key_prefix="modexp",
    )
    order = (
        *output_ports["exponent"].flat,
        *output_ports["x"].flat,
        *cirq.LineQubit.range(3, width),
    )
    assert tuple(order[:3]) == named_source and exported.report.lossless
    operations = tuple(exported.circuit.all_operations())
    assert len(operations) == len(ir.instructions) == 1155
    gates = {
        "x": cirq.X,
        "h": cirq.H,
        "s": cirq.S,
        "t": cirq.T,
        "tdg": cirq.T**-1,
        "cx": cirq.CNOT,
        "cz": cirq.CZ,
    }
    latest: dict[int, str] = {}
    for index, (native, exported_op) in enumerate(
        zip(ir.instructions, operations, strict=True)
    ):
        operands = tuple(order[wire] for wire in native.wires)
        if native.name == "measure":
            key = f"modexp_{index}"
            latest[native.metadata["classical_bit"]] = key
            expected_op = cirq.measure(*operands, key=key)
        elif native.name == "reset":
            expected_op = cirq.reset(*operands)
        else:
            expected_op = gates[native.name].on(*operands)
            if "conditions" in native.metadata:
                ((slot, value),) = native.metadata["conditions"]
                assert value == 1
                expected_op = expected_op.with_classical_controls(latest[slot])
        assert exported_op == expected_op

    counts = Counter(op.name for op in ir.instructions)
    cost = get_cost_value(source, QECGatesCost())
    assert counts["t"] + counts["tdg"] == cost.total_t_count(ts_per_cswap=7) == 206
    assert counts["measure"] == cost.measurement == 48
    assert get_cost_value(source, QubitCount()) == width
    initial = np.zeros(1 << (width + 1), dtype=np.complex128)
    expected = np.zeros_like(initial)
    weights = np.array([1, -2j, 3j, -4], dtype=np.complex128).reshape(2, 2) / np.sqrt(
        30
    )
    for exponent, amplitudes in enumerate(weights):
        start = exponent << width
        initial[start : start + 2] = amplitudes
        value = logical_source.on_classical_vals(exponent=exponent)["x"]
        end = start | (int(value) << (width - 2))
        expected[end : end + 2] = amplitudes
    actual = cirq.Simulator(dtype=np.complex128, seed=34).simulate(
        exported.circuit,
        qubit_order=(*order, cirq.NamedQubit("spectator")),
        initial_state=initial.copy(),
    )
    assert len(actual.measurements) == 48
    np.testing.assert_allclose(actual.final_state_vector, expected, atol=2e-12, rtol=0)


@pytest.mark.ftqc_reference
def test_register_gate_renders_frozen_shaped_wire_glyphs(
    cirq: Any, source_export: Any
) -> None:
    from qualtran import QAny, QBit
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature
    from qualtran.drawing import Circle, LarrowTextBox, ModPlus, RarrowTextBox, TextBox

    from flagquantum.drawer.register_score_data import SymbolClass
    from flagquantum.ecosystem.cirq._register_gate import RegisterCirqGate

    reference_signature = SourceSignature(
        (
            SourceRegister("ctrl", QBit(), shape=(2,)),
            SourceRegister("word", QAny(2), shape=(2,)),
            SourceRegister("target", QBit()),
            SourceRegister("phase", QBit()),
        )
    )

    class ReferenceGlyphs:
        signature = reference_signature

        def wire_symbol(self, register: Any, index: tuple[int, ...] = ()) -> Any:
            if register.name == "ctrl":
                return Circle(filled=index == (0,))
            if register.name == "word":
                return (
                    RarrowTextBox("incoming")
                    if index == (0,)
                    else LarrowTextBox("outgoing")
                )
            return ModPlus() if register.name == "target" else TextBox("phase")

    def glyph(
        register: Register, index: tuple[int, ...]
    ) -> tuple[SymbolClass, dict[str, str | int | bool]]:
        if register.name == "ctrl":
            return "Circle", {"filled": index == (0,)}
        if register.name == "word":
            if index == (0,):
                return "RarrowTextBox", {"text": "incoming"}
            return "LarrowTextBox", {"text": "outgoing"}
        if register.name == "target":
            return "ModPlus", {}
        return "TextBox", {"text": "phase"}

    bit = opaque_register_type(1)
    signature = Signature(
        (
            Register("ctrl", bit, shape=(2,)),
            Register("word", opaque_register_type(2), shape=(2,)),
            Register("target", bit),
            Register("phase", bit),
        )
    )
    gate = RegisterCirqGate(
        CircuitIR(8, (Instruction("x", (6,)),)), signature, wire_symbol=glyph
    )
    original = source_export.BloqAsCirqGate(ReferenceGlyphs())
    expected = (
        "@",
        "(0)",
        "incoming",
        "incoming",
        "outgoing",
        "outgoing",
        "X",
        "phase",
    )
    assert cirq.circuit_diagram_info(original).wire_symbols == expected
    assert cirq.circuit_diagram_info(gate).wire_symbols == expected
    assert cirq.circuit_diagram_info(gate**-1).wire_symbols == expected
    qids = tuple(cirq.LineQubit.range(8))
    assert tuple(cirq.decompose_once(gate.on(*qids))) == (cirq.X(qids[6]),)


@pytest.mark.ftqc_reference
def test_cirq_register_gate_uses_owned_value_identity(
    cirq: Any, source_export: Any
) -> None:
    import torch
    from qualtran.bloqs.basic_gates import Swap

    from flagquantum.ecosystem.cirq._register_gate import RegisterCirqGate

    source_a, source_b = (source_export.BloqAsCirqGate(Swap(1)) for _ in range(2))
    assert source_a == source_b and hash(source_a) == hash(source_b)
    signature = Signature.build(a=1, b=1)
    body = CircuitIR(2, (Instruction("swap", (0, 1)),))
    gate_a = RegisterCirqGate(body, signature)
    gate_b = RegisterCirqGate(
        CircuitIR.from_dict(body.to_dict()), Signature.build(a=1, b=1)
    )
    assert gate_a == gate_b and hash(gate_a) == hash(gate_b)
    assert len({gate_a, gate_b}) == 1
    assert gate_a != RegisterCirqGate(body, Signature.build(data=2))
    assert gate_a != RegisterCirqGate(body, signature, feedback_key_prefix="key")
    assert gate_a != RegisterCirqGate(
        body,
        signature,
        wire_symbol=lambda register, index: ("TextBox", {"text": "custom"}),
    )

    raw = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
    matrix_body = CircuitIR(1, (Instruction("matrix", (0,), matrix=raw),))
    matrix_signature = Signature.build(q=1)
    matrix_gate = RegisterCirqGate(matrix_body, matrix_signature)
    identical = RegisterCirqGate(
        CircuitIR.from_dict(matrix_body.to_dict()), matrix_signature
    )
    before = hash(matrix_gate)
    raw.copy_(torch.eye(2, dtype=torch.complex128))
    assert matrix_gate == identical and hash(matrix_gate) == before
    assert matrix_gate != RegisterCirqGate(matrix_body, matrix_signature)
    np.testing.assert_array_equal(cirq.unitary(matrix_gate), cirq.unitary(cirq.X))

    opaque = CircuitIR(2, body.instructions, metadata={"opaque": object()})
    opaque_gate = RegisterCirqGate(opaque, signature)
    assert opaque_gate == opaque_gate
    assert opaque_gate != RegisterCirqGate(opaque, signature)
    assert [
        op.gate for op in cirq.decompose_once(opaque_gate.on(*cirq.LineQubit.range(2)))
    ] == [cirq.SWAP]


def test_cirq_register_value_identity_includes_workspace_instruction_offsets(
    cirq: Any,
) -> None:
    from dataclasses import replace

    from flagquantum.compiler.registers.register_binding import (
        BoundRegisterProgram,
        WorkspaceEvent,
    )
    from flagquantum.ecosystem.cirq._register_gate import RegisterCirqGate

    body = CircuitIR(2, (Instruction("x", (0,)), Instruction("x", (0,))))
    allocate = WorkspaceEvent(0, "allocate", (1,), False, instruction_offset=0)
    free = WorkspaceEvent(3, "free", (1,), False, instruction_offset=2)
    first = BoundRegisterProgram(body, ((0,),), (allocate, free))
    later = replace(first, events=(replace(allocate, instruction_offset=1), free))
    signature = Signature.build(q=1)
    gate = RegisterCirqGate(first, signature)
    assert gate == RegisterCirqGate(replace(first), signature)
    assert gate != RegisterCirqGate(later, signature)


@pytest.mark.ftqc_reference
def test_cirq_dynamic_workspace_unitary_matches_frozen_cleanup_tensor(
    cirq: Any, source_export: Any
) -> None:
    from dataclasses import replace

    from qualtran.cirq_interop._bloq_to_cirq_test import ComputeUncompute

    from flagquantum.compiler.registers.register_binding import (
        BoundRegisterProgram,
        WorkspaceEvent,
    )
    from flagquantum.ecosystem.cirq._register_gate import RegisterCirqGate

    compute = multi_and_ir((0, 1), (1, 1), (2,))
    erase = multi_and_ir((0, 1), (1, 1), (2,), classical_bits=(0,), uncompute=True)
    body = CircuitIR(3, compute + erase)
    bound = BoundRegisterProgram(
        body,
        ((0,), (1,)),
        (
            WorkspaceEvent(0, "allocate", (2,), False, instruction_offset=0),
            WorkspaceEvent(
                len(body.instructions) + 1,
                "free",
                (2,),
                False,
                instruction_offset=len(body.instructions),
            ),
        ),
    )
    signature = Signature.build(a=1, b=1)
    gate = RegisterCirqGate(bound, signature, feedback_key_prefix="cleanup")
    expected = cirq.unitary(source_export.BloqAsCirqGate(ComputeUncompute()))
    np.testing.assert_allclose(cirq.unitary(gate), expected, atol=2e-12, rtol=0)

    measure_logical = Instruction(
        "measure", (0,), metadata={"is_dynamic": True, "classical_bit": 0}
    )
    negative = RegisterCirqGate(
        replace(bound, circuit=CircuitIR(3, (measure_logical,))),
        signature,
        feedback_key_prefix="measurement",
    )
    assert cirq.unitary(negative, default=None) is None

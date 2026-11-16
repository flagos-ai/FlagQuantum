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
    with pytest.raises(CirqConversionError) as caught:
        export_register_cirq(
            native,
            signature,
            {"ctrl": (0, 1, 2, 3), "junk": (4, 5), "target": (6,)},
            input_quregs=inputs,
        )
    issues = caught.value.report.issues
    assert {issue.code for issue in issues} == {"unsupported_operation"}
    assert all(issue.operation_name in {"sdg", "tdg"} for issue in issues)


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

"""Complete register-gate tutorial and original gate scenarios use native owners."""

import builtins
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from hashlib import sha1
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import torch

try:
    import cirq
except ModuleNotFoundError:
    pytest.importorskip("cirq")

try:
    import numpy as np
except ModuleNotFoundError:
    pytest.importorskip("numpy")

from examples.algorithms.register_gates import RegisterControlledTargets
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.component_control import quantum_control_signature
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.ecosystem.cirq._register_control import control_register_operation
from flagquantum.ecosystem.cirq._register_export import _named_qubits, _register_shape
from flagquantum.ecosystem.cirq._register_gate import RegisterCirqGate
from flagquantum.ecosystem.cirq._register_import import (
    import_cirq_register_gate,
    import_cirq_register_optree,
)
from flagquantum.simulation.unitary import get_unitary
from tests.unit.register_source_graph import (
    SourcePorts,
    SourceResult,
    assert_source_wire_streams,
    walk_source_graph,
)
from tests.unit.test_register_wiring_notebook import execute

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT.parent / "ftqc-reference-env/qualtran/_infra"
NATIVE = ROOT / "examples/notebooks/register_gates.ipynb"


@pytest.fixture(scope="module")
def original() -> dict[str, Any]:
    pytest.importorskip("qualtran")
    return execute(SOURCE / "gate_with_registers.ipynb")


def test_complete_source_identity_and_notebook_replacement(
    original: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, expected in (
        ("gate_with_registers.py", "df21c8cc00ae7cac7a8bfcd4d769f76daac5515f"),
        ("gate_with_registers_test.py", "31fd6f8635cec0b18b1c1f3487d4db9c8fabacf3"),
        ("gate_with_registers.ipynb", "108c73c15f07476d976e0777c4b916bb466670f7"),
    ):
        raw = (SOURCE / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == expected
    source = json.loads((SOURCE / "gate_with_registers.ipynb").read_text())
    native = json.loads(NATIVE.read_text())
    assert len(source["cells"]) == len(native["cells"]) == 17
    assert [cell["cell_type"] for cell in source["cells"]] == [
        cell["cell_type"] for cell in native["cells"]
    ]
    assert [cell["metadata"]["frozen_source_cell"] for cell in native["cells"]] == list(
        range(17)
    )
    real_import = builtins.__import__

    def native_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] | None = (),
        level: int = 0,
    ) -> ModuleType:
        if name.split(".")[0] in {"qualtran", "cirq", "kickmix"}:
            raise AssertionError("The native register tutorial must use native owners")
        return real_import(name, globals, locals, fromlist, level)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", native_import)
        namespace = execute(NATIVE)
    assert (
        original["gate"]._num_qubits_() == namespace["gate"].signature.n_qubits() == 5
    )
    assert namespace["observed_counts"] == {"cx": 6}
    assert verify_program_record(namespace["program"]) == ()


@pytest.mark.parametrize("control,target", [((0, 1), (2, 3, 4)), ((3, 0), (4, 2, 1))])
def test_complete_tutorial_stream_amplitudes_resources_and_append(
    original: dict[str, Any], control: tuple[int, ...], target: tuple[int, ...]
) -> None:
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    operation = RegisterControlledTargets(control, target)
    bound = operation.bind()
    qubits = cirq.LineQubit.range(5)
    source = original["gate"].on_registers(
        control=[qubits[i] for i in control], target=[qubits[i] for i in target]
    )
    decomposed = tuple(cirq.flatten_to_ops(cirq.decompose_once(source)))
    wire_index: dict[cirq.Qid, int] = {qubit: i for i, qubit in enumerate(qubits)}
    assert bound.circuit.instructions == tuple(
        Instruction("cx", tuple(wire_index[q] for q in gate.qubits))
        for gate in decomposed
    )
    assert bound.outputs == (control, target) and bound.events == ()
    assert Counter(g.name for g in bound.circuit.instructions) == {"cx": 6}
    model = get_cost_value(original["gate"], QECGatesCost())
    assert model.clifford == 6 and model.measurement == 0 and model.total_t_count() == 0
    from qualtran.bloqs.basic_gates import CNOT

    def leaf(
        bloq: Any, inputs: SourcePorts, desired: SourcePorts
    ) -> SourceResult | None:
        if isinstance(bloq, CNOT):
            return (Instruction("cx", (*inputs["ctrl"], *inputs["target"])),), inputs
        return None

    ports = {"control": control, "target": target}
    source_graph = original["gate"].decompose_bloq()
    source_instructions, source_outputs = walk_source_graph(
        source_graph, ports, ports, leaf
    )
    assert source_outputs == ports
    assert_source_wire_streams(bound.circuit.instructions, source_instructions)
    assert Counter(
        type(instance.bloq).__name__ for instance in source_graph.bloq_instances
    ) == {
        "Split": 2,
        "CNOT": 6,
        "Join": 2,
    }
    assert len(operation.program().functions[0].blocks[0].operations) == 10
    matrix = cirq.unitary(cirq.Circuit(source))
    np.testing.assert_allclose(
        get_unitary(bound.circuit, dtype=torch.complex128).numpy(), matrix, atol=2e-14
    )
    # A sixth, entangled spectator makes relative phases observable through append.
    state = torch.complex(torch.arange(1, 65).double(), torch.arange(64).double())
    state /= state.norm()
    circuit = Circuit(6, inputs=state, dtype=torch.complex128).append(operation)
    expected = torch.kron(torch.as_tensor(matrix), torch.eye(2)) @ state
    torch.testing.assert_close(circuit.state().flatten(), expected, atol=2e-14, rtol=0)
    circuit.append(operation)
    torch.testing.assert_close(circuit.state().flatten(), state, atol=2e-14, rtol=0)


@pytest.mark.parametrize("atomic", [False, True])
def test_original_register_gate_decomposition_and_unitary_tensor_fallback(
    atomic: bool,
) -> None:
    from qualtran._infra import gate_with_registers_test as source

    original = source._TestGateAtomic() if atomic else source._TestGate()
    width = cirq.num_qubits(original)
    module: ProgramRecord | None
    inputs: tuple[tuple[int, ...], ...]
    if atomic:
        module = import_cirq_register_optree(original.on(*cirq.LineQubit.range(width)))
        inputs = tuple((i,) for i in range(width))
    else:
        qubits = cirq.LineQubit.range(width)
        ports = {"r1": qubits[:5], "r2": qubits[6:], "r3": qubits[5]}
        module = import_cirq_register_gate(
            original.on_registers(**ports),
            signature=Signature.build(r1=5, r2=2, r3=1),
            in_quregs=ports,
            out_quregs=ports,
        )
        inputs = (tuple(range(5)), (5, 6), (7,))
    assert module is not None
    bound = bind_register_program(module, inputs)
    assert bound.events == ()
    expected = cirq.unitary(original)
    np.testing.assert_allclose(
        get_unitary(bound.circuit, dtype=torch.complex128).numpy(), expected, atol=2e-14
    )
    np.testing.assert_allclose(original.tensor_contract(), expected, atol=2e-14)
    counts = Counter(op.name for op in bound.circuit.instructions)
    assert counts == ({"any": 1} if atomic else {"h": 5, "x": 3})
    # Tensor-only atomic storage and the decomposed Clifford stream have distinct granularity.
    signature = Signature.build(q=width)
    exported = RegisterCirqGate(bound.circuit, signature)
    np.testing.assert_allclose(cirq.unitary(exported), expected, atol=2e-14)


def test_original_directional_decomposition_lifetimes_tensor_and_control_metadata() -> (
    None
):
    from qualtran._infra.gate_with_registers_test import BloqWithDecompose
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    source = BloqWithDecompose()
    bit = quantum_bit_type()
    signature = Signature(
        (
            Register("l", bit, side=RegisterSide.LEFT),
            Register("t", bit),
            Register("r", bit, side=RegisterSide.RIGHT),
        )
    )
    module = import_cirq_register_gate(source, signature=signature)
    assert module is not None
    bound = bind_register_program(
        module,
        ((0,), (1,)),
        clean=(2,),
        input_ownership=((0, False),),
        allow_workspace_outputs=True,
    )
    assert bound.outputs == ((1,), (0,))
    assert bound.circuit.instructions == (
        Instruction("x", (0,)),
        Instruction("y", (1,)),
        Instruction("z", (0,)),
    )
    assert [(event.kind, event.wires) for event in bound.events] == [
        ("free", (0,)),
        ("allocate", (0,)),
    ]
    # The released LEFT word is zero after X and is reused as the clean RIGHT port.
    state = torch.tensor([1 + 2j, 3 - 4j, -2 + 1j, 2j], dtype=torch.complex128)
    state /= state.norm()
    initial = torch.zeros((2, 2, 2, 2), dtype=torch.complex128)
    initial[1, :, 0, :] = state.reshape(2, 2)
    actual = Circuit(4, inputs=initial.flatten(), dtype=torch.complex128)
    actual.compose(bound.circuit)
    logical = torch.zeros((2, 2, 2), dtype=torch.complex128)
    logical[1] = state.reshape(2, 2)
    reference = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
    expected = (reference @ logical.reshape(4, 2)).reshape(2, 2, 2)
    result = actual.state().reshape(2, 2, 2, 2).permute(1, 0, 3, 2)
    torch.testing.assert_close(result[..., 0], expected, atol=2e-14, rtol=0)
    assert torch.count_nonzero(result[..., 1]) == 0
    model = get_cost_value(source, QECGatesCost())
    assert model.clifford == 3 and model.measurement == 0
    qids = cirq.NamedQubit.range(3, prefix="q")
    native = RegisterCirqGate(bound, signature, gate_label="BloqWithDecompose")
    assert cirq.Circuit(cirq.decompose_once(native.on(*qids))) == cirq.Circuit(
        cirq.decompose_once(source.on(*qids))
    )
    source_controlled = source.controlled(control_values=[0])
    controlled = quantum_control_signature(signature, ((bit, ()),))
    assert [(r.name, r.side.value) for r in controlled] == [
        (r.name, r.side.value) for r in source_controlled.signature
    ]


def test_original_named_qubits_diagram_controls_and_integer_powers(
    original: dict[str, Any],
) -> None:
    from qualtran import CtrlSpec
    from qualtran._infra.gate_with_registers import get_named_qubits

    gate = original["gate"]
    operation = RegisterControlledTargets((0, 1), (2, 3, 4))
    bound = operation.bind()
    native = RegisterCirqGate(bound.circuit, operation.signature, gate_label="MyGate")
    ports = {
        reg.name: _named_qubits(cirq, reg, _register_shape(reg))
        for reg in operation.signature
    }
    source_ports = get_named_qubits(gate.signature)
    for name in ports:
        np.testing.assert_array_equal(ports[name], source_ports[name])
    qubits = tuple(q for reg in operation.signature for q in ports[reg.name].flat)
    assert (
        cirq.Circuit(native.on(*qubits)).to_text_diagram()
        == cirq.Circuit(gate.on_registers(**source_ports)).to_text_diagram()
    )
    for power in (-10, -1, 1, 10):
        np.testing.assert_allclose(
            cirq.unitary(native**power), cirq.unitary(gate**power), atol=3e-14
        )
        assert cirq.circuit_diagram_info(native**power).wire_symbols[0] == "MyGate"
    with pytest.raises(ValueError):
        _ = native**0
    for value in (0, 1):
        source_controlled = gate.controlled(num_controls=1, control_values=[value])
        native_controlled = native.controlled(num_controls=1, control_values=[value])
        np.testing.assert_allclose(
            cirq.unitary(native_controlled), cirq.unitary(source_controlled), atol=3e-14
        )
        typed, outputs = control_register_operation(
            native.on(*qubits),
            ports,
            (Register("ctrl", quantum_bit_type()),),
            (np.asarray(value),),
            {"ctrl": np.asarray([cirq.NamedQubit("ctrl")], dtype=object)},
        )
        assert outputs["control"] is ports["control"]
        np.testing.assert_allclose(
            cirq.unitary(typed),
            cirq.unitary(gate.controlled(ctrl_spec=CtrlSpec(cvs=value))),
            atol=3e-14,
        )
    with pytest.raises(ValueError):
        native.controlled(num_controls=2, control_values=[1])
    assert native != RegisterCirqGate(bound.circuit, operation.signature)
    assert hash(native) == hash(
        RegisterCirqGate(bound.circuit, operation.signature, gate_label="MyGate")
    )


@pytest.mark.parametrize("widths", [(1, 1), (1, 3), (2, 1)])
def test_register_example_preserves_single_bit_port_types(
    widths: tuple[int, int],
) -> None:
    a, b = widths
    operation = RegisterControlledTargets(tuple(range(a)), tuple(range(a, a + b)))
    module = operation.program()
    assert verify_program_record(module) == ()
    bound = operation.bind()
    assert len(bound.circuit.instructions) == a * b and bound.events == ()


def test_gate_title_validation_and_unchanged_default_labels() -> None:

    body = CircuitIR(1, (Instruction("x", (0,)),))
    assert cirq.circuit_diagram_info(
        RegisterCirqGate(body, Signature.build(q=1))
    ).wire_symbols == ("q",)
    with pytest.raises(ValueError, match="label"):
        RegisterCirqGate(body, Signature.build(q=1), gate_label="")
    with pytest.raises(ValueError, match="label"):
        RegisterCirqGate(body, Signature(()), gate_label="title")

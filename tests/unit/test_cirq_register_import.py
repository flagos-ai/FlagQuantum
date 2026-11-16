"""Verify register-aware Cirq optree import and native workspace lifetimes."""

from hashlib import sha1
from pathlib import Path

import numpy as np
import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler import estimate_resources
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import Instruction
from flagquantum.core.ir.program.register_types import (
    opaque_register_type,
    quantum_bit_type,
)
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.ecosystem.cirq._register_import import import_cirq_register_optree

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_cirq_to_native_source_family_hashes() -> None:
    root = (
        Path(__file__).resolve().parents[3] / "ftqc-reference-env/qualtran/cirq_interop"
    )
    for name, expected in (
        ("_cirq_to_bloq.py", "29c7a89a164836eb2e9af2527c2b78294b952e0e"),
        ("_cirq_to_bloq_test.py", "d2c49ee57cc2680b5b545c6673db63b96333184c"),
    ):
        path = root / name
        if not path.is_file():
            pytest.skip("Frozen Cirq conversion files are not installed")
        data = path.read_bytes()
        assert sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == expected


def test_shaped_ports_preserve_gate_operands_and_release_clean_work() -> None:
    cirq = pytest.importorskip("cirq")
    source = pytest.importorskip("qualtran.cirq_interop._cirq_to_bloq")
    from qualtran import QAny
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature
    from qualtran.bloqs.bookkeeping import Allocate, Free

    a, b, c, d = cirq.LineQubit.range(4)
    work = cirq.NamedQubit("work")
    circuit = cirq.Circuit(
        cirq.Moment(cirq.H(b)),
        cirq.Moment(cirq.CNOT(a, work)),
        cirq.Moment(cirq.T(d)),
        cirq.Moment(cirq.CNOT(a, work)),
        cirq.Moment(cirq.CNOT(b, c)),
    )
    quregs = {"x": np.array(((a, b), (c, d)), dtype=object)}
    signature = Signature((Register("x", opaque_register_type(2), shape=(2,)),))
    source_signature = SourceSignature([SourceRegister("x", QAny(2), shape=(2,))])
    ref = source.cirq_optree_to_cbloq(
        circuit, signature=source_signature, in_quregs=quregs, out_quregs=quregs
    )
    reference_calls = [instance.bloq for instance in ref.bloq_instances]
    assert sum(isinstance(gate, Allocate) for gate in reference_calls) == 1
    assert sum(isinstance(gate, Free) for gate in reference_calls) == 1
    module = import_cirq_register_optree(
        circuit, signature=signature, in_quregs=quregs, out_quregs=quregs
    )
    assert module.functions[0].registers == tuple(signature)
    bound = bind_register_program(module, ((0, 1), (2, 3)), clean=(4,))
    assert [(event.kind, event.wires, event.dirty) for event in bound.events] == [
        ("allocate", (4,), False),
        ("free", (4,), False),
    ]
    assert bound.outputs == ((0, 1), (2, 3))
    reference_gates = (
        Instruction("h", (1,)),
        Instruction("cx", (0, 4)),
        Instruction("t", (3,)),
        Instruction("cx", (0, 4)),
        Instruction("cx", (1, 2)),
    )
    for wire in range(5):
        assert tuple(i for i in bound.circuit.instructions if wire in i.wires) == tuple(
            i for i in reference_gates if wire in i.wires
        )
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    assert get_cost_value(ref, QECGatesCost()).asdict() == {"t": 1, "clifford": 4}
    cost = estimate_resources(bound.circuit)
    assert cost.operation_counts == {"h": 1, "cx": 3, "t": 1}
    source_dense = torch.as_tensor(
        circuit.unitary(qubit_order=(a, b, c, d, work)), dtype=torch.complex128
    )
    state = torch.zeros(32, dtype=torch.complex128)
    state[::2] = torch.tensor(
        [complex(i + 1, i - 2) for i in range(16)], dtype=torch.complex128
    )
    state /= state.norm()
    expected = source_dense @ state
    assert torch.count_nonzero(expected[1::2]) == 0
    native = Circuit(5, inputs=state, dtype=torch.complex128)
    native.append(FieldOperation(5, bound.circuit.instructions))
    torch.testing.assert_close(native.state().flatten(), expected, rtol=0, atol=2e-12)


def test_default_ports_and_sided_output_use_the_owned_binding() -> None:
    cirq = pytest.importorskip("cirq")
    first, second = cirq.LineQubit.range(2)
    default = import_cirq_register_optree([cirq.X(second), cirq.CNOT(second, first)])
    assert default.functions[0].registers == (
        Register("qubits", quantum_bit_type(), shape=(2,)),
    )
    default_bound = bind_register_program(default, ((0,), (1,)))
    assert default_bound.outputs == ((0,), (1,))
    assert default_bound.circuit.instructions == (
        Instruction("x", (1,)),
        Instruction("cx", (1, 0)),
    )

    signature = Signature(
        (
            Register("input", quantum_bit_type(), side=RegisterSide.LEFT),
            Register("output", quantum_bit_type(), side=RegisterSide.RIGHT),
        )
    )
    module = import_cirq_register_optree(
        [cirq.X(first)],
        signature=signature,
        in_quregs={"input": [first]},
        out_quregs={"output": [first]},
    )
    assert module.functions[0].registers == tuple(signature)
    bound = bind_register_program(module, ((0,),))
    assert bound.outputs == ((0,),) and not bound.events
    assert bound.circuit.instructions == (Instruction("x", (0,)),)
    with pytest.raises(ValueError, match="requires qureg shape"):
        import_cirq_register_optree(
            [],
            signature=signature,
            in_quregs={"input": [[first]]},
            out_quregs={"output": [first]},
        )


def test_right_only_port_transfers_newly_allocated_output() -> None:
    cirq = pytest.importorskip("cirq")
    output = cirq.NamedQubit("created")
    signature = Signature(
        (Register("out", quantum_bit_type(), side=RegisterSide.RIGHT),)
    )
    module = import_cirq_register_optree(
        [cirq.X(output)],
        signature=signature,
        in_quregs={},
        out_quregs={"out": [output]},
    )
    bound = bind_register_program(module, (), clean=(0,), allow_workspace_outputs=True)
    assert bound.outputs == ((0,),)
    assert bound.circuit.instructions == (Instruction("x", (0,)),)
    assert [(event.kind, event.wires) for event in bound.events] == [("allocate", (0,))]


def test_concrete_global_phase_uses_an_existing_wire() -> None:
    cirq = pytest.importorskip("cirq")
    source = pytest.importorskip("qualtran.cirq_interop._cirq_to_bloq")
    wire = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.H(wire), cirq.global_phase_operation(-1j), cirq.T(wire))
    reference = source.cirq_optree_to_cbloq(circuit).tensor_contract()
    module = import_cirq_register_optree(circuit)
    bound = bind_register_program(module, ((0,),))
    assert not bound.events
    assert [instruction.name for instruction in bound.circuit.instructions] == [
        "h",
        "x",
        "phase",
        "x",
        "phase",
        "t",
    ]
    state = torch.tensor([1j, 2 - 1j], dtype=torch.complex128)
    state /= state.norm()
    expected = torch.as_tensor(reference, dtype=torch.complex128) @ state
    native = Circuit(1, inputs=state, dtype=torch.complex128)
    native.append(FieldOperation(1, bound.circuit.instructions))
    torch.testing.assert_close(native.state().flatten(), expected, rtol=0, atol=2e-12)
    with pytest.raises(ValueError, match="live wire"):
        import_cirq_register_optree(cirq.global_phase_operation(-1j))

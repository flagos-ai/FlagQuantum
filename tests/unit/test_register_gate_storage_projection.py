"""Raw SDK storage projection reuses quantum import without classical aliasing."""

from collections import Counter
from math import prod
from typing import Any

import pytest
import torch

try:
    import cirq
    import numpy as np
    from qualtran._infra import data_types as q
    from qualtran._infra.gate_with_registers import GateWithRegisters
    from qualtran._infra.registers import Register as SourceRegister
    from qualtran._infra.registers import Signature as SourceSignature
except ModuleNotFoundError:
    pytest.importorskip("numpy")
    pytest.importorskip("cirq")
    pytest.importorskip("qualtran")

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import Instruction
from flagquantum.core.ir.program.register_types import (
    classical_bit_type,
    classical_fixed_point_register_type,
    classical_montgomery_register_type,
    classical_ones_complement_register_type,
    classical_signed_register_type,
    classical_unsigned_register_type,
    quantum_bit_type,
)
from flagquantum.core.ir.program.registers import Register, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.ecosystem.cirq._register_import import import_cirq_register_gate
from flagquantum.simulation.finite_field_encoding import FiniteFieldEncoding
from flagquantum.simulation.finite_field_polynomial_encoding import (
    FiniteFieldPolynomialEncoding,
)
from flagquantum.simulation.unitary import get_unitary
from tests.unit.register_source_graph import assert_source_wire_streams

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
KINDS = (
    "bit",
    "uint",
    "signed",
    "ones_complement",
    "fixed_point",
    "montgomery",
    "field",
    "polynomial",
)


def storage_types(kind: str) -> tuple[Any, RegisterType]:
    if kind in {"field", "polynomial"}:
        codec = FiniteFieldEncoding(3, 1)
        source = q.CGF(3, 1, codec.irreducible_poly)
        if kind == "polynomial":
            return (
                q.CGFPoly(1, q.QGF(3, 1, codec.irreducible_poly)),
                FiniteFieldPolynomialEncoding(1, codec).register_type(classical=True),
            )
        return source, codec.register_type(classical=True)
    return {
        "bit": (q.CBit(), classical_bit_type()),
        "uint": (q.CUInt(2), classical_unsigned_register_type(2)),
        "signed": (q.CInt(2), classical_signed_register_type(2)),
        "ones_complement": (
            q.CIntOnesComp(2),
            classical_ones_complement_register_type(2),
        ),
        "fixed_point": (
            q.CFxp(2, 1, signed=True),
            classical_fixed_point_register_type(2, 1, True),
        ),
        "montgomery": (
            q.CMontgomeryUInt(2, 3),
            classical_montgomery_register_type(2, 3),
        ),
    }[kind]


class StorageProjectionGate(GateWithRegisters):
    """Expose a classical-labelled source port through its raw Cirq qubits."""

    def __init__(self, dtype: Any, shape: tuple[int, ...]) -> None:
        self.dtype = dtype
        self.shape = shape

    @property
    def signature(self) -> SourceSignature:
        return SourceSignature((SourceRegister("c", self.dtype, self.shape),))

    def decompose_from_registers(self, *, context: Any, **quregs: Any) -> Any:
        wires = tuple(quregs["c"].flat)
        if wires:
            yield cirq.H(wires[0])
            yield cirq.T(wires[-1])
            if len(wires) > 1:
                yield cirq.CNOT(wires[0], wires[-1])


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("shape", [(), (2,), (0, 2)])
def test_existing_raw_import_complete_operator_stream_append_and_no_classical_slots(
    kind: str, shape: tuple[int, ...]
) -> None:
    source_type, native_type = storage_types(kind)
    source = StorageProjectionGate(source_type, shape)
    width = int(source_type.num_bits)
    count = width * prod(shape)
    source_wires = np.array(cirq.LineQubit.range(count), dtype=object).reshape(
        (*shape, width)
    )
    operation = source.on_registers(c=source_wires)
    original = cirq.Circuit(cirq.decompose_once(operation))
    module = import_cirq_register_gate(operation)
    assert module is not None
    raw_ports = (
        ()
        if not count
        else (Register("q", quantum_bit_type(), (count,) if count > 1 else ()),)
    )
    assert module.functions[0].registers == raw_ports
    bound = bind_register_program(
        module, tuple((i,) for i in range(count)), borrowed=(count,)
    )
    assert bound.outputs == tuple((i,) for i in range(count))
    assert bound.events == () and bound.classical_outputs == ()
    expected_gates = (
        ()
        if not count
        else (
            Instruction("h", (0,)),
            Instruction("t", (count - 1,)),
            *((Instruction("cx", (0, count - 1)),) if count > 1 else ()),
        )
    )
    assert_source_wire_streams(bound.circuit.instructions, expected_gates)
    matrix = original.unitary(qubit_order=tuple(source_wires.flat))
    torch.testing.assert_close(
        get_unitary(bound.circuit, dtype=torch.complex128),
        torch.kron(
            torch.as_tensor(matrix, dtype=torch.complex128),
            torch.eye(2, dtype=torch.complex128),
        ),
        atol=4e-14,
        rtol=0,
    )
    state = torch.complex(
        torch.arange(1, (1 << (count + 1)) + 1).double(),
        torch.arange(1 << (count + 1)).double(),
    )
    state /= state.norm()
    circuit = Circuit(count + 1, inputs=state, dtype=torch.complex128)
    circuit.append(FieldOperation(count + 1, bound.circuit.instructions))
    expected = (
        torch.as_tensor(matrix, dtype=torch.complex128) @ state.reshape(1 << count, 2)
    ).flatten()
    torch.testing.assert_close(circuit.state().flatten(), expected, atol=4e-14, rtol=0)
    assert Counter(g.name for g in bound.circuit.instructions) == Counter(
        g.name for g in expected_gates
    )
    # A classical owned port remains a distinct contract from this raw SDK view.
    with pytest.raises(ValueError, match="concrete quantum register"):
        import_cirq_register_gate(
            operation, signature=Signature((Register("c", native_type, shape),))
        )

"""Check the remaining pinned X-basis source fixtures at native entry points."""

from collections import Counter
from dataclasses import asdict
from typing import Any, Literal

import numpy as np
import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler._component_trace.compound_control_resource import (
    compound_control_key,
)
from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost
from flagquantum.compiler._component_trace.resource_generalization import CliffordCall
from flagquantum.compiler.registers.control.control_predicate import ControlRegister
from flagquantum.compiler.registers.primitives.basis_state import basis_state_ir
from flagquantum.compiler.registers.primitives.pauli_gate import pauli_gate_ir
from flagquantum.core.ir import Instruction
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.ecosystem.cirq import to_cirq
from flagquantum.ecosystem.pennylane import from_pennylane, to_pennylane
from flagquantum.simulation.classical_pauli_values import x_classical_value
from flagquantum.simulation.classical_truth_table import (
    format_classical_truth_table,
    register_classical_truth_table,
)
from flagquantum.simulation.tensor_network.entrypoints import (
    build_tensor_network,
    tensor_network_basis_effect,
    tensor_network_basis_measurement,
)
from flagquantum.simulation.unitary import get_unitary
from tests.unit.register_source_graph import (
    SourcePorts,
    SourceResult,
    source_control_predicate_leaf,
    walk_source_graph,
)
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("bit", [0, 1])
def test_original_x_basis_preparation_followed_by_x(bit: int) -> None:
    import cirq
    from qualtran import BloqBuilder
    from qualtran.bloqs.basic_gates import MinusState, PlusState, XGate

    source = BloqBuilder()
    q = source.add((MinusState if bit else PlusState)())
    q = source.add(XGate(), q=q)
    composite = source.finalize(q=q)
    original = cirq.decompose(
        composite.to_cirq_circuit(), keep=lambda op: op.gate in (cirq.H, cirq.X)
    )
    circuit = Circuit(1, dtype=torch.complex128)
    circuit.append(
        FieldOperation(1, (*basis_state_ir("x", bit, 0), *pauli_gate_ir("x", 0)))
    )
    exported = to_cirq(circuit.to_ir())
    assert [op.gate for op in original] == [op.gate for op in exported.all_operations()]
    torch.testing.assert_close(
        circuit.state().flatten(),
        torch.as_tensor(composite.tensor_contract()),
        rtol=0,
        atol=2e-12,
    )


def test_original_plus_overlap_and_x_truth_table() -> None:
    from qualtran import BloqBuilder
    from qualtran.bloqs.basic_gates import PlusEffect, PlusState, XGate
    from qualtran.simulation.classical_sim import (
        format_classical_truth_table as format_source,
    )
    from qualtran.simulation.classical_sim import get_classical_truth_table

    source = BloqBuilder()
    source.add(PlusEffect(), q=source.add(PlusState()))
    circuit = Circuit(1, dtype=torch.complex128)
    circuit.append(FieldOperation(1, basis_state_ir("x", 0, 0)))
    plan = build_tensor_network(circuit, dtype=torch.complex128)
    overlap = tensor_network_basis_effect(plan, {0: ("x", 0)}).reshape(())
    torch.testing.assert_close(
        overlap,
        torch.as_tensor(source.finalize().tensor_contract()).reshape(()),
        rtol=0,
        atol=2e-12,
    )

    def evaluate(values: dict[str, object]) -> tuple[object]:
        value = values["q"]
        assert isinstance(value, (int, np.integer))
        return (x_classical_value(value),)

    table = register_classical_truth_table(Signature.build(q=1), evaluate)
    assert format_classical_truth_table(*table) == format_source(
        *get_classical_truth_table(XGate())
    )


def test_original_x_pennylane_matrix_and_native_roundtrip() -> None:
    from qualtran.bloqs.basic_gates import XGate

    qml = pytest.importorskip("pennylane")
    circuit = Circuit(1, dtype=torch.complex128)
    circuit.append(FieldOperation(1, pauli_gate_ir("x", 0)))
    exported = to_pennylane(circuit.to_ir())
    (op,) = exported.operations
    assert op == XGate().as_pl_op(wires=[0])
    assert from_pennylane(exported).instructions == circuit.to_ir().instructions
    torch.testing.assert_close(
        get_unitary(circuit),
        torch.as_tensor(qml.matrix(op), dtype=torch.complex128),
        rtol=0,
        atol=2e-12,
    )


def test_original_eight_bit_x_kept_children_and_physical_feedback() -> None:
    from qualtran import CtrlSpec, QUInt
    from qualtran.bloqs.basic_gates import CNOT, XGate
    from qualtran.bloqs.mcmt import And
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    source = XGate().controlled(CtrlSpec(qdtypes=QUInt(8), cvs=1))
    word = tuple(reversed(range(8)))
    work, slots = tuple(range(9, 16)), tuple(range(4, 11))
    values = ControlRegister(8, (1,)).bits()
    actual = pauli_gate_ir(
        "x", 8, controls=word, values=values, workspace=work, classical_bits=slots
    )

    def leaf(
        bloq: Any, inputs: SourcePorts, desired: SourcePorts
    ) -> SourceResult | None:
        if isinstance(bloq, XGate):
            return (Instruction("x", inputs["q"]),), inputs
        if isinstance(bloq, CNOT):
            return (Instruction("cx", (*inputs["ctrl"], *inputs["target"])),), inputs
        return source_control_predicate_leaf(bloq, inputs, work, slots)

    ports = {source.signature[0].name: word, "q": (8,)}
    reference, outputs = walk_source_graph(source, ports, ports, leaf)
    assert outputs == ports
    assert_source(actual, reference)
    _, counts = source.call_graph(keep=lambda bloq: isinstance(bloq, And))
    emitted = Counter(gate.name for gate in actual)
    assert counts == {And(): 7, And().adjoint(): 7, XGate(): emitted["x"], CNOT(): 1}
    key = compound_control_key(CliffordCall("x", 1), values)
    assert key is not None
    model = query_qec_gate_cost(key)
    assert {
        name: value for name, value in asdict(model).items() if value
    } == get_cost_value(source, QECGatesCost()).asdict()
    assert emitted["t"] + emitted["tdg"] == 28 and emitted["measure"] == 7


@pytest.mark.parametrize("basis,bit", [("z", 0), ("z", 1), ("x", 0)])
def test_original_prepared_x_measurement_superoperator(
    basis: Literal["x", "z"], bit: int
) -> None:
    from qualtran import BloqBuilder
    from qualtran.bloqs.basic_gates import MeasureX, OneState, PlusState, ZeroState

    builder = BloqBuilder()
    source = (PlusState if basis == "x" else OneState if bit else ZeroState)()
    c = builder.add(MeasureX(), q=builder.add(source))
    expected = builder.finalize(c=c).tensor_contract(superoperator=True)
    circuit = Circuit(1, dtype=torch.complex128)
    circuit.append(FieldOperation(1, basis_state_ir(basis, bit, 0)))
    plan = build_tensor_network(circuit, dtype=torch.complex128)
    outcomes = tensor_network_basis_measurement(plan, 0, basis="x")
    weights = torch.stack([outcome.abs().square().sum() for outcome in outcomes])
    torch.testing.assert_close(
        torch.diag(weights).to(torch.complex128),
        torch.as_tensor(expected),
        rtol=0,
        atol=2e-12,
    )

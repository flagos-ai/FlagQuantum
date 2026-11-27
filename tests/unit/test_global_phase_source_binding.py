"""Capture zero-port phase sources with an explicit owned physical carrier."""

from collections import Counter
from dataclasses import replace
from hashlib import sha1
from math import pi
from pathlib import Path

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.algorithms.primitives.black_box_prepare import (
    BlackBoxPrepare,
    _identity_prepare,
)
from flagquantum.algorithms.primitives.select_block_encoding import SelectBlockEncoding
from flagquantum.compiler._component_trace.compound_control_resource import (
    global_phase_cost_key,
)
from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost
from flagquantum.compiler.registers.primitives.global_phase import (
    global_phase_exponent,
    global_phase_ir,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter
from flagquantum.simulation.unitary import get_unitary
from tests._reference_dependencies import unavailable_reference

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_source() -> None:
    from qualtran.bloqs.basic_gates import global_phase

    raw = Path(global_phase.__file__).read_bytes()
    assert (
        sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == "08e0273e6e62562dd50f6af0fb2f30b04bb9033f"
    )


@pytest.mark.parametrize("exponent", (0.0, 0.5, -0.7, 1.7, -2.3))
def test_whole_scalar_source_directions_carrier_and_independent_model(
    exponent: float,
) -> None:
    from qualtran.bloqs.basic_gates.global_phase import GlobalPhase
    from qualtran.resource_counting._bloq_counts import QECGatesCost
    from qualtran.resource_counting._costing import get_cost_value
    from qualtran.resource_counting._qubit_counts import QubitCount

    original = GlobalPhase(exponent=exponent)
    emitter = import_register_emitter(original)
    assert emitter.layout == () and emitter.requirements() == (1, 0)
    assert emitter.binding_swaps == (0, 0)
    emitter = replace(
        emitter,
        snapshots=(
            CircuitIR.from_json(emitter.snapshots[0].to_json()),
            CircuitIR.from_json(emitter.snapshots[1].to_json()),
        ),
    )
    operation = SelectBlockEncoding(
        (), (), emitter, BlackBoxPrepare((), (), _identity_prepare)
    )
    state = torch.tensor(
        [1 + 2j, -2j, 3 - 1j, -2 + 1j, 1j, -3 + 2j, 2 + 3j, 1 - 1j],
        dtype=torch.complex128,
    )
    state /= state.norm()
    for inverse in (False, True):
        source = original.adjoint() if inverse else original
        coefficient = complex(source.tensor_contract())
        expected_gates = global_phase_ir(
            float(global_phase_exponent(coefficient)), anchor=0
        )
        assert emitter.snapshots[int(inverse)].instructions == expected_gates
        inventory = Counter(g.name for g in expected_gates)
        assert inventory == ({} if exponent == 0 else {"x": 2, "phase": 2})
        gates = operation.make_on(
            selection=(), target=(), workspace=(2,), classical_bits=(), adjoint=inverse
        )
        initial = torch.zeros(8, dtype=torch.complex128)
        initial[::2] = state[:4]
        initial /= initial.norm()
        actual = Circuit(3, inputs=initial, dtype=torch.complex128).append(gates)
        torch.testing.assert_close(
            actual.state().flatten(), initial * coefficient, rtol=0, atol=3e-12
        )
        assert not torch.count_nonzero(actual.state().flatten()[1::2])
        # The native scalar recipe also preserves an arbitrary coherent carrier.
        carrier = Circuit(3, inputs=state, dtype=torch.complex128)
        carrier.append(
            FieldOperation(
                3, global_phase_ir(float(global_phase_exponent(coefficient)), anchor=2)
            )
        )
        torch.testing.assert_close(
            carrier.state().flatten(), state * coefficient, rtol=0, atol=3e-12
        )
        assert get_cost_value(source, QECGatesCost()).asdict() == {}
        assert query_qec_gate_cost(global_phase_cost_key(exponent)).asdict() == {}
        assert get_cost_value(source, QubitCount()) == 0
    circuit = Circuit(3, dtype=torch.complex128)
    for inverse in (False, True):
        circuit.append(
            operation.make_on(
                selection=(),
                target=(),
                workspace=(2,),
                classical_bits=(),
                adjoint=inverse,
            )
        )
    torch.testing.assert_close(
        get_unitary(circuit), torch.eye(8, dtype=torch.complex128), rtol=0, atol=3e-12
    )


@pytest.mark.parametrize("value", (0, 1, 2, 3))
def test_actual_word_controlled_phase_source_leaf_and_complex_columns(
    value: int,
) -> None:
    from qualtran._infra.controlled import CtrlSpec
    from qualtran._infra.data_types import QUInt
    from qualtran.bloqs.basic_gates.global_phase import GlobalPhase
    from qualtran.resource_counting._bloq_counts import QECGatesCost
    from qualtran.resource_counting._costing import get_cost_value
    from qualtran.resource_counting._qubit_counts import QubitCount

    source = GlobalPhase(exponent=0.37).controlled(
        CtrlSpec(qdtypes=QUInt(2), cvs=value)
    )
    emitter = import_register_emitter(source)
    assert emitter.requirements() == (0, 0) and emitter.layout == (("ctrl", 2),)
    for inverse in (False, True):
        actual_source = source.adjoint() if inverse else source
        gates = emitter.snapshots[int(inverse)].instructions
        expected_names = {
            0: ("x", "phase", "x", "phase", "rz", "cx", "rz", "cx", "rz"),
            1: ("x", "phase", "any", "cx", "any", "cx", "any", "x"),
            2: ("x", "phase", "x", "phase", "rz", "cx", "rz", "cx", "rz"),
            3: ("cphase",),
        }
        assert tuple(gate.name for gate in gates) == expected_names[value]
        assert not any(gate.name == "measure" for gate in gates)
        matrix = torch.eye(4, dtype=torch.complex128)
        matrix[value, value] = torch.exp(
            torch.tensor((-1 if inverse else 1) * 0.37j * pi, dtype=torch.complex128)
        )
        circuit = Circuit(2, dtype=torch.complex128).append(FieldOperation(2, gates))
        torch.testing.assert_close(get_unitary(circuit), matrix, rtol=0, atol=3e-12)
        torch.testing.assert_close(
            get_unitary(circuit),
            torch.as_tensor(actual_source.tensor_contract()),
            rtol=0,
            atol=3e-12,
        )
        model = query_qec_gate_cost(
            global_phase_cost_key(0.37, (value >> 1, value & 1))
        )
        assert model.asdict() == get_cost_value(actual_source, QECGatesCost()).asdict()
        assert get_cost_value(actual_source, QubitCount()) == 3


def test_original_external_scalar_conventions_and_static_refusal() -> None:
    try:
        import cirq
    except ModuleNotFoundError as _reference_import_error:
        unavailable_reference(_reference_import_error)
    from qualtran.bloqs.basic_gates.global_phase import GlobalPhase
    from qualtran.cirq_interop._cirq_to_bloq import cirq_gate_to_bloq

    from flagquantum.ecosystem.cirq import from_cirq
    from flagquantum.ecosystem.cirq._feedback_import import import_cirq_feedback

    original = GlobalPhase.from_coefficient(1j)
    foreign = cirq.Circuit(cirq.global_phase_operation(1j))
    assert cirq_gate_to_bloq(cirq.GlobalPhaseGate(1j)) == original
    assert cirq.approx_eq(
        original.as_composite_bloq().to_cirq_circuit(), foreign, atol=1e-16
    )
    with pytest.raises(ValueError, match="must reference at least one qubit"):
        from_cirq(foreign)
    assert (
        len(from_cirq(foreign + cirq.Circuit(cirq.I(cirq.LineQubit(0)))).instructions)
        == 5
    )
    captured = import_cirq_feedback(foreign, qubit_order=(cirq.LineQubit(0),)).ir
    assert (
        captured.instructions
        == import_register_emitter(original).snapshots[0].instructions
    )
    qml = pytest.importorskip("pennylane")
    foreign_pl = original.as_pl_op(wires=[0])
    assert foreign_pl == qml.GlobalPhase(phi=0.5 * pi, wires=[0])
    # Preserve the source PennyLane sign convention as an independent assertion.
    torch.testing.assert_close(
        torch.as_tensor(qml.matrix(foreign_pl)),
        -1j * torch.eye(2, dtype=torch.complex128),
        rtol=0,
        atol=2e-12,
    )
    torch.testing.assert_close(
        torch.as_tensor(original.tensor_contract()),
        torch.tensor(1j, dtype=torch.complex128),
        rtol=0,
        atol=2e-12,
    )


@pytest.mark.parametrize("values", ((0,), (0, 0), (1, 0)))
def test_controlled_scalar_retains_source_decomposition(
    values: tuple[int, ...],
) -> None:
    try:
        import cirq
    except ModuleNotFoundError as _reference_import_error:
        unavailable_reference(_reference_import_error)
    from qualtran.bloqs.basic_gates.global_phase import GlobalPhase

    from flagquantum.ecosystem.cirq._feedback_import import (
        _has_explicit_matrix,
        _lower_feedback_gate,
        import_cirq_feedback,
    )
    from tests.unit.test_controlled_unitary_synthesis import assert_source_stream

    source = GlobalPhase(exponent=0.37)
    wires = tuple(cirq.LineQubit.range(len(values)))
    operation = cirq.global_phase_operation(source.coefficient).controlled_by(
        *wires, control_values=values
    )
    assert not _has_explicit_matrix(cirq, operation.gate)
    leaves = cirq.decompose(
        operation,
        keep=lambda op: isinstance(op.gate, cirq.GlobalPhaseGate)
        or _lower_feedback_gate(cirq, op.gate, [], 0) is not None,
    )
    expected = import_cirq_feedback(
        cirq.Circuit(leaves, strategy=cirq.InsertStrategy.NEW), qubit_order=wires
    ).ir
    try:
        import numpy as np
    except ModuleNotFoundError as _reference_import_error:
        unavailable_reference(_reference_import_error)
    from qualtran._infra.controlled import CtrlSpec

    captured = import_register_emitter(
        source.controlled(CtrlSpec(cvs=np.asarray(values)))
    ).snapshots[0]
    assert_source_stream(captured, expected)

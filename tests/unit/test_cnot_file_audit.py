"""Compare the complete original atomic CNOT teaching and execution views."""

from dataclasses import asdict
from hashlib import sha1
from importlib.metadata import version
from pathlib import Path

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from examples.algorithms.cnot_tutorial import (
    CNOT,
    cnot_graph,
    cnot_score_data,
    cnot_signature,
    cnot_truth_table,
)
from flagquantum.circuit import Circuit
from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost
from flagquantum.compiler.registers.primitives.controlled_not import (
    cnot_with_controls_ir,
)
from flagquantum.simulation.bit_encoding import validate_bit_value

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
SOURCE = (
    Path(__file__).resolve().parents[3]
    / "ftqc-reference-env/qualtran/bloqs/basic_gates"
)


def test_frozen_cnot_source_assets() -> None:
    import kickmix

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    for name, digest in (
        ("cnot.py", "d2c2002a226debd2cdf382bbbec5c6f027bf90cc"),
        ("cnot_test.py", "cb4f3767b80d5f3d2220ebb0892964dec725ca92"),
        ("cnot.ipynb", "ab8f97b3f80d8722edf2da502526136d785ccb63"),
    ):
        data = (SOURCE / name).read_bytes()
        assert sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == digest


def test_factory_atomic_cost_truth_table_and_glyphs() -> None:
    from qualtran import DecomposeTypeError, QBit
    from qualtran.bloqs.basic_gates.cnot import _cnot
    from qualtran.drawing import get_musical_score_data
    from qualtran.resource_counting import QECGatesCost, get_cost_value
    from qualtran.simulation.classical_sim import (
        format_classical_truth_table,
        get_classical_truth_table,
    )

    source = _cnot.make()
    assert source.adjoint() is source
    with pytest.raises(DecomposeTypeError, match="atomic"):
        source.decompose_bloq()
    with pytest.raises(ValueError):
        source.call_classically(ctrl=2, target=0)
    with pytest.raises(ValueError):
        validate_bit_value(2)
    assert all(isinstance(r.dtype, QBit) for r in source.signature)
    assert [
        (r.name, r.bitsize, r.shape_symbolic, r.side.value) for r in cnot_signature()
    ] == [(r.name, r.bitsize, r.shape_symbolic, r.side.value) for r in source.signature]
    original_graph, original_leaves = source.call_graph(max_depth=1)
    assert list(original_graph.nodes) == [source]
    assert not original_graph.edges
    graph = cnot_graph()
    assert not graph.direct_calls
    assert graph.leaf_counts == {CNOT: original_leaves[source]}
    expected_cost = get_cost_value(source, QECGatesCost())
    cost = query_qec_gate_cost(CNOT)
    assert cost.clifford == expected_cost.clifford == 1
    assert {name: count for name, count in asdict(cost).items() if count} == {
        "clifford": 1
    }
    assert expected_cost.total_t_count() == 0
    assert cnot_truth_table() == format_classical_truth_table(
        *get_classical_truth_table(source)
    )
    original = get_musical_score_data(source.as_composite_bloq())
    got = cnot_score_data()
    assert (
        [port["symb_cls"] for port in got["soqs"] if port["seq_x"] == 0]
        == [
            port.json_dict()["symb_cls"]
            for port in original.soqs
            if port.json_dict()["seq_x"] == 0
        ]
        == ["Circle", "ModPlus"]
    )


def test_native_bell_append_and_source_inverse() -> None:
    import cirq
    import numpy as np
    from qualtran import BloqBuilder
    from qualtran.bloqs.basic_gates import CNOT as SOURCE_CNOT
    from qualtran.bloqs.basic_gates import PlusState, ZeroState

    from flagquantum.ecosystem.cirq import to_cirq

    builder = BloqBuilder()
    ctrl, target = builder.add(PlusState()), builder.add(ZeroState())
    ctrl, target = builder.add(SOURCE_CNOT(), ctrl=ctrl, target=target)
    source = builder.finalize(ctrl=ctrl, target=target).tensor_contract()
    operation = FieldOperation(2, cnot_with_controls_ir(0, 1))
    assert [(gate.name, gate.wires) for gate in operation.instructions] == [
        ("cx", (0, 1))
    ]
    circuit = Circuit(2, dtype=torch.complex128)
    circuit.append(operation)
    exported = to_cirq(circuit.to_ir())
    original_op, _ = SOURCE_CNOT().as_cirq_op(
        cirq.SimpleQubitManager(),
        ctrl=np.array([cirq.LineQubit(0)]),
        target=np.array([cirq.LineQubit(1)]),
    )
    assert tuple(exported.all_operations()) == (original_op,)
    np.testing.assert_array_equal(exported.unitary(), SOURCE_CNOT().tensor_contract())

    circuit = Circuit(2, dtype=torch.complex128)
    circuit.h(0)
    initial = circuit.state().flatten().clone()
    circuit.append(operation)
    torch.testing.assert_close(
        circuit.state().flatten(),
        torch.from_numpy(np.asarray(source).reshape(-1)),
        atol=1e-12,
        rtol=0,
    )
    circuit.append(operation)
    torch.testing.assert_close(circuit.state().flatten(), initial, atol=1e-12, rtol=0)


@pytest.mark.pennylane
def test_existing_pennylane_export_matches_original_cnot_matrix() -> None:
    import numpy as np

    qml = pytest.importorskip("pennylane")
    from qualtran.bloqs.basic_gates import CNOT as SOURCE_CNOT

    from flagquantum.ecosystem.pennylane import export_pennylane

    circuit = Circuit(2, dtype=torch.complex128)
    circuit.append(FieldOperation(2, cnot_with_controls_ir(0, 1)))
    exported = export_pennylane(circuit.to_ir())
    expected = qml.FromBloq(SOURCE_CNOT(), wires=[0, 1]).matrix()
    np.testing.assert_array_equal(
        qml.matrix(exported.quantum_script, wire_order=[0, 1]), expected
    )

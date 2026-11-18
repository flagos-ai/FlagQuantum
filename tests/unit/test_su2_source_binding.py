"""Close actual SU(2) source binding, presentation and symbolic allocation."""

from functools import partial
from hashlib import sha1
from pathlib import Path

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler._component_trace.call_graph import SymbolAllocator
from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost
from flagquantum.compiler.registers.phase.su2_rotation import (
    su2_rotation_calls,
    su2_rotation_ir,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.core.value_math import SymbolicReal
from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter
from flagquantum.simulation.unitary import get_unitary

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_source() -> None:
    from qualtran.bloqs.basic_gates import su2_rotation

    for name, blob in (
        ("su2_rotation.py", "91283ba1822f1cc3c1b1cedc37a4dfa74ec21085"),
        ("su2_rotation_test.py", "61b4d09dbc41dac7590d4655a6b5ecb6c188228f"),
    ):
        raw = (Path(su2_rotation.__file__).parent / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob


@pytest.mark.parametrize(
    "angles", ((0.2, 0.3, -0.4, 0.1), (0.2, 0.3, -0.4, 7.2), (0.0, 0.0, 0.0, 0.0))
)
def test_actual_su2_directions_source_stream_complex_columns_and_models(
    angles: tuple[float, float, float, float],
) -> None:
    import cirq
    from qualtran.bloqs.basic_gates import SU2RotationGate
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    from flagquantum.ecosystem.cirq._feedback_import import import_cirq_feedback
    from tests.unit.test_controlled_unitary_synthesis import assert_source_stream

    source = SU2RotationGate(*angles)
    emitter = import_register_emitter(source)
    assert emitter.layout == (("q", 1),) and emitter.requirements() == (0, 0)
    assert emitter.binding_swaps == (0, 0)
    initial = torch.tensor([1 + 2j, 3 - 1j, -2j, 4 + 1j], dtype=torch.complex128)
    initial /= initial.norm()
    for inverse in (False, True):
        original = source.adjoint() if inverse else source
        foreign = cirq.Circuit(original.decompose_bloq().to_cirq_circuit())
        direct = import_cirq_feedback(
            foreign, qubit_order=tuple(sorted(foreign.all_qubits()))
        ).ir
        snapshot = CircuitIR.from_json(emitter.snapshots[int(inverse)].to_json())
        assert_source_stream(snapshot, direct)
        matrix = torch.as_tensor(original.rotation_matrix)
        expected = (matrix @ initial.reshape(2, 2).T).T.flatten()
        circuit = Circuit(2, inputs=initial, dtype=torch.complex128)
        circuit.append(FieldOperation(2, emitter({"q": (1,)}, (), (), inverse)))
        torch.testing.assert_close(
            circuit.state().flatten(), expected, rtol=0, atol=3e-12
        )
        circuit.append(FieldOperation(2, emitter({"q": (1,)}, (), (), not inverse)))
        torch.testing.assert_close(
            circuit.state().flatten(), initial, rtol=0, atol=3e-12
        )
        native = su2_rotation_ir(
            0,
            float(original.theta),
            float(original.phi),
            float(original.lambd),
            global_shift=float(original.global_shift),
        )
        assert len(snapshot.instructions) == len(native) == 7
        assert not any(g.name in {"measure", "reset"} for g in native)
        torch.testing.assert_close(
            get_unitary(snapshot, dtype=torch.complex128), matrix, rtol=0, atol=3e-12
        )
        torch.testing.assert_close(
            get_unitary(
                Circuit(1, dtype=torch.complex128).append(FieldOperation(1, native))
            ),
            matrix,
            rtol=0,
            atol=3e-12,
        )
        calls = su2_rotation_calls(
            *map(float, (original.theta, original.phi, original.lambd)),
            global_shift=float(original.global_shift),
        )
        from collections import Counter

        inventory: Counter[str] = Counter()
        for call in calls:
            inventory.update(query_qec_gate_cost(call).asdict())
        assert dict(inventory) == get_cost_value(original, QECGatesCost()).asdict()
        assert get_cost_value(original, QubitCount()) == 1


@pytest.mark.parametrize("mode", ("numeric", "symbolic", "mixed", "precision"))
def test_su2_whole_source_geometry_and_allocator(mode: str) -> None:
    from qualtran import BloqBuilder
    from qualtran.bloqs.basic_gates import SU2RotationGate
    from qualtran.drawing import get_musical_score_data
    from qualtran.resource_counting import SympySymbolAllocator

    from flagquantum.core.ir.program.register_types import opaque_register_type
    from flagquantum.core.ir.program.registers import Register
    from flagquantum.drawer.register_score import RegisterStep, layout_register_score
    from flagquantum.drawer.register_score_data import register_score_data
    from flagquantum.drawer.su2_rotation_symbols import su2_rotation_wire_symbol
    from tests.unit.test_drawer_component_score import assert_source_geometry

    allocator = SymbolAllocator()
    # The existing resource allocator owns deterministic symbol lifetime.
    args: tuple[SymbolicReal, ...] = tuple(
        allocator.new_symbol(name)
        for name in ("theta", "phi", "lambda", "alpha", "eps")
    )
    arbitrary = SU2RotationGate.arbitrary(SympySymbolAllocator())
    assert args == (
        arbitrary.theta,
        arbitrary.phi,
        arbitrary.lambd,
        arbitrary.global_shift,
        arbitrary.eps,
    )
    sources = {
        "numeric": SU2RotationGate(0.21, -0.031, 2.7, -0.0),
        "symbolic": arbitrary,
        "mixed": SU2RotationGate(0.21, args[1], 2.7, -0.0),
        "precision": SU2RotationGate(0.21, -0.031, 2.7, -0.0, eps=args[4]),
    }
    source = sources[mode]
    supplier = partial(
        su2_rotation_wire_symbol,
        theta=source.theta,
        phi=source.phi,
        lambd=source.lambd,
        global_shift=source.global_shift,
    )
    reg = Register("q", opaque_register_type(1))
    step = RegisterStep(
        "SU2", ("q",), ("q",), input_registers=(reg,), output_registers=(reg,)
    )
    score = layout_register_score({"q": 1}, ("q",), (step,))
    builder, inputs = BloqBuilder.from_signature(source.signature)
    composite = builder.finalize(**builder.add_d(source, **inputs))
    assert_source_geometry(
        register_score_data(score, register_wire_symbol=supplier),
        get_musical_score_data(composite),
        ordered_ports=True,
    )
    assert str(source) == "SU_2" + str(supplier(step, None, None)[1]["text"])
    if mode in {"symbolic", "mixed"}:
        assert source._unitary_() is None
        field = "lambda" if mode == "symbolic" else "phi"
        with pytest.raises(ValueError, match=f"Symbolic {field} not allowed"):
            _ = source.rotation_matrix
        calls = su2_rotation_calls(
            args[0], args[1], args[2], global_shift=args[3], precision=args[4]
        )
        assert all(call.precision == args[4] / 3 for call in calls[1:])

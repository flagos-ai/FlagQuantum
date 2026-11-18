"""Capture complete Power wrappers without dropping their child multiplicity."""

from collections import Counter
from dataclasses import asdict, replace
from hashlib import sha1
from itertools import product
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
from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost
from flagquantum.compiler._component_trace.resource_generalization import (
    ControlledZPowerCall,
    PhaseCall,
    PowerCall,
)
from flagquantum.compiler.registers.primitives.controlled_rotation import (
    controlled_z_power_ir,
)
from flagquantum.compiler.registers.primitives.power import register_power_ir
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter
from flagquantum.simulation.unitary import get_unitary
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_source() -> None:
    from qualtran.bloqs.basic_gates import power

    raw = Path(power.__file__).read_bytes()
    assert (
        sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == "5331c3afd2ce0dffba17c4adb92659d15b76311b"
    )


@pytest.mark.parametrize("exponent", (1, 2, -1, -2))
def test_nested_whole_source_power_and_actual_adjoint(exponent: int) -> None:
    from qualtran.bloqs.basic_gates import OnEach, Power, SGate
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    original = Power(OnEach(3, SGate()), 2) ** exponent
    emitter = import_register_emitter(original)
    assert emitter.requirements() == (0, 0)
    count = 2 * abs(exponent)
    for inverse in (False, True):
        backwards = (exponent < 0) != inverse
        code = "sdg" if backwards else "s"
        body = CircuitIR(3, tuple(Instruction(code, (wire,)) for wire in range(3)))
        expected = register_power_ir(body, count)
        snapshot = emitter.snapshots[int(inverse)]
        assert_source(snapshot.instructions, expected)
        assert len(snapshot.instructions) == 3 * count
        assert Counter(g.name for g in snapshot.instructions) == {code: 3 * count}
        circuit = Circuit(3, dtype=torch.complex128)
        circuit.append(FieldOperation(3, snapshot.instructions))
        source = original.adjoint() if inverse else original
        torch.testing.assert_close(
            get_unitary(circuit),
            torch.as_tensor(source.tensor_contract()),
            rtol=0,
            atol=3e-12,
        )
        model = query_qec_gate_cost(
            PowerCall(PhaseCall("s", adjoint=backwards), 3 * count)
        )
        assert (
            model.clifford
            == get_cost_value(source, QECGatesCost()).clifford
            == 3 * count
        )


@pytest.mark.parametrize("count", (2, 3))
def test_measured_power_whole_call_feedback_and_replacement(count: int) -> None:
    from qualtran.bloqs.basic_gates import CZPowGate, Power
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    angle = 0.37
    original = Power(CZPowGate(exponent=angle), count)
    emitter = import_register_emitter(original)
    assert emitter.layout == (("q", 2),) and emitter.requirements() == (1, count)
    emitter = replace(
        emitter,
        snapshots=(
            CircuitIR.from_json(emitter.snapshots[0].to_json()),
            CircuitIR.from_json(emitter.snapshots[1].to_json()),
        ),
    )
    operation = SelectBlockEncoding(
        (), emitter.layout, emitter, BlackBoxPrepare((), (), _identity_prepare)
    )
    for inverse in (False, True):
        expected_gates = tuple(
            gate
            for slot in range(count)
            for gate in controlled_z_power_ir(
                0, 1, -angle if inverse else angle, 2, slot
            )
        )
        actual = (
            Circuit(3)
            .append(
                operation.make_on(
                    selection=(),
                    target=(0, 1),
                    workspace=(2,),
                    classical_bits=tuple(range(count)),
                    adjoint=inverse,
                )
            )
            .to_ir()
        )
        assert actual.instructions == emitter.snapshots[int(inverse)].instructions
        assert_source(actual.instructions, expected_gates)
        for wire in range(3):
            assert [g.params for g in actual.instructions if wire in g.wires] == [
                g.params for g in expected_gates if wire in g.wires
            ]
        inventory = Counter(g.name for g in actual.instructions)
        assert (
            inventory["measure"] == count
            and inventory["t"] + inventory["tdg"] == 4 * count
        )
        source = original.adjoint() if inverse else original
        model = query_qec_gate_cost(
            PowerCall(ControlledZPowerCall(-angle if inverse else angle), count)
        )
        assert {
            key: value for key, value in asdict(model).items() if value
        } == get_cost_value(source, QECGatesCost()).asdict()
        assert get_cost_value(source, QubitCount()) == 3
        matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
        # Every logical column and a coherent input retain the full unnormalized phase.
        vectors = [
            *torch.eye(4, dtype=torch.complex128).unbind(1),
            torch.tensor([1 + 2j, -2j, 3 - 1j, -2 + 1j], dtype=torch.complex128),
        ]
        for vector in vectors:
            vector = vector / vector.norm()
            initial = torch.zeros(8, dtype=torch.complex128)
            expected = torch.zeros_like(initial)
            initial[::2] = vector
            expected[::2] = matrix @ vector / 2 ** (count / 2)
            for history in product((0, 1), repeat=count):
                native = forced_branch(initial, actual.instructions, history)
                torch.testing.assert_close(native, expected, rtol=0, atol=3e-12)
                torch.testing.assert_close(
                    native,
                    forced_branch(initial, expected_gates, history),
                    rtol=0,
                    atol=3e-12,
                )

    coherent = torch.zeros(8, dtype=torch.complex128)
    coherent[::2] = torch.tensor([1 + 1j, -2j, 3 - 2j, 4 + 1j], dtype=torch.complex128)
    coherent /= coherent.norm()
    histories = tuple(product((0, 1), repeat=count))
    for first, second in product(histories, repeat=2):
        intermediate = forced_branch(coherent, emitter.snapshots[0].instructions, first)
        restored = forced_branch(
            intermediate, emitter.snapshots[1].instructions, second
        )
        torch.testing.assert_close(restored, coherent / 2**count, rtol=0, atol=3e-12)

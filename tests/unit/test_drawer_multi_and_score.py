"""Verify the locked notebook's MultiAnd score and native measured inverse."""

from collections import Counter

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.registers.control.multi_control import multi_and_ir
from flagquantum.drawer.register_score import multi_and_register_score
from flagquantum.drawer.register_score_data import register_score_data
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit

_VALUES = (1, 1, 0, 1)
_CONTROLS = (0, 1, 2, 3)
_WORK = (4, 5, 6)
_BITS = (0, 1, 2)


@pytest.mark.ftqc_reference
def test_multi_and_notebook_score_and_resource_parity() -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.mcmt import MultiAnd
    from qualtran.drawing import get_musical_score_data
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    source = MultiAnd(_VALUES)
    expected = get_musical_score_data(source.decompose_bloq())
    data = register_score_data(multi_and_register_score(_VALUES))
    assert data["max_x"] == expected.max_x == 3
    assert data["max_y"] == expected.max_y == 6
    assert data["hlines"] == [line.json_dict() for line in expected.hlines]
    assert data["vlines"] == [line.json_dict() for line in expected.vlines]
    # Native port identities name the original logical register lane, while
    # upstream identifiers name a bloq-local port and shaped index.
    source_ports = {
        (port.rpos.seq_x, port.rpos.y): {
            key: value for key, value in port.json_dict().items() if key != "ident"
        }
        for port in expected.soqs
    }
    assert {
        (port["seq_x"], port["y"]): {
            key: value for key, value in port.items() if key != "ident"
        }
        for port in data["soqs"]
    } == source_ports
    assert len({port["ident"] for port in data["soqs"]}) == len(expected.soqs)

    forward = multi_and_ir(_CONTROLS, _VALUES, _WORK)
    erase = multi_and_ir(
        _CONTROLS, _VALUES, _WORK, classical_bits=_BITS, uncompute=True
    )
    assert_source(
        forward,
        reference_arithmetic_gates(
            source,
            {"ctrl": _CONTROLS, "junk": _WORK[:-1], "target": _WORK[-1:]},
            (),
            {},
            cirq_style=True,
        ),
    )
    assert_source(
        erase,
        reference_arithmetic_gates(
            source.adjoint(),
            {"ctrl": _CONTROLS, "junk": _WORK[:-1], "target": _WORK[-1:]},
            (),
            dict(zip(_WORK, _BITS, strict=True)),
            expected_outputs={"ctrl": _CONTROLS},
        ),
    )
    gates = Counter(gate.name for gate in (*forward, *erase))
    compute_cost = get_cost_value(source, QECGatesCost())
    erase_cost = get_cost_value(source.adjoint(), QECGatesCost())
    assert gates["t"] + gates["tdg"] == compute_cost.total_t_count() == 12
    assert gates["measure"] == erase_cost.measurement == 3
    assert data["max_y"] + 1 == get_cost_value(source, QubitCount()) == 7


def test_multi_and_notebook_pattern_restores_all_measured_branches() -> None:
    state = torch.zeros(128, dtype=torch.complex128)
    control = torch.arange(16, dtype=torch.float64)
    amps = torch.complex(control + 1, 2 * control - 3)
    state[::8] = amps / amps.norm()
    forward = multi_and_ir(_CONTROLS, _VALUES, _WORK)
    erase = multi_and_ir(
        _CONTROLS, _VALUES, _WORK, classical_bits=_BITS, uncompute=True
    )
    circuit = DynamicCircuit(7, inputs=state, dtype=torch.complex128).append(
        FieldOperation(7, forward + erase)
    )
    result = run_dynamic(circuit, shots=128, seed=41, strategy="trajectory")
    assert len(set(map(tuple, result.classical_bits[:, :3].tolist()))) == 8
    torch.testing.assert_close(
        result.final_states.reshape(128, -1), state.expand(128, -1), rtol=0, atol=1e-12
    )

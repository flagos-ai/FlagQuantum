"""Raw source predicates stay separate from valid-bit quantum semantics."""

from itertools import product

import numpy as np
import pytest
import sympy

from flagquantum.simulation.classical_controlled_bits import (
    cnot_classical_values,
    fanout_classical_values,
    toffoli_classical_values,
)
from tests.unit.test_classical_comparison import compare_calls

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
VALUES = [
    0,
    1,
    2,
    -1,
    True,
    np.uint8(255),
    np.int8(127),
    np.array([0, 1]),
    np.array([[1], [0]]),
]


@pytest.mark.parametrize("control,target", list(product(VALUES, repeat=2)))
def test_cnot_original_modulo_types_broadcast_and_overflow(control, target):
    from qualtran.bloqs.basic_gates import CNOT

    compare_calls(
        lambda c, t: tuple(CNOT().on_classical_vals(c, t).values()),
        cnot_classical_values,
        (control, target),
    )


@pytest.mark.parametrize(
    "controls",
    [
        np.array(x)
        for x in [
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
            (2, 1),
            (),
            (0,),
            (1,),
            ((0, 1), (1, 0)),
        ]
    ],
)
@pytest.mark.parametrize("target", VALUES)
def test_toffoli_original_assertion_short_circuit_and_inactive_identity(
    controls, target
):
    from qualtran.bloqs.basic_gates import Toffoli

    compare_calls(
        lambda c, t: tuple(Toffoli().on_classical_vals(c, t).values()),
        toffoli_classical_values,
        (controls, target),
    )


@pytest.mark.parametrize("control,target", list(product(VALUES, repeat=2)))
@pytest.mark.parametrize(
    "width", [0, 1, 3, -1, np.int64(3), sympy.Symbol("n"), sympy.Integer(3)]
)
def test_fanout_original_truthiness_mask_and_unused_width(control, target, width):
    from qualtran.bloqs.mcmt.multi_target_cnot import MultiTargetCNOT

    compare_calls(
        lambda c, t: tuple(MultiTargetCNOT(width).on_classical_vals(c, t).values()),
        lambda c, t: fanout_classical_values(c, t, width),
        (control, target),
    )


def test_diagnostic_nonbits_preserve_different_source_policies():
    target = np.array([0, 1])
    controls = np.array([2, 1])
    assert cnot_classical_values(2, 0) == (2, 0)
    assert cnot_classical_values(target, 0)[0] is target
    assert fanout_classical_values(2, 0, 1) == (2, 1)
    assert toffoli_classical_values(controls, 0)[1] == 0
    assert fanout_classical_values(0, target, sympy.Symbol("n"))[1] is target
    with pytest.raises(AssertionError):
        toffoli_classical_values(np.array([]), 2)
    assert toffoli_classical_values(np.array([0]), 1)[1] == 1


@pytest.mark.parametrize("kind", ["cnot", "toffoli", "fanout"])
@pytest.mark.parametrize("legacy", [False, True])
def test_original_model_categories_reuse_shared_qec_query(kind, legacy):
    from dataclasses import asdict

    from qualtran.bloqs.basic_gates import CNOT, Toffoli
    from qualtran.bloqs.mcmt.multi_target_cnot import MultiTargetCNOT
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost
    from flagquantum.compiler._component_trace.resource_generalization import (
        CliffordCall,
        FanoutCall,
        FixedTCall,
    )

    native, source = {
        "cnot": (CliffordCall("cx", 2), CNOT()),
        "toffoli": (FixedTCall("toffoli"), Toffoli()),
        "fanout": (FanoutCall(2), MultiTargetCNOT(2)),
    }[kind]
    expected = get_cost_value(source, QECGatesCost(legacy_shims=legacy))
    actual = query_qec_gate_cost(native, legacy_shims=legacy)
    assert {
        name: value for name, value in asdict(actual).items() if value
    } == expected.asdict()
    if not legacy:
        assert expected.total_t_count() == (4 if kind == "toffoli" else 0)
        assert expected.clifford == (0 if kind == "toffoli" else 1)


@pytest.mark.parametrize("kind", ["cnot", "toffoli", "fanout"])
def test_complete_complex_valid_bit_action_and_inverse_without_work(kind):
    import torch

    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.circuit import Circuit
    from flagquantum.compiler.registers.bits.bitwise import fanout_ir
    from flagquantum.core.ir import Instruction

    gates = {
        "cnot": (Instruction("cx", (0, 2)),),
        "toffoli": (Instruction("ccx", (0, 1, 2)),),
        "fanout": fanout_ir(0, (1, 2)),
    }[kind]
    grid = torch.arange(1, 17, dtype=torch.float64)
    initial = torch.complex(grid, 3 * grid - 5)
    initial /= initial.norm()
    expected = torch.zeros_like(initial)
    for data, spectator in product(range(8), (0, 1)):
        a, b, target = data >> 2, (data >> 1) & 1, data & 1
        if kind == "cnot":
            updated = (data & ~1) | cnot_classical_values(a, target)[1]
        elif kind == "toffoli":
            updated = (data & ~1) | int(
                toffoli_classical_values(np.array([a, b]), target)[1]
            )
        else:
            updated = (a << 2) | fanout_classical_values(a, data & 3, 2)[1]
        expected[(updated << 1) | spectator] = initial[(data << 1) | spectator]
    circuit = Circuit(4, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(4, gates))
    torch.testing.assert_close(circuit.state().flatten(), expected, rtol=0, atol=2e-12)
    circuit.append(FieldOperation(4, gates))
    torch.testing.assert_close(circuit.state().flatten(), initial, rtol=0, atol=2e-12)
    assert all(g.name not in ("measure", "reset") for g in gates)

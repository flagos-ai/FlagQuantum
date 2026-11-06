"""Check scalar phase and controlled cleanup against the pinned source."""

from collections import Counter
from itertools import product
from math import pi
from typing import Any

import pytest
import torch

import flagquantum as fq
from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler._component_trace.compound_control_resource import (
    global_phase_cost_key,
)
from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost
from flagquantum.compiler.registers.global_phase import (
    global_phase_exponent,
    global_phase_ir,
)
from flagquantum.core.ir import Instruction
from tests.unit.measurement_checks import forced_branch
from tests.unit.register_source_graph import (
    SourcePorts,
    SourceResult,
    source_control_predicate_leaf,
    walk_source_graph,
)
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit


def reference_controlled_phase(
    source: Any, controls: tuple[int, ...], work: tuple[int, ...], bits: tuple[int, ...]
) -> tuple[Instruction, ...]:
    from qualtran.bloqs.basic_gates import XGate, ZPowGate

    def leaf(
        bloq: Any, inputs: SourcePorts, desired: SourcePorts
    ) -> SourceResult | None:
        if isinstance(bloq, XGate):
            return (Instruction("x", inputs["q"]),), inputs
        if isinstance(bloq, ZPowGate):
            return (
                Instruction("phase", inputs["q"], params={"theta": pi * bloq.exponent}),
            ), inputs
        return source_control_predicate_leaf(bloq, inputs, work, bits)

    ports = {next(iter(source.signature)).name: controls}
    gates, returned = walk_source_graph(source, ports, ports, leaf)
    assert returned == ports
    return gates


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("exponent", [0.0, 0.5, -0.7, 2.0])
def test_scalar_source_coefficient_on_entangled_anchor(exponent: float) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.basic_gates import GlobalPhase
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    source = GlobalPhase(exponent=exponent)
    circuit = fq.Circuit(2, dtype=torch.complex128).h(0).cx(0, 1).s(1)
    initial = circuit.state().clone()
    gates = global_phase_ir(exponent, anchor=1)
    assert [gate.name for gate in gates] == (
        ["x", "phase", "x", "phase"] if exponent else []
    )
    circuit.append(FieldOperation(2, gates))
    torch.testing.assert_close(
        circuit.state(), initial * complex(source.tensor_contract()), rtol=0, atol=2e-12
    )
    circuit.append(FieldOperation(2, global_phase_ir(-exponent, anchor=1)))
    torch.testing.assert_close(circuit.state(), initial, rtol=0, atol=2e-12)
    assert query_qec_gate_cost(global_phase_cost_key(exponent)).t == 0
    assert not get_cost_value(source, QECGatesCost()).asdict()
    assert get_cost_value(source, QubitCount()) == 0


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("values", [(1,), (0,), (0, 1), (1, 0), (0, 1, 0)])
@pytest.mark.parametrize("exponent", [0.25, -0.7])
def test_controlled_phase_source_flow_and_every_branch(
    values: tuple[int, ...], exponent: float
) -> None:
    pytest.importorskip("qualtran")
    np = pytest.importorskip("numpy")
    from qualtran import CtrlSpec
    from qualtran.bloqs.basic_gates import GlobalPhase
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    n = len(values)
    # Leave a spectator on wire zero and bind controls in reversed wire order.
    controls = tuple(range(n, 0, -1))
    work = tuple(range(n + 1, 2 * n))
    bits = tuple(range(30, 30 + len(work)))
    ports = dict(controls=controls, values=values, workspace=work, classical_bits=bits)
    source = GlobalPhase(exponent=exponent).controlled(
        CtrlSpec(cvs=values[0] if n == 1 else np.asarray(values))
    )
    forward = global_phase_ir(exponent, **ports)
    reverse = global_phase_ir(-exponent, **ports)
    for gates, bloq in ((forward, source), (reverse, source.adjoint())):
        reference = reference_controlled_phase(bloq, controls, work, bits)
        assert_source(gates, reference)
        assert [g for g in gates if g.name == "phase"] == [
            g for g in reference if g.name == "phase"
        ]

    model = query_qec_gate_cost(global_phase_cost_key(exponent, values))
    source_model = get_cost_value(source, QECGatesCost())
    assert (
        model.t,
        model.rotation,
        model.clifford,
        model.and_bloq,
        model.measurement,
    ) == (
        source_model.t,
        source_model.rotation,
        source_model.clifford,
        source_model.and_bloq,
        source_model.measurement,
    )
    assert get_cost_value(source, QubitCount()) == 2 * n - 1
    counts = Counter(g.name for g in forward)
    assert counts["measure"] == len(work)
    assert counts["t"] + counts["tdg"] == 4 * len(work)

    width = 2 * n
    initial = torch.zeros(1 << width, dtype=torch.complex128)
    diagonal = torch.diag(
        torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
    )
    expected = initial.clone()
    for row in range(1 << (n + 1)):
        index = row << len(work)
        initial[index] = complex(row + 1, row % 3 - 2) / (1 << n)
        source_row = sum(
            ((index >> (width - 1 - wire)) & 1) << (n - 1 - i)
            for i, wire in enumerate(controls)
        )
        expected[index] = initial[index] * diagonal[source_row]
    branch_scale = 2 ** (-len(work) / 2)
    histories = tuple(product((0, 1), repeat=len(work)))
    for first, second in product(histories, repeat=2):
        branch = forced_branch(initial, forward, first)
        torch.testing.assert_close(branch, expected * branch_scale, rtol=0, atol=2e-12)
        torch.testing.assert_close(
            forced_branch(branch, reverse, second),
            initial * branch_scale**2,
            rtol=0,
            atol=2e-12,
        )


def test_global_phase_binding_validation() -> None:
    assert global_phase_ir(0.0, anchor=0) == ()
    with pytest.raises(ValueError, match="requires an anchor"):
        global_phase_ir(0.5)
    with pytest.raises(ValueError, match="phase carriers"):
        global_phase_ir(0.5, anchor=1, controls=(0,), values=(1,))
    with pytest.raises(ValueError, match="classical slots"):
        global_phase_ir(0.5, controls=(0, 1), values=(0, 1), workspace=(2,))
    with pytest.raises(ValueError, match="finite"):
        global_phase_ir(float("inf"), anchor=0)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "coefficient", [0j, -0.0, 2j, -2 - 3j, complex(-1, -0.0), 1 + 1j]
)
def test_phase_coefficient_conversion_and_controlled_application(
    coefficient: complex,
) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.basic_gates import GlobalPhase
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    source = GlobalPhase.from_coefficient(coefficient)
    exponent = float(global_phase_exponent(coefficient))
    assert exponent == pytest.approx(source.exponent, rel=0, abs=2e-15)

    state = torch.tensor([1 + 1j, -2 + 3j], dtype=torch.complex128)
    state /= state.norm()
    expected = torch.as_tensor(source.controlled().tensor_contract()) @ state
    gates = global_phase_ir(exponent, controls=(0,), values=(1,))
    assert [g.name for g in gates] == ["phase"]
    assert gates[0].params["theta"] == pytest.approx(
        pi * source.exponent, rel=0, abs=2e-15
    )
    circuit = fq.Circuit(1, inputs=state, dtype=torch.complex128)
    circuit.append(FieldOperation(1, gates))
    torch.testing.assert_close(circuit.state().flatten(), expected, rtol=0, atol=2e-12)
    model = query_qec_gate_cost(global_phase_cost_key(exponent, (1,)))
    expected_model = get_cost_value(source.controlled(), QECGatesCost())
    assert (model.t, model.clifford, model.rotation) == (
        expected_model.t,
        expected_model.clifford,
        expected_model.rotation,
    )


@pytest.mark.ftqc_reference
def test_symbolic_phase_coefficient_uses_source_argument() -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.basic_gates import GlobalPhase
    from sympy import Symbol

    coefficient = Symbol("coefficient")
    exponent = global_phase_exponent(coefficient)
    assert exponent == GlobalPhase.from_coefficient(coefficient).exponent


def test_nonfinite_coefficient_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        global_phase_exponent(complex(1, float("inf")))

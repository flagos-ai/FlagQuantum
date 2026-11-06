"""Square cost phase against the original Qualtran composition and branches."""

from cmath import exp
from collections import Counter
from itertools import product
from math import pi, sqrt, tau
from typing import Any

import pytest
import torch

from flagquantum.compiler.registers.integer_square import integer_square_ir
from flagquantum.compiler.registers.phase_addition import phase_add_workspace_size
from flagquantum.compiler.registers.phase_gradient import phase_gradient_state_ir
from flagquantum.compiler.registers.scaled_phase_addition import (
    scaled_phase_add_ir,
    scaled_phase_add_terms,
)
from flagquantum.compiler.registers.square_cost_phasing import (
    square_cost_gradient_ir,
    square_cost_zpow_ir,
)
from flagquantum.core.ir import CircuitIR, Instruction
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch
from tests.unit.register_source_graph import (
    SourcePorts,
    SourceResult,
    walk_source_graph,
)
from tests.unit.sparse_register_execution import execute_sparse_register
from tests.unit.test_integer_square import square_layout
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("fraction", [0, 4])
@pytest.mark.parametrize("use_gradient", [False, True])
def test_square_cost_original_graph_phases_and_cleanup(fraction, use_gradient) -> None:
    pytest.importorskip("qualtran")
    from qualtran import QFxp, Register
    from qualtran.bloqs.arithmetic.multiplication import Square
    from qualtran.bloqs.basic_gates import ZPowGate
    from qualtran.bloqs.rotations.phasing_via_cost_function import (
        PhasingViaCostFunction,
    )
    from qualtran.bloqs.rotations.quantum_variable_rotation import (
        QvrPhaseGradient,
        QvrZPow,
    )
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    a, result, square_work = square_layout(2)
    width = 8
    gamma = epsilon = 0.125
    options = dict(cost_fraction_bits=fraction, coefficient=gamma, epsilon=epsilon)
    cost = Register("result", QFxp(len(result), fraction))
    square = Square(2)
    qvr = (QvrPhaseGradient if use_gradient else QvrZPow)(cost, gamma, epsilon)
    source = PhasingViaCostFunction(square, qvr)
    ports = {"a": a}
    square_stream = integer_square_ir(a, result, square_work)
    pre: tuple[Instruction, ...] = ()
    post: tuple[Instruction, ...] = ()
    raw: tuple[Instruction, ...] = ()
    bound: list[Any] = []
    if isinstance(qvr, QvrPhaseGradient):
        m = int(qvr.b_grad)
        graph = qvr.decompose_bloq()
        ((stage, _, _),) = tuple(graph.iter_bloqsoqs())
        scaled = stage.bloq
        terms = scaled_phase_add_terms(
            len(result),
            fraction,
            m,
            scaled.gamma,
            scaled.gamma_dtype.bitsize,
            scaled.gamma_dtype.num_frac,
        )
        required = max(
            (
                phase_add_workspace_size(len(result[part]), m, right_shift=shift)
                for part, shift in terms
            ),
            default=0,
        )
        gradient = tuple(range(width, width + m))
        work = tuple(range(width + m, width + m + required))
        bits = tuple(range(required))
        width += m + required
        ports["phase_grad"] = gradient
        raw = scaled_phase_add_ir(
            result,
            gradient,
            work,
            bits,
            input_fraction_bits=fraction,
            coefficient=scaled.gamma,
            coefficient_width=scaled.gamma_dtype.bitsize,
            coefficient_fraction_bits=scaled.gamma_dtype.num_frac,
        )
        gates = square_cost_gradient_ir(
            a, result, square_work, gradient, work, bits, **options
        )
        increments = tuple(
            source.decompose_bloq().on_classical_vals(a=x, phase_grad=0)["phase_grad"]
            for x in range(4)
        )
        phases = tuple(exp(1j * tau * increment / 2**m) for increment in increments)
        pre = phase_gradient_state_ir(gradient, exponent=-1)
        post = phase_gradient_state_ir(gradient, exponent=-1, inverse=True)
    else:
        gates = square_cost_zpow_ir(a, result, square_work, **options)
        tensor = torch.as_tensor(source.tensor_contract().copy())
        torch.testing.assert_close(
            tensor, torch.diag(tensor.diagonal()), atol=2e-12, rtol=0
        )
        phases = tuple(tensor.diagonal().tolist())

    def leaf(bloq: Any, inputs: SourcePorts, _: SourcePorts) -> SourceResult | None:
        if bloq == square:
            bound.append(bloq)
            assert inputs == {"a": a}
            return square_stream, {"a": a, "result": result}
        if bloq == square.adjoint():
            bound.append(bloq)
            assert inputs == {"a": a, "result": result}
            return square_stream, {"a": a}
        if bloq == qvr:
            bound.append(bloq)
            if isinstance(bloq, QvrPhaseGradient):
                assert inputs == {"result": result, "phase_grad": gradient}
                return raw, inputs
            assert inputs == {"result": result}
        if isinstance(bloq, ZPowGate):
            return (
                Instruction("phase", inputs["q"], {"theta": pi * bloq.exponent}),
            ), inputs
        return None

    expected, outputs = walk_source_graph(source, ports, ports, leaf)
    assert bound == [square, qvr, square.adjoint()]
    assert outputs == ports
    assert_source(gates, expected)
    assert [g for g in gates if g.name == "phase"] == [
        g for g in expected if g.name == "phase"
    ]
    counts = Counter(g.name for g in gates)
    events = counts["measure"]
    model = get_cost_value(source, QECGatesCost())
    assert model == 2 * get_cost_value(square, QECGatesCost()) + get_cost_value(
        qvr, QECGatesCost()
    )
    assert counts["t"] + counts["tdg"] == 4 * events
    assert counts["reset"] == events

    initial = {
        ((x << (width - 1)) | s): exp(1j * (x - 0.37 * s)) / sqrt(8)
        for x, s in product(range(4), (0, 1))
    }
    expected_complex = {
        i: value * phases[i >> (width - 1)] for i, value in initial.items()
    }
    if isinstance(qvr, QvrPhaseGradient):

        def word(value: int) -> tuple[int, ...]:
            return tuple(value >> k & 1 for k in reversed(range(width)))

        incoming = []
        outgoing = []
        for x, y in product(range(4), range(2**m)):
            a_word = x << (width - 2)
            incoming.append(word(a_word | (y << required)))
            outgoing.append(word(a_word | (((y + increments[x]) % 2**m) << required)))
        assert certified_and_channel_map(gates, tuple(incoming)) == (
            tuple(outgoing),
            events,
        )
        for seed in (4, 19, 73):
            observed = execute_sparse_register(
                CircuitIR(width + 1, (*pre, *gates, *post)), initial, seed=seed
            )
            assert all(
                abs(observed.get(i, 0) - expected_complex.get(i, 0)) < 2e-12
                for i in observed.keys() | expected_complex.keys()
            )
    else:
        dense = torch.zeros(1 << (width + 1), dtype=torch.complex128)
        dense_expected = torch.zeros_like(dense)
        for i, value in initial.items():
            dense[i] = value
            dense_expected[i] = expected_complex[i]
        for history in product((0, 1), repeat=events):
            torch.testing.assert_close(
                forced_branch(dense, gates, history),
                dense_expected / sqrt(2**events),
                atol=2e-12,
                rtol=0,
            )


def test_square_cost_append_trajectory() -> None:
    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic

    gates = square_cost_zpow_ir(
        *square_layout(2), cost_fraction_bits=0, coefficient=0.125, epsilon=0.125
    )
    state = torch.zeros(256, dtype=torch.complex128)
    state[0], state[3 << 6] = 0.8, 0.6j
    expected = state.clone()
    expected[3 << 6] *= exp(1j * pi / 4)
    circuit = DynamicCircuit(8, inputs=state, dtype=torch.complex128)
    circuit.append(FieldOperation(8, gates))
    actual = run_dynamic(circuit, shots=8, seed=17, strategy="trajectory")
    torch.testing.assert_close(
        actual.final_states, expected.expand(8, -1), atol=2e-12, rtol=0
    )

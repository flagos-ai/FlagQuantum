"""Pinned cost-function port graph and coherent measured erasure."""

from cmath import exp
from collections import Counter
from itertools import product
from math import pi, tau
from typing import Any

import pytest
import torch

from flagquantum.compiler.registers.cost_function_phasing import (
    RegisterPorts,
    cost_function_phasing_ir,
)
from flagquantum.compiler.registers.hamming_weight import hamming_weight_ir
from flagquantum.compiler.registers.phase_addition import phase_add_workspace_size
from flagquantum.compiler.registers.phase_gradient import phase_gradient_state_ir
from flagquantum.compiler.registers.qvr_phase_gradient import qvr_phase_gradient_ir
from flagquantum.compiler.registers.qvr_zpow import qvr_zpow_ir
from flagquantum.compiler.registers.scaled_phase_addition import (
    scaled_phase_add_ir,
    scaled_phase_add_terms,
)
from flagquantum.core.ir import Instruction
from tests.unit.measurement_checks import forced_branch
from tests.unit.register_source_graph import (
    SourcePorts,
    SourceResult,
    walk_source_graph,
)
from tests.unit.test_hamming_weight import layout
from tests.unit.test_hamming_weight_phasing import source_count_leaf
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("fraction", [0, 2])
@pytest.mark.parametrize("use_gradient", [False, True])
def test_locked_cost_graph_and_all_branches(fraction: int, use_gradient: bool) -> None:
    pytest.importorskip("qualtran")
    from qualtran import QFxp, Register
    from qualtran.bloqs.arithmetic.hamming_weight import HammingWeightCompute
    from qualtran.bloqs.basic_gates import ZPowGate
    from qualtran.bloqs.rotations.phasing_via_cost_function import (
        PhasingViaCostFunction,
    )
    from qualtran.bloqs.rotations.quantum_variable_rotation import (
        QvrPhaseGradient,
        QvrZPow,
    )
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    x, junk, count, count_bits = layout(2)
    parts = {"x": x, "junk": junk, "out": count}
    options = dict(cost_fraction_bits=fraction, coefficient=0.125, epsilon=0.25)
    cost_reg = Register("out", QFxp(len(count), fraction))
    gamma, epsilon = options["coefficient"], options["epsilon"]
    oracle = QvrPhaseGradient if use_gradient else QvrZPow
    qvr = oracle(cost_reg, gamma, epsilon)
    evaluator = HammingWeightCompute(len(x))
    source = PhasingViaCostFunction(evaluator, qvr)
    base_width = len(x) + len(junk) + len(count)
    outer_ports = {"x": x}
    extras: dict[str, tuple[int, ...]] = {}
    before: tuple[Instruction, ...] = ()
    after: tuple[Instruction, ...] = ()
    raw: tuple[Instruction, ...] = ()
    width = base_width
    if isinstance(qvr, QvrPhaseGradient):
        m = int(qvr.b_grad)
        graph = qvr.decompose_bloq()
        ((stage, _, _),) = tuple(graph.iter_bloqsoqs())
        scaled = stage.bloq
        terms = scaled_phase_add_terms(
            len(count),
            fraction,
            m,
            scaled.gamma,
            scaled.gamma_dtype.bitsize,
            scaled.gamma_dtype.num_frac,
        )
        workspace = max(
            (
                phase_add_workspace_size(len(count[part]), m, right_shift=shift)
                for part, shift in terms
            ),
            default=0,
        )
        gradient = tuple(range(base_width, base_width + m))
        work = tuple(range(base_width + m, base_width + m + workspace))
        slots = tuple(range(40, 40 + workspace))
        extras = {"phase_grad": gradient, "work": work}
        outer_ports["phase_grad"] = gradient
        width += m + workspace
        raw = scaled_phase_add_ir(
            count,
            gradient,
            work,
            slots,
            input_fraction_bits=fraction,
            coefficient=scaled.gamma,
            coefficient_width=scaled.gamma_dtype.bitsize,
            coefficient_fraction_bits=scaled.gamma_dtype.num_frac,
        )
        before = phase_gradient_state_ir(gradient, exponent=-1)
        after = phase_gradient_state_ir(gradient, exponent=-1, inverse=True)
        phases = tuple(
            tau * int(graph.on_classical_vals(out=i, phase_grad=0)["phase_grad"]) / 2**m
            for i in range(3)
        )
    else:
        phases = tuple(tau * gamma * i / 2**fraction for i in range(3))

    def phase(ports: RegisterPorts) -> tuple[Instruction, ...]:
        if isinstance(qvr, QvrPhaseGradient):
            return qvr_phase_gradient_ir(
                ports["out"],
                ports["phase_grad"],
                ports["work"],
                slots,
                **options,
            )
        return qvr_zpow_ir(ports["out"], **options)

    def leaf(bloq: Any, ports: SourcePorts, _: SourcePorts) -> SourceResult | None:
        count_leaf = source_count_leaf(bloq, ports, parts, count_bits)
        if count_leaf is not None:
            return count_leaf
        if isinstance(bloq, QvrPhaseGradient):
            assert bloq == qvr and ports == {"out": count, "phase_grad": gradient}
            return raw, ports
        if isinstance(bloq, ZPowGate):
            return (
                Instruction("phase", ports["q"], {"theta": pi * bloq.exponent}),
            ), ports
        return None

    gates = cost_function_phasing_ir(
        {"x": x},
        {"junk": junk, "out": count},
        extras,
        cost_names=("out",),
        evaluate=lambda p: hamming_weight_ir(p["x"], p["junk"], p["out"]),
        phase=phase,
        erase=lambda p: hamming_weight_ir(
            p["x"], p["junk"], p["out"], adjoint=True, classical_bits=count_bits
        ),
    )
    expected, outputs = walk_source_graph(source, outer_ports, outer_ports, leaf)
    assert outputs == outer_ports
    assert_source(gates, expected)
    assert [gate for gate in gates if gate.name == "phase"] == [
        gate for gate in expected if gate.name == "phase"
    ]
    counts = Counter(gate.name for gate in gates)
    model = get_cost_value(source, QECGatesCost())
    assert counts["measure"] == counts["reset"]
    assert counts["t"] + counts["tdg"] == 4 * counts["measure"]
    assert model == (
        get_cost_value(evaluator, QECGatesCost())
        + get_cost_value(qvr, QECGatesCost())
        + get_cost_value(evaluator.adjoint(), QECGatesCost())
    )
    assert model.measurement == 1

    # Full complex output, including a coherent spectator on the last wire.
    initial = torch.zeros(1 << (width + 1), dtype=torch.complex128)
    result = torch.zeros_like(initial)
    for value, spectator in product(range(4), (0, 1)):
        i = (value << (width - 1)) | spectator
        initial[i] = exp(1j * (value - 0.37 * spectator)) / 8**0.5
        result[i] = initial[i] * exp(1j * phases[value.bit_count()])
    for history in product((0, 1), repeat=counts["measure"]):
        torch.testing.assert_close(
            forced_branch(initial, (*before, *gates, *after), history),
            result * 2 ** (-counts["measure"] / 2),
            atol=2e-12,
            rtol=0,
        )


@pytest.mark.parametrize(
    "outputs,extras,names,phase_wire,match",
    [
        ({"x": (1,)}, {}, ("x",), 1, "names must be distinct"),
        ({"out": (1,)}, {}, (), 1, "distinct cost"),
        ({"out": (1,)}, {}, ("out", "out"), 1, "distinct cost"),
        ({"out": (1,)}, {}, ("missing",), 1, "nonempty evaluation output"),
        ({"out": ()}, {}, ("out",), 1, "nonempty evaluation output"),
        ({"out": (1,)}, {"grad": (0,)}, ("out",), 1, "distinct nonnegative"),
        ({"out": (1,), "junk": (2,)}, {}, ("out",), 2, "outside its port"),
    ],
)
def test_cost_phasing_rejects_invalid_binding(
    outputs, extras, names, phase_wire, match
):
    with pytest.raises(ValueError, match=match):
        cost_function_phasing_ir(
            {"x": (0,)},
            outputs,
            extras,
            cost_names=names,
            evaluate=lambda _: (),
            phase=lambda _: (Instruction("z", (phase_wire,)),),
            erase=lambda _: (),
        )

"""Pinned cost-function port graph and coherent measured erasure."""

from cmath import exp
from collections import Counter
from itertools import product
from math import pi, tau
from typing import Any, TypedDict

import pytest
import torch

from flagquantum.compiler.registers.bits.hamming_weight import hamming_weight_ir
from flagquantum.compiler.registers.phase.cost_function_phasing import (
    RegisterPorts,
    cost_function_phasing_ir,
)
from flagquantum.compiler.registers.phase.phase_addition import phase_add_workspace_size
from flagquantum.compiler.registers.phase.phase_gradient import phase_gradient_state_ir
from flagquantum.compiler.registers.phase.qvr_phase_gradient import (
    qvr_phase_gradient_ir,
)
from flagquantum.compiler.registers.phase.qvr_zpow import qvr_zpow_ir
from flagquantum.compiler.registers.phase.scaled_phase_addition import (
    scaled_phase_add_terms,
)
from flagquantum.core.ir import CircuitIR, Instruction
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch
from tests.unit.register_source_graph import (
    SourcePorts,
    SourceResult,
    walk_source_graph,
)
from tests.unit.sparse_register_execution import execute_sparse_register
from tests.unit.test_hamming_phasing_file import _source_qvr_gates
from tests.unit.test_hamming_weight import layout
from tests.unit.test_hamming_weight_phasing import source_count_leaf
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit


class PhaseOptions(TypedDict):
    cost_fraction_bits: int
    coefficient: float
    epsilon: float


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("fraction", [0, 2])
@pytest.mark.parametrize("use_gradient", [False, True])
def test_locked_cost_graph_and_all_branches(fraction: int, use_gradient: bool) -> None:
    _check_hamming_cost(2, fraction, 0.125, 0.25, use_gradient)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("n", [2, 3])
@pytest.mark.parametrize("exponent", [0.1, 1.20345, -1.1934341])
@pytest.mark.parametrize(
    "normalized,use_gradient", list(product((False, True), repeat=2))
)
@pytest.mark.parametrize("fractional", [False, True])
def test_original_hamming_cost_grid(
    n: int, exponent: float, normalized: bool, use_gradient: bool, fractional: bool
) -> None:
    bits = n.bit_length()
    gamma = exponent / 2
    if not normalized:
        gamma *= 1 << bits
    if not fractional:
        gamma /= 1 << bits
    _check_hamming_cost(n, bits if fractional else 0, gamma, 5e-4, use_gradient)


def _check_hamming_cost(
    n: int, fraction: int, gamma: float, epsilon: float, use_gradient: bool
) -> None:
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

    x, junk, count, count_bits = layout(n)
    parts = {"x": x, "junk": junk, "out": count}
    options = PhaseOptions(
        cost_fraction_bits=fraction, coefficient=gamma, epsilon=epsilon
    )
    cost_reg = Register("out", QFxp(len(count), fraction))
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
    m = workspace = 0
    rotations: list[Instruction] = []
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
        slots = tuple(range(workspace))
        extras = {"phase_grad": gradient, "work": work}
        outer_ports["phase_grad"] = gradient
        width += m + workspace
        raw, _ = _source_qvr_gates(
            qvr, {"out": count, "phase_grad": gradient}, work, adjoint=False
        )
        increments = tuple(
            int(graph.on_classical_vals(out=i, phase_grad=0)["phase_grad"])
            for i in range(n + 1)
        )
        phases = tuple(tau * value / 2**m for value in increments)
        if m <= 8:
            before = phase_gradient_state_ir(gradient, exponent=-1)
            after = phase_gradient_state_ir(gradient, exponent=-1, inverse=True)

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
            gate = Instruction("phase", ports["q"], {"theta": pi * bloq.exponent})
            rotations.append(gate)
            return (gate,), ports
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
    assert model.measurement == len(junk)

    if not m:
        phases = tuple(
            sum(
                float(g.params["theta"])
                for g in rotations
                if weight >> (len(count) - 1 - count.index(g.wires[0])) & 1
            )
            for weight in range(n + 1)
        )

    events = counts["measure"]
    if m and (width > 14 or events > 6):
        # Certify exact balanced feedback on all source inputs and representative
        # gradient words, including translation wrap boundaries.
        mask = (1 << m) - 1
        values = range(1 << m) if m <= 6 else (0, 1, mask // 2, mask - 1, mask)
        incoming = []
        outgoing = []
        for value, word in product(range(1 << n), values):
            logical = value << (width - n)
            moved = (word + increments[value.bit_count()]) & mask
            incoming.append(logical | (word << workspace))
            outgoing.append(logical | (moved << workspace))

        def bits(address: int) -> tuple[int, ...]:
            return tuple(address >> i & 1 for i in reversed(range(width)))

        assert certified_and_channel_map(gates, tuple(map(bits, incoming))) == (
            tuple(map(bits, outgoing)),
            events,
        )
        if m > 8:
            state = {
                r: exp(0.37j * i) / len(incoming) ** 0.5 for i, r in enumerate(incoming)
            }
            expected_sparse = dict(zip(outgoing, state.values(), strict=True))
            for seed in (7, 19):
                observed = execute_sparse_register(
                    CircuitIR(width, gates), dict(state), seed=seed
                )
                assert all(
                    abs(observed.get(i, 0) - expected_sparse.get(i, 0)) < 2e-12
                    for i in observed.keys() | expected_sparse.keys()
                )
            return

    semantic = {
        (value << (width + 1 - n))
        | spectator: exp(1j * (value - 0.37 * spectator)) / (2 << n) ** 0.5
        for value, spectator in product(range(1 << n), (0, 1))
    }
    expected_semantic = {
        i: amp * exp(1j * phases[(i >> (width + 1 - n)).bit_count()])
        for i, amp in semantic.items()
    }
    stream = (*before, *gates, *after)
    if width <= 14 and events <= 6:
        initial = torch.zeros(1 << (width + 1), dtype=torch.complex128)
        result = torch.zeros_like(initial)
        for i, amp in semantic.items():
            initial[i] = amp
            result[i] = expected_semantic[i]
        for history in product((0, 1), repeat=events):
            torch.testing.assert_close(
                forced_branch(initial, stream, history),
                result * 2 ** (-events / 2),
                atol=2e-12,
                rtol=0,
            )
    else:
        for seed in (7, 19):
            observed = execute_sparse_register(
                CircuitIR(width + 1, stream), dict(semantic), seed=seed
            )
            assert all(
                abs(observed.get(i, 0) - expected_semantic.get(i, 0)) < 2e-12
                for i in observed.keys() | expected_semantic.keys()
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
    outputs: RegisterPorts,
    extras: RegisterPorts,
    names: tuple[str, ...],
    phase_wire: int,
    match: str,
) -> None:
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

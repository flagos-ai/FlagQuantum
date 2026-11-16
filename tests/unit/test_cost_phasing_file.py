"""Verify tutorial and original Square fixture bindings against locked Qualtran."""

from cmath import exp
from collections import Counter
from dataclasses import fields
from hashlib import sha1
from importlib.metadata import version
from itertools import product
from math import pi, sqrt, tau
from pathlib import Path
from typing import Any

import pytest

from examples.algorithms.square_cost_phasing_catalog import SquareCostPhasingExample
from flagquantum.compiler._component_trace.resource_model import (
    GateCost,
    estimate_component_resources,
)
from flagquantum.compiler.registers.arithmetic.integer_square import integer_square_ir
from flagquantum.compiler.registers.phase.phase_gradient import phase_gradient_state_ir
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.dynamic import DynamicCircuit
from tests.unit.measurement_checks import certified_and_channel_map
from tests.unit.register_source_graph import (
    SourcePorts,
    SourceResult,
    walk_source_graph,
)
from tests.unit.sparse_register_execution import execute_sparse_register
from tests.unit.test_hamming_phasing_file import _source_qvr_gates
from tests.unit.test_relative_controlled_swap import assert_source_dependencies

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module")
def source_module() -> Any:
    import kickmix
    from qualtran.bloqs.rotations import phasing_via_cost_function

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    folder = Path(phasing_via_cost_function.__file__).parent
    for name, digest in (
        ("__init__.py", "e6c63bdf43262fd6e3771a9eebd5c8e7ad75d4b8"),
        ("phasing_via_cost_function.ipynb", "b1c45ff7e36c3d82b43166e98c39d5ba89c68762"),
        ("phasing_via_cost_function.py", "7e4204a0dcc72fa5785b2e58f4c209d0f083dd76"),
        (
            "phasing_via_cost_function_test.py",
            "959d1c2e1b94b83c505aa8d27747002c61540406",
        ),
    ):
        raw = (folder / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == digest
    return phasing_via_cost_function


def verify_square_binding(
    op: SquareCostPhasingExample, source: Any
) -> tuple[int, int, int, int, int]:
    from qualtran.bloqs.basic_gates import ZPowGate

    n, m = op.width, op.gradient_width
    boundary = op.square_end + m
    width = op.qubit_count
    inputs = tuple(range(n))
    cost = tuple(range(n, 3 * n))
    square = integer_square_ir(inputs, cost, tuple(range(3 * n, op.square_end)))
    binding = {"a": inputs}
    gradient = tuple(range(op.square_end, boundary))
    local_phase: tuple[Instruction, ...] = ()
    qvr = source.phase_oracle
    if op.via_gradient:
        assert m == qvr.b_grad
        binding["phase_grad"] = gradient
        local_phase, _ = _source_qvr_gates(
            qvr,
            {"out": cost, "phase_grad": gradient},
            tuple(range(boundary, width)),
            adjoint=False,
        )
    source_rotations: list[Instruction] = []
    stages: list[str] = []
    evaluation = source.cost_eval_oracle

    def leaf(
        bloq: Any, ports: SourcePorts, _desired: SourcePorts
    ) -> SourceResult | None:
        if bloq == evaluation:
            assert ports == {"a": inputs}
            stages.append("Square")
            return square, {"a": inputs, "result": cost}
        if bloq == evaluation.adjoint():
            assert ports == {"a": inputs, "result": cost}
            stages.append("Square†")
            return square, {"a": inputs}
        if bloq == qvr:
            stages.append(type(qvr).__name__)
            if op.via_gradient:
                assert ports == {"result": cost, "phase_grad": gradient}
                return local_phase, ports
            assert ports == {"result": cost}
        if isinstance(bloq, ZPowGate):
            gate = Instruction("phase", ports["q"], {"theta": pi * bloq.exponent})
            source_rotations.append(gate)
            return (gate,), ports
        return None

    assert [(r.name, r.dtype.num_qubits, r.side.name) for r in op.signature] == [
        (r.name, r.bitsize, r.side.name) for r in source.signature
    ]
    reference, outputs = walk_source_graph(source, binding, binding, leaf)
    assert outputs == binding
    assert stages == ["Square", type(qvr).__name__, "Square†"]
    assert op.call_graph().leaf_counts == dict.fromkeys(stages, 1)
    circuit = DynamicCircuit(width)
    circuit.append(op)
    ir = circuit.to_ir()
    gates = ir.instructions
    assert_source_dependencies(gates, reference)
    counts = Counter(g.name for g in gates)
    assert counts["measure"] == counts["reset"]
    logical = range(1 << n)

    def bits(x: int) -> tuple[int, ...]:
        return tuple(x >> i & 1 for i in reversed(range(width)))

    if m:
        mask = (1 << m) - 1
        qvr_graph = qvr.decompose_bloq()
        shifts = {
            x: int(
                qvr_graph.on_classical_vals(result=x * x, phase_grad=0)["phase_grad"]
            )
            for x in logical
        }
        words = (
            range(1 << m)
            if m <= 6
            else (0, 1, mask // 2, mask // 2 + 1, mask - 1, mask)
        )
        pairs = tuple(product(logical, words))
        initial = [
            (x << (width - n)) | (word << (width - boundary)) for x, word in pairs
        ]
        final = [
            (x << (width - n)) | (((word + shifts[x]) & mask) << (width - boundary))
            for x, word in pairs
        ]
        certified = certified_and_channel_map(gates, tuple(map(bits, initial)))
        assert certified == (tuple(map(bits, final)), counts["measure"])
        if m <= 8:
            pre = phase_gradient_state_ir(gradient, exponent=-1)
            post = phase_gradient_state_ir(gradient, exponent=-1, inverse=True)
            kickback = CircuitIR(width, (*pre, *gates, *post))
            state = {
                x << (width - n): exp(0.37j * x) / sqrt(len(logical)) for x in logical
            }
            expected = {
                i: amp * exp(1j * tau * shifts[i >> (width - n)] / (1 << m))
                for i, amp in state.items()
            }
            observed = execute_sparse_register(kickback, state, seed=11)
            assert all(
                abs(observed.get(i, 0) - expected.get(i, 0)) < 2e-12
                for i in observed.keys() | expected.keys()
            )
    else:
        initial = [x << (width - n) for x in logical]
        cost_words = [
            r | ((x * x) << (width - 3 * n))
            for x, r in zip(logical, initial, strict=True)
        ]
        assert certified_and_channel_map(square, tuple(map(bits, initial))) == (
            tuple(map(bits, cost_words)),
            counts["measure"] // 2,
        )
        assert certified_and_channel_map(square, tuple(map(bits, cost_words))) == (
            tuple(map(bits, initial)),
            counts["measure"] // 2,
        )
        state = {r: exp(0.37j * x) / sqrt(len(logical)) for x, r in enumerate(initial)}
        final = initial
    if not m:
        expected = {
            a: state[a]
            * exp(
                1j
                * sum(
                    float(g.params["theta"])
                    for g in source_rotations
                    if cost_words[x] & (1 << (width - g.wires[0] - 1))
                )
            )
            for x, a in enumerate(initial)
        }
    else:
        state = {r: exp(0.37j * j) / sqrt(len(initial)) for j, r in enumerate(initial)}
        expected = dict(zip(final, state.values(), strict=True))
    for seed in (4, 19):
        observed = execute_sparse_register(ir, dict(state), seed=seed)
        assert all(
            abs(observed.get(i, 0) - expected.get(i, 0)) < 2e-12
            for i in observed.keys() | expected.keys()
        )

    return (
        width,
        counts["t"] + counts["tdg"],
        counts["cx"],
        counts["phase"],
        counts["measure"],
    )


@pytest.mark.parametrize("via_gradient", [False, True])
def test_five_bit_tutorial(via_gradient: bool, source_module: Any) -> None:
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    op = SquareCostPhasingExample(via_gradient=via_gradient)
    factory = (
        source_module._square_via_phase_gradient
        if via_gradient
        else source_module._square_via_zpow_phasing
    )
    source = factory.make()
    assert verify_square_binding(op, source) == (
        (98, 1636, 4062, 0, 409) if via_gradient else (35, 160, 394, 10, 40)
    )
    declared = estimate_component_resources(op.resource_model())
    cost = get_cost_value(source, QECGatesCost())
    assert declared.gates == GateCost(
        **{field.name: getattr(cost, field.name) for field in fields(GateCost)}
    )
    assert declared.peak_logical_qubits == get_cost_value(source, QubitCount())


@pytest.mark.parametrize("gamma", [0.1, 1.20345, -1.1934341])
@pytest.mark.parametrize(
    "normalized,via_gradient", list(product((False, True), repeat=2))
)
@pytest.mark.parametrize("fractional", [False, True])
def test_original_square_grid(
    gamma: float,
    normalized: bool,
    via_gradient: bool,
    fractional: bool,
    source_module: Any,
) -> None:
    from qualtran import QFxp, Register
    from qualtran.bloqs.arithmetic import Square
    from qualtran.bloqs.rotations.quantum_variable_rotation import (
        QvrPhaseGradient,
        QvrZPow,
    )

    coefficient = gamma * (1 if normalized else 16) / (1 if fractional else 16)
    op = SquareCostPhasingExample(
        via_gradient, 2, coefficient, 0.05, normalize=fractional
    )
    qvr = QvrPhaseGradient if via_gradient else QvrZPow
    reference = source_module.PhasingViaCostFunction(
        Square(2),
        qvr(Register("result", QFxp(4, op.fraction)), coefficient, 0.05),
    )
    verify_square_binding(op, reference)

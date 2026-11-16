"""Source-backed tutorial, resource sweeps and coherent QVR binding."""

from cmath import exp
from collections import Counter
from hashlib import sha1
from importlib.metadata import version
from itertools import product
from math import ceil, log2, pi, sqrt, tau
from pathlib import Path
from typing import Any

import pytest
import torch
from sympy import symbols

from examples.algorithms.qvr_catalog import QvrExample
from flagquantum.compiler._component_trace.resource_model import (
    estimate_component_resources,
)
from flagquantum.compiler.registers.phase.phase_gradient import phase_gradient_state_ir
from flagquantum.compiler.registers.phase.qvr_phase_gradient import (
    qvr_phase_sizing,
    source_qvr_phase_gradient_cost,
    source_symbolic_qvr_gradient_cost,
)
from flagquantum.compiler.registers.phase.qvr_zpow import (
    source_qvr_zpow_declaration,
    source_qvr_zpow_synthesis_t_count,
)
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.dynamic import DynamicCircuit
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch
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
    from qualtran.bloqs.rotations import quantum_variable_rotation

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    folder = Path(quantum_variable_rotation.__file__).parent
    for name, digest in (
        ("__init__.py", "e6c63bdf43262fd6e3771a9eebd5c8e7ad75d4b8"),
        ("quantum_variable_rotation.ipynb", "e04a9a4f90b717eb8dbd654b445699edc4268773"),
        ("quantum_variable_rotation.py", "8177ea80e8150c79dac75e79e0868054fa2db8a4"),
        (
            "quantum_variable_rotation_test.py",
            "365027236f79d3723e619f6175654f2b285c3d14",
        ),
    ):
        raw = (folder / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == digest
    return quantum_variable_rotation


def assert_signatures_and_calls(op: QvrExample, source: Any) -> None:
    assert [
        (r.name, r.dtype.num_qubits, dict(r.dtype.parameters), r.side.name)
        for r in op.signature
    ] == [
        (
            r.name,
            r.bitsize,
            {"fractional_width": r.dtype.num_frac, "signed": r.dtype.signed},
            r.side.name,
        )
        for r in source.signature
    ]
    ((child, count),) = source.bloq_counts().items()
    if op.via_gradient:
        label = (
            f"AddScaledValIntoPhaseReg({child.x_dtype.bitsize}, {child.x_dtype.num_frac}, "
            f"{child.phase_bitsize}, {child.gamma}, "
            f"{child.gamma_dtype.bitsize}, {child.gamma_dtype.num_frac})"
        )
    else:
        label = f"ZPowGate({child.exponent}, {child.eps})"
    assert op.call_graph().leaf_counts == {label: count}


def check_direct(op: QvrExample, source: Any) -> Counter[str]:
    from qualtran.bloqs.basic_gates import ZPowGate

    assert_signatures_and_calls(op, source)
    n = op.width
    ports = {"x": tuple(range(n))}

    def leaf(
        bloq: Any, inputs: SourcePorts, _desired: SourcePorts
    ) -> SourceResult | None:
        if isinstance(bloq, ZPowGate):
            return (
                Instruction("phase", inputs["q"], {"theta": pi * bloq.exponent}),
            ), inputs
        return None

    stream, _ = walk_source_graph(source, ports, ports, leaf)
    circuit = DynamicCircuit(n)
    circuit.append(op)
    gates = circuit.to_ir().instructions
    assert_source_dependencies(gates, stream)
    addresses = torch.arange(1 << n, dtype=torch.float64)
    state = torch.complex(torch.ones_like(addresses), addresses % 5 - 2)
    state /= state.norm()
    angles = torch.zeros_like(addresses)
    for gate in stream:
        take = addresses.to(torch.int64) >> (n - 1 - gate.wires[0]) & 1
        angles += take.to(torch.float64) * float(gate.params["theta"])
    expected = state * torch.exp(1j * angles)
    torch.testing.assert_close(
        forced_branch(state, gates, ()), expected, atol=2e-12, rtol=0
    )
    exact = torch.exp(1j * (tau * op.coefficient / 2**op.fraction) * addresses)
    assert (
        float(torch.max(torch.abs(torch.exp(1j * angles) - exact)))
        <= op.epsilon + 2e-12
    )
    return Counter(g.name for g in gates)


def check_gradient(
    op: QvrExample,
    source: Any,
    cost_words: tuple[int, ...],
) -> tuple[Counter[str], float]:
    assert_signatures_and_calls(op, source)
    n, m, width = op.width, op.gradient_width, op.qubit_count
    required = width - n - m
    cost = tuple(range(n))
    gradient = tuple(range(n, n + m))
    work = tuple(range(n + m, width))
    reference, _ = _source_qvr_gates(
        source,
        {"out": cost, "phase_grad": gradient},
        work,
        adjoint=op.adjoint,
    )
    circuit = DynamicCircuit(width)
    circuit.append(op)
    ir = circuit.to_ir()
    gates = ir.instructions
    assert_source_dependencies(gates, reference)
    counts = Counter(g.name for g in gates)
    reference_graph = source.decompose_bloq()
    increments = {
        x: int(reference_graph.on_classical_vals(x=x, phase_grad=0)["phase_grad"])
        for x in cost_words
    }
    mask = (1 << m) - 1
    words = range(1 << m) if m <= 4 else (0, 1, mask // 2, mask - 1, mask)
    pairs = tuple(product(cost_words, words))

    def address(x: int, y: int) -> int:
        return x << (width - n) | y << required

    def bits(a: int) -> tuple[int, ...]:
        return tuple(a >> i & 1 for i in reversed(range(width)))

    start = [address(x, y) for x, y in pairs]
    sign = -1 if op.adjoint else 1
    end = [address(x, (y + sign * increments[x]) & mask) for x, y in pairs]
    assert certified_and_channel_map(gates, tuple(map(bits, start))) == (
        tuple(map(bits, end)),
        counts["measure"],
    )
    error = max(
        abs(
            exp(1j * tau * increments[x] / (1 << m))
            - exp(1j * tau * op.coefficient * x / (1 << op.fraction))
        )
        for x in cost_words
    )
    if width <= 12 and counts["measure"] <= 5:
        pre = phase_gradient_state_ir(gradient, exponent=-1)
        post = phase_gradient_state_ir(gradient, exponent=-1, inverse=True)
        initial = torch.zeros(1 << width, dtype=torch.complex128)
        expected = torch.zeros_like(initial)
        for x in cost_words:
            amp = exp(0.19j * x) / sqrt(len(cost_words))
            initial[address(x, 0)] = torch.tensor(amp, dtype=torch.complex128)
            angle = tau * sign * increments[x] / (1 << m)
            expected[address(x, 0)] = torch.tensor(
                amp * exp(1j * angle), dtype=torch.complex128
            )
        for history in product((0, 1), repeat=counts["measure"]):
            torch.testing.assert_close(
                forced_branch(initial, (*pre, *gates, *post), history),
                expected * 2 ** (-counts["measure"] / 2),
                atol=2e-12,
                rtol=0,
            )
    else:
        initial_sparse = {
            a: exp(0.17j * i) / sqrt(len(start)) for i, a in enumerate(start)
        }
        expected_sparse = dict(zip(end, initial_sparse.values(), strict=True))
        for seed in (4, 23):
            actual = execute_sparse_register(
                CircuitIR(width, gates), dict(initial_sparse), seed=seed
            )
            assert all(
                abs(actual.get(i, 0) - expected_sparse.get(i, 0)) < 2e-12
                for i in actual.keys() | expected_sparse.keys()
            )
    return counts, error


@pytest.mark.parametrize("via_gradient", [False, True])
def test_original_factories(via_gradient: bool, source_module: Any) -> None:
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    op = QvrExample(epsilon=1e-4 if via_gradient else 1e-2, via_gradient=via_gradient)
    factory = (
        source_module._qvr_phase_gradient if via_gradient else source_module._qvr_zpow
    )
    source = factory.make()
    if via_gradient:
        counts, error = check_gradient(op, source, (0, 1, 17, 1023, 2047, 4095))
        assert error == pytest.approx(0.00010785802409596983, abs=2e-12)
        assert error > op.epsilon
        assert (
            op.qubit_count,
            counts["t"] + counts["tdg"],
            counts["cx"],
            counts["measure"],
        ) == (49, 484, 1244, 121)
    else:
        assert check_direct(op, source) == {"phase": 6}
    model = estimate_component_resources(op.resource_model())
    original = get_cost_value(source, QECGatesCost())
    assert model.gates.rotation == original.rotation
    assert model.gates.toffoli == original.toffoli
    assert model.peak_logical_qubits == get_cost_value(source, QubitCount())


@pytest.mark.parametrize(
    "epsilon,rotations", [(0.5, 1), (0.1, 3), (0.01, 6), (0.001, 10), (0.0001, 12)]
)
def test_original_direct_precision_grid(
    epsilon: float, rotations: int, source_module: Any
) -> None:
    op = QvrExample(epsilon=epsilon)
    assert check_direct(op, source_module.QvrZPow.from_bitsize(12, 0.1, epsilon)) == {
        "phase": rotations
    }


@pytest.mark.parametrize(
    "gamma,epsilon",
    list(
        product(
            (1 / 2, 1 / 4, 1 / 8, 1 / 16, 15 / 16),
            (1e-2, 1e-3, 1e-4, 1e-5),
        )
    )
    + [(0.1, 1e-2)],
)
def test_original_gradient_angle_grid(
    gamma: float, epsilon: float, source_module: Any
) -> None:
    from qualtran import QFxp, Register

    for fraction in (0, 1):
        coefficient = gamma if fraction else gamma / 2
        op = QvrExample(
            1, coefficient, epsilon, fraction_bits=fraction, via_gradient=True
        )
        source = source_module.QvrPhaseGradient(
            Register("x", QFxp(1, fraction)), coefficient, epsilon
        )
        _, error = check_gradient(op, source, (0, 1))
        assert error <= epsilon + 2e-12


@pytest.mark.parametrize("fraction", [0, 9])
def test_gradient_narrower_than_cost(fraction: int, source_module: Any) -> None:
    from qualtran import QFxp, Register

    gamma = (2**20 - 1) / 2**20
    op = QvrExample(9, gamma, 0.1, fraction_bits=fraction, via_gradient=True)
    source = source_module.QvrPhaseGradient(
        Register("x", QFxp(9, fraction)), gamma, 0.1
    )
    assert op.gradient_width < 9
    _, error = check_gradient(op, source, tuple(range(1 << 9)))
    pinned_error = 0.24178212263757737 if fraction == 0 else 0.2326313220089823
    assert error == pytest.approx(pinned_error, abs=2e-12)
    assert error > op.epsilon


def test_append_gradient_adjoint(source_module: Any) -> None:
    op = QvrExample(2, 0.5, 0.5, via_gradient=True, adjoint=True)
    source = source_module.QvrPhaseGradient.from_bitsize(2, 0.5, 0.5)
    counts, error = check_gradient(op, source, (0, 1, 2, 3))
    assert counts["measure"] > 0
    assert error <= op.epsilon


def test_default_source_precision_overrun(source_module: Any) -> None:
    source = source_module._qvr_phase_gradient.make()
    graph = source.decompose_bloq()
    width = 1 << source.b_grad
    errors = (
        (
            abs(
                exp(
                    1j
                    * tau
                    * graph.on_classical_vals(x=x, phase_grad=0)["phase_grad"]
                    / width
                )
                - exp(1j * tau * source.gamma * x / 4096)
            ),
            x,
        )
        for x in range(4096)
    )
    error, word = max(errors)
    assert word == 4095
    assert error == pytest.approx(0.00010785802409596983, abs=2e-12)
    assert error > source.eps


@pytest.mark.parametrize("gamma", [1.0, 0.1, 0.01])
def test_original_one_bit_direct_rotations(gamma: float, source_module: Any) -> None:
    from qualtran import QFxp, Register

    for fraction in (0, 1):
        coefficient = gamma if fraction else gamma / 2
        op = QvrExample(1, coefficient, 1e-9, fraction_bits=fraction)
        source = source_module.QvrZPow(Register("x", QFxp(1, fraction)), coefficient)
        assert check_direct(op, source) == {"phase": 1}


def test_original_worst_case_gradient_sizing(source_module: Any) -> None:
    for n in range(2, 50):
        epsilon = tau / 2**n
        coefficient = (2**n - 1) / 2**n
        sizing = qvr_phase_sizing(n, n, coefficient, epsilon)
        source = source_module.QvrPhaseGradient.from_bitsize(n, coefficient, epsilon)
        assert sizing.format.coefficient_width == source.gamma_dtype.bitsize == n
        assert sizing.formula_width == source.b_grad_via_formula
        assert sizing.optimized_width == source.b_grad_via_fxp_optimization


@pytest.mark.parametrize(
    "fraction, coefficient, additions",
    [
        (14, 1 / 16, 1),
        (0, 1 / 16, 1),
        (14, (2**20 - 1) / 2**20, 4),
        (0, (2**20 - 1) / 2**20, 11),
    ],
)
def test_original_gradient_t_complexity(
    fraction: int,
    coefficient: float,
    additions: int,
    source_module: Any,
) -> None:
    from qualtran import QFxp, Register

    epsilon = 0.1
    source = source_module.QvrPhaseGradient(
        Register("x", QFxp(14, fraction)), coefficient, epsilon
    )
    cost = estimate_component_resources(
        source_qvr_phase_gradient_cost(14, fraction, coefficient, epsilon)
    )
    width = qvr_phase_sizing(14, fraction, coefficient, epsilon).format.gradient_width
    assert width == source.b_grad and width < 14
    assert (
        4 * cost.gates.toffoli == source.t_complexity().t == 4 * additions * (width - 2)
    )


def test_symbolic_tutorial_costs(source_module: Any) -> None:
    from qualtran.cirq_interop.t_complexity_protocol import TComplexity
    from qualtran.resource_counting.t_counts_from_sigma import t_counts_from_sigma

    n, gamma, epsilon = symbols(r"n \gamma \epsilon")
    direct = source_module.QvrZPow.from_bitsize(n, gamma, epsilon)
    ((leaf, n_rot),) = direct.bloq_counts().items()
    rotation, count = source_qvr_zpow_declaration(n, n, gamma, epsilon)
    assert (rotation.parameter, rotation.precision, count) == (
        leaf.exponent,
        leaf.eps,
        n_rot,
    )
    assert source_qvr_zpow_synthesis_t_count(
        n, n, gamma, epsilon
    ) == n_rot * TComplexity.rotation_cost(leaf.eps)
    gradient = source_module.QvrPhaseGradient.from_bitsize(n, gamma, epsilon)
    model = source_symbolic_qvr_gradient_cost(n, n, gamma, epsilon)
    assert model.logical_qubits == n + gradient.b_grad
    assert 4 * model.toffoli == t_counts_from_sigma(gradient.call_graph()[1])


@pytest.mark.parametrize("chart", ["sizing_eps", "sizing_width", "t_eps", "t_width"])
def test_original_notebook_sweeps(chart: str, source_module: Any) -> None:
    from qualtran.cirq_interop.t_complexity_protocol import TComplexity
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    if chart.endswith("_eps"):
        epsilons = [
            1 / (2 if chart.startswith("sizing") else 10) ** i for i in range(1, 20)
        ]
        points = [(ceil(log2(tau / eps)), eps) for eps in epsilons]
    else:
        points = [
            (width, 1e-12)
            for width in range(1, 50 if chart.startswith("sizing") else 70)
        ]
    for width, epsilon in points:
        if chart.startswith("t_"):
            source = source_module.QvrZPow.from_bitsize(width, 1, epsilon)
            ((leaf, rotations),) = source.bloq_counts().items()
            assert source_qvr_zpow_synthesis_t_count(
                width, width, 1, epsilon
            ) == rotations * TComplexity.rotation_cost(leaf.eps)
        for bits in range(1, 30, 4):
            coefficient = (2**bits - 1) / 2**bits
            op = source_module.QvrPhaseGradient.from_bitsize(
                width, coefficient, epsilon
            )
            sizing = qvr_phase_sizing(width, width, coefficient, epsilon)
            assert sizing.format.gradient_width == op.b_grad
            assert sizing.formula_width == op.b_grad_via_formula
            assert sizing.optimized_width == (
                op.b_grad_via_fxp_optimization if width < 54 else None
            )
            if chart.startswith("t_"):
                cost = estimate_component_resources(
                    source_qvr_phase_gradient_cost(width, width, coefficient, epsilon)
                )
                assert cost.gates.toffoli == get_cost_value(op, QECGatesCost()).toffoli

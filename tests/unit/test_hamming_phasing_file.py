"""Verify the frozen Hamming-phase tutorial and its original numerical grids."""

from cmath import exp
from collections import Counter
from dataclasses import asdict
from hashlib import sha1
from importlib.metadata import version
from itertools import product
from math import pi, tau
from pathlib import Path
from typing import Any

from tests._reference_dependencies import unavailable_reference

try:
    import numpy as np
except ModuleNotFoundError as _reference_import_error:
    unavailable_reference(_reference_import_error)
import pytest
import torch

from examples.algorithms.hamming_phasing_catalog import HammingPhasingExample
from flagquantum.compiler._component_trace.resource_model import (
    estimate_component_resources,
)
from flagquantum.compiler.registers.bits.hamming_weight import (
    hamming_weight_requirements,
)
from flagquantum.compiler.registers.phase.hamming_weight_phasing import (
    source_hamming_weight_phasing_cost,
)
from flagquantum.core.ir import Instruction
from flagquantum.dynamic import DynamicCircuit
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch
from tests.unit.register_source_graph import (
    SourcePorts,
    SourceResult,
    walk_source_graph,
)
from tests.unit.sparse_register_execution import execute_sparse_register
from tests.unit.test_hamming_weight_phasing import source_count_leaf
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_relative_controlled_swap import assert_source_dependencies

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
PHASES = (0.1, 0.2, 1 / 7, pi / 2)
QVR_CASES = (
    (1.0, 0.1),
    (0.5, 0.01),
    (0.1, 0.001),
    (1.20345, 0.001),
    (-1.1934341, 0.001),
)


@pytest.fixture(scope="module")
def source() -> Any:
    import kickmix
    from qualtran.bloqs.rotations import hamming_weight_phasing

    assert version("qualtran") == "0.7.0"
    assert getattr(kickmix, "__version__") == "0.1" and hasattr(kickmix, "Simulator")
    folder = Path(hamming_weight_phasing.__file__).parent
    for name, digest in (
        ("__init__.py", "e6c63bdf43262fd6e3771a9eebd5c8e7ad75d4b8"),
        ("hamming_weight_phasing.ipynb", "3279a78efaa2bd0bbf50eefe43d634612b830d7f"),
        ("hamming_weight_phasing.py", "2c55d84b5b97f9eedbdcf78b9e4ea13d079161b1"),
        ("hamming_weight_phasing_test.py", "bad2b4c5b0475e6b3232f2ebc6a3da34cdf21e6f"),
    ):
        raw = (folder / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == digest
    return hamming_weight_phasing


def _layout(
    op: HammingPhasingExample,
) -> tuple[SourcePorts, SourcePorts, tuple[int, ...]]:
    n, m = op.width, op.gradient_width
    j, w = hamming_weight_requirements(n)
    start = n + m
    ports = {"x": tuple(range(n))}
    if m:
        ports["phase_grad"] = tuple(range(n, start))
    return (
        ports,
        {
            "x": ports["x"],
            "junk": tuple(range(start, start + j)),
            "out": tuple(range(start + j, start + j + w)),
        },
        tuple(range(start + j + w, op.qubit_count)),
    )


def _source_qvr_gates(
    qvr: Any,
    binding: SourcePorts,
    work: tuple[int, ...],
    *,
    adjoint: bool,
) -> tuple[tuple[Instruction, ...], Any]:
    try:
        import cirq
    except ModuleNotFoundError as _reference_import_error:
        unavailable_reference(_reference_import_error)
    from qualtran._infra.data_types import QUInt
    from qualtran.bloqs.arithmetic.addition import Add
    from qualtran.bloqs.rotations.phase_gradient import (
        AddIntoPhaseGrad,
        AddScaledValIntoPhaseReg,
    )

    ((instance, _, _),) = qvr.decompose_bloq().iter_bloqsoqs()
    scaled = instance.bloq
    assert isinstance(scaled, AddScaledValIntoPhaseReg)
    gradient = binding["phase_grad"]
    leaves = tuple(
        cirq.flatten_to_ops(
            scaled.decompose_from_registers(
                context=cirq.DecompositionContext(cirq.SimpleQubitManager()),
                x=np.array([cirq.LineQubit(i) for i in binding["out"]]),
                phase_grad=np.array([cirq.LineQubit(i) for i in gradient]),
            )
        )
    )
    by_qubit: dict[cirq.Qid, int] = {cirq.LineQubit(w): w for w in binding["out"]}
    gates: list[Instruction] = []
    for term in reversed(leaves) if adjoint else leaves:
        child = term.gate
        assert isinstance(child, AddIntoPhaseGrad)
        active = min(int(child.x_bitsize), len(gradient) - child.right_shift)
        prefix = active + child.right_shift
        add = Add(QUInt(active), QUInt(prefix))
        carry = work[: prefix - 1]
        assert len(carry) == prefix - 1
        signed = (child.sign < 0) ^ adjoint
        gates.extend(
            reference_arithmetic_gates(
                add.adjoint() if signed else add,
                {
                    "a": tuple(by_qubit[q] for q in term.qubits[:active]),
                    "b": gradient[:prefix],
                },
                carry,
                dict(zip(carry, range(len(carry)), strict=True)),
            )
        )
    return tuple(gates), scaled


def _compile_against_source(
    op: HammingPhasingExample, original: Any
) -> tuple[DynamicCircuit, Any]:
    from qualtran._infra.adjoint import Adjoint
    from qualtran.bloqs.basic_gates.rotation import ZPowGate
    from qualtran.bloqs.rotations.quantum_variable_rotation import QvrPhaseGradient

    ports, parts, work = _layout(op)
    bits = tuple(range(len(parts["junk"])))
    inner_binding = (
        {"out": parts["out"], "phase_grad": ports["phase_grad"]}
        if op.via_gradient
        else {}
    )
    if op.via_gradient:
        inner, scaled = _source_qvr_gates(
            original.phase_oracle, inner_binding, work, adjoint=op.adjoint
        )
    else:
        inner, scaled = (), None

    def leaf(
        bloq: Any, inputs: SourcePorts, _desired: SourcePorts
    ) -> SourceResult | None:
        counter = source_count_leaf(bloq, inputs, parts, bits)
        if counter is not None:
            return counter
        if isinstance(bloq, ZPowGate):
            return (
                Instruction("phase", inputs["q"], {"theta": pi * bloq.exponent}),
            ), inputs
        base = bloq.subbloq if isinstance(bloq, Adjoint) else bloq
        if isinstance(base, QvrPhaseGradient):
            assert base == original.phase_oracle
            assert inputs == inner_binding
            assert isinstance(bloq, Adjoint) == op.adjoint
            return inner, inputs
        return None

    circuit = DynamicCircuit(op.qubit_count)
    circuit.append(op)
    expected, output = walk_source_graph(
        original.adjoint() if op.adjoint else original, ports, ports, leaf
    )
    assert output == ports
    assert_source_dependencies(circuit.to_ir().instructions, expected)
    return circuit, scaled


def _normalized_state(width: int, inputs: list[int]) -> dict[int, complex]:
    scale = sum(abs(complex(i + 1, width - i)) ** 2 for i in range(len(inputs))) ** 0.5
    return {
        address: complex(i + 1, width - i) / scale for i, address in enumerate(inputs)
    }


@pytest.mark.parametrize("width,exponent", tuple(product(range(2, 9), PHASES)))
def test_direct_original_phase_grid(width: int, exponent: float, source: Any) -> None:
    from qualtran.resource_counting._bloq_counts import QECGatesCost
    from qualtran.resource_counting._costing import get_cost_value

    op = HammingPhasingExample(width, exponent)
    original = source.HammingWeightPhasing(width, exponent)
    circuit, _ = _compile_against_source(op, original)
    ir = circuit.to_ir()
    model = estimate_component_resources(
        source_hamming_weight_phasing_cost(width, exponent)
    )
    actual_model = get_cost_value(original, QECGatesCost())
    assert model.gates.and_bloq == actual_model.and_bloq
    assert model.gates.rotation == actual_model.rotation == width.bit_length()
    inputs = [i << (op.qubit_count - width) for i in range(1 << width)]
    state = _normalized_state(width, inputs)
    expected = {
        address: v * exp(1j * pi * exponent * address.bit_count())
        for address, v in state.items()
    }
    events = model.gates.measurement
    if width <= 4:
        before = torch.zeros(1 << op.qubit_count, dtype=torch.complex128)
        after = torch.zeros_like(before)
        before[inputs] = torch.tensor(list(state.values()), dtype=torch.complex128)
        after[inputs] = torch.tensor(list(expected.values()), dtype=torch.complex128)
        for history in product((0, 1), repeat=events):
            torch.testing.assert_close(
                forced_branch(before, ir.instructions, history),
                after * 2 ** (-events / 2),
                rtol=0,
                atol=2e-12,
            )
    else:
        for seed in (7, 41):
            output = execute_sparse_register(ir, dict(state), seed=seed)
            assert output == pytest.approx(expected, abs=2e-12)


@pytest.mark.parametrize(
    "width,exponent,epsilon",
    [(n, e, eps) for n in (2, 3) for e, eps in QVR_CASES] + [(4, pi / 2, 1e-10)],
)
@pytest.mark.parametrize("adjoint", (False, True))
def test_gradient_literal_child_walk_and_phase(
    width: int, exponent: float, epsilon: float, adjoint: bool, source: Any
) -> None:
    op = HammingPhasingExample(
        width, exponent, epsilon, via_gradient=True, adjoint=adjoint
    )
    original = source.HammingWeightPhasingViaPhaseGradient(width, exponent, epsilon)
    assert op.gradient_width == original.b_grad
    from qualtran.resource_counting._bloq_counts import QECGatesCost
    from qualtran.resource_counting._costing import get_cost_value

    model = estimate_component_resources(op.resource_model())
    cost = get_cost_value(original, QECGatesCost())
    assert {
        name: value for name, value in asdict(model.gates).items() if value
    } == cost.asdict()
    circuit, scaled = _compile_against_source(op, original)
    ir = circuit.to_ir()
    m = op.gradient_width
    modulus = 1 << m
    sign = -1 if adjoint else 1
    increments = [
        sign * int(scaled.on_classical_vals(x=w, phase_grad=0)["phase_grad"])
        for w in range(width + 1)
    ]
    for weight, delta in enumerate(increments):
        assert (
            abs(
                exp(1j * tau * delta / modulus)
                - exp(1j * sign * pi * exponent * weight)
            )
            <= epsilon
        )

    # Translation of the gradient word gives the phase on its Fourier eigenstate.
    # Check the literal channel on every data word and on gradient words covering
    # ordinary addition and modular wraparound, including the coherent phases.
    tail = op.qubit_count - width - m
    values = range(modulus) if m <= 4 else (0, 1, modulus // 2 - 1, modulus - 1)
    pairs = [(word, y) for word, y in product(range(1 << width), values)]
    before = [((word << m) | y) << tail for word, y in pairs]
    after = [
        ((word << m) | ((y + increments[word.bit_count()]) % modulus)) << tail
        for word, y in pairs
    ]

    def binary(address: int) -> tuple[int, ...]:
        return tuple((address >> bit) & 1 for bit in reversed(range(op.qubit_count)))

    counts = Counter(g.name for g in ir.instructions)
    events = counts["measure"]
    assert counts["reset"] == events and counts["t"] + counts["tdg"] == 4 * events
    assert certified_and_channel_map(ir.instructions, tuple(map(binary, before))) == (
        tuple(map(binary, after)),
        events,
    )
    state = _normalized_state(width, before)
    expected = dict(zip(after, state.values(), strict=True))
    for seed in (7, 41):
        output = execute_sparse_register(ir, dict(state), seed=seed)
        assert output == pytest.approx(expected, abs=2e-12)


@pytest.mark.parametrize("via_gradient", (False, True))
def test_original_tutorial_signature_and_direct_calls(
    via_gradient: bool, source: Any
) -> None:
    from qualtran._infra.adjoint import Adjoint
    from qualtran.bloqs.arithmetic.hamming_weight import HammingWeightCompute
    from qualtran.bloqs.basic_gates.rotation import ZPowGate
    from qualtran.bloqs.rotations.quantum_variable_rotation import QvrPhaseGradient
    from qualtran.resource_counting.generalizers import ignore_split_join

    factory = (
        source._hamming_weight_phasing_via_phase_gradient
        if via_gradient
        else source._hamming_weight_phasing
    )
    original = factory.make()
    native = HammingPhasingExample(via_gradient=via_gradient)
    assert (native.width, native.exponent, native.epsilon) == (
        original.bitsize,
        original.exponent,
        original.eps,
    )
    assert [(r.name, r.bitsize, r.side.value) for r in native.signature] == [
        (r.name, r.bitsize, r.side.value) for r in original.signature
    ]
    if via_gradient:
        grad = original.signature.get_left("phase_grad").dtype
        assert dict(native.signature[1].dtype.parameters) == {
            "fractional_width": grad.num_frac,
            "signed": grad.signed,
        }

    def label(bloq: Any) -> str:
        base = bloq.subbloq if isinstance(bloq, Adjoint) else bloq
        if isinstance(base, HammingWeightCompute):
            return f"HammingWeightCompute({base.bitsize})" + (
                "†" if isinstance(bloq, Adjoint) else ""
            )
        if isinstance(base, QvrPhaseGradient):
            return (
                f"QvrPhaseGradient({base.cost_dtype.bitsize}, {base.gamma}, {base.eps})"
            )
        assert isinstance(base, ZPowGate)
        return f"ZPow({base.exponent}, {base.eps})"

    _, leaves = original.call_graph(max_depth=1, generalizer=ignore_split_join)
    assert native.call_graph().leaf_counts == {label(k): v for k, v in leaves.items()}


@pytest.mark.parametrize("width", (10, 32, 64, 100, 1024))
def test_original_large_direct_counts(width: int, source: Any) -> None:
    original = source.HammingWeightPhasing(width, 0.1)
    native = HammingPhasingExample(width, 0.1)
    model = estimate_component_resources(native.resource_model())
    cost = original.t_complexity()
    assert cost.rotations == model.gates.rotation == width.bit_length()
    assert cost.t == model.gates.and_bloq * 4 == 4 * (width - width.bit_count())


def test_original_five_thousand_bit_complexity(source: Any) -> None:
    from qualtran.resource_counting._bloq_counts import QECGatesCost
    from qualtran.resource_counting._costing import get_cost_value

    n, exponent, epsilon = 5000, 0.01, 0.1
    direct = source.HammingWeightPhasing(n, exponent, epsilon)
    gradient = source.HammingWeightPhasingViaPhaseGradient(n, exponent, epsilon)
    native = HammingPhasingExample(n, exponent, epsilon, via_gradient=True)
    model = estimate_component_resources(native.resource_model())
    qec = get_cost_value(gradient, QECGatesCost())
    assert {
        name: value for name, value in asdict(model.gates).items() if value
    } == qec.asdict()
    rotation_epsilon = epsilon / n.bit_length()
    assert gradient.t_complexity().t_incl_rotations(
        eps=rotation_epsilon
    ) < direct.t_complexity().t_incl_rotations(eps=rotation_epsilon)

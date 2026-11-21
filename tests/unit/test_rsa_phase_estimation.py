"""Verify the locked RSA phase circuit, its stated width, and joint projections."""

from cmath import exp
from collections import Counter, defaultdict
from importlib.metadata import version
from itertools import pairwise
from math import tau

import pytest

from flagquantum.algorithms.primitives.modular_exponentiation import (
    ModularExponentiation,
)
from flagquantum.algorithms.primitives.rsa_phase_estimation import RSAPhaseEstimation
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.dynamic import DynamicCircuit
from tests.unit.sparse_register_execution import execute_sparse_register
from tests.unit.test_elliptic_curve_addition_slope import pack

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_original_constructor_and_symbolic_declared_calls() -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("qualtran")
    from qualtran.bloqs.cryptography.rsa.rsa_phase_estimate import RSAPhaseEstimate
    from qualtran.resource_counting import SympySymbolAllocator
    from sympy import symbols

    assert version("qualtran") == "0.7.0"
    n, p, g = symbols("n p g")
    for native, source in (
        (RSAPhaseEstimation(n, p, g), RSAPhaseEstimate(n, p, g)),
        (RSAPhaseEstimation.for_shor(221, 9), RSAPhaseEstimate.make_for_shor(221, g=9)),
        (
            RSAPhaseEstimation.for_shor(91, random_state=np.random.RandomState(52)),
            RSAPhaseEstimate.make_for_shor(91, rs=np.random.RandomState(52)),
        ),
    ):
        assert (native.width, native.modulus, native.base) == (
            source.n,
            source.mod,
            source.base,
        )
        assert tuple(native.source_signature) == tuple(source.signature) == ()
        calls = source.build_call_graph(SympySymbolAllocator())
        modeled = {
            (
                (type(b).__name__, b.n)
                if hasattr(b, "n")
                else (type(b).__name__, b.dtype.bitsize, b.k, b.mod)
            ): count
            for b, count in calls.items()
        }
        assert native.source_call_graph().leaf_counts == modeled
        assert native.phase_width == 2 * native.width

    with pytest.raises(ValueError, match="ceil"):
        RSAPhaseEstimation(4, 3, 2)
    with pytest.raises(ValueError, match="fit the output"):
        RSAPhaseEstimation.for_shor(8, 3).workspace_requirements()
    nonunit = RSAPhaseEstimation.for_shor(15, 3)
    assert nonunit.base == RSAPhaseEstimate.make_for_shor(15, g=3).base
    with pytest.raises(ValueError, match="coprime"):
        nonunit.workspace_requirements()


def layout(n: int, p: int, g: int) -> tuple[
    CircuitIR,
    dict[str, tuple[int, ...]],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
]:
    op = RSAPhaseEstimation(n, p, g)
    q, c = op.workspace_requirements()
    wires = tuple(reversed(range(3 * n + q)))
    phase, target, work = wires[: 2 * n], wires[2 * n : 3 * n], wires[3 * n :]
    slots = tuple(100 + i * 3 for i in range(c))
    results = tuple(11 + i * 2 for i in reversed(range(2 * n)))
    bound = op.make_on(
        phase=phase,
        target=target,
        workspace=work,
        classical_bits=slots,
        result_bits=results,
    )
    return (
        DynamicCircuit(len(wires)).append(bound).to_ir(),
        {"phase": phase, "x": target},
        work,
        slots,
        results,
    )


@pytest.mark.parametrize("n,p,g", [(2, 3, 2), (3, 5, 2)])
def test_source_operand_chain_actual_width_and_all_projected_states(
    n: int, p: int, g: int
) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.basic_gates import IntState, PlusState
    from qualtran.bloqs.bookkeeping import Free
    from qualtran.bloqs.cryptography._factoring_shims import MeasureQFT
    from qualtran.bloqs.cryptography.rsa.rsa_phase_estimate import RSAPhaseEstimate
    from qualtran.bloqs.mod_arithmetic import CModMulK

    source = RSAPhaseEstimate(n, p, g)
    e = 2 * n
    stages = sorted(source.decompose_bloq().iter_bloqsoqs(), key=lambda row: row[0].i)
    positive = tuple(
        outs[0] for item, _, outs in stages if isinstance(item.bloq, PlusState)
    )
    init = [
        (item.bloq, outs[0])
        for item, _, outs in stages
        if isinstance(item.bloq, IntState)
    ]
    body = [
        (item.bloq, inputs, outs)
        for item, inputs, outs in stages
        if isinstance(item.bloq, CModMulK)
    ]
    end = [
        (item.bloq, inputs)
        for item, inputs, _ in stages
        if isinstance(item.bloq, MeasureQFT)
    ]
    release = [
        inputs["reg"]
        for item, inputs, _ in stages
        if isinstance(item.bloq, Free) and item.bloq.dirty
    ]
    assert len(positive) == len(body) == e
    assert len(init) == 1 and (init[0][0].val, init[0][0].bitsize) == (1, n)
    assert body[0][1]["x"] == init[0][1]
    assert tuple(inputs["ctrl"] for _, inputs, _ in reversed(body)) == positive
    assert all(a[2][1] == b[1]["x"] for a, b in pairwise(body))
    assert tuple(item.k for item, _, _ in body) == tuple(
        pow(g, 1 << i, p) for i in range(e)
    )
    assert len(end) == 1 and end[0][0].n == e
    assert tuple(end[0][1]["x"]) == tuple(
        outputs[0] for _, _, outputs in reversed(body)
    )
    assert release == [body[-1][2][1]]
    assert (
        RSAPhaseEstimation(n, p, g).source_call_graph().leaf_counts["MeasureQFT", n]
        == 1
    )

    ir, ports, work, slots, results = layout(n, p, g)
    expected_prefix = DynamicCircuit(ir.n_wires)
    for wire in ports["phase"]:
        expected_prefix.h(wire)
    expected_prefix.append(
        ModularExponentiation(g, p, e, n).make_on(
            exponent=ports["phase"],
            target=ports["x"],
            workspace=work,
            classical_bits=slots,
        )
    )
    prefix = expected_prefix.to_ir().instructions
    assert ir.instructions[: len(prefix)] == prefix
    assert ir.instructions[-e:] == tuple(
        Instruction(
            "measure", (wire,), metadata={"is_dynamic": True, "classical_bit": slot}
        )
        for wire, slot in zip(ports["phase"], results, strict=True)
    )
    counts = Counter(gate.name for gate in ir.instructions[len(prefix) : -e])
    assert counts == {"h": e, "cphase": e * (e - 1) // 2, "swap": n}

    size = 1 << e
    expected: defaultdict[int, complex] = defaultdict(complex)
    for j in range(size):
        modular_value = pow(g, j, p)
        for result in range(size):
            basis = pack({"phase": result, "x": modular_value}, ports, ir.n_wires)
            expected[basis] += exp(1j * tau * j * result / size) / size

    mask = pack({"phase": size - 1}, ports, ir.n_wires)
    body_ir = CircuitIR(ir.n_wires, ir.instructions[:-e])
    for seed in (21, 43):
        actual = execute_sparse_register(body_ir, {0: 1 + 0j}, seed=seed)
        for result in range(size):
            phase = pack({"phase": result}, ports, ir.n_wires)
            assert {
                b: v for b, v in actual.items() if b & mask == phase and abs(v) > 1e-12
            } == pytest.approx(
                {
                    b: v
                    for b, v in expected.items()
                    if b & mask == phase and abs(v) > 1e-12
                },
                abs=1e-12,
            )

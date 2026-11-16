"""Check the two phase readouts, raw initialization and locked dirty release."""

from cmath import exp
from collections import Counter, defaultdict
from importlib.metadata import version
from math import sqrt, tau

import pytest

from flagquantum.algorithms.primitives.elliptic_curve_double_phase import (
    EllipticCurveDoublePhase,
    elliptic_curve_double_phase_source_graph,
)
from flagquantum.algorithms.primitives.elliptic_curve_point import ECPoint
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from tests.unit.sparse_register_execution import execute_sparse_register
from tests.unit.test_elliptic_curve_addition_slope import pack

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_symbolic_and_concrete_source_calls_operands_and_dirty_release() -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.basic_gates import IntState
    from qualtran.bloqs.bookkeeping import Free
    from qualtran.bloqs.cryptography.ecc.ec_phase_estimate_r import ECPhaseEstimateR
    from qualtran.bloqs.cryptography.ecc.ec_point import ECPoint as ReferencePoint
    from qualtran.bloqs.cryptography.ecc.find_ecc_private_key import FindECCPrivateKey
    from qualtran.resource_counting import SympySymbolAllocator
    from sympy import symbols

    assert version("qualtran") == "0.7.0"
    n, p, a, b, c, d = symbols("n p a b c d")
    source = FindECCPrivateKey(n, ReferencePoint(a, b, p), ReferencePoint(c, d, p))
    native = elliptic_curve_double_phase_source_graph()
    declared = {
        type(child).__name__: count
        for child, count in source.build_call_graph(SympySymbolAllocator()).items()
    }
    assert native.leaf_counts == declared == {"ECPhaseEstimateR": 2}
    assert source.cost_attrs() == [("n", n)]
    first, second = ReferencePoint(0, 1, 3, 1), ReferencePoint(1, 0, 3, 1)
    src = FindECCPrivateKey(3, first, second, 3)
    steps = sorted(src.decompose_bloq().iter_bloqsoqs(), key=lambda row: row[0].i)
    assert [
        (node.bloq.val, node.bloq.bitsize)
        for node, _, _ in steps
        if isinstance(node.bloq, IntState)
    ] == [(0, 3), (1, 3)]
    children = [
        (node.bloq, inputs, outputs)
        for node, inputs, outputs in steps
        if isinstance(node.bloq, ECPhaseEstimateR)
    ]
    assert [op.point for op, _, _ in children] == [first, second]
    assert tuple(children[1][1][name] for name in ("x", "y")) == tuple(children[0][2])
    assert tuple(
        ports["reg"]
        for node, ports, _ in steps
        if isinstance(node.bloq, Free) and node.bloq.dirty
    ) == tuple(children[1][2])
    split = EllipticCurveDoublePhase(3, ECPoint(0, 1, 3, 1), ECPoint(1, 0, 3, 1), 3)
    assert not tuple(src.signature) and not tuple(split.source_signature)
    with pytest.raises(ValueError, match="Inconsistent moduli"):
        EllipticCurveDoublePhase(3, ECPoint(0, 1, 3, 1), ECPoint(1, 0, 5, 1), 3)
    with pytest.raises(ValueError, match="Inconsistent curve"):
        EllipticCurveDoublePhase(3, ECPoint(0, 1, 3, 1), ECPoint(1, 0, 3, 2), 3)
    scalar = EllipticCurveDoublePhase(3, ECPoint(0, 1, 3, 1), ECPoint(1, 0, 3, 1), 1)
    with pytest.raises(ValueError, match="no gate decomposition"):
        scalar.workspace_requirements()


def test_joint_readout_all_projected_states_and_shared_zero_work() -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("qualtran")
    from qualtran.bloqs.cryptography.ecc.ec_add_r import ECWindowAddR
    from qualtran.bloqs.cryptography.ecc.ec_point import ECPoint as ReferencePoint

    n, p, k = 3, 3, 3
    op = EllipticCurveDoublePhase(n, ECPoint(0, 1, p, 1), ECPoint(1, 0, p, 1), k)
    q, c = op.workspace_requirements()
    width = 4 * n + q
    wires = tuple(reversed(range(width)))
    ports = {
        name: wires[i * n : (i + 1) * n]
        for i, name in enumerate(("x", "y", "first_phase", "second_phase"))
    }
    scratch = wires[4 * n :]
    slots = tuple(100 + 3 * i for i in range(c))
    first, second = (4, 8, 12), (17, 15, 13)
    ir = (
        DynamicCircuit(width)
        .append(
            op.make_on(
                **ports,
                workspace=scratch,
                classical_bits=slots,
                first_results=first,
                second_results=second,
            )
        )
        .to_ir()
    )
    assert ir.instructions[0].name == "x" and ir.instructions[0].wires == (
        ports["y"][-1],
    )
    starts = [
        i
        for i, g in enumerate(ir.instructions)
        if g.name == "measure" and g.metadata["classical_bit"] in (*first, *second)
    ]
    assert [ir.instructions[i].wires[0] for i in starts] == [
        *ports["first_phase"],
        *ports["second_phase"],
    ]
    assert [ir.instructions[i].metadata["classical_bit"] for i in starts] == list(
        (*first, *second)
    )
    prefix = CircuitIR(width, ir.instructions[: starts[0]])
    suffix = CircuitIR(width, ir.instructions[starts[n - 1] + 1 : starts[n]])
    size = 1 << n
    sources = tuple(
        ECWindowAddR(n, point, k).decompose_bloq()
        for point in (ReferencePoint(0, 1, p, 1), ReferencePoint(1, 0, p, 1))
    )
    translations = tuple(
        {
            (j, x, y): source.on_classical_vals(
                ctrl=np.array([(j >> (n - i - 1)) & 1 for i in range(n)]), x=x, y=y
            )
            for j in range(size)
            for x, y in ((0, 0), (0, 1), (0, 2), (2, 0))
        }
        for source in sources
    )

    def project(
        state: dict[int, complex], word: str, result: int
    ) -> dict[int, complex]:
        mask = pack({word: size - 1}, ports, width)
        chosen = pack({word: result}, ports, width)
        return {
            j: value
            for j, value in state.items()
            if j & mask == chosen and abs(value) > 1e-12
        }

    all_first = execute_sparse_register(prefix, {0: 1 + 0j}, seed=5)
    expected: defaultdict[tuple[int, int], dict[int, complex]] = defaultdict(
        lambda: defaultdict(complex)
    )
    for i0 in range(size):
        a = translations[0][i0, 0, 1]
        for i1 in range(size):
            b = translations[1][i1, a["x"], a["y"]]
            for r0 in range(size):
                for r1 in range(size):
                    basis = pack(
                        dict(x=b["x"], y=b["y"], first_phase=r0, second_phase=r1),
                        ports,
                        width,
                    )
                    expected[r0, r1][basis] += exp(
                        1j * tau * (i0 * r0 + i1 * r1) / size
                    ) / (size * size)
    checked = 0
    for r0 in range(size):
        projected = project(all_first, "first_phase", r0)
        probability = sum(abs(value) ** 2 for value in projected.values())
        out = {}
        if probability > 1e-24:
            normalized = {j: v / sqrt(probability) for j, v in projected.items()}
            out = {
                j: v * sqrt(probability)
                for j, v in execute_sparse_register(
                    suffix, normalized, seed=71 + r0
                ).items()
            }
        for r1 in range(size):
            got = project(out, "second_phase", r1)
            want = {j: v for j, v in expected[r0, r1].items() if abs(v) > 1e-12}
            assert got == pytest.approx(want, abs=1e-12)
            checked += 1
    assert checked == 64
    counts = Counter(g.name for g in ir.instructions)
    assert counts["measure"] == 2692
    assert counts["ccx"] == 38

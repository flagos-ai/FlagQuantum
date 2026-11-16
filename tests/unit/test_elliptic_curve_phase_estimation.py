"""Compare windowed ECC phase estimation to locked strata and the exact DFT."""

from cmath import exp
from collections import Counter, defaultdict
from importlib.metadata import version
from math import tau

import pytest

from flagquantum.algorithms.primitives.elliptic_curve_phase_estimation import (
    EllipticCurvePhaseEstimation,
    elliptic_curve_phase_signature,
    elliptic_curve_phase_source_graph,
    phase_addition_points,
)
from flagquantum.algorithms.primitives.elliptic_curve_point import ECPoint
from flagquantum.algorithms.primitives.elliptic_curve_window_addition import (
    EllipticCurveWindowAddition,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from tests.unit.sparse_register_execution import execute_sparse_register
from tests.unit.test_elliptic_curve_addition_slope import pack

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_fixed_and_windowed_source_declarations() -> None:
    pytest.importorskip("qualtran")
    from qualtran import DecomposeNotImplementedError
    from qualtran.bloqs.cryptography.ecc.ec_phase_estimate_r import ECPhaseEstimateR
    from qualtran.bloqs.cryptography.ecc.ec_point import ECPoint as ReferencePoint
    from qualtran.resource_counting import (
        QECGatesCost,
        SympySymbolAllocator,
        get_cost_value,
    )
    from sympy import symbols

    assert version("qualtran") == "0.7.0"
    n, p, x, y, k = symbols("n p x y k")
    for width, mod, window in ((n, p, 1), (3, p, 1), (n, 3, k), (4, 3, 2)):
        source = ECPhaseEstimateR(width, ReferencePoint(x, y, mod, 2), window)
        signature = elliptic_curve_phase_signature(width)
        assert [(r.name, r.bitsize) for r in source.signature] == [
            (r.name, r.dtype.num_qubits) for r in signature
        ]
        declared = {
            type(child).__name__: count
            for child, count in source.build_call_graph(SympySymbolAllocator()).items()
        }
        graph = elliptic_curve_phase_source_graph(width, window)
        assert graph.leaf_counts == declared
        assert graph.direct_calls == {
            ("ECPhaseEstimateR", name): count for name, count in declared.items()
        }
    scalar = ECPhaseEstimateR(4, ReferencePoint(0, 1, 3, 2))
    with pytest.raises(DecomposeNotImplementedError):
        get_cost_value(scalar, QECGatesCost())
    op = EllipticCurvePhaseEstimation(4, ECPoint(0, 1, 3, 2), 1)
    assert [(p.x, p.y) for p in op.addition_points] == [
        (p.x, p.y) for p in (scalar.point * (1 << i) for i in range(4))
    ]
    with pytest.raises(ValueError, match="no gate decomposition"):
        op.workspace_requirements()
    with pytest.raises(ValueError, match="dividing"):
        phase_addition_points(4, 3, ECPoint(0, 1, 3, 2))


def layout(op: EllipticCurvePhaseEstimation, *, reverse: bool = False) -> tuple[
    CircuitIR,
    dict[str, tuple[int, ...]],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
]:
    n = op.width
    q, c = op.workspace_requirements()
    wires = tuple(range(3 * n + q))
    if reverse:
        wires = wires[::-1]
    words = {
        name: wires[i * n : (i + 1) * n] for i, name in enumerate(("x", "y", "phase"))
    }
    work = wires[3 * n :]
    slots = tuple(51 + 3 * i for i in range(c))
    results = tuple(19 + 2 * i for i in reversed(range(n)))
    ir = (
        DynamicCircuit(len(wires))
        .append(
            op.make_on(
                **words, workspace=work, classical_bits=slots, result_bits=results
            )
        )
        .to_ir()
    )
    return ir, words, work, slots, results


def test_native_dispatch_follows_locked_source_operands_and_physical_order() -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.basic_gates import PlusState
    from qualtran.bloqs.cryptography._factoring_shims import (
        MeasureQFT as ReferenceMeasureQFT,
    )
    from qualtran.bloqs.cryptography.ecc.ec_add_r import ECWindowAddR
    from qualtran.bloqs.cryptography.ecc.ec_phase_estimate_r import ECPhaseEstimateR
    from qualtran.bloqs.cryptography.ecc.ec_point import ECPoint as ReferencePoint

    n, k = 4, 2
    ref = ECPhaseEstimateR(n, ReferencePoint(0, 1, 3, 2), k)
    op = EllipticCurvePhaseEstimation(n, ECPoint(0, 1, 3, 2), k)
    original = ref.decompose_bloq()
    steps = sorted(original.iter_bloqsoqs(), key=lambda row: row[0].i)
    add = [
        (b.bloq, ports, outputs)
        for b, ports, outputs in steps
        if isinstance(b.bloq, ECWindowAddR)
    ]
    assert [(gate.R.x, gate.R.y) for gate, _, _ in add] == [(0, 1), (2, 1)]
    initial = [outputs[0] for b, _, outputs in steps if isinstance(b.bloq, PlusState)]
    assert len(initial) == n and len(add) == n // k
    for i, (_, ports, _) in enumerate(add):
        assert tuple(ports["ctrl"]) == tuple(initial[i * k : (i + 1) * k])
    readouts = [
        ports for b, ports, _ in steps if isinstance(b.bloq, ReferenceMeasureQFT)
    ]
    assert len(readouts) == 1
    assert tuple(readouts[0]["x"]) == tuple(
        bit for _, _, outputs in add for bit in outputs[0]
    )
    ir, ports, work, slots, results = layout(op, reverse=True)
    expected = DynamicCircuit(ir.n_wires)
    for wire in ports["phase"]:
        expected.h(wire)
    for i, (original_add, _, _) in enumerate(add):
        point = original_add.R
        native = EllipticCurveWindowAddition(
            n, ECPoint(point.x, point.y, point.mod, point.curve_a), k
        )
        q, c = native.workspace_requirements()
        expected.append(
            native.make_on(
                ctrl=ports["phase"][i * k : (i + 1) * k],
                x=ports["x"],
                y=ports["y"],
                workspace=work[:q],
                classical_bits=slots[:c],
            )
        )
    prefix = expected.to_ir().instructions
    assert ir.instructions[: len(prefix)] == prefix
    assert [
        (g.name, g.wires, g.metadata["classical_bit"]) for g in ir.instructions[-n:]
    ] == [
        ("measure", (wire,), slot)
        for wire, slot in zip(ports["phase"], results, strict=True)
    ]
    assert Counter(g.name for g in ir.instructions[len(prefix) : -n]) == {
        "cphase": 6,
        "h": 4,
        "swap": 2,
    }


@pytest.mark.parametrize("n,k", [(3, 3), (4, 2)])
def test_every_terminal_result_full_complex_amplitudes_and_clean_work(
    n: int, k: int
) -> None:
    pytest.importorskip("qualtran")
    import numpy as np
    from qualtran.bloqs.cryptography.ecc.ec_add_r import ECWindowAddR
    from qualtran.bloqs.cryptography.ecc.ec_point import ECPoint as ReferencePoint

    p = 3
    ir, ports, _, _, _ = layout(
        EllipticCurvePhaseEstimation(n, ECPoint(0, 1, p, 1), k), reverse=True
    )
    width = ir.n_wires
    multiplier = pow(2, n, p)
    point = ReferencePoint(0, 1, p, 1)
    ref = tuple(
        ECWindowAddR(n, (1 << (k * i)) * point, k).decompose_bloq()
        for i in range(n // k)
    )
    initial = {}
    expected: defaultdict[int, complex] = defaultdict(complex)
    amps = (0.6, 0.8j)
    size = 1 << n
    for (rawx, rawy), amp in zip(((0, 0), (0, 1)), amps, strict=True):
        x, y = rawx * multiplier % p, rawy * multiplier % p
        initial[pack(dict(x=x, y=y), ports, width)] = amp
        for j in range(size):
            coordinate = dict(x=x, y=y)
            for i, addition in enumerate(ref):
                bits = np.array(
                    [(j >> (n - i * k - 1 - offset)) & 1 for offset in range(k)]
                )
                data = addition.on_classical_vals(ctrl=bits, **coordinate)
                coordinate = {name: data[name] for name in ("x", "y")}
            for outcome in range(size):
                basis = pack(dict(**coordinate, phase=outcome), ports, width)
                expected[basis] += amp * exp(1j * tau * j * outcome / size) / size
    expected_support = {basis: v for basis, v in expected.items() if abs(v) > 1e-12}
    combined = CircuitIR(width, ir.instructions[:-n])
    phase_word = ports["phase"]

    def phase_of(basis: int) -> int:
        return sum(
            (basis >> (width - wire - 1) & 1) << (n - i - 1)
            for i, wire in enumerate(phase_word)
        )

    for seed in (11, 23):
        output = execute_sparse_register(combined, initial.copy(), seed=seed)
        for outcome in range(size):
            projected = {
                j: v
                for j, v in output.items()
                if phase_of(j) == outcome and abs(v) > 1e-12
            }
            assert projected == pytest.approx(
                {j: v for j, v in expected_support.items() if phase_of(j) == outcome},
                abs=1e-12,
            )

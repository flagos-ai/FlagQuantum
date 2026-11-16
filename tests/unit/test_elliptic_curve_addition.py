"""Validate the full pinned ECC chain and explicit source cleanup boundaries."""

from collections import Counter
from importlib.metadata import version
from itertools import product
from typing import Any

import pytest

from flagquantum.algorithms.primitives.elliptic_curve_addition import (
    EllipticCurveAddition,
    elliptic_curve_addition_signature,
    elliptic_curve_addition_source_graph,
)
from flagquantum.algorithms.primitives.elliptic_curve_addition_finish import (
    EllipticCurveAdditionFinish,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from tests.unit.measurement_checks import certified_and_channel_map
from tests.unit.sparse_register_execution import execute_sparse_register
from tests.unit.test_elliptic_curve_addition_slope import pack
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def layout(n: int, p: int) -> tuple[CircuitIR, dict[str, tuple[int, ...]]]:
    operation = EllipticCurveAddition(n, p)
    q, c = operation.workspace_requirements()
    width = 5 * n + q
    wires = tuple(reversed(range(width)))
    words = {
        name: wires[i * n : (i + 1) * n]
        for i, name in enumerate(("a", "b", "x", "y", "lam_r"))
    }
    ir = (
        DynamicCircuit(width)
        .append(
            operation.make_on(
                **words,
                workspace=wires[5 * n :],
                classical_bits=tuple(81 + 3 * i for i in range(c)),
            )
        )
        .to_ir()
    )
    flags = {
        name: (wire,)
        for name, wire in zip(
            ("f1", "f2", "f3", "f4", "ctrl"), wires[5 * n : 5 * n + 5], strict=True
        )
    }
    return ir, {**words, **flags}


def finish_with_release_bits(
    graph: Any,
    values: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, int]]:
    """Observe each original Free input while preserving the pinned graph."""
    from qualtran.bloqs.bookkeeping import Free
    from qualtran.simulation.classical_sim import ClassicalSimState

    entries = sorted(graph.iter_bloqsoqs(), key=lambda row: row[0].i)
    release = [
        (node, ports["reg"])
        for node, ports, _ in entries
        if isinstance(node.bloq, Free)
    ]
    assert len(release) == 7
    expected_nodes = dict(
        zip(
            (node for node, _ in release[-5:]),
            ("f1", "f2", "f3", "f4", "ctrl"),
            strict=True,
        )
    )
    sim = ClassicalSimState.from_cbloq(graph, values)
    rejected = {}
    while True:
        try:
            sim.step()
        except StopIteration:
            break
        except ValueError as exc:
            assert "free a non-zero register" in str(exc)
            assert sim.last_binst in expected_nodes
            rejected[expected_nodes[sim.last_binst]] = 1
    return sim.finalize(), {
        name: rejected.get(name, 0) for name in expected_nodes.values()
    }


@pytest.mark.parametrize(
    "n,p,rejected_count,mismatch_count", [(3, 3, 0, 0), (4, 5, 18, 12)]
)
def test_six_stage_curve_pairs_exact_channel_and_locked_release(
    n: int,
    p: int,
    rejected_count: int,
    mismatch_count: int,
) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.cryptography.ecc import ec_add

    assert version("qualtran") == "0.7.0"
    ir, ports = layout(n, p)
    native = EllipticCurveAddition(n, p)
    source = ec_add.ECAdd(n, p)
    assert [(r.name, r.bitsize, r.side.name) for r in source.signature] == [
        (r.name, r.dtype.num_qubits, r.side.name) for r in native.source_signature
    ]
    graphs = tuple(
        getattr(ec_add, f"_ECAddStep{name}")(n, p).decompose_bloq()
        for name in ("One", "Two", "Three", "Four", "Five", "Six")
    )
    points = (
        (0, 0),
        *(
            (x, y)
            for x, y in product(range(p), repeat=2)
            if (y * y - x**3 - x - 1) % p == 0
        ),
    )
    before, after = [], []
    rejected = 0
    coordinate_mismatches = []
    for (a, b), (x, y) in product(points, repeat=2):
        tangent = (3 * a * a + 1) * pow(2 * b, -1, p) % p if b else 0
        raw = {
            name: value * (1 << n) % p
            for name, value in zip(
                ("a", "b", "x", "y", "lam_r"), (a, b, x, y, tangent), strict=True
            )
        }
        before.append(pack(raw, ports, ir.n_wires))
        expected = source.on_classical_vals(**raw)
        data = raw.copy()
        for graph in graphs[:-1]:
            data.update(
                graph.on_classical_vals(
                    **{r.name: data[r.name] for r in graph.signature.lefts()}
                )
            )
        outputs, flags = finish_with_release_bits(
            graphs[-1], {r.name: data[r.name] for r in graphs[-1].signature.lefts()}
        )
        locked = {**outputs, "lam_r": raw["lam_r"]}
        if locked != expected:
            coordinate_mismatches.append(
                (
                    (a, b),
                    (x, y),
                    (locked["x"], locked["y"]),
                    (expected["x"], expected["y"]),
                )
            )
        rejected += any(flags.values())
        if p == 5 and (a, b, x, y) == (0, 0, 2, 1):
            assert flags == {"f1": 1, "f2": 0, "f3": 0, "f4": 0, "ctrl": 0}
        after.append(pack({**locked, **flags}, ports, ir.n_wires))
    assert rejected == rejected_count
    assert len(coordinate_mismatches) == mismatch_count
    if p == 5:
        assert ((0, 1), (2, 4), (2, 4), (4, 3)) in coordinate_mismatches
    width = ir.n_wires
    basis = tuple(tuple(v >> (width - i - 1) & 1 for i in range(width)) for v in before)
    rows, events = certified_and_channel_map(ir.instructions, basis)
    assert events == sum(g.name == "measure" for g in ir.instructions)
    assert [
        sum(v << (width - i - 1) for i, v in enumerate(row)) for row in rows
    ] == after
    indices = (0, len(points) + 1, len(before) // 2, len(before) - 1)
    amps = (0.5, 0.5j, -0.5, -0.5j)
    initial = {before[i]: amp for i, amp in zip(indices, amps, strict=True)}
    result = {after[i]: amp for i, amp in zip(indices, amps, strict=True)}
    for branch in (7, 26):
        observed = execute_sparse_register(ir, initial.copy(), seed=branch)
        assert {k: v for k, v in observed.items() if abs(v) > 1e-12} == pytest.approx(
            result, abs=1e-12
        )


def test_finish_zero_predicates_use_source_control_order_and_measurements() -> None:
    pytest.importorskip("qualtran")
    from qualtran import CtrlSpec, QMontgomeryUInt
    from qualtran.bloqs.basic_gates import XGate
    from qualtran.bloqs.cryptography.ecc.ec_add import _ECAddStepSix
    from qualtran.bloqs.mcmt import MultiControlX
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    n, p = 3, 3
    stage = EllipticCurveAdditionFinish(n, p)
    source = _ECAddStepSix(n, p)
    assert [(r.name, r.bitsize, r.side.name) for r in source.signature] == [
        (r.name, r.dtype.num_qubits, r.side.name) for r in stage.source_signature
    ]
    q, c = stage.workspace_requirements()
    a, b, x, y = (tuple(range(i * n, (i + 1) * n)) for i in range(4))
    flags = tuple(range(4 * n, 4 * n + 5))
    f1, f2, f3, f4, ctrl = flags
    work = tuple(range(4 * n + 5, 4 * n + 5 + q))
    slots = tuple(range(37, 37 + c))
    actual = stage.make_on(
        a=a, b=b, x=x, y=y, flags=flags, workspace=work, classical_bits=slots
    ).instructions
    reset_control = reference_arithmetic_gates(
        MultiControlX((0, 0, 0)),
        {"controls": (f2, f3, f4), "target": (ctrl,)},
        work[:2],
        dict(zip(work[:2], slots[:2], strict=True)),
    )
    zero = XGate().controlled(CtrlSpec(qdtypes=QMontgomeryUInt(n), cvs=[0, 0, 0]))
    multiwork = work[: 3 * n - 1]
    measured = dict(zip(multiwork, slots[: 3 * n - 1], strict=True))
    expected = (
        *reset_control,
        *(
            gate
            for controls, target in (
                (a + x + y, f1),
                (x + a + b, f1),
                (a + b + y, f2),
                (x + y + b, f2),
            )
            for gate in reference_arithmetic_gates(
                zero, {"ctrl": controls, "q": (target,)}, multiwork, measured
            )
        ),
    )
    assert_source(actual[: len(expected)], expected)
    ops = Counter(g.name for g in actual)
    model = get_cost_value(source, QECGatesCost())
    assert ops["measure"] == model.measurement
    assert ops["t"] + ops["tdg"] == 4 * model.and_bloq
    assert ops["ccx"] == model.toffoli == 2 * n + 4


def test_symbolic_tutorial_signature_and_direct_graph() -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.cryptography.ecc.ec_add import ECAdd
    from qualtran.resource_counting import SympySymbolAllocator
    from sympy import symbols

    n, p = symbols("n p")
    source = ECAdd(n, p)
    signature = elliptic_curve_addition_signature(n)
    graph = elliptic_curve_addition_source_graph()
    assert [(r.name, r.bitsize, r.side.name) for r in source.signature] == [
        (r.name, r.dtype.num_qubits, r.side.name) for r in signature
    ]
    source_calls = {
        type(child).__name__.removeprefix("_"): count
        for child, count in source.build_call_graph(SympySymbolAllocator()).items()
    }
    assert graph.leaf_counts == source_calls
    assert graph.direct_calls == {
        ("ECAdd", child): count for child, count in source_calls.items()
    }

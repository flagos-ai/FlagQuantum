"""Check consumed slope, coordinate restoration and coherently clean work."""

from collections import Counter
from importlib.metadata import version
from itertools import product

import pytest

from flagquantum.algorithms.primitives.elliptic_curve_addition_cleanup import (
    EllipticCurveAdditionCleanup,
)
from flagquantum.algorithms.primitives.elliptic_curve_addition_coordinates import (
    EllipticCurveAdditionCoordinates,
)
from flagquantum.algorithms.primitives.elliptic_curve_addition_flags import (
    EllipticCurveAdditionFlags,
)
from flagquantum.algorithms.primitives.elliptic_curve_addition_products import (
    EllipticCurveAdditionProducts,
)
from flagquantum.algorithms.primitives.elliptic_curve_addition_slope import (
    EllipticCurveAdditionSlope,
)
from flagquantum.compiler.registers.modular.modular_inverse import (
    modular_inverse_outputs,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from tests.unit.measurement_checks import certified_and_channel_map
from tests.unit.sparse_register_execution import execute_sparse_register
from tests.unit.test_elliptic_curve_addition_slope import layout as slope_layout
from tests.unit.test_elliptic_curve_addition_slope import pack
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def layout(n: int, p: int, w: int = 1) -> tuple[
    CircuitIR,
    dict[str, tuple[int, ...]],
    tuple[int, ...],
    tuple[int, ...],
]:
    stage = EllipticCurveAdditionCleanup(n, p, w)
    q, c = stage.workspace_requirements()
    words = {
        name: tuple(range(1 + i * n, 1 + (i + 1) * n))
        for i, name in enumerate(("a", "b", "x", "y", "lam_r", "lam"))
    }
    work = tuple(range(1 + 6 * n, 1 + 6 * n + q))
    slots = tuple(43 + 3 * i for i in range(c))
    op = stage.make_on(ctrl=0, **words, workspace=work, classical_bits=slots)
    return (
        DynamicCircuit(1 + 6 * n + q).append(op).to_ir(),
        {"ctrl": (0,), **words},
        work,
        slots,
    )


def test_literal_slope_selection_gates_and_source_model() -> None:
    pytest.importorskip("qualtran")
    from qualtran import CtrlSpec, QMontgomeryUInt
    from qualtran.bloqs.arithmetic import Xor
    from qualtran.bloqs.cryptography.ecc.ec_add import _ECAddStepFive
    from qualtran.bloqs.mcmt import MultiControlX
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    assert version("qualtran") == "0.7.0"
    n, p, w = 3, 3, 1
    source = _ECAddStepFive(n, p, w)
    native = EllipticCurveAdditionCleanup(n, p, w)
    assert [(r.name, r.bitsize, r.side.name) for r in source.signature] == [
        (r.name, r.dtype.num_qubits, r.side.name) for r in native.source_signature
    ]
    ir, ports, work, slots = layout(n, p, w)
    gates = ir.instructions
    inv_x, _ = modular_inverse_outputs(ports["x"], work[4 * n : 9 * n + 8])
    mul_work = work[9 * n + 8 :]
    result = (*mul_work[w:n], *mul_work[2 * n + 1 : 2 * n + 1 + w])
    scratch = mul_work[-(n + w - 1) :]
    register = QMontgomeryUInt(n)
    group = Xor(register).controlled(CtrlSpec(qdtypes=register, cvs=0)).controlled()
    tangent = reference_arithmetic_gates(
        group,
        {
            "ctrl1": ports["ctrl"],
            "ctrl2": inv_x,
            "x": ports["lam_r"],
            "y": ports["lam"],
        },
        scratch[:n],
        dict(zip(scratch[:n], slots[:n], strict=True)),
        keep_exchange_leaves=True,
    )
    selects = tuple(
        gate
        for source_bit, slope_bit in zip(result, ports["lam"], strict=True)
        for gate in reference_arithmetic_gates(
            MultiControlX((1, 1)),
            {"controls": (*ports["ctrl"], source_bit), "target": (slope_bit,)},
            scratch[:1],
            {scratch[0]: slots[0]},
        )
    )
    native_first = next(
        i
        for i, gate in enumerate(gates)
        if gate.name == "ccx" and gate.wires[-1] in ports["lam"]
    )
    source_first = next(i for i, gate in enumerate(tangent) if gate.name == "ccx")
    start = native_first - source_first - len(selects)
    assert_source(
        gates[start : start + len(selects) + len(tangent)], (*selects, *tangent)
    )
    model = get_cost_value(source, QECGatesCost())
    counts = Counter(gate.name for gate in gates)
    assert counts["measure"] == model.measurement
    assert (
        counts["t"] + counts["tdg"]
        == model.total_t_count(ts_per_cswap=7) - 4 * counts["ccx"]
    )
    assert ir.n_wires >= get_cost_value(source, QubitCount())


@pytest.mark.parametrize("n,p", [(3, 3), (4, 5)])
def test_five_stages_on_all_curve_pairs(n: int, p: int) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.cryptography.ecc.ec_add import (
        _ECAddStepFive,
        _ECAddStepFour,
        _ECAddStepOne,
        _ECAddStepThree,
        _ECAddStepTwo,
    )

    _, ports, work, slots = slope_layout(EllipticCurveAdditionSlope(n, p), reverse=True)
    width = 2 + 6 * n + len(work) + 3
    flags = (ports["f1"][0], *range(width - 3, width), ports["ctrl"][0])
    words = {name: ports[name] for name in ("a", "b", "x", "y")}
    circuit = DynamicCircuit(width)
    for op, operands in (
        (EllipticCurveAdditionFlags(n, p), {**words, "flags": flags}),
        (
            EllipticCurveAdditionSlope(n, p),
            {
                **words,
                "f1": flags[0],
                "ctrl": flags[-1],
                "lam": ports["lam"],
                "lam_r": ports["lam_r"],
            },
        ),
        (
            EllipticCurveAdditionCoordinates(n, p),
            {**words, "ctrl": flags[-1], "lam": ports["lam"]},
        ),
        (
            EllipticCurveAdditionProducts(n, p),
            {name: ports[name] for name in ("x", "y", "lam")},
        ),
        (
            EllipticCurveAdditionCleanup(n, p),
            {**words, "ctrl": flags[-1], "lam": ports["lam"], "lam_r": ports["lam_r"]},
        ),
    ):
        q, c = op.workspace_requirements()
        circuit.append(
            op.make_on(**operands, workspace=work[:q], classical_bits=slots[:c])
        )
    ir = circuit.to_ir()
    graphs = tuple(
        stage(n, p).decompose_bloq()
        for stage in (
            _ECAddStepOne,
            _ECAddStepTwo,
            _ECAddStepThree,
            _ECAddStepFour,
            _ECAddStepFive,
        )
    )
    ports.update(
        {
            name: (wire,)
            for name, wire in zip(("f2", "f3", "f4"), flags[1:4], strict=True)
        }
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
    for (a, b), (x, y) in product(points, repeat=2):
        tangent = (3 * a * a + 1) * pow(2 * b, -1, p) % p if b else 0
        data = {
            key: value * (1 << n) % p
            for key, value in zip(
                ("a", "b", "x", "y", "lam_r"), (a, b, x, y, tangent), strict=True
            )
        }
        before.append(pack(data, ports, width))
        for graph in graphs:
            inputs = {reg.name: data[reg.name] for reg in graph.signature.lefts()}
            data.update(graph.on_classical_vals(**inputs))
        data["lam"] = 0
        after.append(pack(data, ports, width))
    basis = tuple(tuple(v >> (width - i - 1) & 1 for i in range(width)) for v in before)
    rows, events = certified_and_channel_map(ir.instructions, basis)
    assert events == sum(g.name == "measure" for g in ir.instructions)
    assert [
        sum(v << (width - i - 1) for i, v in enumerate(row)) for row in rows
    ] == after
    samples = (0, len(points) + 1, len(before) // 2, len(before) - 1)
    amps = (0.5, 0.5j, -0.5, -0.5j)
    original = {before[i]: v for i, v in zip(samples, amps, strict=True)}
    expected = {after[i]: v for i, v in zip(samples, amps, strict=True)}
    for seed in (3, 19):
        output = execute_sparse_register(ir, original.copy(), seed=seed)
        assert {k: v for k, v in output.items() if abs(v) > 1e-12} == pytest.approx(
            expected, abs=1e-12
        )


def test_source_free_rejects_a_dirty_slope_and_native_preserves_evidence() -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.cryptography.ecc.ec_add import _ECAddStepFive

    src = _ECAddStepFive(3, 3)
    ir, ports, _, _ = layout(3, 3)
    inputs = dict(ctrl=1, a=0, b=1, x=0, y=0, lam=2, lam_r=0)
    with pytest.raises(ValueError, match="free a non-zero register"):
        src.decompose_bloq().on_classical_vals(**inputs)
    callback = src.on_classical_vals(**inputs)
    expected = pack({**callback, "lam": 2}, ports, ir.n_wires)
    output = execute_sparse_register(ir, {pack(inputs, ports, ir.n_wires): 1j}, seed=13)
    assert {k: v for k, v in output.items() if abs(v) > 1e-12} == pytest.approx(
        {expected: 1j}, abs=1e-12
    )

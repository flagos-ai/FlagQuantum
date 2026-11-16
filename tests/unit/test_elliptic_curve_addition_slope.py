"""Check the pinned ECAdd slope and its retained-history lifecycle."""

from collections import Counter
from importlib.metadata import version
from itertools import product
from typing import Any

import pytest

from flagquantum.algorithms.primitives.elliptic_curve_addition_flags import (
    EllipticCurveAdditionFlags,
)
from flagquantum.algorithms.primitives.elliptic_curve_addition_slope import (
    EllipticCurveAdditionSlope,
)
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.dynamic import DynamicCircuit
from tests.unit.measurement_checks import certified_and_channel_map
from tests.unit.sparse_register_execution import execute_sparse_register
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def layout(
    stage: EllipticCurveAdditionSlope, *, reverse: bool = False
) -> tuple[CircuitIR, dict[str, tuple[int, ...]], tuple[int, ...], tuple[int, ...]]:
    n = stage.width
    q, c = stage.workspace_requirements()
    width = 2 + 6 * n + q
    wires = tuple(range(width))
    if reverse:
        wires = wires[::-1]
    words = {
        name: wires[2 + i * n : 2 + (i + 1) * n]
        for i, name in enumerate(("a", "b", "x", "y", "lam", "lam_r"))
    }
    work, slots = wires[2 + 6 * n :], tuple(73 + 3 * i for i in range(c))
    op = stage.make_on(
        f1=wires[0], ctrl=wires[1], **words, workspace=work, classical_bits=slots
    )
    ir = DynamicCircuit(width).append(op).to_ir()
    return ir, {"f1": wires[:1], "ctrl": wires[1:2], **words}, work, slots


def pack(values: dict[str, Any], ports: dict[str, tuple[int, ...]], width: int) -> int:
    return sum(
        ((int(values.get(name, 0)) >> (len(wires) - j - 1)) & 1) << (width - wire - 1)
        for name, wires in ports.items()
        for j, wire in enumerate(wires)
    )


def test_locked_controlled_comparison_and_native_resource_counts() -> None:
    pytest.importorskip("qualtran")
    from qualtran import CtrlSpec, QMontgomeryUInt
    from qualtran.bloqs.arithmetic import Equals
    from qualtran.bloqs.cryptography.ecc.ec_add import _ECAddStepTwo
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    assert version("qualtran") == "0.7.0"
    n, p = 3, 3
    op = EllipticCurveAdditionSlope(n, p)
    ir, ports, work, slots = layout(op)
    source = _ECAddStepTwo(n, p, 1)
    assert [(r.name, r.bitsize, r.side.name) for r in source.signature] == [
        (r.name, r.dtype.num_qubits, r.side.name) for r in op.source_signature
    ]

    # Find the literal negative-control conjugation around Equals.
    flip = Instruction("x", (work[-1],))
    gates = ir.instructions
    start = gates.index(flip)
    stop = gates.index(flip, start + 1)
    controlled = Equals(QMontgomeryUInt(n)).controlled(ctrl_spec=CtrlSpec(cvs=0))
    controls = work[-(n + 1) : -1]
    reference = reference_arithmetic_gates(
        controlled,
        {
            "ctrl": work[-1:],
            "x": ports["lam"],
            "y": ports["lam_r"],
            "target": ports["f1"],
        },
        controls,
        dict(zip(controls, slots[:n], strict=True)),
        keep_exchange_leaves=True,
    )
    assert_source(gates[start : stop + 1], reference)
    counts = Counter(g.name for g in gates)
    model = get_cost_value(source, QECGatesCost())
    assert counts["measure"] == model.measurement == 470
    assert counts["t"] + counts["tdg"] == 3056
    assert counts["ccx"] == model.toffoli == 6
    assert model.cswap == 168
    assert ir.n_wires == 71 and get_cost_value(source, QubitCount()) == 55


def test_flags_then_slope_all_small_residues_and_erasure_branches() -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.cryptography.ecc.ec_add import _ECAddStepOne, _ECAddStepTwo

    n, p = 3, 3
    ir, ports, work, slots = layout(EllipticCurveAdditionSlope(n, p), reverse=True)
    width = ir.n_wires + 3
    prior_flags = (ports["f1"][0], *range(ir.n_wires, width), ports["ctrl"][0])
    coordinates = {name: ports[name] for name in ("a", "b", "x", "y")}
    flag_stage = EllipticCurveAdditionFlags(n, p)
    q_flags, c_flags = flag_stage.workspace_requirements()
    initial = flag_stage.make_on(
        **coordinates,
        flags=prior_flags,
        workspace=work[:q_flags],
        classical_bits=slots[:c_flags],
    )
    circuit = DynamicCircuit(width).append(initial)
    slope = EllipticCurveAdditionSlope(n, p).make_on(
        **coordinates,
        f1=ports["f1"][0],
        ctrl=ports["ctrl"][0],
        lam=ports["lam"],
        lam_r=ports["lam_r"],
        workspace=work,
        classical_bits=slots,
    )
    gates = circuit.append(slope).to_ir().instructions
    flag_source = _ECAddStepOne(n, p)
    slope_source = _ECAddStepTwo(n, p).decompose_bloq()
    ports.update(
        {
            name: (wire,)
            for name, wire in zip(("f2", "f3", "f4"), prior_flags[1:4], strict=True)
        }
    )
    before, after = [], []
    for values in product(range(p), repeat=5):
        inputs = dict(zip(("a", "b", "x", "y", "lam_r"), values, strict=True))
        flags = flag_source.on_classical_vals(
            **{name: inputs[name] for name in coordinates}
        )
        slope_inputs = {**inputs, "f1": flags["f1"], "ctrl": flags["ctrl"]}
        result = slope_source.on_classical_vals(**slope_inputs)
        before.append(pack(inputs, ports, width))
        after.append(pack({**flags, **result}, ports, width))
    basis = tuple(
        tuple((v >> (width - i - 1)) & 1 for i in range(width)) for v in before
    )
    observed, events = certified_and_channel_map(gates, basis)
    assert events == 498
    assert [
        sum(v << (width - i - 1) for i, v in enumerate(row)) for row in observed
    ] == after
    # Confirm full complex phases and zero workspace with mixed wire order.
    samples = (0, 21, 47, 109, 201, 242)
    amps = tuple(complex(i + 1, 2 - i) for i in range(len(samples)))
    norm = sum(abs(v) ** 2 for v in amps) ** 0.5
    state = {before[j]: v / norm for j, v in zip(samples, amps, strict=True)}
    expected = {after[j]: v / norm for j, v in zip(samples, amps, strict=True)}
    for seed in (11, 23):
        output = execute_sparse_register(
            CircuitIR(width, gates), state.copy(), seed=seed
        )
        assert {k: v for k, v in output.items() if abs(v) > 1e-12} == pytest.approx(
            expected, abs=1e-12
        )


@pytest.mark.parametrize(
    "n,p,w,x,y,ctrl,expected_f1,expected_slope",
    [
        (3, 3, 1, 1, 1, 0, 1, 0),
        (5, 13, 5, 2, 9, 1, 0, 27),
    ],
)
def test_retains_physical_source_boundary(
    n: int,
    p: int,
    w: int,
    x: int,
    y: int,
    ctrl: int,
    expected_f1: int,
    expected_slope: int,
) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.cryptography.ecc.ec_add import _ECAddStepTwo

    source = _ECAddStepTwo(n, p, w)
    ir, ports, _, _ = layout(EllipticCurveAdditionSlope(n, p, w))
    inputs = dict(a=0, b=0, x=x, y=y, ctrl=ctrl, f1=0, lam_r=0)
    callback = source.on_classical_vals(**inputs)
    expected = dict(**inputs)
    expected.update(f1=expected_f1, lam=expected_slope)
    assert callback != expected
    decomposed = source.decompose_bloq().on_classical_vals(**inputs)
    if w == 1:
        assert decomposed == expected
    else:
        from qualtran.bloqs.mod_arithmetic import (
            DirtyOutOfPlaceMontgomeryModMul,
            KaliskiModInverse,
        )
        from qualtran.resource_counting import QECGatesCost, get_cost_value

        inverse = KaliskiModInverse(n, p).on_classical_vals(x=x)["x"]
        multiply = DirtyOutOfPlaceMontgomeryModMul(n, w, p)
        locked = multiply.decompose_bloq().on_classical_vals(x=inverse, y=y)
        assert inverse == 5 and locked["reduced"] == 1
        assert locked["target"] == expected_slope
        assert callback["lam"] == decomposed["lam"] == 1
        counts = Counter(gate.name for gate in ir.instructions)
        assert counts["measure"] == 1194
        assert counts["t"] + counts["tdg"] == 8296
        assert get_cost_value(source, QECGatesCost()).measurement == 1166
    out = execute_sparse_register(ir, {pack(inputs, ports, ir.n_wires): 1j}, seed=13)
    assert {k: v for k, v in out.items() if abs(v) > 1e-12} == pytest.approx(
        {pack(expected, ports, ir.n_wires): 1j}, abs=1e-12
    )

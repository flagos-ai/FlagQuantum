"""Verify the fourth ECAdd stage against the locked product and XOR chain."""

from collections import Counter
from importlib.metadata import version
from itertools import product

import pytest

from flagquantum.algorithms.primitives.elliptic_curve_addition_products import (
    EllipticCurveAdditionProducts,
)
from flagquantum.compiler.registers.modular.single_window_mod_mul import (
    single_window_qroam_table,
)
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.dynamic import DynamicCircuit
from tests.unit.measurement_checks import certified_and_channel_map
from tests.unit.sparse_register_execution import execute_sparse_register
from tests.unit.test_elliptic_curve_addition_slope import pack
from tests.unit.test_modular_addition import layout as modular_layout
from tests.unit.test_modular_addition import reference_binding
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source
from tests.unit.test_windowed_mod_mul import original_gates

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def layout(
    n: int,
    p: int,
    w: int = 1,
) -> tuple[CircuitIR, dict[str, tuple[int, ...]], tuple[int, ...], tuple[int, ...]]:
    stage = EllipticCurveAdditionProducts(n, p, w)
    q, c = stage.workspace_requirements()
    words = {
        name: tuple(range(i * n, (i + 1) * n))
        for i, name in enumerate(("x", "y", "lam"))
    }
    work = tuple(range(3 * n, 3 * n + q))
    slots = tuple(61 + i * 3 for i in range(c))
    op = stage.make_on(**words, workspace=work, classical_bits=slots)
    return DynamicCircuit(3 * n + q).append(op).to_ir(), words, work, slots


def test_locked_product_sequence_and_feedback() -> None:
    pytest.importorskip("qualtran")
    from qualtran import QMontgomeryUInt
    from qualtran.bloqs.cryptography.ecc.ec_add import _ECAddStepFour
    from qualtran.bloqs.mod_arithmetic import DirtyOutOfPlaceMontgomeryModMul, ModSub
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    assert version("qualtran") == "0.7.0"
    n, p, w = 3, 3, 1
    ir, words, work, slots = layout(n, p, w)
    source = _ECAddStepFour(n, p, w)
    native = EllipticCurveAdditionProducts(n, p, w)
    assert [(r.name, r.bitsize, r.side.name) for r in source.signature] == [
        (r.name, r.dtype.num_qubits, r.side.name) for r in native.source_signature
    ]
    x, y, lam = (words[name] for name in ("x", "y", "lam"))
    slope_copy, target, indices = work[:n], work[n : 2 * n], work[2 * n : 3 * n]
    reduced = work[3 * n]
    padding = work[3 * n + 1 : 3 * n + 1 + w]
    bank_start = 3 * n + 1 + w
    bank_end = bank_start + (n + w) * single_window_qroam_table(n, w, p).block_shape[0]
    bank = tuple(work[i : i + n + w] for i in range(bank_start, bank_end, n + w))
    clean = work[bank_end : bank_end + n + w - 1]
    arith = work[bank_end + n + w - 1 :]
    result = (*target[w:], *padding)
    source_mul = DirtyOutOfPlaceMontgomeryModMul(n, w, p)

    def mul(first: tuple[int, ...], *, adjoint: bool) -> tuple[Instruction, ...]:
        return original_gates(
            source_mul,
            first,
            lam,
            target,
            indices,
            reduced,
            padding,
            bank,
            clean,
            slots[: 2 * n],
            result,
            adjoint=adjoint,
        )

    _, _, old_work, old_slots = modular_layout(n)
    allocation, measured = reference_binding(n, p)
    wires = dict(zip(old_work, arith, strict=True))
    bits = dict(zip(old_slots, slots[:n], strict=True))
    constant = (arith[0], *arith[2 : n + 2])
    carry = arith[n + 2 :]
    outer = (*constant[:n], *carry[: n - 1])
    sub = reference_arithmetic_gates(
        ModSub(QMontgomeryUInt(n), p),
        {"x": result, "y": x},
        (*outer, *(wires[q] for q in allocation), *outer),
        {wires[q]: bits[v] for q, v in measured.items()},
        reuse_allocations=False,
    )
    copy = tuple(Instruction("cx", pair) for pair in zip(lam, slope_copy, strict=True))
    xor = tuple(Instruction("cx", pair) for pair in zip(result, y, strict=True))
    assert_source(
        ir.instructions,
        (
            *copy,
            *mul(slope_copy, adjoint=False),
            *sub,
            *mul(slope_copy, adjoint=True),
            *copy,
            *mul(x, adjoint=False),
            *xor,
            *mul(x, adjoint=True),
        ),
    )
    actual = Counter(g.name for g in ir.instructions)
    model = get_cost_value(source, QECGatesCost())
    assert actual["measure"] == model.measurement == 143
    assert actual["t"] + actual["tdg"] == model.total_t_count() == 572
    assert ir.n_wires == 35 and get_cost_value(source, QubitCount()) == 27


@pytest.mark.parametrize("n,p", [(3, 3), (4, 5)])
def test_exact_xor_for_every_y_and_normalized_complex_branch(n: int, p: int) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.cryptography.ecc.ec_add import _ECAddStepFour

    ir, ports, _, _ = layout(n, p)
    source = _ECAddStepFour(n, p)
    graph = source.decompose_bloq()
    width = ir.n_wires
    dirty = dict(x=0, y=1, lam=1)
    assert (
        graph.on_classical_vals(**dirty)["y"]
        == source.on_classical_vals(**dirty)["y"] ^ 1
    )
    before, after = [], []
    for x, y, lam in product(range(p), range(1 << n), range(p)):
        inputs = dict(x=x, y=y, lam=lam)
        before.append(pack(inputs, ports, width))
        after.append(pack(graph.on_classical_vals(**inputs), ports, width))
    basis = tuple(tuple(v >> (width - i - 1) & 1 for i in range(width)) for v in before)
    rows, events = certified_and_channel_map(ir.instructions, basis)
    assert events == sum(gate.name == "measure" for gate in ir.instructions)
    assert [
        sum(v << (width - i - 1) for i, v in enumerate(row)) for row in rows
    ] == after
    indices = (0, 4, 17, len(before) - 1)
    amplitudes = (0.5, -0.5, 0.5j, -0.5j)
    state = {before[i]: amp for i, amp in zip(indices, amplitudes, strict=True)}
    expected = {after[i]: amp for i, amp in zip(indices, amplitudes, strict=True)}
    for seed in (9, 15):
        out = execute_sparse_register(ir, state.copy(), seed=seed)
        assert {k: v for k, v in out.items() if abs(v) > 1e-12} == pytest.approx(
            expected, abs=1e-12
        )


def test_final_xor_keeps_the_lowered_window_reduction() -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.cryptography.ecc.ec_add import _ECAddStepFour
    from qualtran.bloqs.mod_arithmetic import DirtyOutOfPlaceMontgomeryModMul
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    n, p, w = 5, 13, 5
    source = _ECAddStepFour(n, p, w)
    ir, ports, _, _ = layout(n, p, w)
    inputs = dict(x=12, y=0, lam=9)
    modeled = source.decompose_bloq().on_classical_vals(**inputs)
    assert modeled == dict(x=5, y=1, lam=9)
    multiplication = DirtyOutOfPlaceMontgomeryModMul(n, w, p).decompose_bloq()
    assert multiplication.on_classical_vals(x=9, y=9)["target"] == 7
    assert multiplication.on_classical_vals(x=5, y=9)["target"] == 27
    expected = pack(dict(x=5, y=27, lam=9), ports, ir.n_wires)
    output = execute_sparse_register(ir, {pack(inputs, ports, ir.n_wires): 1j}, seed=11)
    assert {k: v for k, v in output.items() if abs(v) > 1e-12} == pytest.approx(
        {expected: 1j}, abs=1e-12
    )
    assert sum(g.name == "measure" for g in ir.instructions) == 411
    assert get_cost_value(source, QECGatesCost()).measurement == 355

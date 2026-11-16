"""Check source coordinate updates, physical aliases and erasure branches."""

from collections import Counter
from importlib.metadata import version
from itertools import product

import pytest

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
from flagquantum.algorithms.primitives.elliptic_curve_window_product import (
    CurveWindowProduct,
)
from flagquantum.compiler.registers.modular.controlled_modular_addition import (
    controlled_modular_requirements,
)
from flagquantum.compiler.registers.modular.modular_addition import (
    modular_add_requirements,
)
from flagquantum.compiler.registers.modular.modular_doubling import (
    modular_double_requirements,
)
from flagquantum.compiler.registers.modular.single_window_mod_mul import (
    single_window_qroam_table,
)
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.dynamic import DynamicCircuit
from tests.unit.measurement_checks import certified_and_channel_map
from tests.unit.sparse_register_execution import execute_sparse_register
from tests.unit.test_controlled_modular_addition import (
    layout as controlled_layout,
)
from tests.unit.test_controlled_modular_addition import (
    reference_binding as controlled_binding,
)
from tests.unit.test_elliptic_curve_addition_slope import layout as slope_layout
from tests.unit.test_elliptic_curve_addition_slope import pack
from tests.unit.test_modular_addition import layout as add_layout
from tests.unit.test_modular_addition import reference_binding
from tests.unit.test_modular_addition_adjoint import inverse_binding
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source
from tests.unit.test_windowed_mod_mul import original_gates

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def layout(
    n: int,
    p: int,
    window: int = 1,
) -> tuple[CircuitIR, dict[str, tuple[int, ...]], tuple[int, ...], tuple[int, ...]]:
    stage = EllipticCurveAdditionCoordinates(n, p, window)
    q, c = stage.workspace_requirements()
    ports = {
        name: tuple(range(1 + i * n, 1 + (i + 1) * n))
        for i, name in enumerate(("a", "b", "x", "y", "lam"))
    }
    work = tuple(range(1 + 5 * n, 1 + 5 * n + q))
    slots = tuple(83 + 2 * i for i in range(c))
    ir = (
        DynamicCircuit(1 + 5 * n + q)
        .append(stage.make_on(ctrl=0, **ports, workspace=work, classical_bits=slots))
        .to_ir()
    )
    return ir, {"ctrl": (0,), **ports}, work, slots


def test_locked_ordered_body_operands_feedback_and_resource_model() -> None:
    pytest.importorskip("qualtran")
    from qualtran import QMontgomeryUInt
    from qualtran.bloqs.cryptography.ecc.ec_add import _ECAddStepThree
    from qualtran.bloqs.mod_arithmetic import (
        CModAdd,
        CModSub,
        DirtyOutOfPlaceMontgomeryModMul,
        ModAdd,
        ModDbl,
    )
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    assert version("qualtran") == "0.7.0"
    n, p, w = 3, 3, 1
    stage = EllipticCurveAdditionCoordinates(n, p, w)
    source = _ECAddStepThree(n, p, w)
    assert [(r.name, r.bitsize, r.side.name) for r in source.signature] == [
        (r.name, r.dtype.num_qubits, r.side.name) for r in stage.source_signature
    ]
    ir, ports, work, slots = layout(n, p, w)
    ctrl, a, x, y, lam = (ports[name] for name in ("ctrl", "a", "x", "y", "lam"))
    c_q, c_c = controlled_modular_requirements(n)
    a_q, a_c = modular_add_requirements(n)
    d_q, d_c = modular_double_requirements(n)
    product_scratch = work[: CurveWindowProduct(n, p, w).work_size]
    target, indices = product_scratch[:n], product_scratch[n : 2 * n]
    reduced = product_scratch[2 * n]
    padding = product_scratch[2 * n + 1 : 2 * n + 1 + w]
    bank_count = single_window_qroam_table(n, w, p).block_shape[0]
    bank_start = 2 * n + 1 + w
    bank_end = bank_start + bank_count * (n + w)
    bank = tuple(
        product_scratch[i : i + n + w] for i in range(bank_start, bank_end, n + w)
    )
    scratch = product_scratch[bank_end:]
    result = (*target[w:], *padding)
    multiply = DirtyOutOfPlaceMontgomeryModMul(n, w, p)
    args = (
        multiply,
        x,
        lam,
        target,
        indices,
        reduced,
        padding,
        bank,
        scratch,
        slots[: 2 * n],
        result,
    )

    def remap(
        binding: tuple[tuple[int, ...], dict[int, int]],
        old_work: tuple[int, ...],
        old_bits: tuple[int, ...],
        new_work: tuple[int, ...],
        new_bits: tuple[int, ...],
    ) -> tuple[tuple[int, ...], dict[int, int]]:
        alloc, measures = binding
        wire = dict(zip(old_work, new_work, strict=True))
        bit = dict(zip(old_bits, new_bits, strict=True))
        return tuple(wire[q] for q in alloc), {
            wire[q]: bit[v] for q, v in measures.items()
        }

    _, _, canonical_control_work, canonical_control_bits = controlled_layout(n)
    control_alloc, control_measure = remap(
        controlled_binding(n, p, 1),
        canonical_control_work,
        canonical_control_bits,
        work[-c_q:],
        slots[:c_c],
    )
    # CModSub's outer AddK and adjoint use CModAdd's constant/carry wires.
    control_work = work[-c_q:]
    outer_add = (*control_work[1 : n + 1], *control_work[n + 2 : 2 * n + 1])
    register = QMontgomeryUInt(n)
    sub = reference_arithmetic_gates(
        CModSub(register, p),
        {"ctrl": ctrl, "x": result, "y": y},
        (*outer_add, *control_alloc, *outer_add),
        control_measure,
        reuse_allocations=False,
        preserve_components=True,
    )
    double_work = work[n : n + d_q]
    lower, sign = double_work[:2]
    constant, carry = double_work[2 : n + 4], double_work[n + 4 :]
    temporary = work[:n]
    triple = (*temporary[1:], lower)
    double = ModDbl(register, p)
    double_measure = dict(zip(carry, slots[:d_c], strict=True))
    double_forward = reference_arithmetic_gates(
        double,
        {"x": temporary},
        (sign, lower, *constant, *carry, *constant[: n + 1], *carry[:n]),
        double_measure,
        reuse_allocations=False,
        preserve_components=True,
        expected_outputs={"x": triple},
    )
    double_reverse = reference_arithmetic_gates(
        double.adjoint(),
        {"x": triple},
        (sign, temporary[0], *constant[: n + 1], *carry[:n], *constant, *carry),
        double_measure,
        reuse_allocations=False,
        expected_outputs={"x": temporary},
    )

    def add(adjoint: bool) -> tuple[Instruction, ...]:
        proto = ModAdd(n, p)
        binding = (inverse_binding if adjoint else reference_binding)(n, p)
        _, _, canonical_add_work, canonical_add_bits = add_layout(n)
        alloc, measured = remap(
            binding,
            canonical_add_work,
            canonical_add_bits,
            control_work[:a_q],
            slots[:a_c],
        )
        return reference_arithmetic_gates(
            proto.adjoint() if adjoint else proto,
            {"x": a, "y": triple},
            alloc,
            measured,
            reuse_allocations=False,
            preserve_components=adjoint,
        )

    update = reference_arithmetic_gates(
        CModAdd(register, p),
        {"ctrl": ctrl, "x": triple, "y": x},
        control_alloc,
        control_measure,
        reuse_allocations=False,
        preserve_components=True,
    )
    copy = tuple(Instruction("cx", wires) for wires in zip(a, temporary, strict=True))
    expected = (
        *original_gates(*args, adjoint=False),
        *sub,
        *original_gates(*args, adjoint=True),
        *copy,
        *double_forward,
        *add(False),
        *update,
        *add(True),
        *double_reverse,
        *copy,
    )
    assert_source(ir.instructions, expected)
    observed = Counter(g.name for g in ir.instructions)
    modeled = get_cost_value(source, QECGatesCost())
    assert observed["measure"] == modeled.measurement
    assert observed["t"] + observed["tdg"] == modeled.total_t_count()
    assert ir.n_wires >= get_cost_value(source, QubitCount())


@pytest.mark.parametrize("stage_count", [3, 4])
def test_composed_stages_all_small_residues(stage_count: int) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.cryptography.ecc.ec_add import (
        _ECAddStepFour,
        _ECAddStepOne,
        _ECAddStepThree,
        _ECAddStepTwo,
    )

    n, p = 3, 3
    _, ports, work, slots = slope_layout(EllipticCurveAdditionSlope(n, p), reverse=True)
    width = 2 + 6 * n + len(work) + 3
    flags = (ports["f1"][0], *range(width - 3, width), ports["ctrl"][0])
    words = {name: ports[name] for name in ("a", "b", "x", "y")}
    init = EllipticCurveAdditionFlags(n, p)
    q0, c0 = init.workspace_requirements()
    slope = EllipticCurveAdditionSlope(n, p)
    update = EllipticCurveAdditionCoordinates(n, p)
    q2, c2 = update.workspace_requirements()
    circuit = (
        DynamicCircuit(width)
        .append(
            init.make_on(
                **words, flags=flags, workspace=work[:q0], classical_bits=slots[:c0]
            )
        )
        .append(
            slope.make_on(
                **words,
                f1=flags[0],
                ctrl=flags[-1],
                lam=ports["lam"],
                lam_r=ports["lam_r"],
                workspace=work,
                classical_bits=slots,
            )
        )
        .append(
            update.make_on(
                **words,
                ctrl=flags[-1],
                lam=ports["lam"],
                workspace=work[:q2],
                classical_bits=slots[:c2],
            )
        )
    )
    if stage_count == 4:
        products = EllipticCurveAdditionProducts(n, p)
        q3, c3 = products.workspace_requirements()
        circuit.append(
            products.make_on(
                x=ports["x"],
                y=ports["y"],
                lam=ports["lam"],
                workspace=work[:q3],
                classical_bits=slots[:c3],
            )
        )
    gates = circuit.to_ir().instructions
    source_products = _ECAddStepFour(n, p).decompose_bloq()
    source_flags = _ECAddStepOne(n, p)
    source_slope = _ECAddStepTwo(n, p).decompose_bloq()
    source_update = _ECAddStepThree(n, p).decompose_bloq()
    ports.update(
        {
            name: (wire,)
            for name, wire in zip(("f2", "f3", "f4"), flags[1:4], strict=True)
        }
    )
    before, after = [], []
    for row in product(range(p), repeat=5):
        inputs = dict(zip(("a", "b", "x", "y", "lam_r"), row, strict=True))
        described = source_flags.on_classical_vals(
            **{name: inputs[name] for name in words}
        )
        sec = source_slope.on_classical_vals(
            **inputs, f1=described["f1"], ctrl=described["ctrl"]
        )
        third = source_update.on_classical_vals(
            **{name: sec[name] for name in ("ctrl", "a", "b", "x", "y", "lam")}
        )
        result = {**described, **sec, **third}
        if stage_count == 4:
            result.update(
                source_products.on_classical_vals(
                    **{name: result[name] for name in ("x", "y", "lam")}
                )
            )
        before.append(pack(inputs, ports, width))
        after.append(pack(result, ports, width))
    basis = tuple(
        tuple((v >> (width - i - 1)) & 1 for i in range(width)) for v in before
    )
    observed, events = certified_and_channel_map(gates, basis)
    assert events == sum(gate.name == "measure" for gate in gates)
    assert [
        sum(v << (width - i - 1) for i, v in enumerate(row)) for row in observed
    ] == after
    samples = (0, 52, 127, 229)
    amps = (0.5, 0.5j, -0.5j, -0.5)
    state = {before[i]: amp for i, amp in zip(samples, amps, strict=True)}
    expected = {after[i]: amp for i, amp in zip(samples, amps, strict=True)}
    for seed in (13, 29):
        out = execute_sparse_register(CircuitIR(width, gates), state.copy(), seed=seed)
        assert {k: v for k, v in out.items() if abs(v) > 1e-12} == pytest.approx(
            expected, abs=1e-12
        )


def test_tripling_and_inactive_control_at_nontrivial_modulus() -> None:
    pytest.importorskip("qualtran")
    from qualtran import QMontgomeryUInt
    from qualtran.bloqs.cryptography.ecc.ec_add import _ECAddStepThree

    n, p = 4, 5
    ir, ports, _, _ = layout(n, p)
    stage = _ECAddStepThree(n, p)
    word = QMontgomeryUInt(n, p)
    width = ir.n_wires
    before, after = [], []
    for ctrl, a, x, lam in product(range(2), range(p), range(p), range(p)):
        values = dict(
            ctrl=ctrl,
            a=a,
            b=(a + 1) % p,
            x=x,
            y=word.montgomery_product(x, lam),
            lam=lam,
        )
        before.append(pack(values, ports, width))
        after.append(pack(stage.on_classical_vals(**values), ports, width))
    basis = tuple(tuple(v >> (width - i - 1) & 1 for i in range(width)) for v in before)
    output, events = certified_and_channel_map(ir.instructions, basis)
    assert events == sum(gate.name == "measure" for gate in ir.instructions)
    assert [
        sum(v << (width - i - 1) for i, v in enumerate(row)) for row in output
    ] == after


def test_source_product_reduction_can_drive_subtraction_outside_residue_domain() -> (
    None
):
    pytest.importorskip("qualtran")
    from qualtran.bloqs.cryptography.ecc.ec_add import _ECAddStepThree
    from qualtran.bloqs.mod_arithmetic import DirtyOutOfPlaceMontgomeryModMul

    n, p, w = 5, 13, 5
    ir, ports, _, _ = layout(n, p, w)
    inputs = dict(ctrl=1, a=0, b=0, x=5, y=1, lam=9)
    product_source = DirtyOutOfPlaceMontgomeryModMul(n, w, p)
    actual_product = product_source.decompose_bloq().on_classical_vals(x=5, y=9)
    assert actual_product["target"] == 27 and actual_product["reduced"] == 1
    assert _ECAddStepThree(n, p, w).on_classical_vals(**inputs)["y"] == 0
    expected = {**inputs, "y": 6}
    output = execute_sparse_register(ir, {pack(inputs, ports, ir.n_wires): 1j}, seed=11)
    assert {k: v for k, v in output.items() if abs(v) > 1e-12} == pytest.approx(
        {pack(expected, ports, ir.n_wires): 1j}, abs=1e-12
    )

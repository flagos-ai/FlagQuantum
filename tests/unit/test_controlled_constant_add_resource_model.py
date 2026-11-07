"""Single-controlled AddK source graphs, polarity and original shape fallback."""

from dataclasses import fields, replace
from itertools import product

import pytest
import sympy

from flagquantum.compiler._component_trace.arithmetic_call import ArithmeticCall
from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost
from flagquantum.compiler._component_trace.resource_generalization import CliffordCall
from flagquantum.compiler.registers.arithmetic_declarations import arithmetic_call_graph
from tests.unit.test_constant_add_resource_model import CASES
from tests.unit.test_constant_add_resource_model import source as plain_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
CONTROLLED = [
    replace(case, operation="controlled_constant_add", control_value=cv)
    for case in CASES
    for cv in (0, 1)
]


def source(call):
    if not isinstance(call, ArithmeticCall) or call.operation not in (
        "controlled_constant_add",
        "controlled_xor_constant",
    ):
        return plain_source(call)
    from qualtran import CtrlSpec

    operation = (
        "constant_add"
        if call.operation == "controlled_constant_add"
        else "xor_constant"
    )
    plain = replace(call, operation=operation, control_value=1, adjoint=False)
    result = plain_source(plain).controlled(ctrl_spec=CtrlSpec(cvs=call.control_value))
    return result.adjoint() if call.adjoint else result


@pytest.mark.parametrize("root", CONTROLLED)
def test_original_controlled_wrapper_and_each_direct_callee_row(root):
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    rows = arithmetic_call_graph(root)
    for node, row in rows.items():
        assert [(source(child), count) for child, count in row.items()] == list(
            source(node).build_call_graph(None).items()
        )
    if root.control_value == 0:
        assert list(rows[root].items()) == [
            (replace(root, control_value=1), 1),
            (CliffordCall("x", 1), 2),
        ]
    for legacy in (False, True):
        actual = query_qec_gate_cost(root, declared_calls=rows, legacy_shims=legacy)
        expected = get_cost_value(source(root), QECGatesCost(legacy_shims=legacy))
        for field in fields(actual):
            assert (
                sympy.simplify(
                    getattr(actual, field.name) - getattr(expected, field.name)
                )
                == 0
            )


@pytest.mark.parametrize("cv", (0, 1))
def test_existing_query_consumes_replacement_controlled_declarations(cv):
    from flagquantum.compiler._component_trace.resource_generalization import FixedTCall

    root = ArithmeticCall("controlled_constant_add", (3,), constant=5, control_value=cv)
    actual = query_qec_gate_cost(
        root, declared_calls={root: {FixedTCall("fredkin"): 7}}
    )
    assert (
        actual.cswap == 7
        and actual.and_bloq == actual.clifford == actual.measurement == 0
    )


@pytest.mark.parametrize("root", CONTROLLED)
def test_shared_controlled_formula_matches_all_original_fields_and_widths(root):
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    from flagquantum.compiler.registers.constant_add_resource_model import (
        source_controlled_constant_add_resource_formula,
    )

    for legacy in (False, True):
        actual = source_controlled_constant_add_resource_formula(
            root.widths[0],
            root.constant,
            control_value=root.control_value,
            encoding=root.encodings[0],
            adjoint=root.adjoint,
            legacy_shims=legacy,
        )
        expected = get_cost_value(source(root), QECGatesCost(legacy_shims=legacy))
        for field in fields(actual):
            value = (
                get_cost_value(source(root), QubitCount())
                if field.name == "logical_qubits"
                else getattr(expected, field.name)
            )
            assert sympy.simplify(getattr(actual, field.name) - value) == 0


@pytest.mark.parametrize(
    "root", [case for case in CONTROLLED if type(case.widths[0]) is int]
)
def test_controlled_concrete_shape_component_keeps_original_fallback_with_symbolic_constants(
    root,
):
    from flagquantum.compiler._component_trace.resource_model import (
        estimate_component_resources,
    )
    from flagquantum.compiler.registers.constant_add_resource_model import (
        source_controlled_constant_add_cost,
        source_controlled_constant_add_resource_formula,
    )

    opts = dict(
        control_value=root.control_value,
        encoding=root.encodings[0],
        adjoint=root.adjoint,
    )
    actual = estimate_component_resources(
        source_controlled_constant_add_cost(root.widths[0], root.constant, **opts)
    )
    expected = source_controlled_constant_add_resource_formula(
        root.widths[0], root.constant, **opts
    )
    assert actual.peak_logical_qubits == expected.logical_qubits
    for field in fields(actual.gates):
        assert getattr(actual.gates, field.name) == getattr(expected, field.name)


@pytest.mark.parametrize("cv", (0, 1))
def test_original_controlled_peak_python_literal_and_symbolic_reconstruction_are_distinct(
    cv,
):
    from flagquantum.compiler.registers.constant_add_resource_model import (
        source_controlled_constant_add_resource_formula,
    )

    concrete = source_controlled_constant_add_resource_formula(3, 5, control_value=cv)
    constant_literal = source_controlled_constant_add_resource_formula(
        3, sympy.Integer(5), control_value=cv
    )
    width_literal = source_controlled_constant_add_resource_formula(
        sympy.Integer(3), 5, control_value=cv
    )
    assert (
        concrete.logical_qubits == 9
        and constant_literal.logical_qubits == 8
        and width_literal.logical_qubits == 6
    )
    assert concrete.clifford == 15 + 2 * (1 - cv)
    assert constant_literal.clifford == width_literal.clifford == 17 + 2 * (1 - cv)


@pytest.mark.parametrize("cv", (0, 1))
def test_controlled_formula_consumes_shared_add_model_all_categories_and_original_lifetime_rule(
    monkeypatch, cv
):
    import flagquantum.compiler.registers.constant_add_resource_model as module
    from flagquantum.compiler.registers.symbolic_resource_model import (
        SymbolicResourceFormula,
    )

    n, k = sympy.symbols("n k", integer=True, positive=True)
    model = SymbolicResourceFormula(2, 3, 5, 29, t=7, toffoli=11, cswap=13, rotation=17)
    monkeypatch.setattr(
        module, "source_add_resource_formula", lambda *args, **kwargs: model
    )
    actual = module.source_controlled_constant_add_resource_formula(
        n, k, control_value=cv
    )
    assert actual == SymbolicResourceFormula(
        2,
        3 + 2 * n + 2 * (1 - cv),
        5,
        sympy.Max(29, n + 1),
        t=7,
        toffoli=11,
        cswap=13,
        rotation=17,
    )
    assert (
        module.source_controlled_constant_add_resource_formula(
            3, 5, control_value=cv
        ).logical_qubits
        == 30
    )


@pytest.mark.parametrize(
    "options",
    [
        dict(control_value=None),
        dict(control_value=True),
        dict(control_value=2),
        dict(legacy_shims=1),
    ],
)
def test_controlled_formula_refuses_invalid_control_instead_of_becoming_uncontrolled(
    options,
):
    from flagquantum.compiler.registers.constant_add_resource_model import (
        source_controlled_constant_add_resource_formula,
    )

    with pytest.raises(ValueError):
        source_controlled_constant_add_resource_formula(3, 5, **options)


@pytest.mark.parametrize(
    "root",
    [
        case
        for case in CONTROLLED
        if type(case.widths[0]) is int and type(case.constant) is int
    ],
)
def test_controlled_model_and_literal_source_primitive_stream_across_all_encodings(
    root,
):
    from collections import Counter

    from flagquantum.compiler.registers.constant_add_resource_model import (
        source_controlled_constant_add_resource_formula,
    )
    from flagquantum.compiler.registers.controlled_addition import (
        controlled_add_constant_adjoint_ir,
        controlled_add_constant_ir,
    )
    from tests.unit.test_controlled_constant_addition import layout
    from tests.unit.test_register_constant_addition import reference_arithmetic_gates
    from tests.unit.test_register_subtraction import assert_source

    width = root.widths[0]
    target, load, carry, bits = layout(width)
    emitter = (
        controlled_add_constant_adjoint_ir
        if root.adjoint
        else controlled_add_constant_ir
    )
    actual = emitter(
        0, target, root.constant, load, carry, bits, control_value=root.control_value
    )
    expected = reference_arithmetic_gates(
        source(root),
        {"ctrl": (0,), "x": target},
        (*load, *carry),
        dict(zip(carry, bits, strict=True)),
    )
    assert_source(actual, expected)
    counts = Counter(g.name for g in actual)
    model = source_controlled_constant_add_resource_formula(
        width,
        root.constant,
        control_value=root.control_value,
        encoding=root.encodings[0],
        adjoint=root.adjoint,
    )
    assert counts["t"] + counts["tdg"] == 4 * model.and_bloq
    assert counts["measure"] == model.measurement == width - 1
    assert model.logical_qubits == 3 * width


@pytest.mark.parametrize(
    "width,encoding,value,cv,adjoint",
    tuple(
        product(
            (1, 2, 3),
            ("unsigned", "twos_complement", "montgomery"),
            (0, -1),
            (0, 1),
            (False, True),
        )
    ),
)
def test_all_coherent_control_target_spectator_branches_preserve_phase_and_clean_work(
    width, encoding, value, cv, adjoint
):
    import torch

    from flagquantum.compiler.registers.controlled_addition import (
        controlled_add_constant_adjoint_ir,
        controlled_add_constant_ir,
    )
    from tests.unit.measurement_checks import forced_branch
    from tests.unit.test_controlled_constant_addition import layout

    limit = 1 << width
    reference = plain_source(
        ArithmeticCall("constant_add", (width,), (encoding,), constant=value)
    )

    def decoded(x):
        return x - limit if encoding == "twos_complement" and x >= limit // 2 else x

    mapping = {
        x: reference.on_classical_vals(x=decoded(x))["x"] % limit for x in range(limit)
    }
    if adjoint:
        mapping = {b: a for a, b in mapping.items()}
    target, load, carry, bits = layout(width)
    emitter = (
        controlled_add_constant_adjoint_ir if adjoint else controlled_add_constant_ir
    )
    gates = emitter(0, target, value, load, carry, bits, control_value=cv)
    work = 2 * width - 1
    data = torch.arange(1 << (width + 2), dtype=torch.int64)
    coeff = data.to(torch.float64)
    amplitudes = torch.complex(coeff + 1, 3 * coeff - 5)
    amplitudes /= amplitudes.norm()
    state = torch.zeros(1 << (3 * width + 1), dtype=torch.complex128)
    state[((data >> 1) << (work + 1)) | (data & 1)] = amplitudes
    expected = torch.zeros_like(state)
    x = (data >> 1) & (limit - 1)
    active = (data >> (width + 1)) == cv
    mapped = torch.tensor(
        [
            mapping[int(a)] if bool(b) else int(a)
            for a, b in zip(x, active, strict=True)
        ],
        dtype=torch.int64,
    )
    addresses = (((data >> (width + 1)) << width) | mapped) << (work + 1)
    expected[addresses | (data & 1)] = amplitudes
    for history in product((0, 1), repeat=width - 1):
        torch.testing.assert_close(
            forced_branch(state, gates, history),
            expected / 2 ** ((width - 1) / 2),
            rtol=0,
            atol=1e-12,
        )

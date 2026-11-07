"""Unmodified XOR declarations, symbolic fallback and shared outer lifetimes."""

from dataclasses import fields
from itertools import product

import pytest
import sympy

from flagquantum.compiler._component_trace.arithmetic_call import ArithmeticCall
from flagquantum.compiler._component_trace.resource_generalization import (
    CliffordCall,
    FixedTCall,
)
from flagquantum.compiler.registers.arithmetic_declarations import arithmetic_callees
from flagquantum.compiler.registers.xor_resource_model import (
    source_grouped_xor_resource_formula,
    source_xor_resource_formula,
)
from tests.unit.test_grouped_controlled_constant_addition import EXTRA
from tests.unit.test_register_control_predicate import CASES, control_fixture

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
n = sympy.Symbol("n", positive=True, integer=True)
z = sympy.Symbol("z", integer=True)
WIDTHS = (1, 3, sympy.Integer(3), n, z, n + 1)
ENCODINGS = (
    ("QUInt", "unsigned"),
    ("QInt", "twos_complement"),
    ("QMontgomeryUInt", "montgomery"),
)


def compare(actual, source, legacy):
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    expected = get_cost_value(source, QECGatesCost(legacy_shims=legacy))
    for field in fields(actual):
        value = (
            get_cost_value(source, QubitCount())
            if field.name == "logical_qubits"
            else getattr(expected, field.name)
        )
        assert sympy.simplify(getattr(actual, field.name) - value) == 0


@pytest.mark.parametrize(
    "width,encoding,control,legacy",
    tuple(product(WIDTHS, ENCODINGS, (None, 0, 1), (False, True))),
)
def test_plain_and_single_control_all_source_categories_and_peak(
    width, encoding, control, legacy
):
    import qualtran
    from qualtran.bloqs.arithmetic.bitwise import Xor

    source = Xor(getattr(qualtran, encoding[0])(width))
    if control is not None:
        source = source.controlled(qualtran.CtrlSpec(cvs=control))
    compare(
        source_xor_resource_formula(
            width, encoding=encoding[1], control_value=control, legacy_shims=legacy
        ),
        source,
        legacy,
    )


@pytest.mark.parametrize(
    "case,width,encoding,legacy",
    tuple(product(CASES + EXTRA, WIDTHS, ENCODINGS, (False, True))),
)
def test_grouped_source_composition_with_unknown_width_and_both_data_words(
    case, width, encoding, legacy
):
    import qualtran
    from qualtran.bloqs.arithmetic.bitwise import Xor

    spec, regs, _ = control_fixture(case)
    source = Xor(getattr(qualtran, encoding[0])(width)).controlled(ctrl_spec=spec)
    compare(
        source_grouped_xor_resource_formula(
            width, regs, encoding=encoding[1], legacy_shims=legacy
        ),
        source,
        legacy,
    )


@pytest.mark.parametrize(
    "width,encoding,control", tuple(product((3, n), ENCODINGS, (None, 0, 1)))
)
def test_native_direct_edges_retain_original_callee_and_counts(
    width, encoding, control
):
    import qualtran
    from qualtran.bloqs.arithmetic.bitwise import Xor
    from qualtran.bloqs.basic_gates import CNOT, Toffoli, XGate
    from qualtran.resource_counting import SympySymbolAllocator

    root = ArithmeticCall(
        "xor" if control is None else "controlled_xor",
        (width,),
        (encoding[1],),
        control_value=1 if control is None else control,
    )
    source = Xor(getattr(qualtran, encoding[0])(width))
    if control is not None:
        source = source.controlled(qualtran.CtrlSpec(cvs=control))
    expected = dict(source.build_call_graph(SympySymbolAllocator()))
    translated = {}
    for child, count in arithmetic_callees(root).items():
        if isinstance(child, CliffordCall):
            leaf = CNOT() if child.opcode == "cx" else XGate()
        elif isinstance(child, FixedTCall):
            leaf = Toffoli()
        else:
            assert (
                isinstance(child, ArithmeticCall)
                and child.operation == "controlled_xor"
                and child.control_value == 1
            )
            leaf = Xor(getattr(qualtran, encoding[0])(width)).controlled()
        translated[leaf] = count
    assert translated == expected
    assert root.widths[0] is width


@pytest.mark.parametrize("width", (0, sympy.Integer(0)))
@pytest.mark.parametrize("control", (None, 0, 1))
def test_original_zero_width_failure_is_not_repaired_by_formula(width, control):
    from qualtran import CtrlSpec, QUInt
    from qualtran.bloqs.arithmetic.bitwise import Xor
    from qualtran.resource_counting import QubitCount, get_cost_value

    source = Xor(QUInt(width))
    if control is not None:
        source = source.controlled(CtrlSpec(cvs=control))
    with pytest.raises(RuntimeError):
        get_cost_value(source, QubitCount())
    with pytest.raises(ValueError, match="zero-width"):
        source_xor_resource_formula(width, control_value=control)


def test_unconstrained_symbol_does_not_discard_source_callee_peak():
    from flagquantum.compiler.registers.control_predicate import ControlRegister

    assert source_xor_resource_formula(z).logical_qubits == sympy.Max(2, 2 * z)
    grouped = source_grouped_xor_resource_formula(z, (ControlRegister(1, (0, 1)),))
    assert (
        sympy.simplify(
            grouped.logical_qubits - sympy.Max(2 * z + 3, sympy.Max(3, 2 * z + 1) + 2)
        )
        == 0
    )


@pytest.mark.parametrize(
    "kwargs",
    (
        {"width": True},
        {"width": -1},
        {"control_value": True},
        {"control_value": 2},
        {"legacy_shims": 1},
    ),
)
def test_invalid_parameters_are_not_cast(kwargs):
    with pytest.raises(ValueError):
        source_xor_resource_formula(**({"width": n} | kwargs))


def test_concrete_consumer_queries_replaceable_native_declarations(monkeypatch):
    import flagquantum.compiler.registers.controlled_register_xor as consumer
    from flagquantum.compiler._component_trace.resource_model import (
        estimate_component_resources,
    )
    from flagquantum.compiler.registers.control_predicate import ControlRegister

    seen = []

    def recipe(root):
        seen.append(root)
        return {root: {FixedTCall("fredkin"): 2}}

    monkeypatch.setattr(consumer, "arithmetic_call_graph", recipe)
    actual = estimate_component_resources(
        consumer.source_grouped_xor_cost(3, (ControlRegister(1, (1,)),))
    )
    assert seen == [ArithmeticCall("controlled_xor", (3,))]
    assert actual.gates.cswap == 2 and actual.gates.toffoli == 0
    assert actual.peak_logical_qubits == 7

"""Verify declared shape costs against the unmodified Qualtran models."""

import subprocess
import sys

import pytest

from flagquantum.compiler.registers.arithmetic_resource_model import source_add_cost
from flagquantum.compiler.registers.symbolic_resource_model import (
    SymbolicResourceFormula,
    source_add_resource_formula,
    source_permutation_cycle_resource_formula,
    source_phase_gradient_lookup_resource_formula,
    source_qrom_resource_formula,
)

pytestmark = pytest.mark.unit


def assert_resource_formula(formula: SymbolicResourceFormula, source: object) -> None:
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value
    from sympy import simplify

    model = get_cost_value(source, QECGatesCost())
    for field in ("and_bloq", "clifford", "measurement", "toffoli", "t"):
        assert simplify(getattr(formula, field) - getattr(model, field)) == 0, field
    assert simplify(formula.logical_qubits - get_cost_value(source, QubitCount())) == 0


@pytest.mark.ftqc_reference
def test_symbolic_add_uses_signature_fallback_until_widths_are_bound() -> None:
    pytest.importorskip("qualtran")
    sympy = pytest.importorskip("sympy")
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic import Add

    left, right = sympy.symbols("m n", positive=True, integer=True)
    formula = source_add_resource_formula(left, right)
    assert_resource_formula(formula, Add(QUInt(left), QUInt(right)))
    for a, b in ((1, right), (left, 5), (sympy.Integer(2), sympy.Integer(5))):
        assert_resource_formula(
            source_add_resource_formula(a, b), Add(QUInt(a), QUInt(b))
        )
    for m, n in ((1, 1), (1, 2), (2, 5), (8, 8)):
        concrete = source_add_cost(m, n)
        bound = source_add_resource_formula(m, n)
        assert_resource_formula(bound, Add(QUInt(m), QUInt(n)))
        assert (
            formula.measurement.subs({left: m, right: n}) == concrete.gates.measurement
        )
        assert formula.clifford.subs({left: m, right: n}) == concrete.gates.clifford
        assert formula.logical_qubits.subs({left: m, right: n}) == max(3, m + n)
        assert bound.logical_qubits == concrete.qubits


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("mode", ["rank_one", "rank_two", "explicit_widths"])
def test_shape_only_qrom_matches_source_and_concrete_substitution(mode: str) -> None:
    pytest.importorskip("qualtran")
    sympy = pytest.importorskip("sympy")
    from qualtran.bloqs.data_loading.qrom import QROM

    m, n, bits, control = sympy.symbols("m n b c", positive=True, integer=True)
    if mode == "rank_one":
        shape, widths, target_shapes = (n,), (bits,), ((),)
        selections = None
    else:
        shape, widths, target_shapes = (m, n), (bits, 3), ((), (2,))
        selections = (m, bits) if mode == "explicit_widths" else None
    formula = source_qrom_resource_formula(
        shape,
        widths,
        target_shapes=target_shapes,
        control_count=control,
        selection_widths=selections,
    )
    if selections is None:
        source = QROM.build_from_bitsize(
            shape, widths, target_shapes=target_shapes, num_controls=control
        )
    else:
        from qualtran.symbolics import Shaped

        source = QROM(
            tuple(Shaped((*shape, *target)) for target in target_shapes),
            selections,
            widths,
            target_shapes,
            num_controls=control,
        )
    assert_resource_formula(formula, source)
    params = {m: 4, n: 5, bits: 4, control: 2}

    def bind(values: tuple) -> tuple[int, ...]:
        return tuple(int(sympy.sympify(value).subs(params)) for value in values)

    bound = source_qrom_resource_formula(
        bind(shape),
        bind(widths),
        target_shapes=target_shapes,
        control_count=2,
        selection_widths=None if selections is None else bind(selections),
    )
    for key in ("and_bloq", "clifford", "measurement", "logical_qubits"):
        assert int(sympy.sympify(getattr(formula, key)).subs(params)) == getattr(
            bound, key
        )


def test_shape_metadata_validation() -> None:
    with pytest.raises(ValueError, match="at least one target"):
        source_qrom_resource_formula((5,), ())
    with pytest.raises(ValueError, match="target widths and shapes"):
        source_qrom_resource_formula((5,), (2,), target_shapes=())
    with pytest.raises(ValueError, match="selection widths"):
        source_qrom_resource_formula((5,), (2,), selection_widths=())


def test_concrete_formulas_do_not_import_sympy() -> None:
    code = """
import sys
from importlib.abc import MetaPathFinder

class BlockSympy(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'sympy':
            raise AssertionError(f'Unexpected optional import: {fullname}')
        return None

sys.meta_path.insert(0, BlockSympy())
from flagquantum.compiler.registers.symbolic_resource_model import (
    source_add_resource_formula,
    source_permutation_cycle_resource_formula,
    source_phase_gradient_lookup_resource_formula, source_qrom_resource_formula,
)
assert source_add_resource_formula(2, 4).logical_qubits == 9
assert source_qrom_resource_formula((5,), (2,), control_count=1).measurement == 4
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=30)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("controls", [0, 1, 3])
def test_shape_only_phase_gradient_lookup(controls: int) -> None:
    pytest.importorskip("qualtran")
    sympy = pytest.importorskip("sympy")
    from qualtran.bloqs.state_preparation.state_preparation_via_rotation import (
        PRGAViaPhaseGradient,
    )
    from qualtran.symbolics import Shaped, bit_length

    n, b = sympy.symbols("n b", positive=True, integer=True)
    selectors = bit_length(n - 1)
    assert_resource_formula(
        source_phase_gradient_lookup_resource_formula(n, selectors, b, controls),
        PRGAViaPhaseGradient(selectors, b, Shaped((n,)), controls),
    )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("concrete_width", [None, 1, 5])
def test_shape_only_permutation_cycle(concrete_width: int | None) -> None:
    pytest.importorskip("qualtran")
    sympy = pytest.importorskip("sympy")
    from qualtran.bloqs.arithmetic.permutation import PermutationCycle
    from qualtran.symbolics import Shaped

    length, width = sympy.symbols("l w", positive=True, integer=True)
    if concrete_width is not None:
        width = concrete_width
    assert_resource_formula(
        source_permutation_cycle_resource_formula(length, width),
        PermutationCycle(2**width, Shaped((length,)), bitsize=width),
    )


@pytest.mark.parametrize("mode", ("both", "left", "right", "literal"))
@pytest.mark.parametrize("legacy", (False, True))
def test_symbolic_add_queries_replaceable_native_recipe(monkeypatch, mode, legacy):
    import sympy

    import flagquantum.compiler.registers.symbolic_resource_model as module
    from flagquantum.compiler._component_trace.arithmetic_call import ArithmeticCall
    from flagquantum.compiler._component_trace.resource_generalization import FixedTCall

    n = sympy.Symbol("n", integer=True, positive=True)
    left, right = {
        "both": (n, n),
        "left": (n, 3),
        "right": (1, n),
        "literal": (sympy.Integer(2), sympy.Integer(3)),
    }[mode]
    seen = []

    def recipe(root):
        seen.append(root)
        return {FixedTCall("fredkin"): 7 * n}

    monkeypatch.setattr(module, "arithmetic_callees", recipe)
    result = module.source_add_resource_formula(left, right, legacy_shims=legacy)
    assert seen == [ArithmeticCall("add", (left, right))]
    assert result.cswap == 7 * n and result.logical_qubits == sympy.Max(3, left + right)
    assert result.and_bloq == result.clifford == result.measurement == 0


def test_concrete_add_formula_preserves_all_child_categories_and_lifetimes(monkeypatch):
    import flagquantum.compiler.registers.symbolic_resource_model as module
    from flagquantum.compiler._component_trace.resource_model import (
        ComponentCost,
        CostCall,
        GateCost,
    )

    gates = GateCost(
        t=2, toffoli=3, cswap=5, and_bloq=7, clifford=11, rotation=13, measurement=17
    )
    child = ComponentCost("child", 19, gates)
    replacement = ComponentCost("Add", 5, calls=(CostCall(child, 3, 4),))
    monkeypatch.setattr(module, "source_add_cost", lambda *args, **kwargs: replacement)
    actual = module.source_add_resource_formula(2, 3)
    assert actual == SymbolicResourceFormula(
        21, 33, 51, 23, t=6, toffoli=9, cswap=15, rotation=39
    )

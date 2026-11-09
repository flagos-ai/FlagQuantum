"""Compare real and optional symbolic resource arithmetic with the source."""

import pytest

from flagquantum.compiler._component_trace.symbolic_math import (
    bit_length,
    ceil,
    floor,
    log2,
    smax,
    smin,
)

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
def test_symbolic_math_matches_locked_expressions_and_substitutions() -> None:
    pytest.importorskip("qualtran")
    sympy = pytest.importorskip("sympy")
    from qualtran import symbolics as source

    n = sympy.Symbol("n", positive=True, integer=True)
    x = sympy.Symbol("x", nonnegative=True)
    comparisons = (
        (log2(n), source.log2(n)),
        (ceil(x), source.ceil(x)),
        (floor(x), source.floor(x)),
        (bit_length(n - 1), source.bit_length(n - 1)),
        (bit_length(x), source.bit_length(x)),
        (smax(3, n, x), source.smax(3, n, x)),
        (smin(iter((3, n, x))), source.smin(iter((3, n, x)))),
    )
    for actual, expected in comparisons:
        assert actual == expected
        assert actual.subs({n: 17, x: sympy.Rational(7, 2)}) == expected.subs(
            {n: 17, x: sympy.Rational(7, 2)}
        )


def test_concrete_math_and_empty_extrema() -> None:
    for i in range(1024):
        assert bit_length(i) == bit_length(i + 0.5) == i.bit_length()
    assert ceil(-1.25) == -1
    assert floor(-1.25) == -2
    assert log2(2**10000) == 10000
    assert smax((1, 2)) == 2
    assert smin(3) == 3
    with pytest.raises(ValueError, match="at least 1 argument"):
        smax()
    with pytest.raises(ValueError, match="empty sequence"):
        smin(iter(()))


def test_compiler_functions_share_core_authority() -> None:
    from flagquantum.compiler._component_trace import symbolic_math
    from flagquantum.core import value_math

    for name in ("ceil", "floor", "log2", "bit_length", "smax", "smin"):
        assert getattr(symbolic_math, name) is getattr(value_math, name)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("backend", ["source", "native"])
def test_actual_width_consumers_accept_replacement(
    backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sympy
    from qualtran.symbolics import bit_length as source_width

    from flagquantum.compiler.registers.bits import hamming_weight
    from flagquantum.compiler.registers.lookup import qrom_table
    from flagquantum.compiler.registers.resources import symbolic_resource_model
    from flagquantum.core.value_math import SymbolicInt
    from flagquantum.core.value_math import bit_length as native_width

    n = sympy.Symbol("n", integer=True, positive=True)
    baseline = qrom_table.QromTable.from_data([0, 1, 2, 3], num_controls=1)
    work = tuple(range(5, 5 + baseline.workspace_size))
    bits = tuple(range(baseline.workspace_size))
    expected_gates = baseline.bind_packed(tuple(range(5)), work, bits)
    expected_hamming = hamming_weight.hamming_weight_resource_widths(n)
    expected_formula = symbolic_resource_model.source_qrom_resource_formula(
        (n, 3), (2,)
    )
    calls: list[SymbolicInt] = []
    dependency = source_width if backend == "source" else native_width

    def width(value: SymbolicInt) -> SymbolicInt:
        calls.append(value)
        return dependency(value)

    for consumer in (qrom_table, hamming_weight, symbolic_resource_model):
        monkeypatch.setattr(consumer, "bit_length", width)
    table = qrom_table.QromTable.from_data([0, 1, 2, 3], num_controls=1)
    assert table.signature == baseline.signature
    assert table.bind_packed(tuple(range(5)), work, bits) == expected_gates
    assert hamming_weight.hamming_weight_resource_widths(n) == expected_hamming
    assert (
        symbolic_resource_model.source_qrom_resource_formula((n, 3), (2,))
        == expected_formula
    )
    assert n in calls and n - 1 in calls and 2 in calls and 3 in calls

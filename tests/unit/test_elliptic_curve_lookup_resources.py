"""Compare complete point-lookup declarations with the frozen source property."""

from dataclasses import fields
from hashlib import sha1
from importlib.metadata import version
from pathlib import Path

import pytest

from flagquantum.algorithms.primitives.elliptic_curve_lookup_resources import (
    windowed_point_lookup_resources,
)
from flagquantum.algorithms.primitives.elliptic_curve_point import ECPoint
from flagquantum.compiler.registers.resources.symbolic_resource_model import (
    SymbolicResourceFormula,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("profile", ["word", "constant"])
@pytest.mark.parametrize("concrete_window", [1, 4, 5])
def test_original_lookup_property_and_both_independent_cost_models(
    profile: str, concrete_window: int
) -> None:
    import sympy
    from qualtran.bloqs.cryptography.ecc import ec_add_r
    from qualtran.bloqs.cryptography.ecc.ec_point import ECPoint as OriginalPoint
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    assert version("qualtran") == "0.7.0"
    n = sympy.Symbol("n", positive=True, integer=True)
    width = n if profile == "word" else 5
    window = concrete_window
    original = ec_add_r.ECWindowAddR(width, OriginalPoint(15, 13, 17), window)
    table = original.qrom
    actual = windowed_point_lookup_resources(width, window, ECPoint(15, 13, 17))
    assert set(actual) == {"QROAMClean", "QROAMCleanAdjointWrapper"}
    if profile == "word":
        assert table.log_block_sizes == (0,)
    for child in (table, table.adjoint()):
        model = actual[type(child).__name__]
        gates = get_cost_value(child, QECGatesCost())
        for field in fields(SymbolicResourceFormula):
            expected = (
                get_cost_value(child, QubitCount())
                if field.name == "logical_qubits"
                else getattr(gates, field.name)
            )
            assert sympy.simplify(getattr(model, field.name) - expected) == 0, (
                profile,
                field.name,
            )
    # The declared resources carry no substituted data, emitted primitive
    # counts or numerical interpretation for unknown curve coordinates.
    assert all(isinstance(model, SymbolicResourceFormula) for model in actual.values())


def test_symbolic_coordinates_are_irrelevant_to_shape_only_lookup_declaration() -> None:
    import sympy
    from qualtran.bloqs.cryptography.ecc.ec_add_r import ECWindowAddR
    from qualtran.bloqs.cryptography.ecc.ec_point import ECPoint as OriginalPoint

    n, x, y, p = sympy.symbols("n x y p")
    original = ECWindowAddR(n, OriginalPoint(x, y, p), 4)
    assert original.qrom.log_block_sizes == (0,)
    assert windowed_point_lookup_resources(n, 4, ECPoint(x, y, p)) == (
        windowed_point_lookup_resources(n, 4, ECPoint(15, 13, 17))
    )


def test_frozen_source_identity() -> None:
    from qualtran.bloqs.cryptography.ecc import ec_add_r

    raw = Path(ec_add_r.__file__).read_bytes()
    assert (
        sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == "195654d5b4b269d13d07e96a755424755b0196e5"
    )


@pytest.mark.parametrize("symbolic_width", [False, True])
def test_symbolic_window_retains_original_constructor_failure(
    symbolic_width: bool,
) -> None:
    import sympy
    from qualtran.bloqs.cryptography.ecc.ec_add_r import ECWindowAddR
    from qualtran.bloqs.cryptography.ecc.ec_point import ECPoint as OriginalPoint

    n, k = sympy.symbols("n k", positive=True, integer=True)
    width = n if symbolic_width else 5
    source = ECWindowAddR(width, OriginalPoint(15, 13, 17), k)
    with pytest.raises(TypeError, match="NoneType.*not iterable"):
        _ = source.qrom
    with pytest.raises(ValueError, match="no constructible source declaration"):
        windowed_point_lookup_resources(width, k, ECPoint(15, 13, 17))

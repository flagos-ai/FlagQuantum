"""Bind complete original inverse-root source libraries as typed declarations."""

from importlib import import_module
from typing import Any

import pytest
import sympy

from flagquantum.compiler.formats.legacy_component_library import (
    deserialize_component_library,
    serialize_component_library,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_grid_inverse_sqrt_factories import (
    grid_inverse_sqrt_factories,
)
from flagquantum.compiler.registers.fixed_point.grid_inverse_sqrt_signature import (
    grid_inverse_sqrt_signature,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import FunctionRecord, verify_program_record
from flagquantum.core.ir.program.register_types import unsigned_register_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.errors import SerializationError
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_subtraction_foreign_file import native_signature

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
_MODULE = "qualtran.bloqs.chemistry.trotter.grid_ham.inverse_sqrt"


def _source(newton: bool, widths: tuple[Any, Any, Any]) -> Any:
    module = import_module(_MODULE)
    name = (
        "NewtonRaphsonApproxInverseSquareRoot"
        if newton
        else "PolynmomialEvaluationInverseSquareRoot"
    )
    return getattr(module, name)(*widths)


@pytest.mark.parametrize("newton", (False, True))
@pytest.mark.parametrize("depth", (0, 1, 2))
@pytest.mark.parametrize("widths", ((7, 8, 12), (1, 2, 3), (0, 0, 0), (2, 0, 3)))
def test_actual_source_wire_preserves_complete_typed_declaration(
    newton: bool, depth: int, widths: tuple[int, int, int]
) -> None:
    from qualtran import DecomposeNotImplementedError

    source = _source(newton, widths)
    if newton and depth > 0 and widths[1] < 3:
        with pytest.raises(ValueError, match="at least 3 for SquareRealNumber"):
            _library(source, depth)
        # The root itself still has a valid shallow declaration; only recursive
        # source resource discovery constructs the invalid SquareRealNumber.
        library = _library(source, 0)
    else:
        library = _library(source, depth)
    encoded = serialize_component_library(library)
    assert (
        serialize_component_library(deserialize_component_library(encoded)) == encoded
    )
    program = resolve_component_program(library, 0, grid_inverse_sqrt_factories())
    assert verify_program_record(program) == ()
    assert program.functions[0].registers == tuple(native_signature(source.signature))
    assert program.functions[0].registers == tuple(
        grid_inverse_sqrt_signature(*widths, newton=newton)
    )
    assert not program.functions[0].blocks
    with pytest.raises(DecomposeNotImplementedError):
        source.decompose_bloq()
    with pytest.raises(ValueError, match="bodies"):
        bind_register_program(program, ())


@pytest.mark.parametrize("newton", (False, True))
@pytest.mark.parametrize("depth", (0, 1, 2))
def test_symbolic_source_ports_remain_logical_without_physical_fallback(
    newton: bool, depth: int
) -> None:
    n, m, t = sympy.symbols("n m t")
    source = _source(newton, (n, m, t))
    program = resolve_component_program(
        _library(source, depth), 0, grid_inverse_sqrt_factories()
    )
    assert program.functions[0].registers == tuple(
        grid_inverse_sqrt_signature(n, m, t, newton=newton)
    )
    assert verify_program_record(program) == ()
    with pytest.raises(ValueError, match="bodies"):
        bind_register_program(program, ())


@pytest.mark.parametrize("newton", (False, True))
def test_original_factory_docs_and_actual_model_children_keep_their_roles(
    newton: bool,
) -> None:
    from qualtran.resource_counting import SympySymbolAllocator

    module = import_module(_MODULE)
    factory = module._nr_inv_sqrt if newton else module._poly_inv_sqrt
    docs = module._NR_INV_SQRT if newton else module._POLY_INV_SQRT
    source = factory()
    assert source == _source(newton, (7, 8, 12))
    assert docs.bloq_cls is type(source) and docs.examples == (factory,)
    # The source graph is unordered model data, not a gate sequence.
    children = source.build_call_graph(SympySymbolAllocator())
    expected = (
        {"SquareRealNumber": 1, "ScaleIntByReal": 1, "MultiplyTwoReals": 2, "Add": 1}
        if newton
        else {"MultiplyTwoReals": 3, "Add": 3}
    )
    assert {
        type(child).__name__: count for child, count in children.items()
    } == expected
    program = resolve_component_program(
        _library(source, 2), 0, grid_inverse_sqrt_factories()
    )
    assert not program.functions[0].blocks


@pytest.mark.parametrize("newton", (False, True))
def test_invalid_metadata_cannot_change_source_port_types_or_ownership(
    newton: bool,
) -> None:
    source = _source(newton, (2, 3, 4))
    factory = grid_inverse_sqrt_factories()[_MODULE + "." + type(source).__name__]
    output = "target_bitsize" if newton else "out_bitsize"
    good: dict[str, Any] = {"x_sq_bitsize": 2, "poly_bitsize": 3, output: 4}
    direct = factory("root", None, good)
    assert isinstance(direct, FunctionRecord)
    assert direct.registers == tuple(native_signature(source.signature))
    for bad in (
        {},
        {**good, "extra": 0},
        {**good, "poly_bitsize": "3"},
        {**good, "poly_bitsize": True},
    ):
        with pytest.raises(SerializationError):
            factory("bad", None, bad)
    assert direct.registers is not None
    changed = (Register("x_sq", unsigned_register_type(2)), *direct.registers[1:])
    with pytest.raises(SerializationError, match="signature"):
        factory("bad", changed, good)
    with pytest.raises(SerializationError, match="signature"):
        factory("bad", tuple(reversed(direct.registers)), good)


def test_frozen_complete_source_test_assets_and_real_execution_extension() -> None:
    from hashlib import sha1
    from importlib.metadata import version
    from pathlib import Path

    import kickmix

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and callable(kickmix.Simulator)
    root = (
        Path(__file__).resolve().parents[3]
        / "ftqc-reference-env/qualtran/bloqs/chemistry/trotter/grid_ham"
    )
    for name, digest in (
        ("inverse_sqrt.py", "d9beebf87156cbd85f1405eebcb4df402842c1eb"),
        ("inverse_sqrt_test.py", "43ecf9c3178797d6373d6489ceada15fbd289fe6"),
    ):
        data = (root / name).read_bytes()
        assert sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == digest


@pytest.mark.parametrize("newton", (False, True))
@pytest.mark.parametrize(
    "widths", ((7, 8, 12), (5, 15, 22), (4, 3, 1), sympy.symbols("n m t"))
)
def test_generic_resource_formula_preserves_original_categories_and_width_fallback(
    newton: bool, widths: tuple[Any, Any, Any]
) -> None:
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    from flagquantum.algorithms.primitives.grid_inverse_sqrt import (
        source_grid_inverse_sqrt_resource_formula,
    )

    source = _source(newton, widths)
    actual = source_grid_inverse_sqrt_resource_formula(*widths, newton=newton)
    expected = get_cost_value(source, QECGatesCost())
    for category in (
        "t",
        "toffoli",
        "and_bloq",
        "clifford",
        "measurement",
        "cswap",
        "rotation",
    ):
        assert (
            sympy.simplify(getattr(actual, category) - getattr(expected, category)) == 0
        )
    assert (
        sympy.simplify(actual.logical_qubits - get_cost_value(source, QubitCount()))
        == 0
    )


@pytest.mark.parametrize("width", (0, 1, 2))
def test_generic_newton_model_retains_original_small_square_refusal(width: int) -> None:
    from qualtran.resource_counting import SympySymbolAllocator

    from flagquantum.algorithms.primitives.grid_inverse_sqrt import (
        source_grid_inverse_sqrt_resource_formula,
    )

    with pytest.raises(ValueError, match="at least 3 for SquareRealNumber"):
        _source(True, (4, width, 8)).build_call_graph(SympySymbolAllocator())
    with pytest.raises(ValueError, match="at least 3 for SquareRealNumber"):
        source_grid_inverse_sqrt_resource_formula(4, width, 8, newton=True)

"""Existing modular operations expose their owned ordered source libraries."""

import pytest
import sympy

from flagquantum.compiler.formats.symbolic_modular_library import (
    ModularGraphKind,
    modular_component_library,
)
from flagquantum.core.ir.program.register_types import unsigned_register_type
from tests.unit.test_symbolic_modular_library import CASES, _canonical

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "kind,cv", tuple((kind, cv) for kind, cv in CASES if (kind, cv) != ("CModAdd", 0))
)
def test_existing_example_objects_expose_the_native_ordered_graph(
    kind: ModularGraphKind, cv: int
) -> None:
    from examples.algorithms.modular_addition_catalog import ModularAdditionExample
    from examples.algorithms.modular_subtraction_catalog import (
        ModularSubtractionExample,
    )

    p = sympy.Symbol("p", integer=True, positive=True)
    if kind in ("ModAdd", "CModAdd", "CModAddK", "CtrlScaleModAdd"):
        actual = ModularAdditionExample(kind, 3, p, 1).component_library()
    elif kind in ("ModNeg", "CModNeg", "ModSub", "CModSub"):
        actual = ModularSubtractionExample(
            kind, 3, p, control_value=cv
        ).component_library()
    else:
        raise AssertionError("Unknown modular operation")
    expected = modular_component_library(
        kind, unsigned_register_type(3), p, constant=1, control_value=cv
    )
    assert _canonical(actual) == _canonical(expected)

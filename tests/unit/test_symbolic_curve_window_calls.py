"""Retain the complete symbolic point window and every available lookup layer."""

import ast
import sys
from importlib import import_module
from pathlib import Path
from unittest.mock import patch

import pytest

from tests._reference_dependencies import unavailable_reference

try:
    import_module("numpy")
    import sympy

    import_module("cirq")
    import_module("kickmix")
    from qualtran import DecomposeTypeError
    from qualtran.bloqs.cryptography.ecc.ec_add_r import ECWindowAddR
    from qualtran.bloqs.cryptography.ecc.ec_point import ECPoint
except ModuleNotFoundError as error:
    unavailable_reference(error)

from flagquantum.compiler.formats.legacy_component_library import (
    deserialize_component_library,
    serialize_component_library,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.ecosystem._elliptic_curve_window_library import (
    elliptic_curve_window_factories,
)
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_ecc_addition_model import frozen_source as frozen_source
from tests.unit.test_elliptic_curve_window_components import (
    frozen_window_source as frozen_window_source,
)
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_qroam_clean_component import pinned_source as pinned_source

pytestmark = [
    pytest.mark.unit,
    pytest.mark.ftqc_reference,
    pytest.mark.usefixtures("pinned_source"),
]


@pytest.mark.parametrize("window", (1, 4))
@pytest.mark.parametrize("offset_width", (False, True))
@pytest.mark.parametrize("symbolic_point", (False, True))
def test_symbolic_width_keeps_five_root_four_lookup_six_add_and_one_erase_calls(
    window: int, offset_width: bool, symbolic_point: bool
) -> None:
    n, x, y, a = sympy.symbols("n x y a")
    point = ECPoint(x, y, 17, a) if symbolic_point else ECPoint(1, 2, 17, 1)
    source = ECWindowAddR(n + 1 if offset_width else n, point, window)
    encoded = serialize_component_library(_library(source, 0))
    library = deserialize_component_library(encoded)
    assert serialize_component_library(library) == encoded
    with patch(
        "flagquantum.ecosystem._elliptic_curve_window_library.windowed_point_table",
        side_effect=AssertionError("A shape-only source must not sample coordinates"),
    ):
        record = resolve_component_program(
            library, 0, elliptic_curve_window_factories()
        )
    functions = {function.name: function for function in record.functions}
    terminals = {
        function.name
        for function in record.functions
        if not function.blocks
        or any(
            operation.opcode != "call"
            for block in function.blocks
            for operation in block.operations
        )
    }
    pairs = compare_layers(record, source, terminals)
    assert len(pairs) == 17
    assert len(record.functions[0].blocks[0].operations) == 5
    for name, child in pairs.items():
        if name in terminals:
            with pytest.raises(DecomposeTypeError):
                child.decompose_bloq()
        else:
            expected = {
                "ECWindowAddR": 5,
                "ECAdd": 6,
                "QROAMClean": 4,
                "QROAMCleanAdjointWrapper": 1,
            }[type(child).__name__]
            assert len(functions[name].blocks[0].operations) == expected
    opaque_types = sorted(
        type(pairs[name]).__name__ for name in terminals if not functions[name].blocks
    )
    assert opaque_types == [
        "QROAMCleanAdjoint",
        "QROM",
        "_ECAddStepFive",
        "_ECAddStepFour",
        "_ECAddStepOne",
        "_ECAddStepSix",
        "_ECAddStepThree",
        "_ECAddStepTwo",
    ]
    with pytest.raises(ValueError, match="component bodies"):
        bind_register_program(record, ((0,), (1,), (2,)))
    denied = {
        name: None
        for name in sys.modules
        if name.split(".", 1)[0] in {"qualtran", "cirq", "kickmix"}
    }
    with patch.dict(sys.modules, denied):
        assert (
            resolve_component_program(library, 0, elliptic_curve_window_factories())
            == record
        )


def test_complete_fixed_window_source_member_inventory_has_recorded_owners() -> None:
    module = import_module(ECWindowAddR.__module__)
    assert module.__file__ is not None
    tree = ast.parse(Path(module.__file__).read_text())
    audit = (
        Path(__file__).parents[2] / "docs/development/ECC_FIXED_WINDOW_FILE_AUDIT.md"
    ).read_text()
    names = []
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            names.append(node.name)
            if isinstance(node, ast.ClassDef):
                names.extend(
                    node.name + "." + m.name
                    for m in node.body
                    if isinstance(m, ast.FunctionDef)
                )
    assert len(names) == 14
    assert all(f"`{name}`" in audit for name in names)


@pytest.mark.parametrize("width,modulus", ((3, 3), (5, 7)))
def test_fixed_width_symbolic_multiplier_keeps_complete_available_window_hierarchy(
    width: int, modulus: int
) -> None:
    from tests.unit.test_ecc_symbolic_multiplication_window import (
        _compare_available_layers,
    )

    point = ECPoint(0, 1, modulus, 1)
    source = ECWindowAddR(width, point, 1, sympy.Symbol("w"))
    encoded = serialize_component_library(_library(source, 0))
    library = deserialize_component_library(encoded)
    assert serialize_component_library(library) == encoded
    record = resolve_component_program(library, 0, elliptic_curve_window_factories())
    finite = resolve_component_program(
        _library(ECWindowAddR(width, point, 1, 1), 0),
        0,
        elliptic_curve_window_factories(),
    )
    _compare_available_layers(record, source, finite)
    assert len(record.functions[0].blocks[0].operations) == 5
    with pytest.raises(ValueError, match="component bodies"):
        bind_register_program(record, ((0,), (1,), (2,)))
    denied = {
        name: None
        for name in sys.modules
        if name.split(".", 1)[0] in {"qualtran", "cirq", "kickmix"}
    }
    with patch.dict(sys.modules, denied):
        assert (
            resolve_component_program(library, 0, elliptic_curve_window_factories())
            == record
        )

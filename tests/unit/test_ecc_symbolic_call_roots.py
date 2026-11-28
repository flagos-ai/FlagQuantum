"""Keep the source six-call root even when its stage bodies genuinely refuse."""

import sys
from importlib import import_module
from unittest.mock import patch

import pytest

from tests._reference_dependencies import unavailable_reference

try:
    import_module("numpy")
    import sympy

    import_module("cirq")
    import_module("kickmix")
    from qualtran import DecomposeTypeError
    from qualtran.bloqs.cryptography.ecc.ec_add import ECAdd
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
from flagquantum.core.ir.program import verify_program_record
from flagquantum.ecosystem._elliptic_curve_addition_library import (
    elliptic_curve_addition_factories,
)
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_ecc_addition_model import frozen_source as frozen_source
from tests.unit.test_legacy_arithmetic_factories import _library

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("symbolic_modulus", (False, True))
@pytest.mark.parametrize("depth", (0, 1))
def test_actual_symbolic_root_transports_six_calls_and_preserves_stage_refusals(
    symbolic_modulus: bool,
    depth: int,
) -> None:
    n, p = sympy.symbols("n p")
    source = ECAdd(n, p if symbolic_modulus else 17)
    library = _library(source, depth)
    encoded = serialize_component_library(library)
    assert (
        serialize_component_library(deserialize_component_library(encoded)) == encoded
    )
    record = resolve_component_program(library, 0, elliptic_curve_addition_factories())
    assert not verify_program_record(record)
    assert len(record.functions[0].blocks[0].operations) == 6
    leaves = {f.name for f in record.functions if not f.blocks}
    pairs = compare_layers(record, source, leaves)
    assert len(pairs) == 7 and len(leaves) == 6
    for name, child in pairs.items():
        if name in leaves:
            with pytest.raises(DecomposeTypeError, match="symbolic `n`"):
                child.decompose_bloq()
    with pytest.raises(ValueError, match="component bodies"):
        bind_register_program(record, tuple((i,) for i in range(5)))
    denied = {
        name: None
        for name in sys.modules
        if name.split(".", 1)[0] in {"qualtran", "cirq", "kickmix"}
    }
    with patch.dict(sys.modules, denied):
        assert (
            resolve_component_program(library, 0, elliptic_curve_addition_factories())
            == record
        )

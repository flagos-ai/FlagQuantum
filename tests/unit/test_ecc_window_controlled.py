"""Window controls retain available parents and actual directional leaf refusals."""

import sys
from functools import cache
from hashlib import sha1
from importlib import import_module
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from tests._reference_dependencies import unavailable_reference

try:
    for dependency in ("numpy", "cirq", "kickmix"):
        import_module(dependency)
    import sympy
    from qualtran import CtrlSpec, DecomposeTypeError
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
from flagquantum.compiler.registers.component_definition_match import (
    matches_component_definition,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.core.value_math import SymbolicInt
from flagquantum.ecosystem._elliptic_curve_control_library import (
    controlled_curve_addition,
    elliptic_curve_control_factories,
)
from flagquantum.errors import SerializationError
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_ecc_addition_model import frozen_source as frozen_source
from tests.unit.test_legacy_arithmetic_factories import _library

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
_N = sympy.Symbol("n")
_PROFILES = ((3, 1, 1), (3, 2, 1), (4, 2, 2), (4, 4, 2), (_N, 2, 1))


@pytest.fixture(scope="module", autouse=True)
def frozen_window_source() -> None:
    from qualtran.bloqs.cryptography.ecc import ec_add_r

    for filename, digest in (
        ("ec_add_r.py", "195654d5b4b269d13d07e96a755424755b0196e5"),
        ("ec_add_r_test.py", "b47105fd91c4a3b15de73a25eb5a35943b40558d"),
    ):
        raw = (Path(ec_add_r.__file__).parent / filename).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == digest


@cache
def source(width: SymbolicInt, address: int, multiplication: int) -> Any:
    return ECWindowAddR(
        width, ECPoint(2, 2, 7, 3), address, multiplication
    ).controlled()


@cache
def resolved(width: SymbolicInt, address: int, multiplication: int) -> ProgramRecord:
    return resolve_component_program(
        _library(source(width, address, multiplication), 0),
        0,
        elliptic_curve_control_factories(),
    )


@pytest.mark.parametrize("width,address,multiplication", _PROFILES)
def test_complete_window_calls_operands_and_actual_refusal_frontier(
    width: SymbolicInt, address: int, multiplication: int
) -> None:
    record = resolved(width, address, multiplication)
    assert verify_program_record(record) == ()
    calls = record.functions[0].blocks[0].operations
    assert len(calls) == 5
    pairs = compare_layers(
        record,
        source(width, address, multiplication),
        {op.callee for op in calls if op.callee is not None},
    )
    functions = {f.name: f for f in record.functions}
    for index in (1, 3):
        name = calls[index].callee
        assert name is not None and not functions[name].blocks
        with pytest.raises(DecomposeTypeError, match="non-thru registers"):
            pairs[name].decompose_bloq()
    addition = calls[2].callee
    assert addition is not None
    assert matches_component_definition(
        record, addition, controlled_curve_addition(width, 7, multiplication)
    )
    # The six-call child is a complete prior certificate, not a guessed matrix.
    assert len(functions[addition].blocks[0].operations) == 6
    if type(width) is int:
        offset = 0
        inputs = []
        for register in record.functions[0].registers or ():
            size = int(register.total_qubits())
            inputs.append(tuple(range(offset, offset + size)))
            offset += size
        with pytest.raises(ValueError, match="component bodies"):
            bind_register_program(
                record, tuple(inputs), clean=tuple(range(offset, offset + 40))
            )


@pytest.mark.parametrize("depth", (0, 1))
@pytest.mark.parametrize("width", (3, _N))
def test_source_transport_and_sdk_free_resolution(
    depth: int, width: SymbolicInt
) -> None:
    if type(width) is not int and depth == 1:
        with pytest.raises(ValueError, match="Object arrays cannot be saved"):
            _library(source(width, 2, 1), depth)
        return
    library = _library(source(width, 2, 1), depth)
    encoded = serialize_component_library(library)
    library = deserialize_component_library(encoded)
    assert serialize_component_library(library) == encoded
    denied = {
        name: None
        for name in sys.modules
        if name.split(".", 1)[0] in {"qualtran", "cirq", "kickmix"}
    }
    with patch.dict(sys.modules, denied):
        record = resolve_component_program(
            library, 0, elliptic_curve_control_factories()
        )
    assert matches_component_definition(
        record, record.functions[0].name, resolved(width, 2, 1)
    )


@pytest.mark.parametrize("width", (3, _N))
def test_negative_window_control_keeps_three_calls_and_exact_child(
    width: SymbolicInt,
) -> None:
    original = ECWindowAddR(width, ECPoint(2, 2, 7, 3), 2).controlled(CtrlSpec(cvs=0))
    record = resolve_component_program(
        _library(original, 0), 0, elliptic_curve_control_factories()
    )
    calls = record.functions[0].blocks[0].operations
    assert len(calls) == 3
    pairs = compare_layers(
        record, original, {op.callee for op in calls if op.callee is not None}
    )
    child = next(
        name for name, bloq in pairs.items() if type(bloq).__name__ == "Controlled"
    )
    assert matches_component_definition(record, child, resolved(width, 2, 1))


def test_actual_point_table_domain_error_is_not_hidden_by_metadata() -> None:
    original = ECWindowAddR(5, ECPoint(1, 3, 15, 1), 2).controlled()
    with pytest.raises(ValueError) as expected:
        original.decompose_bloq()
    with pytest.raises(type(expected.value)) as actual:
        resolve_component_program(
            _library(original, 0), 0, elliptic_curve_control_factories()
        )
    assert isinstance(actual.value, SerializationError)
    assert str(expected.value) in str(actual.value)
    assert isinstance(actual.value.__cause__, ValueError)


def test_symbolic_address_source_refusal_remains_explicit() -> None:
    window = sympy.Symbol("w", positive=True, integer=True)
    original = ECWindowAddR(3, ECPoint(2, 2, 7, 3), window).controlled()
    with pytest.raises(DecomposeTypeError, match="symbolic shape"):
        original.decompose_bloq()
    with pytest.raises(SerializationError, match="Symbolic curve address"):
        resolve_component_program(
            _library(original, 0), 0, elliptic_curve_control_factories()
        )

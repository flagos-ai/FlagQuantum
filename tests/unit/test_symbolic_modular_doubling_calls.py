"""Symbolic doubling keeps all source calls in both exact directions."""

import sys
from dataclasses import replace
from hashlib import sha1
from importlib import import_module
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from tests._reference_dependencies import unavailable_reference

try:
    import_module("numpy")
    import sympy

    import_module("cirq")
    import_module("kickmix")
    from qualtran import (
        DecomposeNotImplementedError,
        DecomposeTypeError,
        QMontgomeryUInt,
        QUInt,
    )
    from qualtran.bloqs.mod_arithmetic.mod_multiplication import ModDbl
except ModuleNotFoundError as error:
    unavailable_reference(error)

from flagquantum.compiler.formats.legacy_component_library import (
    deserialize_component_library,
    serialize_component_library,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_modular_multiplication_factories import (
    modular_multiplication_factories,
)
from flagquantum.compiler.formats.legacy_word_factories import _rename_program
from flagquantum.compiler.formats.montgomery_port_views import restore_montgomery_ports
from flagquantum.compiler.registers.component_definition_match import (
    matches_component_definition,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.registers import RegisterSide
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_ecc_addition_model import frozen_source as frozen_source
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_symbolic_modular_addition_calls import atomic_views

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def source_body_program(record: ProgramRecord, source: Any) -> ProgramRecord:
    """Certify the existing source-codec Cast bridge before traversing its body."""
    root = record.functions[0]
    if not root.blocks or not any(
        "modulus_input_" in (op.callee or "") for op in root.blocks[0].operations
    ):
        return record
    names = tuple(r.name for r in source.signature)
    functions = {f.name: f for f in record.functions}
    children = [
        functions[op.callee]
        for op in root.blocks[0].operations
        if op.callee is not None
        and tuple(r.name for r in functions[op.callee].registers or ()) == names
    ]
    assert len(children) == 1
    child = children[0]
    subtree = replace(
        record,
        functions=(
            child,
            *(f for f in record.functions if f is not child and f is not root),
        ),
    )
    subtree = _rename_program(subtree, "main")
    assert root.registers is not None
    assert matches_component_definition(
        record, root.name, restore_montgomery_ports(subtree, root.registers)
    )
    return subtree


def assert_unbound(record: ProgramRecord) -> None:
    """Missing source leaves must fail binding, independently of input packing."""
    registers = record.functions[0].registers
    assert registers is not None
    offset = 0
    inputs = []
    for r in registers:
        if r.side is RegisterSide.RIGHT:
            continue
        bits = int(r.bitsize)
        inputs.append(tuple(range(offset, offset + bits)))
        offset += bits
    with pytest.raises(ValueError, match="component bodies"):
        bind_register_program(record, tuple(inputs))


@pytest.fixture(scope="module", autouse=True)
def frozen_doubling_source() -> None:
    module = import_module("qualtran.bloqs.mod_arithmetic.mod_multiplication")
    assert module.__file__ is not None
    raw = Path(module.__file__).read_bytes()
    assert (
        sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == "c0242f9735bf269564c9480c14c008d5f9621fd7"
    )


@pytest.mark.parametrize("width", (3, 5))
@pytest.mark.parametrize("montgomery", (False, True))
@pytest.mark.parametrize("inverse", (False, True))
@pytest.mark.parametrize("depth", (0, 1))
def test_all_fifteen_source_calls_and_true_constant_refusals(
    width: int, montgomery: bool, inverse: bool, depth: int
) -> None:
    p = sympy.Symbol("p")
    source = ModDbl(QMontgomeryUInt(width) if montgomery else QUInt(width), p)
    source = source.adjoint() if inverse else source
    library = _library(source, depth)
    encoded = serialize_component_library(library)
    library = deserialize_component_library(encoded)
    assert serialize_component_library(library) == encoded
    factories = modular_multiplication_factories()
    record = resolve_component_program(library, 0, factories)
    comparison = source_body_program(record, source)
    assert len(comparison.functions[0].blocks[0].operations) == 15
    terminals = atomic_views(comparison)
    pairs = compare_layers(comparison, source, terminals)
    functions = {f.name: f for f in comparison.functions}
    unresolved = []
    for name, child in pairs.items():
        if name not in terminals:
            continue
        with pytest.raises((DecomposeTypeError, DecomposeNotImplementedError)):
            child.decompose_bloq()
        if not functions[name].blocks:
            unresolved.append(child)
            while hasattr(child, "subbloq"):
                child = child.subbloq
            assert type(child).__name__ == "AddK" and child.k in (-p, p)
    assert len(unresolved) == 2
    assert_unbound(record)
    denied = {
        name: None
        for name in sys.modules
        if name.split(".", 1)[0] in {"qualtran", "cirq", "kickmix"}
    }
    with patch.dict(sys.modules, denied):
        assert (
            resolve_component_program(library, 0, modular_multiplication_factories())
            == record
        )

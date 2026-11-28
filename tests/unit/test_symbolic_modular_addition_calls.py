"""Unbound modular constants retain every available source arithmetic layer."""

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
    from qualtran.bloqs.mod_arithmetic.mod_addition import CModAdd, ModAdd
    from qualtran.bloqs.mod_arithmetic.mod_subtraction import CModSub, ModSub
except ModuleNotFoundError as error:
    unavailable_reference(error)

from flagquantum.compiler.formats.legacy_component_library import (
    deserialize_component_library,
    serialize_component_library,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_modular_addition_factories import (
    modular_addition_factories,
)
from flagquantum.compiler.formats.legacy_modular_subtraction_factories import (
    modular_subtraction_factories,
)
from flagquantum.compiler.formats.legacy_word_factories import _rename_program
from flagquantum.compiler.formats.montgomery_port_views import restore_montgomery_ports
from flagquantum.compiler.registers.component_definition_match import (
    matches_component_definition,
)
from flagquantum.compiler.registers.control.component_predicate import (
    control_predicate_program,
)
from flagquantum.compiler.registers.control.conjunction_program import (
    and_component_program,
    conjunction_program,
)
from flagquantum.compiler.registers.control.control_predicate import ControlRegister
from flagquantum.compiler.registers.layout.component_partition import (
    partition_component_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.registers import Register, RegisterSide
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_ecc_addition_model import frozen_source as frozen_source
from tests.unit.test_legacy_arithmetic_factories import _library

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def atomic_views(record: ProgramRecord) -> set[str]:
    """Identify candidates, then require actual source refusal and body proof."""
    terminals = set()
    for function in record.functions:
        registers = function.registers or ()
        if (
            not function.blocks
            or any(op.opcode != "call" for b in function.blocks for op in b.operations)
            or (
                len(registers) == 2
                and registers[0].name == "ctrl"
                and registers[0].shape_symbolic == (2,)
                and registers[1].name == "target"
                and registers[1].side is not RegisterSide.THRU
            )
            or (
                registers
                and registers[0].name == "x"
                and registers[0].side is not RegisterSide.THRU
            )
        ):
            terminals.add(function.name)
    return terminals


@pytest.fixture(scope="module", autouse=True)
def frozen_modular_sources() -> None:
    for module, digest in (
        (
            "qualtran.bloqs.mod_arithmetic.mod_addition",
            "0a167e84fe00d5df50488c4b28925ea6114728f7",
        ),
        (
            "qualtran.bloqs.mod_arithmetic.mod_subtraction",
            "e0d0fbf81a81ec2ed55e02b4b31ad3e75ab85324",
        ),
    ):
        source = import_module(module)
        assert source.__file__ is not None
        raw = Path(source.__file__).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == digest


@pytest.mark.parametrize("width", (3, 5))
@pytest.mark.parametrize("depth", (0, 1))
@pytest.mark.parametrize(
    "kind,montgomery,cv",
    (
        ("ModAdd", True, 1),
        ("CModAdd", False, 0),
        ("CModAdd", False, 1),
        ("CModAdd", True, 0),
        ("CModAdd", True, 1),
        ("ModSub", False, 1),
        ("ModSub", True, 1),
        ("CModSub", False, 0),
        ("CModSub", False, 1),
        ("CModSub", True, 0),
        ("CModSub", True, 1),
    ),
)
def test_complete_symbolic_source_closure_and_reference_free_reconstruction(
    width: int, depth: int, kind: str, montgomery: bool, cv: int
) -> None:
    modulus = sympy.Symbol("p")
    dtype = QMontgomeryUInt(width) if montgomery else QUInt(width)
    source: Any = (
        ModAdd(width, modulus)
        if kind == "ModAdd"
        else (
            CModAdd(dtype, modulus, cv)
            if kind == "CModAdd"
            else (
                ModSub(dtype, modulus)
                if kind == "ModSub"
                else CModSub(dtype, modulus, cv)
            )
        )
    )
    factories = modular_addition_factories() | modular_subtraction_factories()
    library = _library(source, depth)
    encoded = serialize_component_library(library)
    library = deserialize_component_library(encoded)
    assert serialize_component_library(library) == encoded
    record = resolve_component_program(library, 0, factories)
    comparison = record
    if kind == "ModAdd":
        # The frozen codec omits Montgomery's modulus in signature records.
        # Certify every explicit Cast in its existing five-call bridge before
        # traversing the actual seventeen-call arithmetic owner.
        root = record.functions[0]
        assert root.registers is not None
        assert len(root.blocks[0].operations) == 5
        entry = root.blocks[0].operations[2].callee
        inner = next(f for f in record.functions if f.name == entry)
        subtree = replace(
            record,
            functions=(
                inner,
                *(f for f in record.functions if f is not inner and f is not root),
            ),
        )
        comparison = _rename_program(subtree, "main")
        assert matches_component_definition(
            record, root.name, restore_montgomery_ports(comparison, root.registers)
        )
    root_calls = 17 if kind == "ModAdd" else 14 if kind == "CModAdd" else 5
    assert len(comparison.functions[0].blocks[0].operations) == root_calls
    terminals = atomic_views(comparison)
    pairs = compare_layers(comparison, source, terminals)
    assert len(pairs) > root_calls
    functions = {f.name: f for f in comparison.functions}
    unresolved: list[Any] = []
    for name, child in pairs.items():
        if name not in terminals:
            continue
        with pytest.raises((DecomposeTypeError, DecomposeNotImplementedError)):
            child.decompose_bloq()
        if type(child).__name__ == "And":
            registers = functions[name].registers
            assert registers is not None
            values = (int(child.cv1), int(child.cv2))
            assert any(
                matches_component_definition(comparison, name, template)
                for template in (
                    and_component_program(values, name=name, uncompute=child.uncompute),
                    conjunction_program(values, name=name, uncompute=child.uncompute),
                    control_predicate_program(
                        registers[:1],
                        (ControlRegister(1, values),),
                        name=name,
                        uncompute=child.uncompute,
                    ),
                )
            ), (name, child, registers)
        elif type(child).__name__ == "Partition":
            registers = functions[name].registers
            assert registers is not None
            parts = tuple(
                Register(r.name, r.dtype, r.shape_symbolic) for r in registers[1:]
            )
            assert matches_component_definition(
                comparison,
                name,
                partition_component_program(
                    parts, name, registers[0].side is RegisterSide.RIGHT
                ),
            )
        if not functions[name].blocks:
            unresolved.append(child)
            leaf = child.subbloq if hasattr(child, "subbloq") else child
            assert type(leaf).__name__ == "AddK" and leaf.k.free_symbols == {modulus}
    assert len(unresolved) == (2 if kind in {"ModAdd", "CModAdd"} else 4)
    registers = record.functions[0].registers
    assert registers is not None
    inputs = []
    offset = 0
    for register in registers:
        if register.side is RegisterSide.RIGHT:
            continue
        bits = int(register.bitsize)
        inputs.append(tuple(range(offset, offset + bits)))
        offset += bits
    with pytest.raises(ValueError, match="component bodies"):
        bind_register_program(record, tuple(inputs))
    denied = {
        name: None
        for name in sys.modules
        if name.split(".", 1)[0] in {"qualtran", "cirq", "kickmix"}
    }
    with patch.dict(sys.modules, denied):
        assert resolve_component_program(library, 0, factories) == record

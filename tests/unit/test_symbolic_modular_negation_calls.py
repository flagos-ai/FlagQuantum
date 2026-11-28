"""Compare available symbolic negation and ECC flag calls, not opaque prices."""

import sys
from collections.abc import Mapping
from hashlib import sha1
from importlib import import_module
from itertools import product
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
    from qualtran import DecomposeTypeError, QMontgomeryUInt, QUInt
    from qualtran.bloqs.mcmt import And
    from qualtran.bloqs.mod_arithmetic import mod_subtraction
except ModuleNotFoundError as error:
    unavailable_reference(error)

from flagquantum.compiler.formats.legacy_component_library import (
    deserialize_component_library,
    serialize_component_library,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    ComponentFactory,
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_modular_subtraction_factories import (
    modular_subtraction_factories,
)
from flagquantum.compiler.registers.component_definition_match import (
    matches_component_definition,
)
from flagquantum.compiler.registers.control.conjunction_program import (
    and_component_program,
)
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


def _terminals(record: ProgramRecord) -> set[str]:
    """Certify native typed views of the source atomic And and Partition leaves.

    Directional roots and predicates must still be traversed. Direction alone
    does not establish an atomic boundary.
    """
    terminals: set[str] = set()
    for function in record.functions:
        if not function.blocks or any(
            operation.opcode != "call"
            for block in function.blocks
            for operation in block.operations
        ):
            terminals.add(function.name)
            continue
        registers = function.registers or ()
        if (
            len(registers) == 2
            and registers[0].name == "ctrl"
            and registers[0].shape_symbolic == (2,)
            and registers[1].name == "target"
            and registers[1].side is not RegisterSide.THRU
        ):
            assert any(
                matches_component_definition(
                    record,
                    function.name,
                    and_component_program(
                        (values[0], values[1]),
                        name=function.name,
                        uncompute=registers[1].side is RegisterSide.LEFT,
                    ),
                )
                for values in product((0, 1), repeat=2)
            )
            terminals.add(function.name)
        elif (
            registers
            and registers[0].name == "x"
            and registers[0].side is not RegisterSide.THRU
        ):
            parts = tuple(
                Register(r.name, r.dtype, r.shape_symbolic) for r in registers[1:]
            )
            assert matches_component_definition(
                record,
                function.name,
                partition_component_program(
                    parts, function.name, registers[0].side is RegisterSide.RIGHT
                ),
            )
            terminals.add(function.name)
    return terminals


def assert_symbolic_negation_dependencies(
    source: Any,
    factories: Mapping[str, ComponentFactory],
    *,
    width: int,
    depth: int,
    root_calls: int,
    input_words: int,
) -> None:
    """Compare every available layer and prove all stopped source refusals."""
    assert mod_subtraction.__file__ is not None
    raw = Path(mod_subtraction.__file__).read_bytes()
    assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == (
        "e0d0fbf81a81ec2ed55e02b4b31ad3e75ab85324"
    )
    library = _library(source, depth)
    encoded = serialize_component_library(library)
    library = deserialize_component_library(encoded)
    assert serialize_component_library(library) == encoded
    record = resolve_component_program(library, 0, factories)
    assert len(record.functions[0].blocks[0].operations) == root_calls
    terminals = _terminals(record)
    pairs: dict[str, Any] = compare_layers(record, source, terminals)
    assert len(pairs) > root_calls
    functions = {f.name: f for f in record.functions}
    opaque = []
    for name, child in pairs.items():
        if name not in terminals:
            continue
        # Every boundary really is source-atomic or source-refused. A native
        # header must never hide an available source decomposition.
        with pytest.raises((DecomposeTypeError, NotImplementedError)):
            child.decompose_bloq()
        if isinstance(child, And):
            assert matches_component_definition(
                record,
                name,
                and_component_program(
                    (int(child.cv1), int(child.cv2)),
                    name=name,
                    uncompute=child.uncompute,
                ),
            )
        if not functions[name].blocks:
            opaque.append(child)
            assert type(child).__name__ == "Controlled"
            assert child.subbloq == mod_subtraction.AddK(
                (
                    source.dtype
                    if isinstance(source, mod_subtraction.CModNeg)
                    else QUInt(width)
                ),
                sympy.Symbol("p") + 1,
            )
    assert len(opaque) == 1  # Repeated calls share the same source constant child.
    registers = record.functions[0].registers
    assert registers is not None
    lefts = tuple(r for r in registers if r.side is not RegisterSide.RIGHT)
    assert len(lefts) == input_words
    inputs = []
    offset = 0
    for register in lefts:
        size = int(register.bitsize)
        inputs.append(tuple(range(offset, offset + size)))
        offset += size
    with pytest.raises(ValueError, match="component bodies"):
        bind_register_program(record, tuple(inputs))
    denied = {
        name: None
        for name in sys.modules
        if name.split(".", 1)[0] in {"qualtran", "cirq", "kickmix"}
    }
    with patch.dict(sys.modules, denied):
        assert resolve_component_program(library, 0, factories) == record


@pytest.mark.parametrize("width", (3, 5))
@pytest.mark.parametrize("montgomery", (False, True))
@pytest.mark.parametrize("depth", (0, 1))
def test_all_available_symbolic_calls_operands_and_true_terminal_refusals(
    width: int, montgomery: bool, depth: int
) -> None:
    dtype = QMontgomeryUInt(width) if montgomery else QUInt(width)
    assert_symbolic_negation_dependencies(
        mod_subtraction.ModNeg(dtype, sympy.Symbol("p")),
        modular_subtraction_factories(),
        width=width,
        depth=depth,
        root_calls=8,
        input_words=1,
    )


@pytest.mark.parametrize("width", (3, 5))
@pytest.mark.parametrize("montgomery", (False, True))
@pytest.mark.parametrize("depth", (0, 1))
@pytest.mark.parametrize("control_value", (0, 1))
def test_controlled_symbolic_negation_retains_every_available_source_layer(
    width: int, montgomery: bool, depth: int, control_value: int
) -> None:
    dtype = QMontgomeryUInt(width) if montgomery else QUInt(width)
    assert_symbolic_negation_dependencies(
        mod_subtraction.CModNeg(dtype, sympy.Symbol("p"), control_value),
        modular_subtraction_factories(),
        width=width,
        depth=depth,
        root_calls=14,
        input_words=2,
    )

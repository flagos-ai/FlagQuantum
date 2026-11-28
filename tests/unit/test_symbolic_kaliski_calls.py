"""Kaliski symbolic modulus retains available parents around true source refusals."""

import sys
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
    from qualtran import DecomposeNotImplementedError, DecomposeTypeError
    from qualtran.bloqs.mod_arithmetic import mod_division
except ModuleNotFoundError as error:
    unavailable_reference(error)

from flagquantum.compiler.formats.legacy_component_library import (
    deserialize_component_library,
    serialize_component_library,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_modular_inverse_factories import (
    _SIMPLE_ADJOINT_BUILDERS,
    modular_inverse_factories,
)
from flagquantum.compiler.registers.comparison.half_component import (
    half_greater_program,
)
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
from flagquantum.core.ir.program.register_types import montgomery_register_type
from flagquantum.core.ir.program.registers import Register, RegisterSide
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_ecc_addition_model import frozen_source as frozen_source
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_symbolic_modular_addition_calls import atomic_views
from tests.unit.test_symbolic_modular_doubling_calls import (
    assert_unbound,
    source_body_program,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_inverse_source() -> None:
    assert mod_division.__file__ is not None
    raw = Path(mod_division.__file__).read_bytes()
    assert (
        sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == "947f40eb5d17101ad91ff4d4432ac5287f867bd5"
    )


@pytest.mark.parametrize("width", (3, 4))
@pytest.mark.parametrize(
    "kind", ("_KaliskiIteration", "_KaliskiModInverseImpl", "KaliskiModInverse")
)
@pytest.mark.parametrize("inverse", (False, True))
@pytest.mark.parametrize("depth", (0, 1))
def test_complete_available_symbolic_kaliski_layers_and_actual_refusals(
    width: int, kind: str, inverse: bool, depth: int
) -> None:
    p = sympy.Symbol("p")
    source: Any = getattr(mod_division, kind)(width, p)
    source = source.adjoint() if inverse else source
    library = _library(source, depth)
    encoded = serialize_component_library(library)
    library = deserialize_component_library(encoded)
    assert serialize_component_library(library) == encoded
    record = resolve_component_program(library, 0, modular_inverse_factories())
    if depth:
        # Source serialization can renumber helper definitions. Reuse traversal
        # only after the complete reachable closure matches, including sharing,
        # every operand and every field except names/unreachable definitions.
        assert matches_component_definition(
            record,
            "main",
            resolve_component_program(
                _library(source, 0), 0, modular_inverse_factories()
            ),
        )
        assert_unbound(record)
        denied = {
            name: None
            for name in sys.modules
            if name.split(".", 1)[0] in {"qualtran", "cirq", "kickmix"}
        }
        with patch.dict(sys.modules, denied):
            assert (
                resolve_component_program(library, 0, modular_inverse_factories())
                == record
            )
        return
    comparison = source_body_program(record, source)
    calls = (
        10
        if kind == "_KaliskiIteration"
        else 2 * width + (14 if kind == "_KaliskiModInverseImpl" else 27)
    )
    assert len(comparison.functions[0].blocks[0].operations) == calls
    terminals = atomic_views(comparison)
    # Reuse the entire accepted half-comparator, including its exact inverse.
    # Its independent input Split schedule differs from source creation order;
    # the complete stored definition and existing primitive/phase proofs apply.
    reused = {}
    for direction in (False, True):
        template = half_greater_program(
            montgomery_register_type(width), adjoint=direction
        )
        for function in comparison.functions:
            if function.registers == template.functions[0].registers:
                assert matches_component_definition(comparison, function.name, template)
                reused[function.name] = direction
                terminals.add(function.name)
    reused_steps: dict[str, Any] = {}
    if inverse:
        # Source generic adjoints can schedule independent Split/CNOT calls
        # differently. Reuse the complete already-accepted inverse step owners,
        # not an opaque signature or a claim that the source refuses them.
        for step, builder in _SIMPLE_ADJOINT_BUILDERS.items():
            template = builder(width)
            for function in comparison.functions:
                if function.registers == template.functions[
                    0
                ].registers and matches_component_definition(
                    comparison, function.name, template
                ):
                    reused_steps[function.name] = getattr(mod_division, step)(
                        width
                    ).adjoint()
                    terminals.add(function.name)
    pairs = compare_layers(comparison, source, terminals)
    assert len(pairs) > calls
    functions = {f.name: f for f in comparison.functions}
    stopped = []
    for name, child in pairs.items():
        if name not in terminals:
            continue
        if name in reused_steps:
            assert child == reused_steps[name]
            assert len(child.decompose_bloq().bloq_instances) > 0
            continue
        if name in reused:
            assert len(child.decompose_bloq().bloq_instances) > 0
            assert type(child).__name__ == "LinearDepthHalfGreaterThan"
            assert child.uncompute == reused[name]
            assert child.dtype.bitsize == width and child.dtype.modulus is None
            continue
        with pytest.raises((DecomposeTypeError, DecomposeNotImplementedError)):
            child.decompose_bloq()
        if type(child).__name__ == "And":
            registers = functions[name].registers
            assert registers is not None
            values = int(child.cv1), int(child.cv2)
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
            ), (name, child)
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
            stopped.append(child)
            while hasattr(child, "subbloq"):
                child = child.subbloq
            assert type(child).__name__ in {"AddK", "XorK", "_KaliskiIterationStep6"}
            if hasattr(child, "k"):
                assert child.k.free_symbols == {p}
    assert len(stopped) == (1 if kind == "_KaliskiIteration" else 5)
    assert_unbound(record)
    denied = {
        name: None
        for name in sys.modules
        if name.split(".", 1)[0] in {"qualtran", "cirq", "kickmix"}
    }
    with patch.dict(sys.modules, denied):
        assert (
            resolve_component_program(library, 0, modular_inverse_factories()) == record
        )

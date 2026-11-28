"""Available controlled ECC parents preserve exact arithmetic and refusal frontiers."""

from functools import cache
from importlib import import_module
from typing import Any, Literal

import pytest

from tests._reference_dependencies import unavailable_reference

try:
    for dependency in ("numpy", "cirq", "kickmix"):
        import_module(dependency)
    import sympy
    from qualtran import CtrlSpec, DecomposeTypeError
    from qualtran.bloqs.cryptography.ecc import ec_add
except ModuleNotFoundError as error:
    unavailable_reference(error)

from flagquantum.compiler.formats.legacy_component_library import (
    deserialize_component_library,
    serialize_component_library,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.registers.arithmetic.constant_addition_component import (
    constant_addition_program,
)
from flagquantum.compiler.registers.component_definition_match import (
    matches_component_definition,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.value_math import SymbolicInt
from flagquantum.ecosystem._elliptic_curve_control_library import (
    controlled_curve_addition,
    controlled_curve_arithmetic_stage,
    elliptic_curve_control_factories,
)
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_ecc_addition_model import frozen_source as frozen_source
from tests.unit.test_ecc_symbolic_multiplication_window import _reachable
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_nested_controlled_constant_component import base_for_always
from tests.unit.test_symbolic_modular_addition_calls import atomic_views

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
_PROFILES = (
    (3, 3, 1),
    (3, 3, 3),
    (6, 13, 2),
    (3, 3, sympy.Symbol("w", positive=True, integer=True)),
    (3, sympy.Symbol("p"), 1),
)


@cache
def stage(
    kind: Literal["coordinates", "products"],
    width: int,
    modulus: SymbolicInt,
    window: SymbolicInt,
) -> ProgramRecord:
    return controlled_curve_arithmetic_stage(kind, width, modulus, window)


def available_layers(record: ProgramRecord, source: Any) -> dict[str, Any]:
    record = _reachable(record, record.functions[0].name)
    reused = set()
    for function in record.functions:
        if "constant_always" in function.name:
            assert function.registers is not None
            dtype = function.registers[0].dtype
            assert any(
                matches_component_definition(
                    record,
                    function.name,
                    base_for_always(
                        constant_addition_program(dtype, 0, adjoint=inverse)
                    ),
                )
                for inverse in (False, True)
            )
            reused.add(function.name)
    pairs = compare_layers(record, source, atomic_views(record) | reused)
    functions = {function.name: function for function in record.functions}
    for name in reused & pairs.keys():
        assert type(pairs[name]).__name__ == "Always"
    for name, child in pairs.items():
        if not functions[name].blocks:
            assert type(child).__name__ == "Controlled"
            with pytest.raises(
                DecomposeTypeError, match="non-thru registers|symbolic AddK"
            ):
                child.decompose_bloq()
    return pairs


@pytest.mark.parametrize("kind", ("coordinates", "products"))
@pytest.mark.parametrize("width,modulus,window", _PROFILES)
def test_every_available_stage_layer_and_actual_directional_refusal(
    kind: Literal["coordinates", "products"],
    width: int,
    modulus: SymbolicInt,
    window: SymbolicInt,
) -> None:
    record = stage(kind, width, modulus, window)
    source = (
        ec_add._ECAddStepThree if kind == "coordinates" else ec_add._ECAddStepFour
    )(width, modulus, window).controlled()
    calls = record.functions[0].blocks[0].operations
    assert len(calls) == (2 * width + 18 if kind == "coordinates" else 3 * width + 19)
    pairs = available_layers(record, source)
    multiplies = [
        child
        for child in pairs.values()
        if type(child).__name__ == "Controlled"
        and type(child.subbloq).__name__ == "DirtyOutOfPlaceMontgomeryModMul"
    ]
    assert len(multiplies) == 2
    for child in multiplies:
        assert child.subbloq.window_size == window and child.subbloq.mod == modulus
        assert (
            child.signature.get_right("qrom_indices").bitsize
            == ((width + window - 1) // window) * window
            if not child.subbloq.uncompute
            else child.signature.get_left("qrom_indices").bitsize
            == ((width + window - 1) // window) * window
        )
    offset = 0
    inputs = []
    for register in record.functions[0].registers or ():
        size = int(register.bitsize)
        inputs.append(tuple(range(offset, offset + size)))
        offset += size
    with pytest.raises(ValueError, match="component bodies"):
        bind_register_program(
            record, tuple(inputs), clean=tuple(range(offset, offset + 30))
        )


@pytest.mark.parametrize(
    "window", (1, 3, sympy.Symbol("w", positive=True, integer=True))
)
def test_controlled_root_six_calls_and_complete_stage_reuse(
    window: SymbolicInt,
) -> None:
    record = controlled_curve_addition(3, 3, window)
    source = ec_add.ECAdd(3, 3, window).controlled()
    calls = record.functions[0].blocks[0].operations
    assert len(calls) == 6
    pairs = compare_layers(
        record, source, {op.callee for op in calls if op.callee is not None}
    )
    functions = {function.name: function for function in record.functions}
    for i, operation in enumerate(calls):
        assert operation.callee is not None
        if i in (2, 3):
            assert matches_component_definition(
                record,
                operation.callee,
                stage("coordinates" if i == 2 else "products", 3, 3, window),
            )
        else:
            assert not functions[operation.callee].blocks
            with pytest.raises(DecomposeTypeError, match="non-thru registers"):
                pairs[operation.callee].decompose_bloq()


@pytest.mark.parametrize("root", (False, True))
@pytest.mark.parametrize("depth", (0, 1))
def test_constructor_control_roundtrip_and_sdk_free_resolution(
    root: bool, depth: int
) -> None:
    import sys
    from unittest.mock import patch

    source = (
        ec_add.ECAdd(3, 3, 1) if root else ec_add._ECAddStepThree(3, 3, 3)
    ).controlled()
    library = _library(source, depth)
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
    expected = (
        controlled_curve_addition(3, 3, 1) if root else stage("coordinates", 3, 3, 3)
    )
    assert matches_component_definition(record, record.functions[0].name, expected)


def test_negative_root_wrapper_retains_three_original_calls() -> None:
    source = ec_add.ECAdd(3, 3, 1).controlled(CtrlSpec(cvs=0))
    record = resolve_component_program(
        _library(source, 0), 0, elliptic_curve_control_factories()
    )
    calls = record.functions[0].blocks[0].operations
    assert len(calls) == 3
    compare_layers(record, source, {op.callee for op in calls if op.callee is not None})


@pytest.mark.parametrize("kind", ("coordinates", "products"))
@pytest.mark.parametrize(
    "width,modulus,window", ((3, 3, 2), (3, 5, 1), (3, 2, 1), (3, 1, 1), (3, 3, 0))
)
def test_original_dirty_multiplier_constructor_failures_are_preserved(
    kind: Literal["coordinates", "products"], width: int, modulus: int, window: int
) -> None:
    source = (
        ec_add._ECAddStepThree if kind == "coordinates" else ec_add._ECAddStepFour
    )(width, modulus, window).controlled()
    with pytest.raises((AssertionError, ZeroDivisionError)) as expected:
        source.decompose_bloq()
    with pytest.raises(type(expected.value)) as actual:
        controlled_curve_arithmetic_stage(kind, width, modulus, window)
    assert str(actual.value) == str(expected.value)


@pytest.mark.parametrize("width,modulus,window", _PROFILES)
@pytest.mark.parametrize("adjoint", (False, True))
def test_shared_history_signature_is_original_constructor_metadata(
    width: int, modulus: int, window: SymbolicInt, adjoint: bool
) -> None:
    from qualtran.bloqs.mod_arithmetic.mod_multiplication import (
        DirtyOutOfPlaceMontgomeryModMul,
    )

    from flagquantum.compiler.registers.modular.montgomery_signature import (
        montgomery_product_signature,
    )
    from tests.unit.test_component_graph import _source

    signature = montgomery_product_signature(width, window, modulus, adjoint=adjoint)
    original = DirtyOutOfPlaceMontgomeryModMul(
        width, window, modulus, uncompute=adjoint
    )
    assert _source(signature).signature(signature) == original.signature


@pytest.mark.parametrize(
    "width,modulus,window",
    (
        (3, 5, 1),
        (3, 3, 2),
        (3, 3, 0),
        (sympy.Symbol("n", positive=True, integer=True), 3, 1),
        (sympy.Integer(3), 3, 1),
        (sympy.Symbol("n", positive=True, integer=True), sympy.Symbol("p"), 1),
    ),
)
def test_available_six_call_parent_retained_when_actual_children_fail(
    width: SymbolicInt, modulus: SymbolicInt, window: SymbolicInt
) -> None:
    source = ec_add.ECAdd(width, modulus, window).controlled()
    record = controlled_curve_addition(width, modulus, window)
    calls = record.functions[0].blocks[0].operations
    assert len(calls) == 6
    pairs = compare_layers(
        record, source, {op.callee for op in calls if op.callee is not None}
    )
    functions = {function.name: function for function in record.functions}
    for operation in calls:
        assert operation.callee is not None
        assert not functions[operation.callee].blocks
        with pytest.raises((DecomposeTypeError, AssertionError, ZeroDivisionError)):
            pairs[operation.callee].decompose_bloq()


def test_available_symbolic_modulus_control_is_not_mislabeled_as_source_refusal() -> (
    None
):
    p = sympy.Symbol("p", positive=True, integer=True)
    source = ec_add._ECAddStepThree(3, p, 1).controlled()
    assert len(source.decompose_bloq().bloq_instances) == 24
    record = controlled_curve_addition(3, p, 1)
    calls = record.functions[0].blocks[0].operations
    assert len(calls) == 6
    compare_layers(
        record,
        ec_add.ECAdd(3, p, 1).controlled(),
        {op.callee for op in calls if op.callee is not None},
    )
    for index, kind in ((2, "coordinates"), (3, "products")):
        callee = calls[index].callee
        assert callee is not None
        assert matches_component_definition(record, callee, stage(kind, 3, p, 1))

"""Additional modular controls retain source children, inverse dispatch and refusals."""

from functools import cache
from importlib import import_module
from typing import Any

import pytest

from tests._reference_dependencies import unavailable_reference

try:
    for dependency in ("numpy", "cirq", "kickmix"):
        import_module(dependency)
    from qualtran import DecomposeTypeError, QMontgomeryUInt, QUInt
    from qualtran.bloqs.mod_arithmetic.mod_addition import CModAdd, ModAdd
    from qualtran.bloqs.mod_arithmetic.mod_subtraction import CModSub, ModSub
except ModuleNotFoundError as error:
    unavailable_reference(error)

from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_modular_control_factories import (
    modular_control_factories,
)
from flagquantum.compiler.registers.arithmetic.constant_addition_component import (
    constant_addition_program,
)
from flagquantum.compiler.registers.component_definition_match import (
    matches_component_definition,
)
from flagquantum.compiler.registers.modular.additional_control_component import (
    additional_modular_addition_program,
    additional_modular_subtraction_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.register_types import (
    montgomery_register_type,
    unsigned_register_type,
)
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_ecc_addition_model import frozen_source as frozen_source
from tests.unit.test_ecc_symbolic_multiplication_window import _reachable
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_nested_controlled_constant_component import base_for_always
from tests.unit.test_symbolic_modular_addition_calls import (
    atomic_views,
)
from tests.unit.test_symbolic_modular_addition_calls import (
    frozen_modular_sources as frozen_modular_sources,
)
from tests.unit.test_symbolic_modular_doubling_calls import source_body_program

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@cache
def case(
    subtract: bool, control_value: int | None, adjoint: bool, montgomery: bool
) -> tuple[ProgramRecord, Any]:
    dtype = montgomery_register_type(3, 3) if montgomery else unsigned_register_type(3)
    source_dtype = QMontgomeryUInt(3, 3) if montgomery else QUInt(3)
    if subtract:
        source = (
            ModSub(source_dtype, 3)
            if control_value is None
            else CModSub(source_dtype, 3, cv=control_value)
        )
        record = additional_modular_subtraction_program(
            dtype, 3, control_value=control_value, adjoint=adjoint
        )
    else:
        source = (
            ModAdd(3, 3)
            if control_value is None
            else CModAdd(source_dtype, 3, cv=control_value)
        )
        record = additional_modular_addition_program(
            montgomery_register_type(3, 3) if control_value is None else dtype,
            3,
            control_value=control_value,
            adjoint=adjoint,
        )
    return record, (source.adjoint() if adjoint else source).controlled()


@pytest.mark.parametrize("subtract", (False, True))
@pytest.mark.parametrize("control_value", (None, 0, 1))
@pytest.mark.parametrize("adjoint", (False, True))
@pytest.mark.parametrize("montgomery", (False, True))
def test_complete_available_call_layers_and_actual_refused_leaves(
    subtract: bool, control_value: int | None, adjoint: bool, montgomery: bool
) -> None:
    record, source = case(subtract, control_value, adjoint, montgomery)
    record = _reachable(record, record.functions[0].name)
    calls = record.functions[0].blocks[0].operations
    assert len(calls) == (5 if subtract else 17 if control_value is None else 14)
    terminals = atomic_views(record)
    reused = set()
    for function in record.functions:
        if "constant_always" not in function.name:
            continue
        assert function.registers is not None
        word = function.registers[0].dtype
        assert any(
            matches_component_definition(
                record,
                function.name,
                base_for_always(constant_addition_program(word, 0, adjoint=inverse)),
            )
            for inverse in (False, True)
        )
        reused.add(function.name)
    pairs = compare_layers(record, source, terminals | reused)
    for name in reused & pairs.keys():
        assert type(pairs[name]).__name__ == "Always"
    functions = {function.name: function for function in record.functions}
    declarations = [name for name in pairs if not functions[name].blocks]
    assert declarations
    for name in declarations:
        with pytest.raises(DecomposeTypeError, match="non-thru registers"):
            pairs[name].decompose_bloq()
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


@pytest.mark.parametrize("subtract", (False, True))
@pytest.mark.parametrize("control_value", (None, 0, 1))
@pytest.mark.parametrize("adjoint", (False, True))
@pytest.mark.parametrize("depth", (0, 1))
def test_source_constructor_roundtrip_and_sdk_free_resolution(
    subtract: bool, control_value: int | None, adjoint: bool, depth: int
) -> None:
    import sys
    from unittest.mock import patch

    from flagquantum.compiler.formats.legacy_component_library import (
        deserialize_component_library,
        serialize_component_library,
    )

    expected, source = case(subtract, control_value, adjoint, False)
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
        record = resolve_component_program(library, 0, modular_control_factories())
    record = source_body_program(record, source)
    assert matches_component_definition(record, record.functions[0].name, expected)


@pytest.mark.parametrize("subtract", (False, True))
@pytest.mark.parametrize("inner", (False, True))
def test_negative_added_control_wrapper(subtract: bool, inner: bool) -> None:
    from qualtran import CtrlSpec

    _, source = case(subtract, 0 if inner else None, False, False)
    source = source.subbloq.controlled(CtrlSpec(cvs=0))
    record = resolve_component_program(
        _library(source, 0), 0, modular_control_factories()
    )
    record = source_body_program(record, source)
    calls = record.functions[0].blocks[0].operations
    assert len(calls) == 3
    compare_layers(record, source, {op.callee for op in calls if op.callee is not None})


@pytest.mark.parametrize("control_value", (None, 0, 1))
def test_shared_subtraction_exact_inverse_primitives_complex_residues_and_work(
    control_value: int | None,
) -> None:
    from collections import Counter

    import numpy as np

    from flagquantum.compiler.registers.modular.controlled_modular_subtraction_component import (
        controlled_modular_subtraction_program,
    )
    from flagquantum.compiler.registers.modular.modular_subtraction_component import (
        modular_subtraction_program,
    )
    from tests.unit.measurement_checks import certified_and_channel_map
    from tests.unit.sparse_register_execution import execute_sparse_register
    from tests.unit.test_modular_subtraction_component import source_stream
    from tests.unit.test_register_subtraction import assert_source

    dtype = unsigned_register_type(3)
    ports: dict[str, tuple[int, ...]]
    words: list[tuple[int, ...]]
    if control_value is None:
        record = modular_subtraction_program(dtype, 3, adjoint=True)
        source = ModSub(QUInt(3), 3).adjoint()
        ports = {"x": (0, 1, 2), "y": (3, 4, 5)}
        words = [(x, y) for x in range(3) for y in range(3)]
    else:
        record = controlled_modular_subtraction_program(
            dtype, 3, control_value=control_value, adjoint=True
        )
        source = CModSub(QUInt(3), 3, cv=control_value).adjoint()
        ports = {"ctrl": (0,), "x": (1, 2, 3), "y": (4, 5, 6)}
        words = [(c, x, y) for c in (0, 1) for x in range(3) for y in range(3)]
    logical = sum(len(wires) for wires in ports.values())
    bound = bind_register_program(
        record, tuple(ports.values()), clean=tuple(range(logical, logical + 25))
    )
    assert_source(
        bound.circuit.instructions,
        source_stream(source, bound, ports, expand_modular=True),
    )
    matrix = np.asarray(source.tensor_contract())
    width = bound.circuit.n_wires

    def logical_word(values: tuple[int, ...]) -> int:
        result = 0
        for value, wires in zip(values, ports.values(), strict=True):
            result = (result << len(wires)) | value
        return result

    coefficients = np.array([complex(i + 1, 5 - i) for i in range(len(words))])
    coefficients /= np.linalg.norm(coefficients)
    logical_state = np.zeros(1 << logical, dtype=np.complex128)
    for values, amp in zip(words, coefficients, strict=True):
        logical_state[logical_word(values)] = amp
    expected_logical = matrix @ logical_state
    before = {
        logical_word(values) << (width - logical): complex(amp)
        for values, amp in zip(words, coefficients, strict=True)
    }
    expected = {
        i << (width - logical): complex(amp)
        for i, amp in enumerate(expected_logical)
        if abs(amp) > 1e-12
    }
    basis = tuple(
        tuple((word >> (width - i - 1)) & 1 for i in range(width)) for word in before
    )
    mapped, measurements = certified_and_channel_map(bound.circuit.instructions, basis)
    assert (
        measurements == Counter(op.name for op in bound.circuit.instructions)["measure"]
    )
    for bits in mapped:
        assert not any(bits[logical:])
    for seed in (0, 19):
        assert execute_sparse_register(
            bound.circuit, before, seed=seed
        ) == pytest.approx(expected, abs=4e-12)

"""Controlled comparison parents stop only at actual source-refused leaves."""

from hashlib import sha1
from importlib import import_module
from pathlib import Path

import pytest

from tests._reference_dependencies import unavailable_reference

try:
    for dependency in ("numpy", "cirq", "kickmix"):
        import_module(dependency)
    from qualtran import DecomposeTypeError, QInt, QUInt
    from qualtran.bloqs.arithmetic.comparison import (
        CLinearDepthGreaterThan,
        LinearDepthGreaterThan,
    )
except ModuleNotFoundError as error:
    unavailable_reference(error)

from flagquantum.compiler.registers.comparison.controlled_linear_component import (
    controlled_linear_greater_program,
    nested_controlled_greater_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.register_types import (
    montgomery_register_type,
    signed_register_type,
    unsigned_register_type,
)
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_ecc_addition_model import frozen_source as frozen_source
from tests.unit.test_symbolic_modular_addition_calls import atomic_views

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_comparison_source() -> None:
    module = import_module("qualtran.bloqs.arithmetic.comparison")
    assert module.__file__ is not None
    raw = Path(module.__file__).read_bytes()
    assert (
        sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == "ae1f0275d081eae99fcf7786659dd33873083f18"
    )


def assert_original_refused_leaves(record: ProgramRecord, source: object) -> None:
    leaves = atomic_views(record)
    pairs = compare_layers(record, source, leaves)
    functions = {function.name: function for function in record.functions}
    declarations = [name for name in pairs if not functions[name].blocks]
    assert len(declarations) == 2
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
            record, tuple(inputs), clean=tuple(range(offset, offset + 20))
        )


@pytest.mark.parametrize("width", (2, 3))
@pytest.mark.parametrize("signed", (False, True))
@pytest.mark.parametrize("adjoint", (False, True))
def test_controlled_linear_complete_available_source_layers(
    width: int, signed: bool, adjoint: bool
) -> None:
    source = LinearDepthGreaterThan(width, signed=signed)
    source = (source.adjoint() if adjoint else source).controlled()
    record = controlled_linear_greater_program(width, signed=signed, adjoint=adjoint)
    assert_original_refused_leaves(record, source)


@pytest.mark.parametrize("width", (1, 3))
@pytest.mark.parametrize("control_value", (0, 1))
@pytest.mark.parametrize("adjoint", (False, True))
def test_nested_controlled_unsigned_complete_available_source_layers(
    width: int, control_value: int, adjoint: bool
) -> None:
    source = CLinearDepthGreaterThan(QUInt(width), cv=control_value)
    source = (source.adjoint() if adjoint else source).controlled()
    record = nested_controlled_greater_program(
        unsigned_register_type(width), control_value=control_value, adjoint=adjoint
    )
    assert_original_refused_leaves(record, source)


@pytest.mark.parametrize("control_value", (0, 1))
def test_signed_nested_root_and_source_directional_boundaries(
    control_value: int,
) -> None:
    source = CLinearDepthGreaterThan(QInt(3), cv=control_value).controlled()
    record = nested_controlled_greater_program(
        signed_register_type(3), control_value=control_value
    )
    pairs = compare_layers(record, source, atomic_views(record))
    functions = {function.name: function for function in record.functions}
    declarations = [name for name in pairs if not functions[name].blocks]
    assert len(declarations) == 4
    for name in declarations:
        with pytest.raises(DecomposeTypeError, match="non-thru registers"):
            pairs[name].decompose_bloq()
    with pytest.raises(ValueError, match="signed comparator adjoint"):
        nested_controlled_greater_program(signed_register_type(3), adjoint=True)


@pytest.mark.parametrize("control_value", (0, 1))
@pytest.mark.parametrize("adjoint", (False, True))
def test_nested_montgomery_available_layers(control_value: int, adjoint: bool) -> None:
    from qualtran import QMontgomeryUInt

    source = CLinearDepthGreaterThan(QMontgomeryUInt(3, 5), cv=control_value)
    source = (source.adjoint() if adjoint else source).controlled()
    record = nested_controlled_greater_program(
        montgomery_register_type(3, 5), control_value=control_value, adjoint=adjoint
    )
    assert_original_refused_leaves(record, source)


@pytest.mark.parametrize("nested", (False, True))
@pytest.mark.parametrize("adjoint", (False, True))
@pytest.mark.parametrize("depth", (0, 1))
def test_constructor_context_roundtrip_without_reference_imports(
    nested: bool, adjoint: bool, depth: int
) -> None:
    import sys
    from unittest.mock import patch

    from flagquantum.compiler.formats.legacy_comparison_factories import (
        comparison_factories,
    )
    from flagquantum.compiler.formats.legacy_component_resolution import (
        resolve_component_program,
    )
    from flagquantum.compiler.registers.component_definition_match import (
        matches_component_definition,
    )
    from tests.unit.test_legacy_arithmetic_factories import _library

    source = (
        CLinearDepthGreaterThan(QUInt(3), cv=0) if nested else LinearDepthGreaterThan(3)
    )
    source = (source.adjoint() if adjoint else source).controlled()
    library = _library(source, depth)
    expected = (
        nested_controlled_greater_program(
            unsigned_register_type(3), control_value=0, adjoint=adjoint
        )
        if nested
        else controlled_linear_greater_program(3, adjoint=adjoint)
    )
    denied = {
        name: None
        for name in sys.modules
        if name.split(".", 1)[0] in {"qualtran", "cirq", "kickmix"}
    }
    with patch.dict(sys.modules, denied):
        record = resolve_component_program(library, 0, comparison_factories())
    assert matches_component_definition(record, record.functions[0].name, expected)


@pytest.mark.parametrize("nested", (False, True))
def test_negative_outer_predicate_wrapper(nested: bool) -> None:
    from qualtran import CtrlSpec

    from flagquantum.compiler.formats.legacy_comparison_factories import (
        comparison_factories,
    )
    from flagquantum.compiler.formats.legacy_component_resolution import (
        resolve_component_program,
    )
    from tests.unit.test_legacy_arithmetic_factories import _library

    source = CLinearDepthGreaterThan(QUInt(3)) if nested else LinearDepthGreaterThan(3)
    source = source.controlled(CtrlSpec(cvs=0))
    record = resolve_component_program(_library(source, 0), 0, comparison_factories())
    calls = record.functions[0].blocks[0].operations
    assert len(calls) == 3
    compare_layers(record, source, {op.callee for op in calls if op.callee is not None})


@pytest.mark.parametrize("signed", (False, True))
@pytest.mark.parametrize("adjoint", (False, True))
def test_one_bit_complete_operator_every_complex_branch_and_append(
    signed: bool, adjoint: bool
) -> None:
    from collections import Counter
    from itertools import product

    import numpy as np
    import torch
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic
    from tests.unit.measurement_checks import forced_branch
    from tests.unit.test_multi_control_component import MeasurementSlots
    from tests.unit.test_register_constant_addition import reference_arithmetic_gates
    from tests.unit.test_register_subtraction import assert_source

    source = LinearDepthGreaterThan(1, signed=signed)
    source = (source.adjoint() if adjoint else source).controlled()
    record = controlled_linear_greater_program(1, signed=signed, adjoint=adjoint)
    compare_layers(record, source, atomic_views(record))
    inputs = ((0,), (1,), (2,), (3,))
    bound = bind_register_program(record, inputs, clean=(4, 5))
    counts = Counter(op.name for op in bound.circuit.instructions)
    ports: dict[str, tuple[int, ...]] = dict(
        zip(("ctrl", "a", "b", "target"), inputs, strict=True)
    )
    allocations = tuple(
        w for event in bound.events if event.kind == "allocate" for w in event.wires
    )
    assert_source(
        bound.circuit.instructions,
        reference_arithmetic_gates(
            source,
            ports,
            allocations,
            MeasurementSlots(bound.circuit.instructions),
            reuse_allocations=False,
            keep_pauli_leaves=True,
        ),
    )
    model = get_cost_value(source, QECGatesCost())
    assert counts["measure"] == model.measurement
    assert counts["t"] + counts["tdg"] + 4 * counts["ccx"] == model.total_t_count()
    matrix = torch.as_tensor(
        np.asarray(source.tensor_contract()), dtype=torch.complex128
    )
    values = torch.arange(1, 33, dtype=torch.float64)
    amplitudes = torch.complex(values, 7 - 2 * values).reshape(16, 2)
    amplitudes /= amplitudes.norm()
    physical = bound.circuit.n_wires + 1
    initial = torch.zeros(1 << physical, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    after = matrix @ amplitudes
    for word in range(16):
        for spectator in (0, 1):
            index = (word << (physical - 4)) | spectator
            initial[index], expected[index] = (
                amplitudes[word, spectator],
                after[word, spectator],
            )
    for history in product((0, 1), repeat=counts["measure"]):
        torch.testing.assert_close(
            forced_branch(initial, bound.circuit.instructions, history),
            expected / 2 ** (counts["measure"] / 2),
            rtol=0,
            atol=3e-12,
        )
    circuit = DynamicCircuit(physical, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(physical, bound.circuit.instructions))
    result = run_dynamic(circuit, shots=2, seed=19, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(2, -1), expected.expand(2, -1), rtol=0, atol=3e-12
    )

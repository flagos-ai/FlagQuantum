"""Resolve the original specialized control of an already-controlled LCU."""

from itertools import product
from math import sqrt
from typing import Any

import pytest
import torch

from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_linear_combination_factories import (
    linear_combination_factories,
)
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.ecosystem._linear_combination_component_library import (
    capture_linear_combination_context,
)
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_default_linear_combination_oracles import bind
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_linear_combination_component import native_oracle_fixture
from tests.unit.test_multi_control_component import atomic_leaves
from tests.unit.test_multiplexed_source_binding import logical_operator

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def repeated_fixture(
    pattern: tuple[int, ...], inverse: bool
) -> tuple[ProgramRecord, Any]:
    from qualtran import CtrlSpec
    from qualtran._infra.adjoint import Adjoint

    positive, preparation, selection = native_oracle_fixture(True)
    context = capture_linear_combination_context(
        positive, prepare_owner=lambda _: preparation, select_owner=lambda _: selection
    )
    source = positive.controlled(
        CtrlSpec(cvs=pattern[0] if len(pattern) == 1 else pattern)
    )
    if inverse:
        source = Adjoint(source)
    library = _library(source, 0)
    roots = [
        entry
        for entry in library.entries
        if entry.declaration.name
        == "qualtran.bloqs.block_encoding.linear_combination.LinearCombination"
    ]
    assert len(roots) == 1
    binding = f"legacy_component_{roots[0].identifier}"
    return (
        resolve_component_program(
            library, 0, linear_combination_factories({binding: context})
        ),
        source,
    )


@pytest.mark.parametrize("pattern", ((0,), (1,), (1, 0), (0, 1, 0)))
@pytest.mark.parametrize("inverse", (False, True))
def test_repeated_control_source_calls_complex_operator_and_erasure(
    pattern: tuple[int, ...], inverse: bool
) -> None:
    record, source = repeated_fixture(pattern, inverse)
    assert not verify_program_record(record)
    compare_layers(record, source, atomic_leaves(record))
    bound, logical = bind(record)
    torch.testing.assert_close(
        logical_operator(bound.circuit, logical),
        torch.as_tensor(source.tensor_contract(), dtype=torch.complex128),
        atol=1e-10,
        rtol=0,
    )
    count = sum(g.name == "measure" for g in bound.circuit.instructions)
    index = torch.arange(1 << logical, dtype=torch.float64)
    state = torch.complex(index % 5 - 2, index % 7 - 3)
    state /= state.norm()
    stride = 1 << (bound.circuit.n_wires - logical)
    initial = torch.zeros(1 << bound.circuit.n_wires, dtype=torch.complex128)
    initial[::stride] = state
    expected = torch.zeros_like(initial)
    expected[::stride] = (
        torch.as_tensor(source.tensor_contract(), dtype=torch.complex128) @ state
    )
    for history in product((0, 1), repeat=count):
        actual = forced_branch(initial, bound.circuit.instructions, history) * sqrt(
            2**count
        )
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=0)


@pytest.mark.parametrize("values", ((0, 1, 1), (1, 0, 1), (0, 1, 0, 1)))
@pytest.mark.parametrize("inverse", (False, True))
def test_specialized_control_array_extension_exact_inverse(
    values: tuple[int, ...], inverse: bool
) -> None:
    from qualtran._infra.adjoint import Adjoint
    from qualtran.bloqs.mcmt.specialized_ctrl import (
        _MultiControlledFromSinglyControlled,
    )

    positive, preparation, selection = native_oracle_fixture(True)
    context = capture_linear_combination_context(
        positive, prepare_owner=lambda _: preparation, select_owner=lambda _: selection
    )
    source = _MultiControlledFromSinglyControlled(values, positive, "ctrl")
    if inverse:
        source = Adjoint(source)
    library = _library(source, 0)
    root = next(
        e for e in library.entries if e.declaration.name.endswith(".LinearCombination")
    )
    record = resolve_component_program(
        library,
        0,
        linear_combination_factories({f"legacy_component_{root.identifier}": context}),
    )
    compare_layers(record, source, atomic_leaves(record))
    bound, logical = bind(record)
    torch.testing.assert_close(
        logical_operator(bound.circuit, logical),
        torch.as_tensor(source.tensor_contract(), dtype=torch.complex128),
        atol=1e-10,
        rtol=0,
    )


def test_repeated_default_constructor_without_source_imports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    from qualtran import CtrlSpec

    from flagquantum import Circuit
    from flagquantum.algorithms.primitives._block_binding import _BoundBlockOperation
    from flagquantum.ecosystem._default_linear_combination_oracles import (
        default_linear_combination_prepare_pair,
        default_linear_combination_select_pair,
    )
    from tests.unit.test_default_linear_combination_oracles import default_source

    positive = default_source().controlled(CtrlSpec())
    context = capture_linear_combination_context(
        positive,
        prepare_owner=default_linear_combination_prepare_pair,
        select_owner=default_linear_combination_select_pair,
    )
    source = positive.controlled(CtrlSpec(cvs=0))
    library = _library(source, 0)
    root = next(
        e for e in library.entries if e.declaration.name.endswith(".LinearCombination")
    )
    factories = linear_combination_factories(
        {f"legacy_component_{root.identifier}": context}
    )
    expected = resolve_component_program(library, 0, factories)
    compare_layers(expected, source, atomic_leaves(expected))
    before, _ = bind(expected)
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "qualtran", None)
        patch.setitem(sys.modules, "cirq", None)
        record = resolve_component_program(
            library,
            0,
            linear_combination_factories(
                {f"legacy_component_{root.identifier}": context}
            ),
        )
        bound, _ = bind(record)
        circuit = Circuit(bound.circuit.n_wires)
        circuit.append(
            _BoundBlockOperation(bound.circuit.instructions, bound.circuit.n_wires)
        )
        assert circuit.to_ir().instructions == before.circuit.instructions


def test_original_helper_further_control_callback_failure_is_retained() -> None:
    from qualtran import BloqBuilder, CtrlSpec

    from flagquantum.errors import SerializationError

    positive, preparation, selection = native_oracle_fixture(True)
    helper = positive.controlled(CtrlSpec())
    _, adder = helper.get_ctrl_system(CtrlSpec())
    builder, inputs = BloqBuilder.from_signature(
        helper.signature, add_registers_allowed=True
    )
    control = builder.add_register("outer", 1)
    with pytest.raises(TypeError, match="integer scalar arrays"):
        adder(builder, [control], inputs)
    context = capture_linear_combination_context(
        positive, prepare_owner=lambda _: preparation, select_owner=lambda _: selection
    )
    source = helper.controlled(CtrlSpec(cvs=(1, 0)))
    with pytest.raises(TypeError, match="integer scalar arrays"):
        source.decompose_bloq()
    library = _library(source, 0)
    root = next(
        e for e in library.entries if e.declaration.name.endswith(".LinearCombination")
    )
    with pytest.raises(SerializationError, match="Missing owned"):
        resolve_component_program(
            library,
            0,
            linear_combination_factories(
                {f"legacy_component_{root.identifier}": context}
            ),
        )

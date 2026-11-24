"""Resolve original polynomial constructors through explicit exact owners."""

import sys
from unittest.mock import patch

import pytest
import torch

from flagquantum import Circuit
from flagquantum.algorithms.primitives._block_binding import _BoundBlockOperation
from flagquantum.compiler.formats.legacy_chebyshev_factories import chebyshev_factories
from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_linear_combination_factories import (
    ExactPair,
    linear_combination_factories,
)
from flagquantum.compiler.registers.component_definition_match import (
    matches_component_definition,
)
from flagquantum.core.ir.program import FunctionRecord, verify_program_record
from flagquantum.core.value_math import SymbolicInt
from flagquantum.ecosystem._chebyshev_component_library import capture_chebyshev_pair
from flagquantum.ecosystem._default_linear_combination_oracles import (
    default_linear_combination_prepare_pair,
    default_linear_combination_select_pair,
)
from flagquantum.ecosystem._linear_combination_component_library import (
    capture_linear_combination_context,
)
from tests.unit.test_chebyshev_component import (
    assert_all_small_raw_histories,
    chebyshev_terminals,
)
from tests.unit.test_chebyshev_external_file import (
    frozen_chebyshev_sources as frozen_chebyshev_sources,
)
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_default_linear_combination_oracles import bind
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_multiplexed_source_binding import logical_operator

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("odd", (False, True))
@pytest.mark.parametrize("scaled", (False, True))
def test_all_original_constructor_roots_with_explicit_source_context(
    odd: bool, scaled: bool
) -> None:
    from qualtran._infra.adjoint import Adjoint
    from qualtran.bloqs.block_encoding.chebyshev_polynomial import (
        _chebyshev_poly_even,
        _chebyshev_poly_odd,
        _scaled_chebyshev_poly_even,
        _scaled_chebyshev_poly_odd,
    )
    from qualtran.bloqs.block_encoding.linear_combination import LinearCombination

    examples = (
        (_chebyshev_poly_even, _chebyshev_poly_odd),
        (_scaled_chebyshev_poly_even, _scaled_chebyshev_poly_odd),
    )
    base = examples[int(scaled)][int(odd)].make()
    child_pair = capture_chebyshev_pair(base.block_encoding)
    property_pair = capture_chebyshev_pair(base.linear_combination) if scaled else None
    for inverse in (False, True):
        source = Adjoint(base) if inverse else base
        library = _library(source, 0)
        contexts = {}
        if isinstance(base.block_encoding, LinearCombination):
            context = capture_linear_combination_context(
                base.block_encoding,
                prepare_owner=default_linear_combination_prepare_pair,
                select_owner=default_linear_combination_select_pair,
            )
            for entry in library.entries:
                if entry.declaration.name.endswith(
                    "linear_combination.LinearCombination"
                ):
                    contexts[f"legacy_component_{entry.identifier}"] = context

        def exact_child(child: FunctionRecord) -> ExactPair:
            assert child.registers == child_pair[0].functions[0].registers
            return child_pair

        def coefficients(
            child: FunctionRecord, order: SymbolicInt, precision: SymbolicInt
        ) -> ExactPair:
            assert child.registers == child_pair[0].functions[0].registers
            assert order == base.order and precision == base.lambd_bits
            assert property_pair is not None
            return property_pair

        factories = {
            **linear_combination_factories(contexts),
            **chebyshev_factories(child_owner=exact_child, scaled_owner=coefficients),
        }
        record = resolve_component_program(library, 0, factories)
        assert not verify_program_record(record)
        # The complete child graphs are independently source-checked by the
        # component suite. Reuse only literal matching definitions here; the
        # constructor's own call order and named operands are checked anew.
        terminals = chebyshev_terminals(record)
        dependencies = property_pair if scaled else child_pair
        assert dependencies is not None
        matched = set()
        for function in record.functions[1:]:
            for dependency in dependencies:
                if function.registers != dependency.functions[0].registers:
                    continue
                if matches_component_definition(record, function.name, dependency):
                    matched.add(function.name)
        assert matched
        compare_layers(record, source, terminals | matched)


@pytest.mark.parametrize("values", (1, 0, (1, 0)))
@pytest.mark.parametrize(
    "direction", ("forward", "inverse_then_control", "control_then_inverse")
)
@pytest.mark.parametrize("scaled", (False, True))
def test_constructor_controls_preserve_literal_child_and_inverse_calls(
    values: int | tuple[int, ...], direction: str, scaled: bool
) -> None:
    from qualtran import CtrlSpec
    from qualtran._infra.adjoint import Adjoint
    from qualtran.bloqs.basic_gates import TGate
    from qualtran.bloqs.block_encoding.chebyshev_polynomial import (
        ChebyshevPolynomial,
        ScaledChebyshevPolynomial,
    )
    from qualtran.bloqs.block_encoding.unitary import Unitary

    child = Unitary(TGate(), ancilla_bitsize=1)
    original = (
        ScaledChebyshevPolynomial(child, 1) if scaled else ChebyshevPolynomial(child, 3)
    )
    subject = Adjoint(original) if direction == "inverse_then_control" else original
    source = subject.controlled(CtrlSpec(cvs=values))
    if direction == "control_then_inverse":
        source = Adjoint(source)
    record = resolve_component_program(
        _library(source, 0),
        0,
        chebyshev_factories(
            positive_child_owner=lambda _: capture_chebyshev_pair(
                child, controlled=True
            ),
            scaled_owner=lambda *_: capture_chebyshev_pair(original.linear_combination),
            positive_scaled_owner=lambda *_: capture_chebyshev_pair(
                original.linear_combination, controlled=True
            ),
        ),
    )
    assert not verify_program_record(record)
    compare_layers(record, source, chebyshev_terminals(record))
    with patch.dict(sys.modules, {"qualtran": None}):
        bound, logical = bind(record)
        appended = Circuit(bound.circuit.n_wires).append(
            _BoundBlockOperation(bound.circuit.instructions, bound.circuit.n_wires)
        )
        assert appended.to_ir().instructions == bound.circuit.instructions
    expected = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
    torch.testing.assert_close(
        logical_operator(bound.circuit, logical),
        expected,
        atol=1e-10,
        rtol=0,
    )
    assert_all_small_raw_histories(bound.circuit, logical, expected)


@pytest.mark.parametrize("inverse", (False, True))
@pytest.mark.parametrize("depth", (0, 1))
def test_original_odd_constructor_calls_full_complex_operator(
    inverse: bool, depth: int
) -> None:
    from qualtran._infra.adjoint import Adjoint
    from qualtran.bloqs.block_encoding.chebyshev_polynomial import _chebyshev_poly_odd

    source = _chebyshev_poly_odd.make()
    if inverse:
        source = Adjoint(source)
    record = resolve_component_program(
        _library(source, depth), 0, chebyshev_factories()
    )
    assert not verify_program_record(record)
    compare_layers(record, source, chebyshev_terminals(record))
    bound, logical = bind(record)
    torch.testing.assert_close(
        logical_operator(bound.circuit, logical),
        torch.as_tensor(source.tensor_contract(), dtype=torch.complex128),
        atol=1e-10,
        rtol=0,
    )


@pytest.mark.parametrize("inverse", (False, True))
def test_explicit_scaled_property_owner_preserves_constructor_boundary(
    inverse: bool,
) -> None:
    from qualtran._infra.adjoint import Adjoint
    from qualtran.bloqs.basic_gates import Hadamard
    from qualtran.bloqs.block_encoding.chebyshev_polynomial import (
        ScaledChebyshevPolynomial,
    )
    from qualtran.bloqs.block_encoding.unitary import Unitary

    source = ScaledChebyshevPolynomial(Unitary(Hadamard(), alpha=3.14), 1)
    property_pair = capture_chebyshev_pair(source.linear_combination)
    original = source
    if inverse:
        source = Adjoint(source)

    def coefficient_owner(
        child: FunctionRecord, order: SymbolicInt, precision: SymbolicInt
    ) -> ExactPair:
        assert child.registers == property_pair[0].functions[0].registers
        assert order == original.order and precision == original.lambd_bits
        return property_pair

    record = resolve_component_program(
        _library(source, 0),
        0,
        chebyshev_factories(scaled_owner=coefficient_owner),
    )
    compare_layers(record, source, chebyshev_terminals(record))
    bound, logical = bind(record)
    torch.testing.assert_close(
        logical_operator(bound.circuit, logical),
        torch.as_tensor(source.tensor_contract(), dtype=torch.complex128),
        atol=1e-10,
        rtol=0,
    )

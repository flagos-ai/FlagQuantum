"""Retain explicitly captured LCU contexts despite lossy original payloads."""

import pytest
import torch

from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_linear_combination_factories import (
    ExactPair,
    linear_combination_factories,
)
from flagquantum.compiler.formats.legacy_word_factories import _rename_program
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.layout.packed_oracle_component import (
    packed_oracle_component_program,
)
from flagquantum.compiler.registers.preparation.linear_combination_component import (
    linear_combination_signature,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import (
    FunctionRecord,
    ProgramRecord,
    verify_program_record,
)
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.ecosystem._linear_combination_component_library import (
    capture_linear_combination_context,
)
from flagquantum.errors import SerializationError
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_legacy_block_encoding_factories import _oracles
from tests.unit.test_linear_combination_component import native_oracle_fixture
from tests.unit.test_multi_control_component import atomic_leaves
from tests.unit.test_multiplexed_source_binding import logical_operator

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("controlled", (False, True))
@pytest.mark.parametrize("inverse", (False, True))
def test_captured_context_resolves_original_shallow_root_and_exact_inverse(
    controlled: bool, inverse: bool
) -> None:
    from qualtran import CtrlSpec
    from qualtran._infra.adjoint import Adjoint

    source, prepare_pair, select_pair = native_oracle_fixture(controlled)
    actual_select = (
        source.select.get_ctrl_system(CtrlSpec())[0] if controlled else source.select
    )
    seen = []

    def preparation(child: object) -> ExactPair:
        assert child == source.prepare
        seen.append("prepare")
        return prepare_pair

    def selection(child: object) -> ExactPair:
        assert child == actual_select
        seen.append("select")
        return select_pair

    context = capture_linear_combination_context(
        source, prepare_owner=preparation, select_owner=selection
    )
    assert seen == ["prepare", "select"]
    if inverse:
        source = Adjoint(source)
    library = _library(source, 0)
    binding = "legacy_component_1" if inverse else "main"
    record = resolve_component_program(
        library, 0, linear_combination_factories({binding: context})
    )
    assert not verify_program_record(record)
    compare_layers(record, source, atomic_leaves(record))
    logical = 3 + int(controlled)
    bound = bind_register_program(
        record,
        tuple((i,) for i in range(logical)),
        clean=(logical,) if controlled else (),
    )
    direct = bind_register_program(
        context.program(adjoint=inverse),
        tuple((i,) for i in range(logical)),
        clean=(logical,) if controlled else (),
    )
    assert bound.circuit.instructions == direct.circuit.instructions
    assert tuple((event.kind, event.wires) for event in bound.events) == tuple(
        (event.kind, event.wires) for event in direct.events
    )
    torch.testing.assert_close(
        logical_operator(bound.circuit, logical),
        torch.as_tensor(source.tensor_contract(), dtype=torch.complex128),
        atol=5e-11,
        rtol=0,
    )


def test_context_binding_is_explicit_and_surviving_fields_are_checked() -> None:
    from dataclasses import replace

    source, prepare_pair, select_pair = native_oracle_fixture(False)
    context = capture_linear_combination_context(
        source, prepare_owner=lambda _: prepare_pair, select_owner=lambda _: select_pair
    )
    library = _library(source, 0)
    for contexts, message in (
        ({}, "Missing explicit"),
        ({"some_other_definition": context}, "Missing explicit"),
        ({"main": replace(context, precision=2)}, "disagrees"),
        ({"main": replace(context, controlled=True)}, "disagrees"),
        (
            {
                "main": replace(
                    context,
                    signature=linear_combination_signature(
                        2, (0, 0), (0, 0), prepare_selection=1, prepare_junk=1
                    ),
                )
            },
            "signature",
        ),
    ):
        with pytest.raises((SerializationError, ValueError), match=message):
            resolve_component_program(
                library, 0, linear_combination_factories(contexts)
            )


@pytest.mark.parametrize("precision", (0, -3))
def test_supplied_oracle_unused_precision_remains_accepted(precision: int) -> None:
    from attrs import evolve

    source, prepare_pair, select_pair = native_oracle_fixture(False)
    source = evolve(source, lambd_bits=precision)
    context = capture_linear_combination_context(
        source, prepare_owner=lambda _: prepare_pair, select_owner=lambda _: select_pair
    )
    record = resolve_component_program(
        _library(source, 0), 0, linear_combination_factories({"main": context})
    )
    bound = bind_register_program(record, ((0,), (1,), (2,)))
    torch.testing.assert_close(
        logical_operator(bound.circuit, 3),
        torch.as_tensor(source.tensor_contract(), dtype=torch.complex128),
        atol=5e-11,
        rtol=0,
    )


def test_capture_is_native_after_conversion_and_source_free_append(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    from flagquantum import Circuit
    from flagquantum.algorithms.primitives._block_binding import _BoundBlockOperation

    source, prepare_pair, select_pair = native_oracle_fixture(False)
    context = capture_linear_combination_context(
        source, prepare_owner=lambda _: prepare_pair, select_owner=lambda _: select_pair
    )
    library = _library(source, 0)
    expected = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "qualtran", None)
        patch.setitem(sys.modules, "cirq", None)
        record = resolve_component_program(
            library, 0, linear_combination_factories({"main": context})
        )
        bound = bind_register_program(record, ((0,), (1,), (2,)))
        circuit = Circuit(3)
        circuit.append(_BoundBlockOperation(bound.circuit.instructions, 3))
        torch.testing.assert_close(
            logical_operator(circuit.to_ir(), 3), expected, atol=5e-11, rtol=0
        )


def test_symbolic_unused_precision_and_owned_boundary() -> None:
    from attrs import evolve
    from sympy import Symbol

    source, prepare_pair, select_pair = native_oracle_fixture(False)
    source = evolve(source, lambd_bits=Symbol("mu", positive=True, integer=True))
    context = capture_linear_combination_context(
        source, prepare_owner=lambda _: prepare_pair, select_owner=lambda _: select_pair
    )
    record = resolve_component_program(
        _library(source, 0), 0, linear_combination_factories({"main": context})
    )
    assert not verify_program_record(record)
    bound = bind_register_program(record, ((0,), (1,), (2,)))
    torch.testing.assert_close(
        logical_operator(bound.circuit, 3),
        torch.as_tensor(source.tensor_contract(), dtype=torch.complex128),
        atol=5e-11,
        rtol=0,
    )
    with pytest.raises(SerializationError, match="logical native Programs"):
        capture_linear_combination_context(
            source,
            prepare_owner=lambda _: (source, source),
            select_owner=lambda _: select_pair,
        )
    with pytest.raises(SerializationError, match="explicit source"):
        capture_linear_combination_context(
            object(),
            prepare_owner=lambda _: prepare_pair,
            select_owner=lambda _: select_pair,
        )


def test_identical_lossy_payloads_retain_distinct_explicit_native_owners() -> None:
    from typing import Any

    from attrs import evolve, frozen
    from qualtran.bloqs.state_preparation.black_box_prepare import BlackBoxPrepare
    from qualtran.bloqs.state_preparation.prepare_base import PrepareOracle

    first, first_prepare, selection = native_oracle_fixture(False)
    original, _, raw_prepare, _, _ = _oracles()

    class InversePrepare:
        @property
        def selection_registers(self) -> tuple[Any, ...]:
            return tuple(original.selection_registers)

        @property
        def junk_registers(self) -> tuple[Any, ...]:
            return tuple(original.junk_registers)

        @property
        def l1_norm_of_coeffs(self) -> float:
            return 3.0

        def build_composite_bloq(self, bb: Any, **values: Any) -> dict[str, Any]:
            return dict(bb.add_d(original.adjoint(), **values))

    inverse_type = frozen(
        type(
            "InversePrepare", (InversePrepare, PrepareOracle), {"__module__": __name__}
        )
    )
    second = evolve(first, prepare=BlackBoxPrepare(inverse_type()))
    wrapped = []
    for child in reversed(raw_prepare):
        owner = ComponentProgramBuilder(Signature(child.functions[0].registers or ()))
        wrapped.append(
            owner.finalize(
                owner.call_program(child, owner.inputs, entry=child.functions[0].name)
            )
        )
    inverse_pair = wrapped[0], wrapped[1]
    ports = inverse_pair[0].functions[0].registers or ()
    packed_signature = Signature(
        _library(second.prepare, 0).entries[0].declaration.registers or ()
    )
    forward, reverse = (
        packed_oracle_component_program(
            packed_signature,
            (("selection", (ports[0],)), ("junk", (ports[1],))),
            inverse_pair[0],
            inverse=inverse_pair[1],
            adjoint=direction,
        )
        for direction in (False, True)
    )
    second_prepare = forward, reverse
    first_context = capture_linear_combination_context(
        first, prepare_owner=lambda _: first_prepare, select_owner=lambda _: selection
    )
    second_context = capture_linear_combination_context(
        second, prepare_owner=lambda _: second_prepare, select_owner=lambda _: selection
    )
    libraries = _library(first, 0), _library(second, 0)
    assert (
        libraries[0].entries[0].declaration.arguments
        == libraries[1].entries[0].declaration.arguments
    )
    assert (
        libraries[0].entries[0].declaration.registers
        == libraries[1].entries[0].declaration.registers
    )
    factories = linear_combination_factories(
        {"first": first_context, "second": second_context}
    )
    matrices = []
    for name, source, library in zip(
        ("first", "second"), (first, second), libraries, strict=True
    ):
        record = resolve_component_program(library, 0, factories, entry_name=name)
        comparable = _rename_program(record, "main")
        compare_layers(comparable, source, atomic_leaves(comparable))
        bound = bind_register_program(record, ((0,), (1,), (2,)), entry=name)
        matrix = logical_operator(bound.circuit, 3)
        torch.testing.assert_close(
            matrix,
            torch.as_tensor(source.tensor_contract(), dtype=torch.complex128),
            atol=5e-11,
            rtol=0,
        )
        matrices.append(matrix)
    assert not torch.allclose(matrices[0], matrices[1], atol=1e-8, rtol=0)


def test_symbolic_source_widths_preserve_resolved_maxima_and_wire_projection() -> None:
    from typing import Any

    from attrs import evolve, frozen
    from qualtran import QAny, QBit
    from qualtran import Register as SourceRegister
    from qualtran.bloqs.basic_gates import XGate
    from qualtran.bloqs.block_encoding.unitary import Unitary
    from qualtran.bloqs.multiplexers.black_box_select import BlackBoxSelect
    from qualtran.bloqs.multiplexers.select_base import SelectOracle
    from sympy import Symbol

    source, preparation, _ = native_oracle_fixture(False)
    width = Symbol("n", positive=True, integer=True)

    class Select:
        @property
        def control_registers(self) -> tuple[Any, ...]:
            return ()

        @property
        def selection_registers(self) -> tuple[Any, ...]:
            return (SourceRegister("address", QBit()),)

        @property
        def target_registers(self) -> tuple[Any, ...]:
            return (SourceRegister("data", QAny(width + 1)),)

    select_type = frozen(
        type("SymbolicSelect", (Select, SelectOracle), {"__module__": __name__})
    )
    source = evolve(
        source,
        block_encodings=(Unitary(XGate()), Unitary(XGate(), ancilla_bitsize=width)),
        select=BlackBoxSelect(select_type()),
    )
    ports = _library(source.select, 0).entries[0].declaration.registers
    selection = ProgramRecord((FunctionRecord("main", (), registers=ports),))
    context = capture_linear_combination_context(
        source,
        prepare_owner=lambda _: preparation,
        select_owner=lambda _: (selection, selection),
    )
    library = _library(source, 0)
    record = resolve_component_program(
        library, 0, linear_combination_factories({"main": context})
    )
    assert record.functions[0].registers == library.entries[0].declaration.registers
    assert not record.functions[0].blocks and not verify_program_record(record)
    with pytest.raises(ValueError, match="bodies"):
        bind_register_program(record, ((0,), (1, 2), (3,)))


@pytest.mark.parametrize("pattern", ((0,), (1, 0), (1, 0, 1)))
@pytest.mark.parametrize("inverse", (False, True))
def test_context_predicate_controls_exact_inverse_and_every_raw_history(
    pattern: tuple[int, ...], inverse: bool
) -> None:
    from itertools import product
    from math import sqrt

    from qualtran import CtrlSpec
    from qualtran._infra.adjoint import Adjoint

    from flagquantum.compiler.formats.legacy_linear_combination_factories import (
        LinearCombinationContext,
    )
    from tests.unit.measurement_checks import forced_branch

    base, prepare_pair, select_pair = native_oracle_fixture(False)
    context = capture_linear_combination_context(
        base, prepare_owner=lambda _: prepare_pair, select_owner=lambda _: select_pair
    )
    positive_source, positive_prepare, positive_select = native_oracle_fixture(True)
    positive = capture_linear_combination_context(
        positive_source,
        prepare_owner=lambda _: positive_prepare,
        select_owner=lambda _: positive_select,
    )

    def positive_owner(original: LinearCombinationContext) -> LinearCombinationContext:
        assert original is context
        return positive

    source = base.controlled(CtrlSpec(cvs=pattern[0] if len(pattern) == 1 else pattern))
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
    record = resolve_component_program(
        library,
        0,
        linear_combination_factories(
            {binding: context}, positive_context=positive_owner
        ),
    )
    compare_layers(record, source, atomic_leaves(record))
    logical = 3 + len(pattern)
    bound = bind_register_program(
        record,
        tuple((i,) for i in range(logical)),
        clean=tuple(range(logical, logical + len(pattern))),
    )
    matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
    torch.testing.assert_close(
        logical_operator(bound.circuit, logical), matrix, atol=7e-11, rtol=0
    )
    index = torch.arange(1 << logical, dtype=torch.float64)
    state = torch.complex(index % 7 - 3, index % 11 - 5)
    state /= state.norm()
    clean = torch.zeros(1 << len(pattern), dtype=torch.complex128)
    clean[0] = 1
    measurements = sum(g.name == "measure" for g in bound.circuit.instructions)
    assert measurements == len(pattern)
    for history in product((0, 1), repeat=measurements):
        torch.testing.assert_close(
            forced_branch(
                torch.kron(state, clean), bound.circuit.instructions, history
            ),
            torch.kron(matrix @ state, clean) / sqrt(2**measurements),
            atol=7e-11,
            rtol=0,
        )


@pytest.mark.parametrize("gate", ("XGate", "YGate", "ZGate", "Hadamard"))
def test_other_control_owners_are_delegated_without_context_inference(
    gate: str,
) -> None:
    from qualtran import CtrlSpec
    from qualtran.bloqs import basic_gates

    from flagquantum.compiler.formats.legacy_controlled_factories import (
        controlled_via_and_factories,
    )
    from flagquantum.compiler.formats.legacy_fixed_gate_factories import (
        fixed_gate_factories,
    )

    source = getattr(basic_gates, gate)().controlled(CtrlSpec(cvs=0))
    fallback = controlled_via_and_factories()[
        "qualtran.bloqs.mcmt.controlled_via_and.ControlledViaAnd"
    ]
    factories = fixed_gate_factories() | linear_combination_factories(
        {}, controlled_factory=fallback
    )
    record = resolve_component_program(_library(source, 0), 0, factories)
    compare_layers(record, source, atomic_leaves(record))
    bound = bind_register_program(record, ((0,), (1,)))
    torch.testing.assert_close(
        logical_operator(bound.circuit, 2),
        torch.as_tensor(source.tensor_contract(), dtype=torch.complex128),
        atol=5e-11,
        rtol=0,
    )


def test_predicate_context_resolution_and_append_without_source_imports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    from qualtran import CtrlSpec

    from flagquantum import Circuit
    from flagquantum.algorithms.primitives._block_binding import _BoundBlockOperation

    base, preparation, selection = native_oracle_fixture(False)
    positive_source, positive_preparation, positive_selection = native_oracle_fixture(
        True
    )
    context = capture_linear_combination_context(
        base, prepare_owner=lambda _: preparation, select_owner=lambda _: selection
    )
    positive = capture_linear_combination_context(
        positive_source,
        prepare_owner=lambda _: positive_preparation,
        select_owner=lambda _: positive_selection,
    )
    source = base.controlled(CtrlSpec(cvs=[1, 0]))
    library = _library(source, 0)
    expected = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "qualtran", None)
        patch.setitem(sys.modules, "cirq", None)
        record = resolve_component_program(
            library,
            0,
            linear_combination_factories(
                {"legacy_component_1": context}, positive_context=lambda _: positive
            ),
        )
        bound = bind_register_program(
            record, tuple((i,) for i in range(5)), clean=(5, 6)
        )
        circuit = Circuit(bound.circuit.n_wires)
        circuit.append(
            _BoundBlockOperation(bound.circuit.instructions, bound.circuit.n_wires)
        )
        torch.testing.assert_close(
            logical_operator(circuit.to_ir(), 5), expected, atol=7e-11, rtol=0
        )

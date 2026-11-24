"""Resolve complete walk roots using actual source signal-state contexts."""

import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch

from flagquantum import Circuit
from flagquantum.algorithms.primitives._block_binding import _BoundBlockOperation
from flagquantum.compiler.formats.legacy_component_resolution import (
    ComponentFactory,
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_qubitization_walk_factories import (
    ExactPair,
    WalkContext,
    qubitization_walk_factories,
)
from flagquantum.compiler.registers.component_control import quantum_control_signature
from flagquantum.core.ir.program import FunctionRecord, verify_program_record
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.ecosystem._default_linear_combination_oracles import _signature
from flagquantum.ecosystem._qubitization_walk_component_library import (
    capture_qubitization_walk_context,
)
from flagquantum.errors import SerializationError
from tests.unit.test_chebyshev_component import (
    assert_all_small_raw_histories,
    chebyshev_terminals,
)
from tests.unit.test_component_graph import _source
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_default_linear_combination_oracles import bind
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_legacy_block_encoding_factories import _factories, _oracles
from tests.unit.test_multiplexed_source_binding import logical_operator
from tests.unit.test_qubitization_walk_component import (
    reference_runtime as reference_runtime,
)
from tests.unit.test_qubitization_walk_component import (
    walk_fixture,
)
from tests.unit.test_qubitization_walk_file import (
    complete_frozen_sources as complete_frozen_sources,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_retained_signal_context_binds_native_ports_and_rejects_lossy_source_abi() -> (
    None
):
    from math import pi

    from flagquantum.compiler.registers.preparation.prepared_reflection_component import (
        prepared_reflection_program,
    )
    from flagquantum.compiler.registers.primitives.global_phase_component import (
        scalar_phase_program,
    )

    oracles = _oracles()
    with patch(
        "tests.unit.test_qubitization_walk_component._oracles", return_value=oracles
    ):
        source, blocks, _ = walk_fixture(False)
    retained = (Register("signal_junk", quantum_bit_type()),)
    reflection = prepared_reflection_program(
        *oracles[2],
        ("address",),
        global_phase=-1,
        scalar_phase=scalar_phase_program(pi),
        signal_junk_names=("signal_junk",),
    )
    context = capture_qubitization_walk_context(
        source,
        block_owner=lambda _: blocks,
        reflection_owner=lambda _: (reflection, reflection),
        signal_junk=retained,
    )
    assert context.source_signature == Signature(_signature(source.signature))
    for record in context.pair():
        assert not verify_program_record(record)
        bound, _ = bind(record)
        assert bound.circuit.instructions
    factories = _factories(*oracles[2:])
    factories.update(qubitization_walk_factories(lambda _: context))
    with pytest.raises(SerializationError, match="omit retained signal junk"):
        resolve_component_program(_library(source, 0), 0, factories)
    with pytest.raises(SerializationError, match="must not alias"):
        capture_qubitization_walk_context(
            source,
            block_owner=lambda _: blocks,
            reflection_owner=lambda _: (reflection, reflection),
            signal_junk=(Register("address", quantum_bit_type()),),
        )


def context_fixture(lcu: bool) -> tuple[Any, WalkContext, dict[str, ComponentFactory]]:
    from qualtran import CtrlSpec

    oracles = _oracles()
    with patch(
        "tests.unit.test_qubitization_walk_component._oracles", return_value=oracles
    ):
        source, blocks, reflection = walk_fixture(lcu)
        positive_source, positive_blocks, positive_reflection = walk_fixture(
            lcu, controlled=True
        )

    def block_owner(child: object) -> ExactPair:
        assert child == source.block_encoding
        return blocks

    def reflection_owner(child: object) -> ExactPair:
        assert child == source.reflect
        return reflection, reflection

    def positive_block_owner(child: object) -> ExactPair:
        assert child == source.block_encoding.get_ctrl_system(CtrlSpec())[0]
        return positive_blocks

    def positive_reflection_owner(child: object) -> ExactPair:
        assert child == source.reflect.get_ctrl_system(CtrlSpec())[0]
        return positive_reflection, positive_reflection

    context = capture_qubitization_walk_context(
        source,
        block_owner=block_owner,
        reflection_owner=reflection_owner,
        positive_block_owner=positive_block_owner,
        positive_reflection_owner=positive_reflection_owner,
    )
    assert positive_source.subbloq == source
    assert (source.reflect.prepare_gate != source.prepare) is lcu
    return source, context, _factories(*oracles[2:])


@pytest.mark.parametrize(
    "example", ("_walk_op", "_thc_walk_op", "_walk_op_chem_sparse")
)
def test_original_examples_metadata_control_names_and_unmodified_producer_boundaries(
    example: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from qualtran import CtrlSpec
    from qualtran._infra.adjoint import Adjoint
    from qualtran.bloqs.qubitization import qubitization_walk_operator as module

    extra = (
        Path(__file__).resolve().parents[3] / "ftqc-reference-env/openfermion-reference"
    )
    if extra.exists():
        monkeypatch.syspath_prepend(str(extra))
    pytest.importorskip("openfermion")
    original = getattr(module, example).make()
    observer = _source(Signature(()))
    for child in (
        original,
        original.block_encoding,
        original.reflect,
        original.block_encoding.get_ctrl_system(CtrlSpec())[0],
        original.reflect.get_ctrl_system(CtrlSpec())[0],
    ):
        # This is descriptor admission, not an executable child-body witness.
        assert tuple(
            observer.signature(Signature(_signature(child.signature)))
        ) == tuple(child.signature)
    control_signature = quantum_control_signature(
        Signature(_signature(original.signature)), ((quantum_bit_type(), ()),)
    )
    assert tuple(observer.signature(control_signature)) == tuple(
        original.controlled().signature
    )
    for inverse in (False, True):
        source = Adjoint(original) if inverse else original
        for depth in (0, 1, 2):
            if example == "_walk_op_chem_sparse":
                expected = (5, 7, 23) if inverse else (4, 5, 20)
                assert len(_library(source, depth).entries) == expected[depth]
            else:
                diagnostic = (
                    "Object arrays" if example == "_walk_op" else "Cannot serialize 32"
                )
                with pytest.raises(ValueError, match=diagnostic):
                    _library(source, depth)


def test_capture_rejects_wrong_pair_ports_and_incomplete_positive_ownership() -> None:
    source, context, _ = context_fixture(False)
    positive = context.positive_block_pair
    assert positive is not None
    with pytest.raises(SerializationError, match="supplied together"):
        capture_qubitization_walk_context(
            source,
            block_owner=lambda _: context.block_pair,
            reflection_owner=lambda _: context.reflection_pair,
            positive_block_owner=lambda _: positive,
        )
    with pytest.raises(SerializationError, match="matching native logical pairs"):
        capture_qubitization_walk_context(
            source,
            block_owner=lambda _: context.reflection_pair,
            reflection_owner=lambda _: context.reflection_pair,
        )


@pytest.mark.parametrize("lcu", (False, True))
@pytest.mark.parametrize(
    "values,direction",
    [(None, "forward"), (None, "after")]
    + [
        (value, direction)
        for value in (1, 0, (1, 0))
        for direction in ("forward", "before", "after")
    ],
)
def test_source_context_controls_exact_inverse_complex_histories_and_append(
    lcu: bool, values: int | tuple[int, ...] | None, direction: str
) -> None:
    from qualtran import CtrlSpec
    from qualtran._infra.adjoint import Adjoint

    original, context, factories = context_fixture(lcu)
    source = Adjoint(original) if direction == "before" else original
    if values is not None:
        source = source.controlled(CtrlSpec(cvs=values))
    if direction == "after":
        source = Adjoint(source)

    def owner(child: FunctionRecord) -> WalkContext:
        assert child.registers == context.block_pair[0].functions[0].registers
        return context

    factories.update(
        qubitization_walk_factories(
            owner, adjoint_factory=factories["qualtran._infra.adjoint.Adjoint"]
        )
    )
    record = resolve_component_program(_library(source, 0), 0, factories)
    assert not verify_program_record(record)
    compare_layers(record, source, chebyshev_terminals(record))
    with patch.dict(sys.modules, {"qualtran": None, "cirq": None}):
        bound, logical = bind(record)
        circuit = Circuit(bound.circuit.n_wires).append(
            _BoundBlockOperation(bound.circuit.instructions, bound.circuit.n_wires)
        )
        assert circuit.to_ir().instructions == bound.circuit.instructions
    expected = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
    torch.testing.assert_close(
        logical_operator(bound.circuit, logical), expected, atol=3e-11, rtol=0
    )
    assert_all_small_raw_histories(bound.circuit, logical, expected)

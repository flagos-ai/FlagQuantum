"""Original QPE constructors retain source powers and exact prepared-image erasure."""

from itertools import product
from math import sqrt
from typing import Any

import pytest
import torch

from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_textbook_qpe_factories import (
    textbook_qpe_library_factories,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.errors import SerializationError
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_component_graph import _native_trace, _source_trace
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_multi_control_component import atomic_leaves
from tests.unit.test_multiplexed_source_binding import logical_operator
from tests.unit.test_register_component_calls import SourceCalls

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def _resolve(source: Any, depth: int = 0) -> ProgramRecord:
    return resolve_component_program(
        _library(source, depth), 0, textbook_qpe_library_factories()
    )


@pytest.mark.parametrize(
    "name",
    (
        "_textbook_qpe_small",
        "_textbook_qpe_using_m_bits",
        "_textbook_qpe_from_precision_and_delta",
        "_textbook_qpe_from_standard_deviation_eps",
    ),
)
@pytest.mark.parametrize("inverse", (False, True))
def test_all_original_qpe_constructor_examples(name: str, inverse: bool) -> None:
    from qualtran.bloqs.phase_estimation import text_book_qpe as module

    source = getattr(module, name).make()
    if inverse:
        source = source.adjoint()
    record = _resolve(source)
    assert not verify_program_record(record)
    deeper = _resolve(source, 1)
    assert not verify_program_record(deeper)
    assert deeper.functions[0].registers == record.functions[0].registers
    if record.functions[0].blocks:
        assert _native_trace(deeper) == _native_trace(record)
    else:
        assert not deeper.functions[0].blocks
    assert (
        record.functions[0].registers
        == _library(source).entries[0].declaration.registers
    )
    if name != "_textbook_qpe_small":
        assert not record.functions[0].blocks
        with pytest.raises(ValueError):
            bind_register_program(record, ())
    else:
        assert _native_trace(record) == _source_trace(
            SourceCalls, source.decompose_bloq()
        )


@pytest.mark.parametrize(
    "width,reverse,qft_inverse", tuple(product((1, 2, 3), (False, True), (False, True)))
)
def test_qpe_exact_source_calls_complex_isometry_and_inverse_cleanup(
    width: int, reverse: bool, qft_inverse: bool
) -> None:
    from qualtran.bloqs.basic_gates.rotation import ZPowGate
    from qualtran.bloqs.phase_estimation.qpe_window_state import RectangularWindowState
    from qualtran.bloqs.phase_estimation.text_book_qpe import TextbookQPE
    from qualtran.bloqs.qft.qft_text_book import QFTTextBook

    fourier = QFTTextBook(width, with_reverse=reverse)
    if qft_inverse:
        fourier = fourier.adjoint()
    source = TextbookQPE(ZPowGate(0.468), RectangularWindowState(width), fourier)
    record = _resolve(source)
    inverse = _resolve(source.adjoint())
    assert not verify_program_record(record)
    assert not verify_program_record(inverse)
    for obj, program in ((source, record), (source.adjoint(), inverse)):
        assert _native_trace(program) == _source_trace(
            SourceCalls, obj.decompose_bloq()
        )
        compare_layers(program, obj, atomic_leaves(program))
    bound = bind_register_program(
        record,
        ((width,),),
        clean=(*range(width), width + 1),
        allow_workspace_outputs=True,
    )
    expected = torch.tensor(source.tensor_contract(), dtype=torch.complex128)
    torch.testing.assert_close(
        logical_operator(bound.circuit, width + 1)[:, :2], expected, atol=8e-12, rtol=0
    )
    erased = bind_register_program(
        inverse,
        (tuple(range(width)), (width,)),
        clean=(width + 1,),
        input_ownership=tuple((i, False) for i in range(width)),
    )
    measurements = sum(
        i.name == "measure"
        for i in (*bound.circuit.instructions, *erased.circuit.instructions)
    )
    # Free is the prepared-image promise. Test that domain rather than treating
    # it as an arbitrary-input projection or discarding nonzero phase amplitude.
    initial = torch.zeros(1 << (width + 2), dtype=torch.complex128)
    initial[0] = 0.6
    initial[2] = torch.tensor(0.8j, dtype=torch.complex128)
    histories = (
        product((0, 1), repeat=measurements) if width <= 2 else ((0,) * measurements,)
    )
    for history in histories:
        result = forced_branch(
            initial,
            (*bound.circuit.instructions, *erased.circuit.instructions),
            history,
        ) * sqrt(2**measurements)
        torch.testing.assert_close(result, initial, atol=8e-12, rtol=0)


def test_qpe_rejects_unowned_same_signature_children() -> None:
    from qualtran.bloqs.basic_gates import XGate, ZPowGate
    from qualtran.bloqs.phase_estimation.qpe_window_state import RectangularWindowState
    from qualtran.bloqs.phase_estimation.text_book_qpe import TextbookQPE

    from flagquantum.compiler.formats.legacy_fixed_gate_factories import (
        fixed_gate_factories,
    )

    factories = {**textbook_qpe_library_factories(), **fixed_gate_factories()}
    with pytest.raises(SerializationError, match="context"):
        resolve_component_program(
            _library(TextbookQPE(XGate(), RectangularWindowState(2))), 0, factories
        )
    source = TextbookQPE(ZPowGate(0.468), RectangularWindowState(2))
    library = _library(source)
    factory = textbook_qpe_library_factories()[
        "qualtran.bloqs.phase_estimation.text_book_qpe.TextbookQPE"
    ]
    with pytest.raises(SerializationError):
        factory("bad", library.entries[0].declaration.registers, {})


def test_qpe_source_free_resolution_and_sdk_free_append(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    from qualtran.bloqs.phase_estimation.text_book_qpe import _textbook_qpe_small

    from flagquantum import Circuit
    from flagquantum.algorithms.primitives._block_binding import _BoundBlockOperation

    source = _textbook_qpe_small.make()
    expected = torch.tensor(source.tensor_contract(), dtype=torch.complex128)
    library = _library(source)
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "qualtran", None)
        patch.setitem(sys.modules, "cirq", None)
        patch.setitem(sys.modules, "flagquantum.algorithms", None)
        patch.setitem(sys.modules, "flagquantum.simulation", None)
        record = resolve_component_program(library, 0, textbook_qpe_library_factories())
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "qualtran", None)
        patch.setitem(sys.modules, "cirq", None)
        bound = bind_register_program(
            record, ((3,),), clean=(0, 1, 2, 4), allow_workspace_outputs=True
        )
        circuit = Circuit(bound.circuit.n_wires)
        circuit.append(
            _BoundBlockOperation(bound.circuit.instructions, bound.circuit.n_wires)
        )
        assert (
            circuit.to_ir().to_dict()["instructions"]
            == bound.circuit.to_dict()["instructions"]
        )
        actual = logical_operator(bound.circuit, 4)[:, :2]
    torch.testing.assert_close(actual, expected, atol=8e-12, rtol=0)


def test_exact_constructor_context_is_local_to_one_compilation() -> None:
    first = textbook_qpe_library_factories()
    second = textbook_qpe_library_factories()
    unitary = first["qualtran.bloqs.basic_gates.rotation.ZPowGate"](
        "u", None, {"exponent": 0.468}
    )
    window = first[
        "qualtran.bloqs.phase_estimation.qpe_window_state.RectangularWindowState"
    ]("window", None, {"bitsize": 2})
    assert isinstance(window, ProgramRecord)
    arguments = {"unitary": unitary, "ctrl_state_prep": window.functions[0]}
    assert not isinstance(unitary, ProgramRecord)
    # Same ports are insufficient: provenance belongs to this compilation.
    identifier = "qualtran.bloqs.phase_estimation.text_book_qpe.TextbookQPE"
    first[identifier]("qpe", None, arguments)
    with pytest.raises(SerializationError, match="context"):
        second[identifier]("qpe", None, arguments)

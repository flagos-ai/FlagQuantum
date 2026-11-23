"""Textbook Fourier libraries retain their typed graph, feedback and model scope."""

from collections import Counter
from itertools import product

import pytest
import sympy
import torch

from flagquantum.compiler.formats.legacy_component_resolution import (
    ResolvedArgument,
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_qft_factories import qft_factories
from flagquantum.compiler.registers.phase.textbook_qft_component import (
    textbook_qft_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import FunctionRecord, verify_program_record
from flagquantum.core.value_math import SymbolicInt
from flagquantum.errors import SerializationError
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_component_graph import _native_trace, _source_trace
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_register_component_calls import SourceCalls

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "width,reverse", tuple(product((1, 2, 3, 5, 8), (False, True)))
)
def test_textbook_constructor_preserves_complete_bound_graph_and_source(
    width: int, reverse: bool
) -> None:
    from qualtran.bloqs.qft.qft_text_book import QFTTextBook
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    source = QFTTextBook(width, reverse)
    library = _library(source, 0)
    record = resolve_component_program(library, 0, qft_factories())
    for depth in (1, 2):
        assert (
            resolve_component_program(_library(source, depth), 0, qft_factories())
            == record
        )
    assert verify_program_record(record) == ()
    assert record.functions[0].registers == library.entries[0].declaration.registers
    wires = tuple(reversed(range(width)))
    clean = (width + 1,) if width > 1 else ()
    bound = bind_register_program(record, (wires,), clean=clean)
    previous = bind_register_program(
        textbook_qft_program(width, with_reverse=reverse), (wires,), clean=clean
    )
    assert bound.circuit == previous.circuit
    assert bound.outputs == previous.outputs == (wires,)
    assert bound.events == previous.events
    assert _native_trace(record, record.functions[0].name) == _source_trace(
        SourceCalls, source.decompose_bloq()
    )
    pairs = width * (width - 1) // 2
    counts = Counter(g.name for g in bound.circuit.instructions)
    for gate, count in {
        "h": width + 3 * pairs,
        "cx": 6 * pairs,
        "t": 2 * pairs,
        "tdg": 2 * pairs,
        "s": pairs,
        "phase": pairs,
        "measure": pairs,
        "cz": pairs,
        "reset": pairs,
        "swap": width // 2 if reverse else 0,
    }.items():
        assert counts[gate] == count
    model = get_cost_value(source, QECGatesCost())
    assert model.and_bloq == model.measurement == pairs
    assert model.t == max(width - 2, 0)
    if width > 3:
        return
    # Include a coherent spectator and inspect every unnormalized erasure branch.
    total = width + 1 + len(clean)
    values = torch.arange(1, 2 ** (width + 1) + 1, dtype=torch.float64)
    amplitudes = torch.complex(values, values.flip(0))
    amplitudes /= amplitudes.norm()
    cirq = pytest.importorskip("cirq")

    order = (*wires, width, *clean)
    restore = tuple(order.index(wire) for wire in range(total))
    initial_view = torch.zeros((2**width, 2, 2 ** len(clean)), dtype=torch.complex128)
    initial_view[:, :, 0] = amplitudes.reshape(2**width, 2)
    target_view = torch.zeros_like(initial_view)
    matrix = torch.tensor(cirq.unitary(source), dtype=torch.complex128)
    target_view[:, :, 0] = matrix @ amplitudes.reshape(2**width, 2)
    initial = initial_view.reshape((2,) * total).permute(restore).flatten()
    target = target_view.reshape((2,) * total).permute(restore).flatten()
    for history in product((0, 1), repeat=pairs):
        actual = forced_branch(initial, bound.circuit.instructions, history)
        torch.testing.assert_close(
            actual, target / 2 ** (pairs / 2), atol=1e-12, rtol=0
        )
        if clean:
            torch.testing.assert_close(
                actual.reshape(-1, 2)[:, 1],
                torch.zeros(2 ** (total - 1), dtype=actual.dtype),
                atol=1e-12,
                rtol=0,
            )


@pytest.mark.parametrize("width", (0, sympy.Symbol("n"), sympy.Integer(3)))
def test_direct_symbolic_and_empty_declarations_retain_source_refusal(
    width: SymbolicInt,
) -> None:
    from qualtran.bloqs.qft.qft_text_book import QFTTextBook

    with pytest.raises(TypeError):
        QFTTextBook(width).decompose_bloq()
    factory = next(iter(qft_factories().values()))
    record = factory("declared", None, {"bitsize": width})
    assert isinstance(record, FunctionRecord) and not record.blocks
    with pytest.raises(ValueError, match="bodies"):
        from flagquantum.core.ir.program import ProgramRecord

        bind_register_program(ProgramRecord((record,)), ())


def test_textbook_original_examples_wire_defaults_and_invalid_metadata() -> None:
    from qualtran.bloqs.qft.qft_text_book import _qft_text_book, _symbolic_qft

    for example in (_qft_text_book, _symbolic_qft):
        library = _library(example.make(), 0)
        result = resolve_component_program(library, 0, qft_factories())
        assert result.functions[0].registers == library.entries[0].declaration.registers
    factory = next(iter(qft_factories().values()))
    default = factory("default", None, {"bitsize": 3})
    explicit = factory("default", None, {"bitsize": 3, "with_reverse": 1})
    assert default == explicit
    cases: tuple[dict[str, ResolvedArgument], ...] = (
        {},
        {"bitsize": True},
        {"bitsize": 3, "other": 1},
        {"bitsize": 3, "with_reverse": 2},
    )
    for args in cases:
        with pytest.raises(SerializationError):
            factory("bad", None, args)
    with pytest.raises(SerializationError, match="signature"):
        factory("bad", (), {"bitsize": 3})

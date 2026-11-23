"""Stored gradient Fourier roots retain recursive operands and work ownership."""

from collections import Counter
from itertools import product

import pytest
import sympy
import torch

from flagquantum.algorithms.primitives.qft_phase_gradient import (
    qft_phase_gradient_requirements,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    ResolvedArgument,
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_gradient_qft_factories import (
    gradient_qft_factories,
)
from flagquantum.compiler.registers.phase.gradient_qft_component import (
    gradient_qft_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import (
    FunctionRecord,
    ProgramRecord,
    verify_program_record,
)
from flagquantum.core.value_math import SymbolicInt
from flagquantum.errors import SerializationError
from tests.unit.test_component_graph import _native_trace, _source_trace
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_multiplexed_source_binding import logical_operator
from tests.unit.test_register_component_calls import SourceCalls

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "width,reverse", tuple(product((1, 2, 3, 4, 5), (False, True)))
)
def test_gradient_constructor_retains_complete_product_graph(
    width: int, reverse: bool
) -> None:
    from qualtran.bloqs.qft.qft_phase_gradient import QFTPhaseGradient

    source = QFTPhaseGradient(width, reverse)
    library = _library(source, 0)
    record = resolve_component_program(library, 0, gradient_qft_factories())
    assert verify_program_record(record) == ()
    assert record.functions[0].registers == library.entries[0].declaration.registers
    for depth in (1, 2):
        assert (
            resolve_component_program(
                _library(source, depth), 0, gradient_qft_factories()
            )
            == record
        )
    assert _native_trace(record, record.functions[0].name) == _source_trace(
        SourceCalls, source.decompose_bloq()
    )
    wires = tuple(reversed(range(2 * width)))
    inputs = (wires[:width], wires[width:])
    work = qft_phase_gradient_requirements(width)[0]
    clean = tuple(range(2 * width, 2 * width + work))
    bound = bind_register_program(record, inputs, clean=clean)
    accepted = bind_register_program(
        gradient_qft_program(width, with_reverse=reverse), inputs, clean=clean
    )
    assert bound.circuit == accepted.circuit
    assert bound.events == accepted.events
    assert bound.outputs == accepted.outputs == inputs
    if width == 3 and reverse:
        counts = Counter(g.name for g in bound.circuit.instructions)
        assert counts["t"] + counts["tdg"] == 24
        assert counts["cx"] == 46
        assert counts["measure"] == counts["reset"] == 6
    if width <= 3:
        cirq = pytest.importorskip("cirq")
        contiguous = bind_register_program(
            record,
            (tuple(range(width)), tuple(range(width, 2 * width))),
            clean=clean,
        )
        torch.testing.assert_close(
            logical_operator(contiguous.circuit, 2 * width),
            torch.tensor(cirq.unitary(source), dtype=torch.complex128),
            atol=1e-11,
            rtol=0,
        )


@pytest.mark.parametrize("width", (0, sympy.Symbol("n"), sympy.Integer(3)))
def test_gradient_direct_empty_and_symbolic_source_boundaries(
    width: SymbolicInt,
) -> None:
    from qualtran.bloqs.qft.qft_phase_gradient import QFTPhaseGradient

    with pytest.raises((KeyError, TypeError)):
        QFTPhaseGradient(width).decompose_bloq()
    factory = next(iter(gradient_qft_factories().values()))
    declaration = factory("declared", None, {"bitsize": width})
    assert isinstance(declaration, FunctionRecord) and not declaration.blocks
    with pytest.raises(ValueError, match="bodies"):
        bind_register_program(ProgramRecord((declaration,)), ())


def test_gradient_original_example_default_and_metadata_validation() -> None:
    from qualtran.bloqs.qft.qft_phase_gradient import _qft_phase_gradient_small

    source = _qft_phase_gradient_small.make()
    library = _library(source, 0)
    record = resolve_component_program(library, 0, gradient_qft_factories())
    assert record.functions[0].registers == library.entries[0].declaration.registers
    factory = next(iter(gradient_qft_factories().values()))
    assert factory("default", None, {"bitsize": 3}) == factory(
        "default", None, {"bitsize": 3, "with_reverse": 1}
    )
    cases: tuple[dict[str, ResolvedArgument], ...] = (
        {},
        {"bitsize": True},
        {"bitsize": 3, "other": 1},
        {"bitsize": 3, "with_reverse": 2},
    )
    for arguments in cases:
        with pytest.raises(SerializationError):
            factory("bad", None, arguments)
    with pytest.raises(SerializationError, match="signature"):
        factory("bad", (), {"bitsize": 3})

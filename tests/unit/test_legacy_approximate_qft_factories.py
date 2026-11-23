"""Stored approximate Fourier roots retain precision slices and work ownership."""

import pytest
import sympy
import torch

from flagquantum.algorithms.primitives.approximate_qft import (
    approximate_qft_workspace_size,
)
from flagquantum.compiler.formats.legacy_approximate_qft_factories import (
    approximate_qft_factories,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    ResolvedArgument,
    resolve_component_program,
)
from flagquantum.compiler.registers.phase.approximate_qft_component import (
    approximate_qft_program,
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
    "width,precision,reverse",
    tuple(
        (n, b, r)
        for n, b in (
            (1, 0),
            (1, 1),
            (2, 1),
            (2, 2),
            (3, 2),
            (3, 3),
            (3, 5),
            (4, 3),
            (5, 2),
        )
        for r in (False, True)
    ),
)
def test_approximate_constructor_retains_complete_addition_graph(
    width: int, precision: int, reverse: bool
) -> None:
    from qualtran.bloqs.qft.approximate_qft import ApproximateQFT

    source = ApproximateQFT(width, precision, reverse)
    library = _library(source, 0)
    record = resolve_component_program(library, 0, approximate_qft_factories())
    assert verify_program_record(record) == ()
    assert record.functions[0].registers == library.entries[0].declaration.registers
    for depth in (1, 2):
        assert (
            resolve_component_program(
                _library(source, depth), 0, approximate_qft_factories()
            )
            == record
        )
    assert _native_trace(record, record.functions[0].name) == _source_trace(
        SourceCalls, source.decompose_bloq()
    )
    wires = tuple(reversed(range(width + precision)))
    inputs = (wires[:width], wires[width:]) if precision else (wires,)
    work = approximate_qft_workspace_size(width, precision)
    clean = tuple(range(width + precision, width + precision + work))
    bound = bind_register_program(record, inputs, clean=clean)
    accepted = bind_register_program(
        approximate_qft_program(width, precision, with_reverse=reverse),
        inputs,
        clean=clean,
    )
    assert bound.circuit == accepted.circuit
    assert bound.events == accepted.events
    assert bound.outputs == accepted.outputs == inputs
    if width + precision <= 6:
        cirq = pytest.importorskip("cirq")
        contiguous_inputs: tuple[tuple[int, ...], ...] = (tuple(range(width)),)
        if precision:
            contiguous_inputs += (tuple(range(width, width + precision)),)
        contiguous = bind_register_program(record, contiguous_inputs, clean=clean)
        torch.testing.assert_close(
            logical_operator(contiguous.circuit, width + precision),
            torch.tensor(cirq.unitary(source), dtype=torch.complex128),
            atol=1e-11,
            rtol=0,
        )


@pytest.mark.parametrize("width", (0, sympy.Symbol("n"), sympy.Integer(3)))
def test_approximate_direct_empty_and_symbolic_source_boundaries(
    width: SymbolicInt,
) -> None:
    from qualtran.bloqs.qft.approximate_qft import ApproximateQFT

    with pytest.raises((KeyError, TypeError)):
        ApproximateQFT(width, 2).decompose_bloq()
    factory = next(iter(approximate_qft_factories().values()))
    declaration = factory("declared", None, {"bitsize": width, "phase_bitsize": 2})
    assert isinstance(declaration, FunctionRecord) and not declaration.blocks
    with pytest.raises(ValueError, match="bodies"):
        bind_register_program(ProgramRecord((declaration,)), ())


def test_approximate_original_example_default_and_metadata_validation() -> None:
    from qualtran.bloqs.qft.approximate_qft import _approximate_qft_small

    source = _approximate_qft_small.make()
    library = _library(source, 0)
    record = resolve_component_program(library, 0, approximate_qft_factories())
    assert record.functions[0].registers == library.entries[0].declaration.registers
    factory = next(iter(approximate_qft_factories().values()))
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


def test_approximate_symbolic_and_missing_gradient_declarations() -> None:
    from qualtran.bloqs.qft.approximate_qft import ApproximateQFT

    factory = next(iter(approximate_qft_factories().values()))
    n, b = sympy.symbols("n b", integer=True, positive=True)
    for source in (ApproximateQFT(n, b), ApproximateQFT(3, 0)):
        with pytest.raises((KeyError, TypeError)):
            source.decompose_bloq()
        record = resolve_component_program(
            _library(source, 0), 0, approximate_qft_factories()
        )
        assert not record.functions[0].blocks
        with pytest.raises(ValueError, match="bodies"):
            bind_register_program(record, ())
    default = factory("default", None, {"bitsize": n})
    explicit = factory(
        "default",
        None,
        {"bitsize": n, "phase_bitsize": ApproximateQFT(n).phase_bitsize},
    )
    assert default == explicit


def test_approximate_epsilon_example_uses_existing_full_program() -> None:
    from qualtran.bloqs.qft.approximate_qft import _approximate_qft_from_epsilon

    source = _approximate_qft_from_epsilon.make()
    record = resolve_component_program(
        _library(source, 0), 0, approximate_qft_factories()
    )
    factory = next(iter(approximate_qft_factories().values()))
    expected = factory(
        "main", None, {"bitsize": 50, "phase_bitsize": int(source.phase_bitsize)}
    )
    assert record == expected
    assert _native_trace(record) == _source_trace(SourceCalls, source.decompose_bloq())

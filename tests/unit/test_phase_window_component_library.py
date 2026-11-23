"""Stored phase windows preserve output ownership and original leaf boundaries."""

import pytest
import sympy
import torch

from flagquantum.algorithms.primitives.phase_estimation_window import (
    rectangular_window_program,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import verify_program_record
from flagquantum.ecosystem._phase_window_component_library import (
    phase_window_library_factories,
)
from flagquantum.errors import SerializationError
from tests.unit.test_component_graph import _native_trace, _source_trace
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_multiplexed_source_binding import logical_operator
from tests.unit.test_register_component_calls import SourceCalls

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("width", (1, 2, 3, 5, sympy.Symbol("n", positive=True)))
def test_rectangular_complete_source_graph_and_clean_output(
    width: int | sympy.Expr,
) -> None:
    from qualtran.bloqs.phase_estimation.qpe_window_state import RectangularWindowState

    source = RectangularWindowState(width)
    record = resolve_component_program(
        _library(source, 0), 0, phase_window_library_factories()
    )
    assert verify_program_record(record) == ()
    for depth in (1, 2):
        assert (
            resolve_component_program(
                _library(source, depth), 0, phase_window_library_factories()
            )
            == record
        )
    assert _native_trace(record) == _source_trace(SourceCalls, source.decompose_bloq())
    if type(width) is not int:
        with pytest.raises(ValueError):
            bind_register_program(record, ())
        return
    clean = tuple(reversed(range(width)))
    bound = bind_register_program(record, (), clean=clean, allow_workspace_outputs=True)
    accepted = bind_register_program(
        rectangular_window_program(width), (), clean=clean, allow_workspace_outputs=True
    )
    assert bound == accepted
    assert bound.outputs == (clean,)
    assert [g.name for g in bound.circuit.instructions] == ["h"] * width
    contiguous = bind_register_program(
        record, (), clean=tuple(range(width)), allow_workspace_outputs=True
    )
    torch.testing.assert_close(
        logical_operator(contiguous.circuit, width)[:, 0],
        torch.tensor(source.tensor_contract(), dtype=torch.complex128),
        atol=3e-12,
        rtol=0,
    )


@pytest.mark.parametrize(
    "width,alpha",
    ((0, 0.0), (1, 2.0), (3, 1.5), (sympy.Symbol("n"), sympy.Symbol("alpha"))),
)
def test_kaiser_retains_original_tensor_only_boundary(
    width: int | sympy.Expr, alpha: float | sympy.Expr
) -> None:
    from qualtran import DecomposeNotImplementedError
    from qualtran.bloqs.phase_estimation.kaiser_window_state import KaiserWindowState

    source = KaiserWindowState(width, alpha)
    with pytest.raises(DecomposeNotImplementedError):
        source.decompose_bloq()
    record = resolve_component_program(
        _library(source, 0), 0, phase_window_library_factories()
    )
    assert verify_program_record(record) == ()
    assert (
        record.functions[0].registers
        == _library(source, 0).entries[0].declaration.registers
    )
    assert not record.functions[0].blocks
    with pytest.raises(ValueError, match="bodies"):
        bind_register_program(record, ())


def test_phase_window_original_examples_and_invalid_metadata() -> None:
    from qualtran.bloqs.phase_estimation.kaiser_window_state import (
        _kaiser_window_state_small,
        _kaiser_window_state_symbolic,
    )
    from qualtran.bloqs.phase_estimation.qpe_window_state import (
        _rectangular_window_state_small,
        _rectangular_window_state_symbolic,
    )

    for example in (
        _kaiser_window_state_small,
        _kaiser_window_state_symbolic,
        _rectangular_window_state_small,
        _rectangular_window_state_symbolic,
    ):
        library = _library(example.make(), 0)
        record = resolve_component_program(library, 0, phase_window_library_factories())
        assert record.functions[0].registers == library.entries[0].declaration.registers
    for identifier, factory in phase_window_library_factories().items():
        for arguments in ({}, {"bitsize": True}, {"bitsize": 3, "extra": 1}):
            with pytest.raises(SerializationError):
                factory("bad", None, arguments)
        arguments = (
            {"bitsize": 3, "alpha": 2} if "kaiser" in identifier else {"bitsize": 3}
        )
        with pytest.raises(SerializationError, match="signature"):
            factory("bad", (), arguments)


def test_rectangular_empty_graph_keeps_original_tensor_and_storage_failures() -> None:
    from qualtran.bloqs.phase_estimation.qpe_window_state import RectangularWindowState

    source = RectangularWindowState(0)
    record = resolve_component_program(
        _library(source, 0), 0, phase_window_library_factories()
    )
    assert _native_trace(record) == _source_trace(SourceCalls, source.decompose_bloq())
    with pytest.raises(ValueError, match="not enough values"):
        source.tensor_contract()
    with pytest.raises(ValueError, match="unsupported executable register"):
        bind_register_program(record, (), allow_workspace_outputs=True)

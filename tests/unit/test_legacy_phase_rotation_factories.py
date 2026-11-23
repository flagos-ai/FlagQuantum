"""Stored kickback rotations retain source calls and accepted execution paths."""

from collections import Counter
from itertools import product

import pytest
import sympy
import torch

from flagquantum.compiler.formats.legacy_component_resolution import (
    ResolvedArgument,
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_phase_rotation_factories import (
    phase_rotation_factories,
)
from flagquantum.compiler.registers.phase.rz_phase_gradient_component import (
    rz_via_phase_gradient_program,
)
from flagquantum.compiler.registers.phase.zpow_phase_gradient_component import (
    zpow_via_phase_gradient_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import (
    FunctionRecord,
    ProgramRecord,
    verify_program_record,
)
from flagquantum.core.ir.program.register_types import fixed_point_register_type
from flagquantum.errors import SerializationError
from tests.unit.test_component_graph import _native_trace, _source_trace
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_multiplexed_source_binding import logical_operator
from tests.unit.test_register_component_calls import SourceCalls

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("constant,width", tuple(product((False, True), (2, 3, 4))))
def test_complete_stored_rotation_graph_and_accepted_execution(
    constant: bool, width: int
) -> None:
    from qualtran import QFxp
    from qualtran.bloqs.rotations.rz_via_phase_gradient import RzViaPhaseGradient
    from qualtran.bloqs.rotations.zpow_via_phase_gradient import (
        ZPowConstViaPhaseGradient,
    )

    source = (
        ZPowConstViaPhaseGradient(0.75, width)
        if constant
        else RzViaPhaseGradient(QFxp(width, width), QFxp(width, width))
    )
    record = resolve_component_program(
        _library(source, 0), 0, phase_rotation_factories()
    )
    assert verify_program_record(record) == ()
    assert _native_trace(record) == _source_trace(SourceCalls, source.decompose_bloq())
    for depth in (1, 2):
        assert (
            resolve_component_program(
                _library(source, depth), 0, phase_rotation_factories()
            )
            == record
        )
    dtype = fixed_point_register_type(width, width)
    accepted = (
        zpow_via_phase_gradient_program(0.75, width)
        if constant
        else rz_via_phase_gradient_program(dtype, dtype)
    )
    logical = 1 + width * (1 if constant else 2)
    wires = tuple(reversed(range(logical)))
    ports = (
        ((wires[0],), wires[1:])
        if constant
        else ((wires[0],), wires[1 : width + 1], wires[width + 1 :])
    )
    work = 2 * width - 1 if constant else width - 1
    clean = tuple(range(logical, logical + work))
    bound = bind_register_program(record, ports, clean=clean)
    reference = bind_register_program(accepted, ports, clean=clean)
    assert bound.circuit == reference.circuit
    assert bound.events == reference.events
    assert bound.outputs == reference.outputs == ports
    counts = Counter(op.name for op in bound.circuit.instructions)
    assert counts["measure"] == counts["reset"] == width - 1
    assert counts["t"] + counts["tdg"] == 4 * (width - 1)
    if width <= 3:
        contiguous = (
            ((0,), tuple(range(1, logical)))
            if constant
            else ((0,), tuple(range(1, width + 1)), tuple(range(width + 1, logical)))
        )
        bound = bind_register_program(record, contiguous, clean=clean)
        torch.testing.assert_close(
            logical_operator(bound.circuit, logical),
            torch.tensor(source.tensor_contract(), dtype=torch.complex128),
            atol=1e-11,
            rtol=0,
        )


@pytest.mark.parametrize("signed", (False, True))
def test_source_success_logical_rz_calls_survive_unbound_storage(signed: bool) -> None:
    from qualtran import QFxp, QUInt
    from qualtran.bloqs.rotations.rz_via_phase_gradient import RzViaPhaseGradient

    dtype = (
        QFxp(3, 2, signed=True)
        if signed
        else QUInt(sympy.Symbol("n", positive=True, integer=True))
    )
    source = RzViaPhaseGradient(dtype, dtype)
    record = resolve_component_program(
        _library(source, 0), 0, phase_rotation_factories()
    )
    assert verify_program_record(record) == ()
    assert _native_trace(record) == _source_trace(SourceCalls, source.decompose_bloq())
    assert len(record.functions[0].blocks[0].operations) == 5
    with pytest.raises(ValueError):
        bind_register_program(record, ((0,), (1, 2, 3), (4, 5, 6)), clean=(7, 8))


def test_symbolic_precision_and_original_example_dispositions() -> None:
    from qualtran.bloqs.rotations.zpow_via_phase_gradient import (
        _zpow_const_via_phase_grad,
        _zpow_const_via_phase_grad_symb_angle,
        _zpow_const_via_phase_grad_symb_prec,
    )

    for example in (
        _zpow_const_via_phase_grad,
        _zpow_const_via_phase_grad_symb_prec,
        _zpow_const_via_phase_grad_symb_angle,
    ):
        source = example.make()
        library = _library(source, 0)
        record = resolve_component_program(library, 0, phase_rotation_factories())
        assert verify_program_record(record) == ()
        assert record.functions[0].registers == library.entries[0].declaration.registers
        if isinstance(source.exponent, sympy.Expr):
            assert not record.functions[0].blocks
        else:
            assert _native_trace(record) == _source_trace(
                SourceCalls, source.decompose_bloq()
            )


def test_empty_negative_and_invalid_metadata_boundaries() -> None:
    from qualtran.bloqs.rotations.zpow_via_phase_gradient import (
        ZPowConstViaPhaseGradient,
    )

    factories = phase_rotation_factories()
    rz, zpow = tuple(factories.values())
    empty = zpow("empty", None, {"exponent": 0.5, "phase_grad_bitsize": 0})
    assert isinstance(empty, FunctionRecord) and not empty.blocks
    with pytest.raises(ValueError):
        bind_register_program(ProgramRecord((empty,)), ())
    with pytest.raises(ValueError):
        ZPowConstViaPhaseGradient(-0.5, 3).decompose_bloq()
    with pytest.raises(ValueError):
        zpow("negative", None, {"exponent": -0.5, "phase_grad_bitsize": 3})
    cases: tuple[dict[str, ResolvedArgument], ...] = (
        {},
        {"exponent": True, "phase_grad_bitsize": 3},
        {"exponent": 0.5, "phase_grad_bitsize": True},
        {"exponent": 0.5, "phase_grad_bitsize": 3, "other": 0},
    )
    for arguments in cases:
        with pytest.raises(SerializationError):
            zpow("invalid", None, arguments)
    with pytest.raises(SerializationError):
        rz("invalid", None, {"angle_dtype": 3, "phasegrad_dtype": 3})
    with pytest.raises(SerializationError, match="signature"):
        zpow("invalid", (), {"exponent": 0.5, "phase_grad_bitsize": 3})


@pytest.mark.parametrize("exponent", (-0.5, 0.75, 2.5))
def test_symbolic_width_preserves_unencoded_load_domain(exponent: float) -> None:
    from qualtran.bloqs.rotations.zpow_via_phase_gradient import (
        ZPowConstViaPhaseGradient,
    )

    source = ZPowConstViaPhaseGradient(
        exponent, sympy.Symbol("n", positive=True, integer=True)
    )
    record = resolve_component_program(
        _library(source, 0), 0, phase_rotation_factories()
    )
    assert verify_program_record(record) == ()
    assert _native_trace(record) == _source_trace(SourceCalls, source.decompose_bloq())
    assert len(record.functions[0].blocks[0].operations) == 5
    with pytest.raises(ValueError):
        bind_register_program(record, ((0,), (1, 2, 3)), clean=tuple(range(4, 9)))

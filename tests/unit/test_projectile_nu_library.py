"""Original momentum constructors retain opaque leaves and literal flag wiring."""

from importlib import import_module
from typing import Any

import pytest

from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.registers.preparation.uniform_preparation_component import (
    uniform_preparation_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.core.ir.program.registers import RegisterSide
from flagquantum.ecosystem._nu_preparation_library import nu_preparation_factories
from tests.unit.test_cirq_decomposition_file import locked_source as locked_source
from tests.unit.test_component_graph import _native_trace, _source_trace
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_register_component_calls import SourceCalls

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]

_BASE = "qualtran.bloqs.chemistry.pbc.first_quantization."


def _named_inputs(rows: list[object]) -> list[object]:
    result: list[object] = []
    for row in rows:
        assert isinstance(row, tuple) and len(row) == 3
        index, inputs, outputs = row
        # Source graph dictionaries do not preserve signature insertion order.
        # Keep every producer, port, array index and output order intact.
        result.append((index, dict(inputs), outputs))
    return result


def _source(kind: str, width: int, *, adjoint: bool = False) -> Any:
    module = import_module(
        _BASE + ("projectile.prepare_nu" if kind.endswith("WithProj") else "prepare_nu")
    )
    if kind == "PrepareMuUnaryEncodedOneHotWithProj":
        return getattr(module, kind)(width, max(1, width - 1), is_adjoint=adjoint)
    if kind == "PrepareMuUnaryEncodedOneHot":
        return getattr(module, kind)(width)
    if kind == "TestNuInequality":
        return getattr(module, kind)(width, 2, is_adjoint=adjoint)
    return getattr(module, kind)(width, is_adjoint=adjoint)


@pytest.mark.parametrize("width", (1, 2, 4))
@pytest.mark.parametrize(
    "kind,adjoint",
    [("PrepareMuUnaryEncodedOneHot", False)]
    + [
        (kind, inverse)
        for kind in (
            "PrepareMuUnaryEncodedOneHotWithProj",
            "PrepareNuSuperPositionState",
            "FlagZeroAsFailure",
            "TestNuLessThanMu",
            "TestNuInequality",
        )
        for inverse in (False, True)
    ],
)
def test_original_leaf_serialization_and_directional_ownership(
    width: int, kind: str, adjoint: bool
) -> None:
    source = _source(kind, width, adjoint=adjoint)
    factories = nu_preparation_factories()
    for depth in (0, 1, 2):
        library = _library(source, depth)
        record = resolve_component_program(library, 0, factories)
        assert verify_program_record(record) == ()
        assert record.functions[0].registers == library.entries[0].declaration.registers
        assert not record.functions[0].blocks
    if kind == "PrepareMuUnaryEncodedOneHotWithProj":
        assert record.functions[0].registers is not None
        assert record.functions[0].registers[-1].side == RegisterSide.RIGHT


@pytest.mark.parametrize("projectile", (False, True))
@pytest.mark.parametrize(
    "electron,width,sample", ((1, 2, 3), (2, 3, 4), (6, 8, 524288))
)
def test_complete_literal_root_constructor_and_original_flag_operands(
    projectile: bool, electron: int, width: int, sample: int
) -> None:
    module = import_module(
        _BASE + ("projectile.prepare_nu" if projectile else "prepare_nu")
    )
    source = (
        module.PrepareNuStateWithProj(electron, width, sample)
        if projectile
        else module.PrepareNuState(electron, sample)
    )
    factories = nu_preparation_factories()
    baseline = None
    for depth in (0, 1, 2):
        library = _library(source, depth)
        record = resolve_component_program(library, 0, factories)
        assert verify_program_record(record) == ()
        assert record.functions[0].registers == library.entries[0].declaration.registers
        assert _named_inputs(_native_trace(record)) == _named_inputs(
            _source_trace(SourceCalls, source.decompose_bloq())
        )
        if baseline is None:
            baseline = record
        else:
            assert record == baseline
    # Five physical leaves stay opaque. The sixth uniform-m child is executable
    # and must not be erased just because the analytical call graph omits it.
    assert sum(not function.blocks for function in record.functions) == 5
    assert len(source.build_call_graph(None)) == 5
    assert len(record.functions[0].blocks[0].operations) == 6
    with pytest.raises(ValueError, match="physical lowering requires component bodies"):
        bind_register_program(record, (), clean=tuple(range(100)))


def test_projectile_source_adjoint_failure_is_not_replaced_by_corrected_model() -> None:
    source_error = import_module("qualtran").BloqError

    module = import_module(_BASE + "projectile.prepare_nu")
    source = module.PrepareNuStateWithProj(1, 2, 3)
    with pytest.raises((source_error, ValueError)):
        source.adjoint().decompose_bloq()
    # The admitted opaque inverse leaf itself still has the original RIGHT port.
    leaf = source.decompose_bloq()
    assert any(
        type(instance.bloq).__name__ == "PrepareMuUnaryEncodedOneHotWithProj"
        for instance in leaf.bloq_instances
    )
    original = module.PrepareMuUnaryEncodedOneHotWithProj(2, 1).adjoint()
    program = resolve_component_program(
        _library(original, 0), 0, nu_preparation_factories()
    )
    with pytest.raises(ValueError, match="physical lowering requires component bodies"):
        bind_register_program(program, ((0, 1),), clean=(2, 3))


@pytest.mark.parametrize("sample", (3, 4, 8))
def test_actual_constructor_uniform_child_reuses_existing_execution_with_sdks_blocked(
    sample: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    source = import_module(_BASE + "projectile.prepare_nu").PrepareNuStateWithProj(
        1, 2, sample
    )
    library = _library(source, 0)
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "qualtran", None)
        patch.setitem(sys.modules, "cirq", None)
        program = resolve_component_program(library, 0, nu_preparation_factories())
        entry = program.functions[0].blocks[0].operations[4].callee
        assert entry is not None
        # Only the uniform child and its owned helper bodies form this executable
        # submodule; dropping opaque siblings does not execute the entire root.
        uniform = ProgramRecord(
            tuple(function for function in program.functions[1:] if function.blocks)
        )
        assert verify_program_record(uniform) == ()
        width = (sample - 1).bit_length()
        ports = (tuple(reversed(range(width))),)
        clean = tuple(range(width, width + 8))
        actual = bind_register_program(uniform, ports, clean=clean, entry=entry)
        accepted = bind_register_program(
            uniform_preparation_program(sample), ports, clean=clean
        )
        assert actual == accepted

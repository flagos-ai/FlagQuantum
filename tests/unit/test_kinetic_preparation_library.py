"""Kinetic constructors preserve ordinary calls and projectile opaque boundaries."""

from importlib import import_module

import pytest

from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.registers.preparation.uniform_preparation_component import (
    uniform_preparation_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.ecosystem._kinetic_preparation_library import (
    kinetic_preparation_factories,
)
from tests.unit.test_cirq_decomposition_file import locked_source as locked_source
from tests.unit.test_component_graph import _native_trace, _source_trace
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_projectile_nu_library import _named_inputs
from tests.unit.test_register_component_calls import SourceCalls

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
_BASE = "qualtran.bloqs.chemistry.pbc.first_quantization."


@pytest.mark.parametrize("width", (1, 2, 5))
@pytest.mark.parametrize(
    "kind,inverse",
    (
        ("PreparePowerTwoState", False),
        ("PreparePowerTwoStateWithProj", False),
        ("PreparePowerTwoStateWithProj", True),
        ("PrepareTFirstQuantizationWithProj", False),
        ("PrepareTFirstQuantizationWithProj", True),
    ),
)
def test_original_opaque_leaf_fields_signatures_and_refusal(
    width: int, kind: str, inverse: bool
) -> None:
    module = import_module(
        _BASE + ("projectile.prepare_t" if kind.endswith("WithProj") else "prepare_t")
    )
    if kind == "PreparePowerTwoState":
        source = module.PreparePowerTwoState(width)
    elif kind == "PreparePowerTwoStateWithProj":
        source = module.PreparePowerTwoStateWithProj(width, 1, inverse)
    else:
        source = module.PrepareTFirstQuantizationWithProj(
            1, width, 10, is_adjoint=inverse
        )
    for depth in (0, 1, 2):
        library = _library(source, depth)
        program = resolve_component_program(library, 0, kinetic_preparation_factories())
        assert (
            program.functions[0].registers == library.entries[0].declaration.registers
        )
        assert verify_program_record(program) == ()
        assert not program.functions[0].blocks
        with pytest.raises(
            ValueError, match="physical lowering requires component bodies"
        ):
            bind_register_program(program, (), clean=(0, 1))


@pytest.mark.parametrize("width", (1, 2, 5))
def test_literal_three_call_root_and_actual_coordinate_replacement(
    width: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    source = import_module(_BASE + "prepare_t").PrepareTFirstQuantization(width, 10)
    baseline = None
    for depth in (0, 1, 2):
        library = _library(source, depth)
        with monkeypatch.context() as patch:
            patch.setitem(sys.modules, "qualtran", None)
            patch.setitem(sys.modules, "cirq", None)
            program = resolve_component_program(
                library, 0, kinetic_preparation_factories()
            )
            assert verify_program_record(program) == ()
            assert (
                program.functions[0].registers
                == library.entries[0].declaration.registers
            )
            root = program.functions[0].blocks[0]
            assert len(root.operations) == 3
            assert root.operations[1].callee == root.operations[2].callee
            assert root.operations[1].operands != root.operations[2].operands
            entry = root.operations[0].callee
            assert entry is not None
            uniform = ProgramRecord(tuple(f for f in program.functions[1:] if f.blocks))
            ports, clean = ((1, 0),), tuple(range(2, 10))
            assert bind_register_program(
                uniform, ports, clean=clean, entry=entry
            ) == bind_register_program(
                uniform_preparation_program(3), ports, clean=clean
            )
        assert _named_inputs(_native_trace(program)) == _named_inputs(
            _source_trace(SourceCalls, source.decompose_bloq())
        )
        assert sum(not f.blocks for f in program.functions) == 1
        if baseline is None:
            baseline = program
        else:
            assert program == baseline


def test_source_negative_narrow_cost_and_unused_precision_are_not_reinterpreted() -> (
    None
):
    import attrs

    costs = import_module("qualtran.resource_counting")

    module = import_module(_BASE + "prepare_t")
    assert (
        costs.get_cost_value(
            module.PreparePowerTwoState(1), costs.QECGatesCost()
        ).toffoli
        == -1
    )
    source = module._prepare_t.make()
    program = resolve_component_program(
        _library(source, 0), 0, kinetic_preparation_factories()
    )
    changed = attrs.evolve(source, eta=-7, num_bits_rot_aa=-2)
    library = _library(changed, 0)
    fields = {arg.name: arg.value for arg in library.entries[0].declaration.arguments}
    assert fields["eta"] == -7 and fields["num_bits_rot_aa"] == -2
    assert (
        resolve_component_program(library, 0, kinetic_preparation_factories())
        == program
    )


@pytest.mark.parametrize("example", ("_prep_power_two_proj", "_prep_t_proj"))
def test_original_projectile_examples_remain_declarations(example: str) -> None:
    source = getattr(import_module(_BASE + "projectile.prepare_t"), example).make()
    program = resolve_component_program(
        _library(source, 0), 0, kinetic_preparation_factories()
    )
    assert verify_program_record(program) == ()
    assert not program.functions[0].blocks

"""Potential constructor composition retains source fields and opaque children."""

from importlib import import_module

import pytest

from flagquantum.compiler.formats.legacy_component_library import (
    deserialize_component_library,
    serialize_component_library,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import verify_program_record
from flagquantum.ecosystem._potential_preparation_library import (
    potential_preparation_factories,
)
from tests.unit.test_cirq_decomposition_file import locked_source as locked_source
from tests.unit.test_component_graph import _native_trace, _source_trace
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_projectile_nu_library import _named_inputs
from tests.unit.test_register_component_calls import SourceCalls

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]

_BASE = "qualtran.bloqs.chemistry.pbc.first_quantization."


@pytest.mark.parametrize("atoms", (1, 2, 10))
@pytest.mark.parametrize("charge", (0, 10))
@pytest.mark.parametrize("adjoint", (False, True))
def test_nuclear_leaf_constructor_and_direction_metadata_survive_codec(
    atoms: int, charge: int, adjoint: bool
) -> None:
    source = import_module(_BASE + "prepare_zeta").PrepareZetaState(
        atoms, charge, 8, adjoint
    )
    for depth in (0, 1, 2):
        library = _library(source, depth)
        encoded = serialize_component_library(library)
        assert (
            serialize_component_library(deserialize_component_library(encoded))
            == encoded
        )
        assert (
            next(
                arg.value
                for arg in library.entries[0].declaration.arguments
                if arg.name == "is_adjoint"
            )
            == adjoint
        )
        program = resolve_component_program(
            library, 0, potential_preparation_factories()
        )
        assert (
            program.functions[0].registers == library.entries[0].declaration.registers
        )
        assert not program.functions[0].blocks
        assert verify_program_record(program) == ()
        with pytest.raises(
            ValueError, match="physical lowering requires component bodies"
        ):
            bind_register_program(program, (), clean=(0, 1))


@pytest.mark.parametrize("projectile", (False, True))
@pytest.mark.parametrize("atoms,sample", ((1, 3), (3, 4), (10, 8)))
def test_full_source_root_with_single_atom_empty_index_and_nu_closure(
    projectile: bool, atoms: int, sample: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    module = import_module(
        _BASE + ("projectile.prepare_uv" if projectile else "prepare_uv")
    )
    fields = {
        "num_bits_p": 1,
        "eta": 10,
        "num_atoms": atoms,
        "m_param": sample,
        "lambda_zeta": 10,
        "num_bits_nuc_pos": 8,
    }
    if projectile:
        fields["num_bits_n"] = 2
    source = getattr(
        module,
        (
            "PrepareUVFirstQuantizationWithProj"
            if projectile
            else "PrepareUVFirstQuantization"
        ),
    )(**fields)
    baseline = None
    for depth in (0, 1, 2):
        library = _library(source, depth)
        with monkeypatch.context() as patch:
            patch.setitem(sys.modules, "qualtran", None)
            patch.setitem(sys.modules, "cirq", None)
            program = resolve_component_program(
                library, 0, potential_preparation_factories()
            )
            assert verify_program_record(program) == ()
            with pytest.raises(
                ValueError, match="physical lowering requires component bodies"
            ):
                bind_register_program(program, (), clean=tuple(range(100)))
        assert (
            program.functions[0].registers == library.entries[0].declaration.registers
        )
        assert _named_inputs(_native_trace(program)) == _named_inputs(
            _source_trace(SourceCalls, source.decompose_bloq())
        )
        assert len(program.functions[0].blocks[0].operations) == 2
        assert sum(not function.blocks for function in program.functions) == 6
        if baseline is None:
            baseline = program
        else:
            assert program == baseline


@pytest.mark.parametrize("projectile", (False, True))
def test_original_factories_and_unused_eta_have_no_fabricated_gate_effect(
    projectile: bool,
) -> None:
    import attrs

    module = import_module(
        _BASE + ("projectile.prepare_uv" if projectile else "prepare_uv")
    )
    source = getattr(module, "_prep_uv_proj" if projectile else "_prepare_uv").make()
    program = resolve_component_program(
        _library(source, 0), 0, potential_preparation_factories()
    )
    _named = _named_inputs(_source_trace(SourceCalls, source.decompose_bloq()))
    assert _named_inputs(_native_trace(program)) == _named
    changed = attrs.evolve(source, eta=-7)
    library = _library(changed, 0)
    assert (
        next(
            arg.value
            for arg in library.entries[0].declaration.arguments
            if arg.name == "eta"
        )
        == -7
    )
    assert (
        resolve_component_program(library, 0, potential_preparation_factories())
        == program
    )

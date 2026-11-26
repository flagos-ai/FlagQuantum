"""Ordinary root constructors preserve source domains and shared dependency calls."""

from importlib import import_module

import pytest
import sympy

from flagquantum.algorithms.first_quantization_resource_model import (
    FirstQuantizationPrepareDeclaration,
)
from flagquantum.compiler.formats.legacy_component_library import (
    deserialize_component_library,
    serialize_component_library,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import verify_program_record
from flagquantum.ecosystem._first_quantization_component_library import (
    first_quantization_component_factories,
)
from tests.unit.test_cirq_decomposition_file import locked_source as locked_source
from tests.unit.test_component_graph import _native_trace, _source_trace
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_projectile_nu_library import _named_inputs
from tests.unit.test_register_component_calls import SourceCalls

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
_BASE = "qualtran.bloqs.chemistry.pbc.first_quantization."


@pytest.mark.parametrize("select", (False, True))
@pytest.mark.parametrize("example", (False, True))
def test_full_root_stored_constructor_and_source_node_connections(
    select: bool, example: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    module = import_module(_BASE + "select_and_prepare")
    source = (
        getattr(module, "_sel_first_quant" if select else "_prep_first_quant").make()
        if example
        else getattr(
            module, "SelectFirstQuantization" if select else "PrepareFirstQuantization"
        )(2, 2, 3, 3, m_param=4, num_bits_nuc_pos=8, num_bits_t=2, num_bits_rot_aa=3)
    )
    expected = _named_inputs(_source_trace(SourceCalls, source.decompose_bloq()))
    baseline = None
    for depth in (0, 1, 2):
        library = _library(source, depth)
        encoded = serialize_component_library(library)
        assert (
            serialize_component_library(deserialize_component_library(encoded))
            == encoded
        )
        with monkeypatch.context() as patch:
            patch.setitem(sys.modules, "qualtran", None)
            patch.setitem(sys.modules, "cirq", None)
            program = resolve_component_program(
                library, 0, first_quantization_component_factories()
            )
            assert verify_program_record(program) == ()
            root = program.functions[0]
            assert root.registers == library.entries[0].declaration.registers
            assert _named_inputs(_native_trace(program)) == expected
            assert len(root.blocks[0].operations) == (20 if select else 4)
            with pytest.raises(
                ValueError, match="physical lowering requires component bodies"
            ):
                bind_register_program(program, (), clean=tuple(range(100)))
            assert root.registers is not None
            index = next(r for r in root.registers if r.name == "l")
            bound = dict(index.dtype.parameters)["bound"]
            assert bound == (2 ** int(index.bitsize) if select else index.bitsize)
            if not select:
                offset = 0
                for register in root.registers:
                    if register.name == "plus_t":
                        body = root.blocks[0]
                        assert body.return_values is not None
                        assert body.return_values[offset] == body.arguments[offset]
                        break
                    offset += sum(1 for _ in register.all_idxs())
        if baseline is None:
            baseline = program
        else:
            assert baseline == program


@pytest.mark.parametrize("norm", (None, 0, 0.75, sympy.Symbol("lambda_norm")))
def test_normalization_metadata_and_existing_declaration_queries(
    norm: int | float | sympy.Expr | None,
) -> None:
    module = import_module(_BASE + "select_and_prepare")
    source = module.PrepareFirstQuantization(
        2, 2, 3, 3, m_param=4, sum_of_l1_coeffs=norm
    )
    native = FirstQuantizationPrepareDeclaration(
        2, 2, 3, 3, m_param=4, sum_of_l1_coeffs=norm
    )
    library = _library(source, 0)
    program = resolve_component_program(
        library, 0, first_quantization_component_factories()
    )
    default = module.PrepareFirstQuantization(2, 2, 3, 3, m_param=4)
    assert program == resolve_component_program(
        _library(default, 0), 0, first_quantization_component_factories()
    )
    if norm is None:
        with pytest.raises(ValueError):
            _ = source.l1_norm_coeffs
        with pytest.raises(ValueError):
            _ = native.l1_norm_of_coeffs
    else:
        assert source.l1_norm_coeffs == native.l1_norm_of_coeffs == norm
        assert (
            next(
                a.value
                for a in library.entries[0].declaration.arguments
                if a.name == "sum_of_l1_coeffs"
            )
            == norm
        )


def test_original_leaf_examples_and_capitalized_nuclear_position_port() -> None:
    cases = (
        import_module(_BASE + "select_t")._select_t.make(),
        import_module(_BASE + "select_uv")._select_uv.make(),
        import_module(_BASE + "select_uv").ApplyNuclearPhase(2, 8),
        import_module(_BASE + "select_and_prepare").PrepareTUVSuperpositions(3, 2, 3),
    )
    for source in cases:
        for depth in (0, 1, 2):
            library = _library(source, depth)
            program = resolve_component_program(
                library, 0, first_quantization_component_factories()
            )
            assert (
                program.functions[0].registers
                == library.entries[0].declaration.registers
            )
            assert not program.functions[0].blocks
            assert verify_program_record(program) == ()
    phase = resolve_component_program(
        _library(cases[2], 0), 0, first_quantization_component_factories()
    )
    assert tuple(r.name for r in phase.functions[0].registers or ()) == (
        "l",
        "Rl",
        "nu",
    )

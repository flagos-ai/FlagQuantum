"""Match complete THC PREPARE graphs while retaining source execution boundaries."""

import pytest

from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_thc_preparation_factories import (
    thc_preparation_factories,
)
from flagquantum.compiler.registers.component_definition_match import (
    matches_component_definition,
)
from flagquantum.compiler.registers.primitives.identity import register_identity_program
from flagquantum.compiler.registers.sorting.contiguous_index_component import (
    contiguous_index_program,
)
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.errors import SerializationError
from tests.unit.preparation_reference import preparation_terminals
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_qubitization_walk_component import (
    reference_runtime as reference_runtime,
)
from tests.unit.test_sparse_chemistry_prepare_component import bind_used_workspace
from tests.unit.test_thc_uniform_source_component import (
    pinned_thc_source as pinned_thc_source,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def thc_source_terminals(
    record: ProgramRecord, width: int, address_width: int
) -> set[str]:
    templates = (
        register_identity_program(1),
        register_identity_program(width),
        contiguous_index_program(width, address_width),
    )
    terminals = preparation_terminals(record)
    for template in templates:
        terminals.update(
            function.name
            for function in record.functions
            if function.registers == template.functions[0].registers
            and matches_component_definition(record, function.name, template)
        )
    return terminals


@pytest.mark.parametrize("original_example", (False, True))
@pytest.mark.parametrize("inverse", (False, True))
def test_complete_original_eight_bit_calls_constructor_depths_and_refusal(
    original_example: bool, inverse: bool
) -> None:
    from qualtran.bloqs.chemistry.thc.prepare import PrepareTHC, _thc_prep
    from qualtran.bloqs.chemistry.thc.prepare_test import build_random_test_integrals

    original = (
        _thc_prep.make()
        if original_example
        else PrepareTHC.from_hamiltonian_coeffs(
            *build_random_test_integrals(2, 2, seed=7),
            num_bits_state_prep=8,
            log_block_size=1,
        )
    )
    assert original.keep_bitsize == 8
    source = original.adjoint() if inverse else original
    assert len(source.decompose_bloq().bloq_instances) == 17
    rows = len(original.keep)
    for depth in (0, 1):
        record = resolve_component_program(
            _library(source, depth), 0, thc_preparation_factories()
        )
        assert not verify_program_record(record)
        terminals = thc_source_terminals(
            record, original.num_mu.bit_length(), (rows - 1).bit_length()
        )
        paired = compare_layers(record, source, terminals)
        models = {
            type(bloq).__name__
            for name, bloq in paired.items()
            if not next(f for f in record.functions if f.name == name).blocks
        }
        assert "GreaterThanConstant" in models or inverse and "Adjoint" in models
        if inverse:
            assert "QROAMCleanAdjoint" in models
        with pytest.raises(ValueError, match="requires component bodies"):
            bind_used_workspace(record, allow_workspace_outputs=True)


def test_zero_block_source_missing_junk_is_not_silently_repaired() -> None:
    from attrs import evolve
    from qualtran.bloqs.chemistry.thc.prepare import _thc_prep

    original = evolve(_thc_prep.make(), log_block_size=0)
    with pytest.raises(KeyError, match="junk_target0_"):
        original.decompose_bloq()
    with pytest.raises(SerializationError, match="zero-block.*junk_target0_"):
        resolve_component_program(_library(original, 0), 0, thc_preparation_factories())

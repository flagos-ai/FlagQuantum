"""Double walk controls preserve per-child predicates and exact source views."""

from importlib import import_module
from unittest.mock import patch

import pytest
import torch

from tests._reference_dependencies import unavailable_reference

try:
    import cirq as cirq

    import_module("numpy")
    import_module("qualtran")
    import_module("kickmix")
except ModuleNotFoundError as _reference_import_error:
    unavailable_reference(_reference_import_error)

from flagquantum.compiler.registers.preparation.qubitization_walk_component import (
    doubly_controlled_select_walk_pair,
)
from flagquantum.core.ir.program import verify_program_record
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.ecosystem._default_linear_combination_oracles import _signature
from flagquantum.ecosystem._hubbard_walk_component_library import (
    capture_hubbard_walk_context,
)
from tests.unit.test_chebyshev_component import assert_all_small_raw_histories
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_hubbard_stored_graph import source_terminals
from tests.unit.test_legacy_block_encoding_factories import _oracles
from tests.unit.test_legacy_phased_block_factories import scalar_terminals
from tests.unit.test_multiplexed_source_binding import logical_operator
from tests.unit.test_qubitization_walk_component import walk_fixture
from tests.unit.test_qubitization_walk_file import (
    complete_frozen_sources as complete_frozen_sources,
)
from tests.unit.test_sparse_chemistry_prepare_component import bind_used_workspace

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_actual_hubbard_double_controlled_forward_and_inverse() -> None:
    from qualtran import Controlled, CtrlSpec
    from qualtran.bloqs.hamiltonian_simulation.hamiltonian_simulation_by_gqsp import (
        _hubbard_time_evolution_by_gqsp,
    )

    from flagquantum.compiler.registers.control.hubbard_select_component import (
        hubbard_select_program,
    )

    source = _hubbard_time_evolution_by_gqsp.make().walk_operator
    context = capture_hubbard_walk_context(source)
    assert context.positive_reflection_pair is not None
    assert context.block_pair[0].functions[0].registers is not None
    pair = doubly_controlled_select_walk_pair(
        context.signature,
        Signature(context.block_pair[0].functions[0].registers),
        (
            hubbard_select_program(2, controlled=True),
            hubbard_select_program(2, controlled=True, adjoint=True),
        ),
        context.positive_reflection_pair[0],
        select_control="control",
        reflection_control="control",
    )
    reference = Controlled(source.controlled(), CtrlSpec())
    for native, original in zip(pair, (reference, reference.adjoint()), strict=True):
        assert verify_program_record(native) == ()
        pairs = compare_layers(
            native, original, source_terminals(native) | scalar_terminals(native)
        )
        assert len(pairs) == 124
        # This is literal Hubbard composition. Its known private temp release
        # defect remains; graph equality is not a clean-work certificate.


@pytest.mark.parametrize("inverse", (False, True))
def test_clean_double_walk_full_complex_operator_all_histories_and_work_zero(
    inverse: bool,
) -> None:
    from qualtran import Controlled, CtrlSpec

    oracles = _oracles()
    with patch(
        "tests.unit.test_qubitization_walk_component._oracles", return_value=oracles
    ):
        source, blocks, _ = walk_fixture(False)
        _, _, reflection = walk_fixture(False, controlled=True)
    assert blocks[0].functions[0].registers is not None
    signature = Signature(_signature(source.signature))
    pair = doubly_controlled_select_walk_pair(
        signature,
        Signature(blocks[0].functions[0].registers),
        oracles[4],
        reflection,
        select_control="control",
        reflection_control="control",
    )
    native = pair[int(inverse)]
    original = Controlled(source.controlled(), CtrlSpec())
    if inverse:
        original = original.adjoint()
    compare_layers(
        native, original, source_terminals(native) | scalar_terminals(native)
    )
    bound, logical = bind_used_workspace(native)
    assert logical == 4
    expected = torch.as_tensor(original.tensor_contract(), dtype=torch.complex128)
    torch.testing.assert_close(
        logical_operator(bound.circuit, logical), expected, atol=2e-10, rtol=0
    )
    assert_all_small_raw_histories(bound.circuit, logical, expected)

"""Exercise sparse Walk capture with explicit retained bank signal storage."""

import sys
from unittest.mock import patch

import pytest

from flagquantum import Circuit
from flagquantum.algorithms.primitives._block_binding import _BoundBlockOperation
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.core.ir.program import verify_program_record
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.ecosystem._sparse_walk_component_library import (
    capture_sparse_walk_context,
)
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_multi_control_component import atomic_leaves
from tests.unit.test_qubitization_walk_component import (
    reference_runtime as reference_runtime,
)
from tests.unit.test_qubitization_walk_file import (
    complete_frozen_sources as complete_frozen_sources,
)
from tests.unit.test_sparse_chemistry_prepare import _branch
from tests.unit.test_sparse_chemistry_prepare_component import bind_used_workspace
from tests.unit.test_sparse_preparation_bank_unitary import bank_preparation_fixture

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("log", (0, 1))
@pytest.mark.parametrize("controlled", (False, True))
def test_sparse_walk_native_context_source_block_and_joint_inverse(
    log: int, controlled: bool
) -> None:
    from qualtran import CtrlSpec
    from qualtran.bloqs.block_encoding.lcu_block_encoding import SelectBlockEncoding
    from qualtran.bloqs.chemistry.sparse.select_bloq import SelectSparse
    from qualtran.bloqs.qubitization.qubitization_walk_operator import (
        QubitizationWalkOperator,
    )

    prepare, _, _ = bank_preparation_fixture(log)
    source = QubitizationWalkOperator(SelectBlockEncoding(SelectSparse(4), prepare))
    context = capture_sparse_walk_context(source)
    assert context.source_signature is not None
    assert context.signature != context.source_signature
    source_block = (
        source.block_encoding.get_ctrl_system(CtrlSpec())[0]
        if controlled
        else source.block_encoding
    )
    blocks = context.positive_block_pair if controlled else context.block_pair
    assert blocks is not None
    compare_layers(blocks[0], source_block, atomic_leaves(blocks[0]))
    with patch.dict(sys.modules, {"qualtran": None, "cirq": None}):
        pair = context.pair(controlled=controlled)
        assert all(not verify_program_record(record) for record in pair)
        builder = ComponentProgramBuilder(
            Signature(pair[0].functions[0].registers or ())
        )
        output = builder.call_program(pair[0], builder.inputs)
        restored = builder.call_program(pair[1], output)
        bound, logical = bind_used_workspace(builder.finalize(restored))
        circuit = Circuit(bound.circuit.n_wires).append(
            _BoundBlockOperation(bound.circuit.instructions, bound.circuit.n_wires)
        )
        assert circuit.to_ir().instructions == bound.circuit.instructions
    work = bound.circuit.n_wires - logical
    initial = {0: 0.6 + 0j, ((1 << logical) - 1) << work: 0.8j}
    count = sum(g.name == "measure" for g in bound.circuit.instructions)
    for outcome in (0, 1):
        actual = _branch(
            bound.circuit.instructions,
            bound.circuit.n_wires,
            (outcome,) * count,
            initial,
            normalize_measurements=True,
        )
        for key in actual.keys() | initial.keys():
            assert actual.get(key, 0) == pytest.approx(initial.get(key, 0), abs=4e-10)


def test_original_65_row_twelve_bit_walk_native_binding_scope() -> None:
    from qualtran.bloqs.qubitization.qubitization_walk_operator import (
        _walk_op_chem_sparse,
    )

    source = _walk_op_chem_sparse.make()
    assert source.prepare.num_non_zero == 65
    assert source.prepare.num_bits_state_prep == 12
    context = capture_sparse_walk_context(source)
    assert context.signature.get_left("p").bitsize == 2
    assert context.signature.get_left("signal_p").bitsize == 2
    assert context.signature.thru_registers_only
    for controlled in (False, True):
        for record in context.pair(controlled=controlled):
            assert not verify_program_record(record)
            with patch.dict(sys.modules, {"qualtran": None, "cirq": None}):
                bound, _ = bind_used_workspace(record)
                circuit = Circuit(bound.circuit.n_wires).append(
                    _BoundBlockOperation(
                        bound.circuit.instructions, bound.circuit.n_wires
                    )
                )
                assert circuit.to_ir().instructions == bound.circuit.instructions

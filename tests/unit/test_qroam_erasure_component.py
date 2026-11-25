"""Exercise explicit typed native erasure on original signed lookup outputs."""

from itertools import product
from math import sqrt

import pytest
import torch

from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.lookup.qroam_clean_component import (
    qroam_clean_program,
)
from flagquantum.compiler.registers.lookup.qroam_erasure_component import (
    qroam_measurement_erasure_program,
)
from flagquantum.compiler.registers.lookup.source_qroam_clean import QroamCleanTable
from flagquantum.core.ir.program import verify_program_record
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_default_linear_combination_oracles import bind
from tests.unit.test_qubitization_walk_component import (
    reference_runtime as reference_runtime,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_wrapper_matches_existing_source_call_without_filling_its_model_leaf() -> None:
    np = pytest.importorskip("numpy")
    from qualtran._infra.bloq import DecomposeNotImplementedError
    from qualtran.bloqs.data_loading.qroam_clean import QROAMClean

    from flagquantum.compiler.formats.legacy_component_resolution import (
        resolve_component_program,
    )
    from flagquantum.compiler.formats.legacy_qroam_factories import qroam_factories
    from tests.unit.test_component_hierarchy_control import compare_layers
    from tests.unit.test_legacy_arithmetic_factories import _library

    table = QroamCleanTable((5,), (1,), ((1, 0, 1, 1, 0),))
    source = QROAMClean.build_from_data(
        np.array(table.datasets[0]), target_bitsizes=(1,), log_block_sizes=(1,)
    ).adjoint()
    erase = qroam_measurement_erasure_program(table, (1,))
    leaf_name = erase.functions[0].blocks[0].operations[0].callee
    assert leaf_name is not None
    pairs = compare_layers(erase, source, {leaf_name})
    assert type(pairs[leaf_name]).__name__ == "QROAMCleanAdjoint"
    with pytest.raises(DecomposeNotImplementedError):
        pairs[leaf_name].decompose_bloq()
    default = resolve_component_program(_library(source, 0), 0, qroam_factories())
    assert any(
        not function.blocks and function.registers for function in default.functions
    )


@pytest.mark.parametrize(
    "shape,logs,data,widths,controls",
    (
        ((2,), (1,), ((0, 1),), (1,), 0),
        ((4,), (1,), ((1, 0, 1, 1),), (1,), 0),
        ((3,), (0,), ((0, 1, 1),), (1,), 1),
        ((2, 2), (1, 1), ((0, 1, 1, 0),), (1,), 0),
        ((2,), (1,), ((0, 0), (1, 0)), (0, 1), 0),
        ((2,), (1,), ((1, 0), (2, 3)), (1, 2), 0),
    ),
)
def test_forward_erasure_joint_complex_state_every_small_history(
    shape: tuple[int, ...],
    logs: tuple[int, ...],
    data: tuple[tuple[int, ...], ...],
    widths: tuple[int, ...],
    controls: int,
) -> None:
    table = QroamCleanTable(shape, logs, data)
    forward = qroam_clean_program(table, widths, num_controls=controls)
    erase = qroam_measurement_erasure_program(table, widths, num_controls=controls)
    assert not verify_program_record(forward)
    assert not verify_program_record(erase)
    assert (
        tuple(Signature(forward.functions[0].registers or ()).adjoint())
        == erase.functions[0].registers
    )
    builder = ComponentProgramBuilder(
        Signature(tuple(Signature(forward.functions[0].registers or ()).lefts()))
    )
    loaded = builder.call_program(forward, builder.inputs)
    result = builder.call_program(erase, loaded)
    record = builder.finalize(result)
    bound, logical = bind(record)
    count = sum(g.name == "measure" for g in bound.circuit.instructions)
    assert count <= 12
    stride = 1 << (bound.circuit.n_wires - logical)
    weights = torch.arange(1, (1 << logical) + 1, dtype=torch.float64).to(
        torch.complex128
    )
    weights[1::2] *= 1j
    weights /= weights.norm()
    initial = torch.zeros(1 << bound.circuit.n_wires, dtype=torch.complex128)
    initial[::stride] = weights
    for history in product((0, 1), repeat=count):
        recovered = forced_branch(initial, bound.circuit.instructions, history) * sqrt(
            2**count
        )
        torch.testing.assert_close(recovered, initial, atol=2e-11, rtol=0)

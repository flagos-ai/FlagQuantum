"""Compare the locked placeholder QROAM's two directional formulae."""

import pytest

from flagquantum.algorithms.primitives.chemistry_qroam_model import (
    chemistry_qroam_block_score,
    chemistry_qroam_signature,
    source_chemistry_qroam_cost,
    source_chemistry_qroam_two_axis_cost,
)

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "rows,width,block",
    [
        (51, 20, None),
        (64, 8, None),
        (19, 8, 1),
        (19, 8, 3),
        (2, 12, 2),
        (1, 8, 1),
        (100, 10, None),
        (100, 3, None),
        (1_000, 13, None),
        (1_000_000, 20, None),
    ],
)
@pytest.mark.parametrize("adjoint", [False, True])
def test_one_axis_chemistry_qroam(rows, width, block, adjoint):
    pytest.importorskip("qualtran")
    from qualtran.bloqs.basic_gates import Toffoli
    from qualtran.bloqs.chemistry.black_boxes import QROAM
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    source = QROAM(rows, width, is_adjoint=adjoint, qroam_block_size=block)
    actual = source_chemistry_qroam_cost(rows, width, block_size=block, adjoint=adjoint)
    assert actual.gates.toffoli == get_cost_value(source, QECGatesCost()).toffoli
    assert actual.qubits == get_cost_value(source, QubitCount())
    assert source.build_call_graph(None) == {Toffoli(): actual.gates.toffoli}
    assert tuple(
        (reg.name, reg.bitsize, reg.side.value)
        for reg in chemistry_qroam_signature(rows, width)
    ) == tuple((reg.name, reg.bitsize, reg.side.value) for reg in source.signature)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "shape,blocks,width", [((51, 15), (4, 8), 20), ((5, 10), (3, 2), 4)]
)
@pytest.mark.parametrize("adjoint", [False, True])
def test_two_axis_chemistry_qroam(shape, blocks, width, adjoint):
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.black_boxes import QROAMTwoRegs
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    source = QROAMTwoRegs(*shape, *blocks, width, is_adjoint=adjoint)
    actual = source_chemistry_qroam_two_axis_cost(
        *shape, *blocks, width, adjoint=adjoint
    )
    assert actual.gates.toffoli == get_cost_value(source, QECGatesCost()).toffoli
    assert actual.qubits == get_cost_value(source, QubitCount())
    assert tuple(
        (reg.name, reg.bitsize, reg.side.value)
        for reg in chemistry_qroam_signature(shape[0], width)
    ) == tuple((reg.name, reg.bitsize, reg.side.value) for reg in source.signature)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("adjoint", [False, True])
@pytest.mark.parametrize("dirty", [False, True])
def test_original_unrounded_clean_and_dirty_block_scores(adjoint, dirty):
    from qualtran.bloqs.chemistry.black_boxes import qroam_cost, qroam_cost_dirty

    reference = qroam_cost_dirty if dirty else qroam_cost
    for rows, width, block in ((51, 20, 2), (19, 8, 3), (100, 10, 4)):
        assert chemistry_qroam_block_score(
            rows, width, block, adjoint=adjoint, dirty=dirty
        ) == reference(block, rows, width, adjoint=adjoint)

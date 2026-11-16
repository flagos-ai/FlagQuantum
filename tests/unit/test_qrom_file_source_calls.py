"""Compare immediate source QROM calls before primitive lowering."""

from math import prod
from typing import Any

import pytest

from flagquantum.compiler.registers.lookup.qrom_resource_model import (
    QromCall,
    source_qrom_calls,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def call_key(child: Any) -> QromCall:
    from qualtran import Adjoint, Controlled
    from qualtran.bloqs.arithmetic.bitwise import XorK
    from qualtran.bloqs.basic_gates import CNOT, XGate
    from qualtran.bloqs.mcmt.and_bloq import And, MultiAnd

    if isinstance(child, XGate):
        return "X", (), False
    if isinstance(child, CNOT):
        return "CNOT", (), False
    if isinstance(child, And):
        return "And", (child.cv1, child.cv2), child.uncompute
    if isinstance(child, MultiAnd):
        return "MultiAnd", child.cvs, False
    if isinstance(child, Adjoint) and isinstance(child.subbloq, MultiAnd):
        return "MultiAnd", child.subbloq.cvs, True
    if isinstance(child, Controlled) and isinstance(child.subbloq, XorK):
        return "XorK", (child.subbloq.dtype.bitsize, child.subbloq.k, 1), False
    assert isinstance(child, XorK), child
    return "XorK", (child.dtype.bitsize, child.k, 0), False


@pytest.mark.parametrize(
    "shape,target_shapes,widths,datasets",
    [
        ((5,), ((),), (3,), ((0, 1, 2, 3, 4),)),
        ((3, 2), ((), (2,)), (2, 3), ((0, 1, 3, 3, 2, 2), tuple(range(6)) * 2)),
        ((2,), ((),), (3,), ((1, 0),)),
        ((1,), ((2,),), (3,), ((0, 0),)),
        ((3,), ((2,),), (3,), ((1, 1) * 3,)),
    ],
)
def test_direct_calls_match_reference(
    shape: tuple[int, ...],
    target_shapes: tuple[tuple[int, ...], ...],
    widths: tuple[int, ...],
    datasets: tuple[tuple[int, ...], ...],
) -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("qualtran")
    from qualtran.bloqs.data_loading import QROM
    from qualtran.resource_counting.generalizers import ignore_split_join

    selectors = tuple((extent - 1).bit_length() for extent in shape)
    layout = tuple(
        (prod(sh), width) for sh, width in zip(target_shapes, widths, strict=True)
    )
    for controls in range(5):
        source = QROM.build_from_data(
            *(
                np.array(values).reshape(shape + sh)
                for values, sh in zip(datasets, target_shapes, strict=True)
            ),
            target_shapes=target_shapes,
            target_bitsizes=widths,
            num_controls=controls,
        )
        graph, _ = source.call_graph(max_depth=1, generalizer=ignore_split_join)
        expected = {
            call_key(child): attributes["n"]
            for _, child, attributes in graph.edges(data=True)
        }
        assert (
            source_qrom_calls(controls, selectors, shape, layout, datasets) == expected
        )

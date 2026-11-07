"""QROM factories against unmodified source metadata and existing IR binding."""

from dataclasses import replace

import numpy as np
import pytest

from flagquantum.compiler.registers.qrom_table import QromTable

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "data,targets,widths",
    [
        (([0, 0, 0],), (), None),
        (([0, 1, 3],), (), None),
        (([0.9, 2.8, 1.1],), (), None),
        (([[1, 0], [0, 1], [1, 1]], [2, 0, 1]), ((2,), ()), (1, 2)),
        (([[[1, 2], [0, 3]], [[3, 0], [2, 1]]],), ((2, 2),), (2,)),
        (([[0, 1, 2], [3, 2, 1]],), (), None),
        (([4, 1],), (), (1,)),
        (([-1, -2],), (), None),
        ((3,), (), None),
    ],
)
@pytest.mark.parametrize("controls", [0, 1, 3])
def test_source_data_factory(data, targets, widths, controls):
    from qualtran.bloqs.data_loading.qrom import QROM

    source = QROM.build_from_data(
        *data, target_shapes=targets, target_bitsizes=widths, num_controls=controls
    )
    native = QromTable.from_data(
        *data, target_shapes=targets, target_bitsizes=widths, num_controls=controls
    )
    assert native.shape == source.data_shape
    assert native.selection_bitsizes == source.selection_bitsizes
    assert native.target_bitsizes == source.target_bitsizes
    assert native.target_shapes == source.target_shapes
    assert native.num_controls == source.num_controls
    assert native.datasets == tuple(tuple(map(int, a.flat)) for a in source.data)
    assert native.target_layout == tuple(
        (int(np.prod(t)), w)
        for t, w in zip(source.target_shapes, source.target_bitsizes, strict=True)
    )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("shape", [3, (2, 3), (), (1, 2)])
@pytest.mark.parametrize("targets", [((), ()), ((2,), (2, 2))])
def test_shape_factory_and_replacement(shape, targets):
    from qualtran.bloqs.data_loading.qrom import QROM

    source = QROM.build_from_bitsize(
        shape, (2, 3), target_shapes=targets, num_controls=2
    )
    native = QromTable.from_shape(shape, (2, 3), target_shapes=targets, num_controls=2)
    assert native.datasets is None
    assert not source.has_data()
    assert native.shape == source.data_shape
    assert native.selection_bitsizes == source.selection_bitsizes
    data = tuple(
        np.arange(int(np.prod(native.shape + t))).reshape(native.shape + t) % 4
        for t in targets
    )
    loaded, original = native.with_data(*data), source.with_data(*data)
    assert loaded.datasets == tuple(tuple(map(int, a.flat)) for a in original.data)
    assert native.datasets is None
    with pytest.raises(ValueError, match="unavailable"):
        native.resource_model()
    with pytest.raises(ValueError, match="unavailable"):
        _ = native.workspace_size


def test_binding_and_replacement_reject_invalid_layouts():
    table = QromTable.from_data([0, 3])
    with pytest.raises(ValueError, match="shape"):
        table.with_data([1, 0, 1])
    with pytest.raises(ValueError, match="width"):
        table.with_data([4, 0])
    with pytest.raises(ValueError, match="ports"):
        table.bind((0,), ((1,),), (((2, 3),),), (), ())
    with pytest.raises(ValueError, match="ports"):
        table.bind((), ((0,),), (((1,),),), (), ())
    with pytest.raises(ValueError, match="array shapes"):
        replace(table, datasets=((1,),))
    with pytest.raises(ValueError, match="common selection"):
        QromTable.from_data([1, 0], [1, 0, 1])
    # Original construction does not enforce the unsigned emitted-value domain.
    for data, widths in (([-1, -2], (2,)), ([4, 0], (1,))):
        metadata = QromTable.from_data(data, target_bitsizes=widths)
        with pytest.raises(ValueError, match="dataset"):
            metadata.resource_model()
        with pytest.raises(ValueError, match="dataset"):
            metadata.bind((), ((0,),), ((tuple(range(1, 1 + widths[0])),),), (), ())

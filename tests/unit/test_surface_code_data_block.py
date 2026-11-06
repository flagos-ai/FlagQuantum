"""Check the space, consumption and storage-error rules of each data layout."""

from dataclasses import asdict

import pytest

from flagquantum.compiler._component_trace.resource_model import GateCost
from flagquantum.qec.resources.data_block import DataBlock, DataLayout

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("layout", "tiles", "cycles"),
    [
        ("simple", 8, 0),
        ("compact", 8, 945),
        ("intermediate", 10, 525),
        ("fast", 18, 105),
    ],
)
def test_layouts(layout: DataLayout, tiles: int, cycles: int) -> None:
    block = DataBlock(layout, 7)
    assert block.tiles(5) == tiles
    assert block.physical_qubits(5) == 98 * tiles
    assert block.consumption_cycles(GateCost(t=1, toffoli=3, rotation=1)) == cycles
    assert block.storage_error(5, 40, 1e-5) == pytest.approx(tiles * 40e-5)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    ("layout", "source_name"),
    [
        ("simple", "SimpleDataBlock"),
        ("compact", "CompactDataBlock"),
        ("intermediate", "IntermediateDataBlock"),
        ("fast", "FastDataBlock"),
    ],
)
@pytest.mark.parametrize("width", [0, 1, 100, 12581])
def test_layout_matches_source(
    layout: DataLayout, source_name: str, width: int
) -> None:
    source = pytest.importorskip("qualtran.surface_code")
    from qualtran.resource_counting import GateCounts

    block = DataBlock(layout, 3)
    reference = getattr(source, source_name)(data_d=3)
    error = source.LogicalErrorModel(
        physical_error=1e-3, qec_scheme=source.QECScheme.make_gidney_fowler()
    )
    gates = GateCost(t=9, cswap=3, and_bloq=5, rotation=1, measurement=7)
    assert block.tiles(width) == reference.n_tiles(width)
    assert block.physical_qubits(width) == reference.n_physical_qubits(width)
    assert block.consumption_cycles(gates) == reference.n_cycles(
        GateCounts(**asdict(gates)), error
    )
    assert block.storage_error(width, 3, error(3)) == reference.data_error(
        width, 3, error
    )


def test_data_block_rejects_invalid_width_and_distance() -> None:
    with pytest.raises(ValueError, match="distance"):
        DataBlock("fast", 0)
    with pytest.raises(ValueError, match="width"):
        DataBlock("fast", 3).tiles(-1)

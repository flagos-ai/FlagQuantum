"""Check ownership and exact reconstruction in the public DEM example."""

from unittest.mock import Mock

import pytest

from examples.qec import dem_model_and_chunks as example
from flagquantum.qec import dem_close_all

pytestmark = pytest.mark.unit


def test_round_trip_keeps_shared_fault_once(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    close = Mock(wraps=dem_close_all)
    monkeypatch.setattr(example, "dem_close_all", close)

    example._round_trip()

    close.assert_called_once()
    (chunks,) = close.call_args.args
    first, second = chunks
    assert [(error.probability, error.observables) for error in first.model.errors] == [
        (0.125, ()),
        (0.25, (0,)),
    ]
    assert [error.probability for error in second.model.errors] == [0.375]
    assert capsys.readouterr().out.splitlines() == ["[2, 1]", "True"]


def test_sparse_projection_shares_fault_columns(
    capsys: pytest.CaptureFixture[str],
) -> None:
    example._sparse_projection()
    assert capsys.readouterr().out.splitlines() == [
        "((0,), (0, 1, 2), (2,))",
        "((1,),)",
    ]

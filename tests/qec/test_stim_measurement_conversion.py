"""Exercise the public conversion contract without loading Stim."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from flagquantum.qec.adapters import stim as bridge

pytestmark = pytest.mark.unit


def test_consumer_accepts_replacement_conversion_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = "R 0\nX 0\nM 0\nDETECTOR rec[-1]"
    raw = torch.tensor([[1], [0]], dtype=torch.uint8)
    sweeps = torch.empty((2, 0), dtype=torch.bool)

    def convert(
        circuit_text: str,
        measurements: torch.Tensor,
        *,
        sweep_bits: torch.Tensor | None,
        bit_packed_measurements: bool,
        bit_packed_sweep_bits: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert circuit_text == text
        assert measurements is raw
        assert sweep_bits is sweeps
        assert bit_packed_measurements and not bit_packed_sweep_bits
        return torch.tensor([[False], [True]]), torch.empty((2, 0), dtype=torch.bool)

    monkeypatch.setattr(bridge, "_convert_stim_measurements", convert)
    batch = bridge.convert_stim_measurements(
        text, raw, sweep_bits=sweeps, bit_packed_measurements=True
    )
    assert batch.detectors.tolist() == [[False], [True]]
    assert batch.observables.shape == (2, 0)


@pytest.mark.parametrize(
    ("measurements", "kwargs", "error"),
    [
        (torch.zeros(2), {}, ValueError),
        (torch.zeros((2, 1)), {}, TypeError),
        (torch.tensor([[2]]), {}, ValueError),
        (torch.tensor([[False]]), {"bit_packed_measurements": True}, TypeError),
        (torch.tensor([[False]]), {"bit_packed_measurements": 1}, TypeError),
        (torch.tensor([[False]]), {"bit_packed_sweep_bits": True}, ValueError),
        (
            torch.tensor([[False]]),
            {"sweep_bits": torch.zeros((2, 1), dtype=torch.bool)},
            ValueError,
        ),
    ],
)
def test_invalid_records_fail_before_engine(
    monkeypatch: pytest.MonkeyPatch,
    measurements: torch.Tensor,
    kwargs: dict[str, Any],
    error: type[Exception],
) -> None:
    def convert(*args: object, **options: object) -> None:
        pytest.fail("invalid record reached the conversion engine")

    monkeypatch.setattr(bridge, "_convert_stim_measurements", convert)
    with pytest.raises(error):
        bridge.convert_stim_measurements("M 0", measurements, **kwargs)

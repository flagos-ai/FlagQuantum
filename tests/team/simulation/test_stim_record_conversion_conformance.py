"""Compare fixed measurement records against Stim and independent parity."""

from __future__ import annotations

import pytest
import torch

from flagquantum.errors import CapabilityError
from flagquantum.qec.execution.measurement_conversion import convert_measurement_records
from flagquantum.simulation.stabilizer import _record_conversion

pytestmark = pytest.mark.integration

SWEEP = "R 0\nCX sweep[0] 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]"


@pytest.mark.parametrize("packed_records", [False, True])
@pytest.mark.parametrize("packed_sweeps", [False, True])
def test_sweep_controls_agree_with_xor_and_upstream(
    packed_records: bool, packed_sweeps: bool
) -> None:
    stim = pytest.importorskip("stim")
    raw = torch.tensor([[0], [1], [0], [1]], dtype=torch.bool)
    sweeps = torch.tensor([[0], [1], [1], [0]], dtype=torch.bool)
    records = raw.to(torch.uint8) if packed_records else raw
    controls = sweeps.to(torch.uint8) if packed_sweeps else sweeps
    output = convert_measurement_records(
        SWEEP,
        records,
        circuit_format="stim",
        sweep_bits=controls,
        bit_packed_measurements=packed_records,
        bit_packed_sweep_bits=packed_sweeps,
    )
    detectors, observables = (
        stim.Circuit(SWEEP)
        .compile_m2d_converter()
        .convert(
            measurements=records.numpy(),
            sweep_bits=controls.numpy(),
            separate_observables=True,
        )
    )
    assert torch.equal(output.detectors, raw ^ sweeps)
    assert torch.equal(output.observables, raw ^ sweeps)
    assert torch.equal(output.detectors, torch.from_numpy(detectors))
    assert torch.equal(output.observables, torch.from_numpy(observables))


def test_lsb_byte_boundary_and_missing_observable_columns() -> None:
    pytest.importorskip("stim")
    text = "R 0\nX 0\nM 0 0 0 0 0 0 0 0 0\nDETECTOR rec[-1] rec[-9]\nOBSERVABLE_INCLUDE(2) rec[-1]"
    dense = torch.ones((9, 2), dtype=torch.bool).T
    dense[1, 8] = False
    expected_detectors = [[False], [True]]
    expected_observables = [[False, False, False], [False, False, True]]
    for record, packed in (
        (dense, False),
        (torch.tensor([[255, 1], [255, 0]], dtype=torch.uint8), True),
    ):
        output = convert_measurement_records(
            text, record, circuit_format="stim", bit_packed_measurements=packed
        )
        assert output.detectors.tolist() == expected_detectors
        assert output.observables.tolist() == expected_observables


def test_omitted_sweeps_and_zero_shots() -> None:
    pytest.importorskip("stim")
    assert convert_measurement_records(
        SWEEP, torch.tensor([[1]]), circuit_format="stim"
    ).detectors.tolist() == [[True]]
    batch = convert_measurement_records(
        "R 0\nM 0\nDETECTOR rec[-1]",
        torch.empty((0, 1), dtype=torch.bool),
        circuit_format="stim",
    )
    assert batch.detectors.shape == (0, 1)
    assert batch.observables.shape == (0, 0)


def test_refuses_width_padding_and_nondeterminism() -> None:
    pytest.importorskip("stim")
    with pytest.raises(ValueError, match="width"):
        convert_measurement_records(
            SWEEP, torch.zeros((2, 2), dtype=torch.bool), circuit_format="stim"
        )
    with pytest.raises(ValueError, match="padding"):
        convert_measurement_records(
            SWEEP,
            torch.tensor([[2]], dtype=torch.uint8),
            circuit_format="stim",
            bit_packed_measurements=True,
        )
    with pytest.raises(ValueError, match="padding"):
        convert_measurement_records(
            SWEEP,
            torch.zeros((1, 1), dtype=torch.bool),
            circuit_format="stim",
            sweep_bits=torch.tensor([[128]], dtype=torch.uint8),
            bit_packed_sweep_bits=True,
        )
    with pytest.raises(CapabilityError, match="nondeterministic") as caught:
        convert_measurement_records(
            "H 0\nM 0\nDETECTOR rec[-1]",
            torch.zeros((1, 1), dtype=torch.bool),
            circuit_format="stim",
        )
    assert isinstance(caught.value.__cause__, ValueError)


def test_sdk_buffers_are_validated_and_owned() -> None:
    pytest.importorskip("stim")
    source = torch.tensor([[True]])
    owned = _record_conversion._owned_stim_conversion_bits(source.numpy(), (1, 1))
    source.zero_()
    assert owned.tolist() == [[True]]
    with pytest.raises(CapabilityError, match="dtype or shape"):
        _record_conversion._owned_stim_conversion_bits(source.numpy(), (2, 1))
    with pytest.raises(CapabilityError, match="dtype or shape"):
        _record_conversion._owned_stim_conversion_bits(
            torch.tensor([[float("nan")]]).numpy(), (1, 1)
        )
    with pytest.raises(CapabilityError, match="invalid conversion buffer") as caught:
        _record_conversion._owned_stim_conversion_bits([[True]], (1, 1))
    assert isinstance(caught.value.__cause__, TypeError)

"""Convert externally recorded measurement data into QEC events."""

from __future__ import annotations

from typing import Literal

import torch

from ...errors import CapabilityError
from ...simulation.stabilizer.engine import _convert_stim_measurements
from ..dem.model import DemSample

__all__ = ("convert_measurement_records",)


def _check_record(record: torch.Tensor, packed: bool, name: str) -> None:
    if not isinstance(record, torch.Tensor) or record.layout != torch.strided:
        raise TypeError(f"{name} must be a dense tensor")
    if record.device.type != "cpu":
        raise CapabilityError(f"{name} must be on CPU")
    if record.ndim != 2:
        raise ValueError(f"{name} must have shot and record dimensions")
    if packed:
        if record.dtype != torch.uint8:
            raise TypeError(f"packed {name} must be uint8")
        return
    if record.dtype == torch.bool:
        return
    if record.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
        torch.uint16,
        torch.uint32,
        torch.uint64,
    ):
        raise TypeError(f"{name} must have bool or integer dtype")
    if not bool(((record == 0) | (record == 1)).all()):
        raise ValueError(f"{name} must be binary")


def convert_measurement_records(
    circuit_text: str,
    measurements: torch.Tensor,
    *,
    circuit_format: Literal["stim"],
    sweep_bits: torch.Tensor | None = None,
    bit_packed_measurements: bool = False,
    bit_packed_sweep_bits: bool = False,
) -> DemSample:
    """Convert external CPU records to reference-normalized detector and logical flips.

    Each input uses shot-major rows. Packed inputs use LSB-first uint8 bytes
    with zero padding. Omitted sweep controls are zero. The circuit format
    identifies the source text dialect. The current implementation requires
    the Stim extra and refuses nondeterministic declared parities.

    Examples:
        >>> from importlib.util import find_spec
        >>> text = "R 0\\nX 0\\nM 0\\nDETECTOR rec[-1]"
        >>> if find_spec("stim") is not None:
        ...     batch = convert_measurement_records(
        ...         text, torch.tensor([[1], [0]]), circuit_format="stim"
        ...     )
        ...     assert batch.detectors.tolist() == [[False], [True]]
    """
    if circuit_format != "stim":
        raise CapabilityError("unsupported measurement circuit format")
    if (
        type(bit_packed_measurements) is not bool
        or type(bit_packed_sweep_bits) is not bool
    ):
        raise TypeError("packed flags must be bool")
    _check_record(measurements, bit_packed_measurements, "measurements")
    if sweep_bits is None:
        if bit_packed_sweep_bits:
            raise ValueError("packed sweep controls require sweep_bits")
    else:
        _check_record(sweep_bits, bit_packed_sweep_bits, "sweep_bits")
        if sweep_bits.shape[0] != measurements.shape[0]:
            raise ValueError(
                "sweep_bits and measurements must have the same shot count"
            )
    detectors, observables = _convert_stim_measurements(
        circuit_text,
        measurements,
        sweep_bits=sweep_bits,
        bit_packed_measurements=bit_packed_measurements,
        bit_packed_sweep_bits=bit_packed_sweep_bits,
    )
    return DemSample(detectors, observables)

"""Annotated Stim experiments using FlagQuantum's existing sampling lifecycle.

QEC supplies memory placement and parity maps. Simulation owns SDK parsing,
numerical execution and text conversion; no vendor object leaves that boundary.
"""

from __future__ import annotations

import torch

from ...errors import CapabilityError
from ...noise import NoiseModel
from ...simulation.stabilizer.engine import (
    _annotated_stim_text,
    _convert_stim_measurements,
    _prepare_stim_experiment,
    _stim_experiment_dem_text,
)
from ..circuits.memory import MemoryCircuit
from ..dem.model import DemSample
from ..execution.sampling import (
    MemorySampler,
    _checked_seed,
    _measurement_plan,
    _memory_sampler,
    _parity_columns,
    _positioned_noise_program,
)
from ..noise import PhenomenologicalNoise

__all__ = (
    "memory_to_stim_text",
    "compile_stim_sampler",
    "stim_circuit_to_dem_text",
    "convert_stim_measurements",
)


def memory_to_stim_text(
    circuit: MemoryCircuit,
    *,
    noise: PhenomenologicalNoise | NoiseModel,
) -> str:
    """Export the positioned memory experiment and its declared parities.

    Noise is read from the actual executed Kraus law. No geometric coordinates
    are invented for native layouts. Requires the optional Stim extra.

    Examples:
        >>> from importlib.util import find_spec
        >>> from flagquantum.qec import RepetitionCode, build_memory_circuit
        >>> if find_spec("stim") is not None:
        ...     memory = build_memory_circuit(RepetitionCode(distance=3), rounds=3)
        ...     text = memory_to_stim_text(memory, noise=PhenomenologicalNoise())
        ...     assert "DETECTOR" in text
    """
    if not isinstance(circuit, MemoryCircuit):
        raise TypeError("circuit must be a MemoryCircuit")
    if not isinstance(noise, (PhenomenologicalNoise, NoiseModel)):
        raise TypeError("noise must be a PhenomenologicalNoise or a NoiseModel")
    plan = _measurement_plan(circuit)
    program = _positioned_noise_program(circuit, plan, noise)
    detectors, observables = _parity_columns(circuit, plan)
    return _annotated_stim_text(
        program,
        terminal_qubits=tuple(circuit.code.data_qubits),
        detector_columns=detectors,
        observable_columns=observables,
    )


def compile_stim_sampler(
    circuit_text: str,
    *,
    seed: int | None = None,
) -> MemorySampler:
    """Compile original external text into reference-normalized flip batches.

    Requires deterministic noiseless detector/logical parities and at least one
    detector. Returns the existing owner-bound MemorySampler, including its
    close, poisoning, seed and no-copy contracts. Noisy DEM analysis is separate.

    Examples:
        >>> from importlib.util import find_spec
        >>> text = "R 0\\nX 0\\nM 0\\nDETECTOR rec[-1]"
        >>> if find_spec("stim") is not None:
        ...     with compile_stim_sampler(text, seed=53) as sampler:
        ...         batch = sampler.sample(shots=4)
        ...         assert not batch.detectors.any()  # Ideal raw ones are not flips.
    """
    seed = _checked_seed(seed)
    prepared, detectors, observables = _prepare_stim_experiment(circuit_text, seed=seed)
    try:
        return _memory_sampler(
            prepared,
            tuple((index,) for index in range(detectors)),
            tuple((detectors + index,) for index in range(observables)),
        )
    except BaseException:
        prepared.close()
        raise


def stim_circuit_to_dem_text(
    circuit_text: str,
    *,
    decompose_errors: bool = False,
) -> str:
    """Convert strictly to DEM text, preserving coordinates and complete shape.

    Missing detector/logical declarations are supplied for the existing reader.
    Unsupported exact conversion raises CapabilityError; no disjoint-error
    approximation is enabled. Decomposition affects decoding, never sampling.

    Examples:
        >>> from importlib.util import find_spec
        >>> from flagquantum.qec import DetectorErrorModel
        >>> text = "R 0\\nX_ERROR(0.1) 0\\nM 0\\nDETECTOR rec[-1]"
        >>> if find_spec("stim") is not None:
        ...     model = DetectorErrorModel.from_stim_text(stim_circuit_to_dem_text(text))
        ...     assert model.num_detectors == 1
    """
    if not isinstance(decompose_errors, bool):
        raise TypeError("decompose_errors must be a boolean")
    return _stim_experiment_dem_text(circuit_text, decompose_errors=decompose_errors)


def _check_stim_record(record: torch.Tensor, packed: bool, name: str) -> None:
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


def convert_stim_measurements(
    circuit_text: str,
    measurements: torch.Tensor,
    *,
    sweep_bits: torch.Tensor | None = None,
    bit_packed_measurements: bool = False,
    bit_packed_sweep_bits: bool = False,
) -> DemSample:
    """Convert external CPU records to reference-normalized detector and logical flips.

    Each input uses shot-major rows. Packed inputs use LSB-first uint8 bytes
    with zero padding. Omitted sweep controls are zero. The optional Stim
    extra is required and nondeterministic declared parities are refused.

    Examples:
        >>> from importlib.util import find_spec
        >>> text = "R 0\nX 0\nM 0\nDETECTOR rec[-1]"
        >>> if find_spec("stim") is not None:
        ...     batch = convert_stim_measurements(text, torch.tensor([[1], [0]]))
        ...     assert batch.detectors.tolist() == [[False], [True]]
    """
    if (
        type(bit_packed_measurements) is not bool
        or type(bit_packed_sweep_bits) is not bool
    ):
        raise TypeError("packed flags must be bool")
    _check_stim_record(measurements, bit_packed_measurements, "measurements")
    if sweep_bits is None:
        if bit_packed_sweep_bits:
            raise ValueError("packed sweep controls require sweep_bits")
    else:
        _check_stim_record(sweep_bits, bit_packed_sweep_bits, "sweep_bits")
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

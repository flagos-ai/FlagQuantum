"""Annotated Stim experiments using FlagQuantum's existing sampling lifecycle.

QEC supplies memory placement and parity maps. Simulation owns SDK parsing,
numerical execution and text conversion; no vendor object leaves that boundary.
"""

from __future__ import annotations

from ...noise import NoiseModel
from ...simulation.stabilizer.engine import (
    _annotated_stim_text,
    _prepare_stim_experiment,
    _stim_experiment_dem_text,
)
from ..circuits.memory import MemoryCircuit
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

"""Accept the specified Surface/Repetition distances with fixed noisy CPU shots.

Run this example explicitly in an embedded build with Stim 1.16.0 installed
for memory-to-text export. Each invocation runs one configuration and emits JSON.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from importlib.metadata import version

from flagquantum.qec import (
    DetectorErrorModel,
    Pauli,
    PhenomenologicalNoise,
    RepetitionCode,
    RotatedSurfaceCode,
    build_memory_circuit,
)
from flagquantum.qec.adapters.stim import (
    compile_stim_sampler,
    memory_to_stim_text,
    stim_circuit_to_dem_text,
)
from flagquantum.qec.decoders.sparse_blossom import SparseBlossomDecoder
from flagquantum.qec.dem.graph import DecodingGraph

SURFACE_DISTANCES = (*range(3, 28, 2), 30)
REPETITION_DISTANCES = tuple(range(3, 30, 2))
ROUNDS = 3
SHOTS = 64
BATCH_SIZE = 16


def configurations() -> tuple[tuple[str, int, str], ...]:
    return tuple(
        [
            ("surface", distance, basis)
            for distance in SURFACE_DISTANCES
            for basis in ("x", "z")
        ]
        + [("repetition", distance, "z") for distance in REPETITION_DISTANCES]
    )


def accept_configuration(family: str, distance: int, basis: str) -> dict[str, object]:
    """Run original noisy shots, strict DEM analysis and bundled batch decoding."""
    if (family, distance, basis) not in configurations():
        raise ValueError("configuration is outside the declared acceptance list")
    start = time.monotonic()
    code = (
        RotatedSurfaceCode(distance)
        if family == "surface"
        else RepetitionCode(distance)
    )
    product = Pauli(x_qubits=code.data_qubits) if basis == "x" else None
    memory = build_memory_circuit(code, rounds=ROUNDS, product=product)
    noise = PhenomenologicalNoise(
        data_flip=0.005, phase_flip=0.007, measurement_flip=0.009
    )
    text = memory_to_stim_text(memory, noise=noise)
    # Analyze without decomposition first. The decoding model explicitly requests
    # graphlike decomposition, with disjoint-error approximation still disabled.
    exact_text = stim_circuit_to_dem_text(text, decompose_errors=False)
    exact = DetectorErrorModel.from_stim_text(exact_text)
    decoded_text = stim_circuit_to_dem_text(text, decompose_errors=True)
    model = DetectorErrorModel.from_stim_text(
        decoded_text, use_decomp_suggestions=True
    ).merge_duplicate_mechanisms()
    expected_shape = (len(memory.detectors), len(memory.observables))
    assert (exact.num_detectors, exact.num_observables) == expected_shape
    assert (model.num_detectors, model.num_observables) == expected_shape
    decoder = SparseBlossomDecoder(DecodingGraph.from_detector_error_model(model))
    failures = 0
    observed_events = 0
    seed = 120500 + configurations().index((family, distance, basis))
    with compile_stim_sampler(text, seed=seed) as sampler:
        for _ in range(SHOTS // BATCH_SIZE):
            batch = sampler.sample(shots=BATCH_SIZE)
            predictions = decoder.decode_batch(batch.detectors)
            assert predictions.shape == batch.observables.shape
            assert batch.detectors.shape == (BATCH_SIZE, exact.num_detectors)
            residual = predictions.bool() ^ batch.observables.bool()
            failures += int(residual.any(dim=1).sum())
            observed_events += int(batch.detectors.sum())
    assert observed_events > 0, "the noisy witness must contain detector events"
    return {
        "family": family,
        "distance": distance,
        "basis": basis,
        "rounds": ROUNDS,
        "shots": SHOTS,
        "batch_size": BATCH_SIZE,
        "seed": seed,
        "noise": {"data_flip": 0.005, "phase_flip": 0.007, "measurement_flip": 0.009},
        "qubits": code.num_data_qubits + code.num_ancilla_qubits,
        "detectors": exact.num_detectors,
        "observables": exact.num_observables,
        "exact_mechanisms": len(exact.errors),
        "decoder": "sparse_blossom",
        "decoding_decomposition": True,
        "approximate_disjoint_errors": False,
        "logical_failures": failures,
        "detector_events": observed_events,
        "source_bytes": len(text.encode()),
        "strict_dem_bytes": len(exact_text.encode()),
        "seconds": time.monotonic() - start,
        "python": platform.python_version(),
        "torch": version("torch"),
        "stim": version("stim"),
        "sampling_source": "original_memory_circuit",
        "status": "passed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", choices=("surface", "repetition"))
    parser.add_argument("distance", type=int)
    parser.add_argument("basis", choices=("x", "z"))
    args = parser.parse_args()
    print(
        json.dumps(accept_configuration(args.family, args.distance, args.basis)),
        flush=True,
    )


if __name__ == "__main__":
    main()

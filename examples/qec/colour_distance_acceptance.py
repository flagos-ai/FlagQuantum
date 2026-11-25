"""Check one triangular 6.6.6 colour patch with original noisy CPU shots.

Requires an embedded FlagQuantum build and Stim 1.16.0 for text export and the
independent strict-DEM reference. Run: python -m examples.qec.colour_distance_acceptance 15 z
"""

from __future__ import annotations

import argparse
import json
import math
import time
from typing import Literal, TypedDict

from flagquantum.qec import (
    DetectorErrorModel,
    Pauli,
    PhenomenologicalNoise,
    StabilizerCode,
    build_memory_circuit,
    triangular_colour_code,
)
from flagquantum.qec.adapters.stim import (
    compile_stim_sampler,
    memory_to_stim_text,
    stim_circuit_to_dem_text,
)
from flagquantum.qec.decoders.belief_propagation import BeliefPropagationDecoder

DISTANCES = (3, 5, 7, 9, 11, 13, 15)
ROUNDS = 3
SHOTS = 64
BATCH_SIZE = 16


class ColourAcceptance(TypedDict):
    distance: int
    basis: str
    data_qubits: int
    ancilla_qubits: int
    detectors: int
    observables: int
    mechanisms: int
    hyperedges: int
    rounds: int
    shots: int
    seed: int
    logical_failures: int
    detector_events: int
    osd_fallback_shots: int
    syndrome_reproduced: int
    constructor_seconds: float
    seconds: float
    decoder: str
    approximate_disjoint_errors: bool
    stim_version: str
    stim_reference: str
    status: str


def accept_configuration(distance: int, basis: Literal["x", "z"]) -> ColourAcceptance:
    """Verify strict DEM mechanisms and decode every original-circuit syndrome."""
    if distance not in DISTANCES or basis not in ("x", "z"):
        raise ValueError("configuration is outside the declared colour distance list")
    import stim

    if stim.__version__ != "1.16.0":
        raise RuntimeError("this acceptance requires Stim 1.16.0")
    started = time.monotonic()
    code = triangular_colour_code(distance)
    constructor_seconds = time.monotonic() - started
    assert code.distance is not None
    assert isinstance(code, StabilizerCode)
    assert (code.distance, code.x_distance, code.z_distance) == (distance,) * 3
    memory = build_memory_circuit(
        code,
        rounds=ROUNDS,
        product=Pauli(x_qubits=code.data_qubits) if basis == "x" else None,
    )
    noise = PhenomenologicalNoise(
        data_flip=0.005, phase_flip=0.007, measurement_flip=0.009
    )
    text = memory_to_stim_text(memory, noise=noise)
    model = DetectorErrorModel.from_stim_text(
        stim_circuit_to_dem_text(text, decompose_errors=False)
    ).merge_duplicate_mechanisms()
    reference = stim.Circuit(text).detector_error_model(
        decompose_errors=False, approximate_disjoint_errors=False
    )
    assert (
        (model.num_detectors, model.num_observables)
        == (
            reference.num_detectors,
            reference.num_observables,
        )
        == (len(memory.detectors), len(memory.observables))
    )
    actual_errors = sorted(
        (error.detectors, error.observables, error.probability)
        for error in model.errors
    )
    # Read the SDK instructions directly: its printer may omit declarations for
    # detectors already named by errors, which our strict text reader refuses.
    expected_errors = sorted(
        (
            tuple(
                target.val
                for target in instruction.targets_copy()
                if target.is_relative_detector_id()
            ),
            tuple(
                target.val
                for target in instruction.targets_copy()
                if target.is_logical_observable_id()
            ),
            instruction.args_copy()[0],
        )
        for instruction in reference.flattened()
        if instruction.type == "error"
    )
    assert len(actual_errors) == len(expected_errors)
    for actual, wanted in zip(actual_errors, expected_errors, strict=True):
        assert actual[:2] == wanted[:2]
        assert math.isclose(actual[2], wanted[2], rel_tol=1e-13, abs_tol=1e-15)
    decoder = BeliefPropagationDecoder.from_detector_error_model(model)
    failures = events = fallbacks = reproduced = 0
    seed = 120800 + distance * 2 + int(basis == "x")
    with compile_stim_sampler(text, seed=seed) as sampler:
        for _ in range(SHOTS // BATCH_SIZE):
            batch = sampler.sample(shots=BATCH_SIZE)
            assert batch.detectors.shape == (BATCH_SIZE, model.num_detectors)
            assert batch.observables.shape == (BATCH_SIZE, model.num_observables)
            for row in range(BATCH_SIZE):
                syndrome = tuple(batch.detectors[row].nonzero().flatten().tolist())
                correction = decoder.decode(syndrome)
                reconstructed: set[int] = set()
                for index in correction.mechanisms:
                    reconstructed.symmetric_difference_update(
                        model.errors[index].detectors
                    )
                assert reconstructed == set(syndrome)
                observed = set(batch.observables[row].nonzero().flatten().tolist())
                failures += int(bool(observed ^ set(correction.observables)))
                events += len(syndrome)
                fallbacks += int(not correction.converged)
                reproduced += 1
    assert events > 0 and reproduced == SHOTS
    return {
        "distance": distance,
        "basis": basis,
        "data_qubits": code.num_data_qubits,
        "ancilla_qubits": code.num_ancilla_qubits,
        "detectors": model.num_detectors,
        "observables": model.num_observables,
        "mechanisms": len(model.errors),
        "hyperedges": sum(len(error.detectors) > 2 for error in model.errors),
        "rounds": ROUNDS,
        "shots": SHOTS,
        "seed": seed,
        "logical_failures": failures,
        "detector_events": events,
        "osd_fallback_shots": fallbacks,
        "syndrome_reproduced": reproduced,
        "constructor_seconds": constructor_seconds,
        "seconds": time.monotonic() - started,
        "decoder": "belief_propagation_osd0",
        "approximate_disjoint_errors": False,
        "stim_version": stim.__version__,
        "stim_reference": "exact_undecomposed_mechanisms_and_shape",
        "status": "passed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("distance", type=int, choices=DISTANCES)
    parser.add_argument("basis", choices=("x", "z"))
    args = parser.parse_args()
    basis: Literal["x", "z"] = "x" if args.basis == "x" else "z"
    print(json.dumps(accept_configuration(args.distance, basis)), flush=True)


if __name__ == "__main__":
    main()

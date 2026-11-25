"""Check original noisy Toric shots, strict DEMs and both logical observables.

Requires an embedded FlagQuantum build and Stim 1.16.0 for memory text export
and the independent reference. Run: python -m examples.qec.toric_distance_acceptance 15 z
"""

from __future__ import annotations

import argparse
import json
import math
import time
from typing import Literal, TypedDict

from flagquantum.qec import (
    DetectorErrorModel,
    PhenomenologicalNoise,
    StabilizerCode,
    build_memory_circuit,
    toric_code,
)
from flagquantum.qec.adapters.stim import (
    compile_stim_sampler,
    memory_to_stim_text,
    stim_circuit_to_dem_text,
)
from flagquantum.qec.decoders.belief_propagation import BeliefPropagationDecoder

DISTANCES = (3, 5, 7, 9, 11, 13, 15)


class ToricAcceptance(TypedDict):
    distance: int
    basis: str
    noise_profile: str
    data_qubits: int
    ancilla_qubits: int
    detectors: int
    observables: int
    mechanisms: int
    hyperedges: int
    rounds: int
    shots: int
    seed: int
    logical_residual_shots: int
    logical_residuals: list[int]
    detector_events: int
    osd_fallback_shots: int
    syndrome_reproduced: int
    constructor_seconds: float
    seconds: float
    approximate_disjoint_errors: bool
    stim_version: str
    status: str


def accept_configuration(
    distance: int,
    basis: Literal["x", "z"],
    noise_profile: Literal["independent", "correlated"] = "independent",
) -> ToricAcceptance:
    """Compare every strict DEM mechanism and decode each original-circuit shot."""
    if distance not in DISTANCES or basis not in ("x", "z"):
        raise ValueError("configuration is outside the declared Toric distance list")
    if noise_profile not in ("independent", "correlated"):
        raise ValueError("unknown Toric noise profile")
    import stim

    if stim.__version__ != "1.16.0":
        raise RuntimeError("this acceptance requires Stim 1.16.0")
    started = time.monotonic()
    code = toric_code(distance)
    constructor_seconds = time.monotonic() - started
    assert code.distance is not None
    assert isinstance(code, StabilizerCode)
    assert (code.distance, code.x_distance, code.z_distance) == (distance,) * 3
    memory = build_memory_circuit(code, rounds=3, readout_basis=basis)
    noise = PhenomenologicalNoise(
        data_flip=0.005,
        phase_flip=0.007,
        both_flip=0.003 if noise_profile == "correlated" else 0.0,
        measurement_flip=0.009,
    )
    text = memory_to_stim_text(memory, noise=noise)
    model = DetectorErrorModel.from_stim_text(
        stim_circuit_to_dem_text(text, decompose_errors=False)
    ).merge_duplicate_mechanisms()
    reference = stim.Circuit(text).detector_error_model(
        decompose_errors=False, approximate_disjoint_errors=False
    )
    assert (
        model.num_observables
        == reference.num_observables
        == len(memory.observables)
        == 2
    )
    assert model.num_detectors == reference.num_detectors == len(memory.detectors)
    actual = sorted((e.detectors, e.observables, e.probability) for e in model.errors)
    expected = sorted(
        (
            tuple(
                t.val for t in instruction.targets_copy() if t.is_relative_detector_id()
            ),
            tuple(
                t.val
                for t in instruction.targets_copy()
                if t.is_logical_observable_id()
            ),
            instruction.args_copy()[0],
        )
        for instruction in reference.flattened()
        if instruction.type == "error"
    )
    assert len(actual) == len(expected)
    for observed, wanted in zip(actual, expected, strict=True):
        assert observed[:2] == wanted[:2]
        assert math.isclose(observed[2], wanted[2], rel_tol=1e-13, abs_tol=1e-15)
    hyperedges = sum(len(e.detectors) > 2 for e in model.errors)
    assert (hyperedges > 0) == (noise_profile == "correlated")
    decoder = BeliefPropagationDecoder.from_detector_error_model(model)
    failures = events = fallbacks = reproduced = 0
    residuals = [0, 0]
    seed = 121200 + distance * 2 + int(basis == "x")
    with compile_stim_sampler(text, seed=seed) as sampler:
        for _ in range(4):
            batch = sampler.sample(shots=16)
            assert batch.detectors.shape == (16, model.num_detectors)
            assert batch.observables.shape == (16, 2)
            for row in range(16):
                syndrome = tuple(batch.detectors[row].nonzero().flatten().tolist())
                correction = decoder.decode(syndrome)
                reconstructed: set[int] = set()
                for index in correction.mechanisms:
                    reconstructed.symmetric_difference_update(
                        model.errors[index].detectors
                    )
                assert reconstructed == set(syndrome)
                observed_flips = set(
                    batch.observables[row].nonzero().flatten().tolist()
                )
                residual = observed_flips ^ set(correction.observables)
                failures += int(bool(residual))
                for index in residual:
                    residuals[index] += 1
                events += len(syndrome)
                fallbacks += int(not correction.converged)
                reproduced += 1
    assert events > 0 and reproduced == 64
    return {
        "distance": distance,
        "basis": basis,
        "noise_profile": noise_profile,
        "data_qubits": code.num_data_qubits,
        "ancilla_qubits": code.num_ancilla_qubits,
        "detectors": model.num_detectors,
        "observables": model.num_observables,
        "mechanisms": len(model.errors),
        "hyperedges": hyperedges,
        "rounds": 3,
        "shots": 64,
        "seed": seed,
        "logical_residual_shots": failures,
        "logical_residuals": residuals,
        "detector_events": events,
        "osd_fallback_shots": fallbacks,
        "syndrome_reproduced": reproduced,
        "constructor_seconds": constructor_seconds,
        "seconds": time.monotonic() - started,
        "approximate_disjoint_errors": False,
        "stim_version": stim.__version__,
        "status": "passed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("distance", type=int, choices=DISTANCES)
    parser.add_argument("basis", choices=("x", "z"))
    parser.add_argument(
        "--noise-profile", choices=("independent", "correlated"), default="independent"
    )
    args = parser.parse_args()
    basis: Literal["x", "z"] = "x" if args.basis == "x" else "z"
    profile: Literal["independent", "correlated"] = (
        "correlated" if args.noise_profile == "correlated" else "independent"
    )
    print(json.dumps(accept_configuration(args.distance, basis, profile)), flush=True)


if __name__ == "__main__":
    main()

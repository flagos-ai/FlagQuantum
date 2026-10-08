"""Decode a Stim circuit's detector error model with FlagQuantum's own decoder.

The workflow this script demonstrates is the one a Stim user runs today:
generate a memory experiment with :mod:`stim`, ask it for a detector error
model, sample detection events, and decode them into logical corrections.  The
script replaces the last three steps with this package's own reader, model
sampler, and matcher, and keeps Stim for the circuit layer, which is the
boundary `docs/guides/STIM_USER_MIGRATION.md` documents.

What it is worth watching for is the reading of the model text.  Stim writes a
detector error model whose mechanisms can flip three or more detectors, and its
decomposed variant splits those into graphlike pieces with a ``^`` separator.
The two readings are not equivalent, and the script prints both: the default
reading states the line as written and reproduces Stim's own sampler's
observable marginal to three parts in ten thousand, while the graphlike reading
is the one a minimum-weight matcher accepts and carries a marginal that is
measurably higher.  The script also reports the per-shot agreement between the
in-tree matcher and the PyMatching cross-check, because a logical error rate at
this distance does not separate the two routes and the agreement does.

This is a local, single-process demonstration of an experimental QEC surface.
It establishes no threshold and no scalability claim.

Run it with:

    python -m examples.qec.stim_user_migration

The ``stim`` extra is required, and the ``pymatching`` extra is required for the
cross-check section; both are declared in ``pyproject.toml``.

"""

from __future__ import annotations

import argparse

import numpy as np
import stim

from flagquantum.errors import CapabilityError
from flagquantum.qec import (
    BELIEF_PROPAGATION_OSD_NAME,
    DetectorErrorModel,
    MatchingDependencyError,
    decoder_names,
    get_decoder,
)

LABEL_WIDTH = 34
SURFACE_CODE = "surface_code:rotated_memory_z"
REPETITION_CODE = "repetition_code:memory"
#: The shot count the reference marginal is estimated at. The estimate's own
#: standard error is reported beside it, because a marginal that agrees with the
#: sampler to within a few standard errors agrees and one that misses by fifty
#: does not, and the difference between those two verdicts is the shot count.
REFERENCE_SHOTS = 200000


def report(label: str, value: object) -> None:
    print(f"  {label:<{LABEL_WIDTH}}: {value}")


def decode_all(decoder: object, syndromes: np.ndarray) -> np.ndarray:
    """Decode one bit row per shot into a logical prediction per shot."""

    return np.array(
        [
            bool(decoder.decode(np.flatnonzero(row)).observables)  # type: ignore[attr-defined]
            for row in syndromes
        ]
    )


def repetition_route(distance: int, rounds: int, shots: int, seed: int) -> None:
    """Read a Stim-written model and decode it without Stim's sampler."""

    print("repetition code: read the model Stim wrote")
    circuit = stim.Circuit.generated(
        REPETITION_CODE,
        distance=distance,
        rounds=rounds,
        before_round_data_depolarization=0.05,
        before_measure_flip_probability=0.02,
        after_reset_flip_probability=0.01,
    )
    model = circuit.detector_error_model()
    raw_text = str(model)
    report("stim text lines", len(raw_text.splitlines()))
    report(
        "repeat blocks",
        sum(1 for line in raw_text.splitlines() if line.strip().startswith("repeat")),
    )
    try:
        DetectorErrorModel.from_stim_text(raw_text)
    except ValueError as error:
        report("reader refusal", str(error)[:52] + "...")
    flattened = DetectorErrorModel.from_stim_text(str(model.flattened()))
    report(
        "detectors / observables",
        f"{flattened.num_detectors} / {flattened.num_observables}",
    )
    report("mechanisms", len(flattened.errors))
    report("signatures unique", flattened.mechanisms_are_unique())

    merged = flattened.merge_duplicate_mechanisms()
    report("after merge_duplicate", f"{len(merged.errors)} mechanisms")
    sample = merged.dem_sampling(shots=shots, seed=seed)
    report("sampled detection events", tuple(sample.detectors.shape))
    matcher = get_decoder("minimum_weight_matching", merged)
    predictions = decode_all(matcher, sample.detectors.numpy().astype(bool))
    answers = sample.observables.numpy()[:, 0].astype(bool)
    report("logical error rate", round(float(np.mean(predictions != answers)), 4))
    report("registered decoders", ", ".join(decoder_names()))


def surface_route(distance: int, rounds: int, shots: int, seed: int) -> None:
    """Compare the two readings of Stim's ``^`` separator."""

    print()
    print("surface code: the two readings of a decomposed mechanism")
    circuit = stim.Circuit.generated(
        SURFACE_CODE,
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=0.01,
        before_measure_flip_probability=0.01,
        after_reset_flip_probability=0.01,
    )
    combined = DetectorErrorModel.from_stim_text(str(circuit.detector_error_model()))
    expanded = DetectorErrorModel.from_stim_text(
        str(circuit.detector_error_model(decompose_errors=True)),
        use_decomp_suggestions=True,
    ).merge_duplicate_mechanisms()
    hyperedges = sum(1 for error in combined.errors if len(error.detectors) > 2)
    report(
        "default reading", f"{len(combined.errors)} mechanisms, {hyperedges} hyperedges"
    )
    report(
        "graphlike reading",
        f"{len(expanded.errors)} mechanisms, unique {expanded.mechanisms_are_unique()}",
    )
    try:
        get_decoder("minimum_weight_matching", combined)
        report("matcher on default", "accepted")
    except CapabilityError as error:
        report("matcher on default", f"{type(error).__name__}: {str(error)[:40]}...")

    report("marginal, default reading", round(float(combined.observable_rates()[0]), 6))
    report(
        "marginal, graphlike reading", round(float(expanded.observable_rates()[0]), 6)
    )
    reference = (
        circuit.detector_error_model()
        .compile_sampler(seed=seed)
        .sample(shots=REFERENCE_SHOTS)[1]
    )
    sampled = float(reference[:, 0].mean())
    standard_error = float(np.sqrt(sampled * (1.0 - sampled) / REFERENCE_SHOTS))
    report("stim's own sampler", round(sampled, 6))
    report("  its standard error", round(standard_error, 6))
    report(
        "default reading misses by",
        f"{abs(float(combined.observable_rates()[0]) - sampled) / standard_error:.1f} standard errors",
    )
    report(
        "graphlike reading misses by",
        f"{abs(float(expanded.observable_rates()[0]) - sampled) / standard_error:.0f} standard errors",
    )

    events, observables = circuit.compile_detector_sampler(seed=seed + 5).sample(
        shots=shots, separate_observables=True
    )
    syndromes = events.astype(bool)
    answers = observables[:, 0].astype(bool)
    matcher = get_decoder("minimum_weight_matching", expanded)
    matching = decode_all(matcher, syndromes)
    report("matching LER", round(float(np.mean(matching != answers)), 4))
    print()
    print("surface code: what a rate comparison cannot see")


def cross_check(distance: int, rounds: int, shots: int, seed: int) -> None:
    """Separate the routes by per-shot agreement rather than by rate."""

    circuit = stim.Circuit.generated(
        SURFACE_CODE,
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=0.01,
        before_measure_flip_probability=0.01,
        after_reset_flip_probability=0.01,
    )
    combined = DetectorErrorModel.from_stim_text(str(circuit.detector_error_model()))
    expanded = DetectorErrorModel.from_stim_text(
        str(circuit.detector_error_model(decompose_errors=True)),
        use_decomp_suggestions=True,
    ).merge_duplicate_mechanisms()
    events, observables = circuit.compile_detector_sampler(seed=seed).sample(
        shots=shots, separate_observables=True
    )
    syndromes = events.astype(bool)
    answers = observables[:, 0].astype(bool)
    try:
        cross = get_decoder("pymatching", expanded)
    except MatchingDependencyError as error:
        report("pymatching cross-check", f"unavailable: {error}")
        return
    matching = decode_all(get_decoder("minimum_weight_matching", expanded), syndromes)
    crossed = decode_all(cross, syndromes)
    hyperedge = decode_all(get_decoder(BELIEF_PROPAGATION_OSD_NAME, combined), syndromes)
    report("shots", shots)
    report("matcher LER", round(float(np.mean(matching != answers)), 4))
    report("pymatching LER", round(float(np.mean(crossed != answers)), 4))
    report("BP+OSD on default LER", round(float(np.mean(hyperedge != answers)), 4))
    report("matcher == pymatching", float(np.mean(matching == crossed)))
    report("matcher == BP+OSD", round(float(np.mean(matching == hyperedge)), 3))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--distance", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--shots", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    print("Stim user migration: local, single process, experimental QEC surface")
    print()
    repetition_route(args.distance, args.rounds, args.shots, args.seed)
    surface_route(args.distance, 3, args.shots // 2, args.seed)
    cross_check(args.distance, 3, 2000, args.seed - 2)


if __name__ == "__main__":
    main()

"""Run a patch whose checks are all mixed, and read what the mixing costs.

Every code record this package shipped before the mixed-check record declares its
checks in two families: a check is an ``X`` product or a ``Z`` product, never both.
That split is what the parity-check matrix route rests on -- an X row and a Z row
are two rows -- and it is also what the memory-circuit readout rests on. A Z-type
check on a ``|0...0>`` preparation has a known value, so its first-round syndrome
is a detector, and its value at the last round is recoverable from a Z-basis data
readout, so the round-0 preparation and the final measurement are both inside the
decoder's model.

A mixed check has neither property, and this script measures the consequence
rather than asserting it. The ZXXZ patch is the rotated surface patch seen after a
Hadamard on the data checkerboard: same qubits, same counts, same distance, but
every check carries an ``X`` factor and a ``Z`` factor on one ancilla, so no check
is pure and nothing anchors the ends of the experiment.

The script holds two patches of the same distance against each other. They occupy
the same nine data qubits and the same eight ancillas, and they are decoded by the
same matcher over the same noise model at the same fault rate and seed, so the only
difference left is the check families. What the two numbers then show is the honest
ceiling of this record: the rotated patch's logical rate falls by an order of
magnitude under decoding and the ZXXZ patch's does not, because a fault that
strikes the preparation or the readout is invisible to a decoder whose model has no
detector there.

This is a record of the shape of a mixed-check experiment, not a distance claim and
not a decoding result: no threshold is computed, no suppression claim is made, and
the contrast is a statement about the two check families rather than about the
matcher's quality.
"""

from __future__ import annotations

import argparse
from collections import Counter

from flagquantum.qec import (
    DetectorErrorModel,
    MinimumWeightMatchingDecoder,
    PhenomenologicalNoise,
    RotatedSurfaceCode,
    ZxxzSurfaceCode,
    build_memory_circuit,
    sample_memory_circuit,
)

#: The distance both patches are built at, so the comparison holds the code's size
#: fixed and varies only the check families.
DISTANCE = 3
#: Syndrome rounds, including the preparation and the readout.
ROUNDS = 3
#: One data rate on both bases and one measurement rate: a depolarizing data fault
#: beside a readout fault, which is the noise both patches are decoded under.
FAULT_RATE = 0.01
SEED = 0
LABEL_WIDTH = 34


def report(label: str, value: object) -> None:
    """Print one indented ``label: value`` line, so the two sections line up.

    The line is indented and carries a colon because that is the shape a test reads
    a transcript in: the values below are the example's claims, and a claim that no
    reader can parse is a claim no test can falsify.
    """

    print(f"  {label + ':':<{LABEL_WIDTH + 1}}{value}")


def check_kinds(code: object) -> Counter[str]:
    """Return how many of a record's checks are pure X, pure Z, and mixed."""

    kinds: Counter[str] = Counter()
    for check in code.checks:  # type: ignore[attr-defined]
        has_x = bool(check.stabilizer.x_qubits)
        has_z = bool(check.stabilizer.z_qubits)
        kinds["mixed" if (has_x and has_z) else ("pure X" if has_x else "pure Z")] += 1
    return kinds


def noise() -> PhenomenologicalNoise:
    """Return the one noise model both patches are sampled and decoded under."""

    return PhenomenologicalNoise(
        data_flip=FAULT_RATE,
        phase_flip=FAULT_RATE,
        both_flip=0.0,
        measurement_flip=FAULT_RATE,
    )


def section(label: str, code: object, shots: int) -> tuple[float, float]:
    """Print one patch's layout and its raw and decoded logical rates.

    Returns:
        The raw and the decoded rate, so the caller compares them after both
        sections have printed rather than interleaving the comparison.
    """

    observable = code.logical_observables[0]  # type: ignore[attr-defined]
    memory = build_memory_circuit(code, rounds=ROUNDS, product=observable)  # type: ignore[arg-type]
    profile = noise()
    model = DetectorErrorModel.from_memory_circuit(memory, noise=profile)
    decoder = MinimumWeightMatchingDecoder.from_detector_error_model(model)
    sample = sample_memory_circuit(memory, noise=profile, shots=shots, seed=SEED)

    # The bulk of the model is one detector per check per round boundary. Anything
    # above that is a detector whose value is known in advance, which is what ties
    # the preparation and the readout into the model.
    bulk = len(code.checks) * (ROUNDS - 1)  # type: ignore[attr-defined]
    anchored = len(memory.detectors) - bulk

    raw = 0
    decoded = 0
    for detector_row, observable_row in zip(
        sample.detectors.tolist(), sample.observables.tolist(), strict=True
    ):
        events = tuple(index for index, bit in enumerate(detector_row) if bit)
        correction = decoder.decode(events)
        # The decoded shot is wrong when the correction and the draw disagree about
        # the observable, which is the correction's flip count modulo two.
        raw += observable_row[0]
        decoded += (len(correction.observables) % 2) ^ observable_row[0]

    print(f"{label}, distance {DISTANCE}, {ROUNDS} rounds")
    report("data qubits, ancillas", f"{code.num_data_qubits}, {code.num_ancilla_qubits}")  # type: ignore[attr-defined]
    report("checks", f"{len(code.checks)} ({dict(check_kinds(code))})")  # type: ignore[attr-defined]
    report("detectors", f"{len(memory.detectors)} = {bulk} bulk + {anchored} anchored")
    report("mechanisms, observables", f"{len(model.errors)}, {model.num_observables}")
    report("shots", shots)
    print()
    return raw / shots, decoded / shots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shots", type=int, default=4000)
    args = parser.parse_args()

    print("A code whose checks are all mixed: local, single process, experimental")
    print()
    mixed = section("ZXXZ patch", ZxxzSurfaceCode(DISTANCE), args.shots)
    rotated = section("Rotated patch", RotatedSurfaceCode(DISTANCE), args.shots)

    print("Comparison")
    report("ZXXZ raw, decoded", f"{mixed[0]:.4f}, {mixed[1]:.4f}")
    report("rotated raw, decoded", f"{rotated[0]:.4f}, {rotated[1]:.4f}")
    print()
    print(
        "The two raw rates agree to "
        f"{abs(mixed[0] - rotated[0]) / max(rotated[0], 1e-12):.1%} of each other, "
        "because the two patches are the same code up to a local Clifford."
    )
    print(
        "Decoding divides the rotated patch's rate by "
        f"{rotated[0] / max(rotated[1], 1e-12):.1f} and the ZXXZ patch's by "
        f"{mixed[0] / max(mixed[1], 1e-12):.1f}, and the anchored detectors above "
        "are why."
    )
    print()
    print("No threshold, no logical-suppression claim, and no scalability claim.")


if __name__ == "__main__":
    main()

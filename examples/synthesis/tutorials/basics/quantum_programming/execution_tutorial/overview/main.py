"""Run the complete local execution tutorial and print measured results."""

from collections.abc import Mapping

import flagquantum as fq
from flagquantum.synth import Program

from . import reference, workflow


def print_square(counts: Mapping[tuple[int, int], int]) -> None:
    shots = sum(counts.values())
    print(f"x  x²+1  count / {shots}")
    for (x, y), count in sorted(counts.items()):
        print(f"{x}  {y:4d}  {count:5d}  {count / shots:.6f}")


def print_ideal(program: Program) -> None:
    state = workflow.ideal_state(program)
    expected = reference.ghz_state()
    error = float((state - expected).abs().max())
    print(
        f"Ideal GHZ: 000={complex(state[0])}, 111={complex(state[-1])}, max error={error:.3g}"
    )


def print_noisy(program: Program) -> None:
    profile = workflow.calibration()
    circuit = workflow.native_basis(program, profile)
    counts, probabilities, density = workflow.noisy_ghz(circuit, profile)
    expected_density, expected_probabilities = reference.noisy_state(circuit, profile)
    print(
        f"Local noise approximation: {profile.device} {profile.release} ({profile.date})"
    )
    print("word  count  exact observed probability")
    for word, probability in enumerate(probabilities):
        bits = f"{word:03b}"
        print(f"{bits}   {counts.get(bits, 0):5d}  {float(probability):.9f}")
    print(
        f"Density matrix max error: {float((density - expected_density).abs().max()):.3g}"
    )
    print(
        f"Readout probability max error: {float((probabilities - expected_probabilities).abs().max()):.3g}"
    )


def print_square_error(square: Program) -> None:
    state = fq.run(square).statevector().reshape(-1)
    print(
        f"Full square-state max error: {float((state - reference.square_state()).abs().max()):.3g}"
    )


def print_mps(counts: Mapping[str | int, int]) -> None:
    words = dict(sorted((str(word), count) for word, count in counts.items()))
    print(f"MPS counts ({sum(words.values())} draws): {words}")


def main() -> None:
    square = workflow.build_square()
    print(f"Square circuit resources: {square.report['resources']}")
    print_square(workflow.square_counts(square, workflow.DEFAULT_SHOTS))
    print_square(workflow.square_counts(square, workflow.LARGE_SHOTS))
    print_square_error(square)
    ghz = workflow.build_ghz()
    print_noisy(ghz)
    print_ideal(ghz)
    print(f"Installed local backends: {', '.join(workflow.backend_names())}")
    print_mps(workflow.mps_counts(ghz))


if __name__ == "__main__":
    main()

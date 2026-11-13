"""Run the complete local QMCI Pi Notebook workflow."""

from .workflow import grid_pi, run_iterative, sample_qpe


def run_example() -> None:
    for size in (3, 5, 6):
        print("Classical grid axis:", size, "pi:", grid_pi(size))
    phase = sample_qpe()
    print("Three-bit QPE counts:", phase.observations)
    print("Source 4*sin(pi*phase):", phase.source_pi)
    print("Squared-probability comparison:", phase.squared_probability_pi)
    for bits in range(1, 5):
        result, sampler = run_iterative(bits, 1800 + bits)
        print(
            "Coordinate bits:",
            bits,
            "pi:",
            4 * result.estimate,
            "Probability interval:",
            result.confidence_interval,
            "Actual local shots:",
            sum(row.shots for row in result.iterations),
            "Visited Grover powers:",
            tuple(sampler.checks),
        )
        for row in result.iterations:
            print("Grover power, shots, successes:", row.k, row.shots, row.successes)
        for check in sampler.checks.values():
            print(check)


if __name__ == "__main__":
    run_example()

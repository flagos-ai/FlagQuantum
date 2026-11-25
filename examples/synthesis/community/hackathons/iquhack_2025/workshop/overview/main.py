"""Execute and verify the local iQuHack state-preparation workshop."""

from . import workflow


def print_amplitude_counts(sample: workflow.Sampling) -> None:
    print("Actual joint draws:", sum(count for _, count in sample.joint))
    print("Joint counts:", sample.joint)
    print("Marginal counts:", sample.marginal)
    print(
        "Observed indicator ones:",
        sum(count for (_, bit), count in sample.joint if bit),
    )
    print("Exact indicator-one probability:", sample.expected_indicator_probability)


def main() -> None:
    print("Coherent unsigned arithmetic:", workflow.arithmetic())
    prepared, distance = workflow.prepare_random(workflow.random_probabilities())
    print("Seeded 32-state preparation:", prepared)
    print("Preparation probability L2 error:", distance)
    values, signed = workflow.fixed_point()
    print("Signed fixed-point operands and sum:", values)
    print("Signed fixed-point all-branch check:", signed)
    amplitude, sample = workflow.amplitude_assignment()
    print("Fractional square amplitude loading:", amplitude)
    print_amplitude_counts(sample)
    exponential, distribution, outcome = workflow.exponential_qft()
    print("Exponential QFT state check:", exponential)
    print("Exponential QFT probability at zero:", distribution[0])
    print("Exponential QFT bonus draw:", outcome)


if __name__ == "__main__":
    main()

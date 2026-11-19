"""Run all original exponential, interval and shifted-interval preparations."""

from pathlib import Path

from .model import make_case, source_cases
from .plots import plot_amplitudes
from .workflow import prepare_distribution, prepare_interval


def run_example() -> None:
    first, shifted = source_cases()
    base = prepare_distribution(first.amplitudes)
    direct_small = make_case(0.5, 3, 13)
    output = Path("artifacts/development/partial_exponential_state_preparation")
    print("Base preparation:", base.check, base.resources)
    print(plot_amplitudes(base.probabilities, output, "full-exponential"))
    for name, case in (("first-interval", first), ("shifted-interval", shifted)):
        result = prepare_interval(case)
        print(
            "Source hint versus discrete overlap:",
            case.declared_overlap,
            case.observed_overlap,
        )
        print("Actual preparation:", result.check, result.resources)
        assert result.check.source_amplitude_error < 0.01
        print(
            plot_amplitudes(
                result.probabilities,
                output,
                name,
                interval=(case.lower + case.shift, case.upper + case.shift),
            )
        )
    print("Direct small interval source hint:", direct_small.declared_overlap)
    print(
        "Exact overlap-derived iteration counts:",
        direct_small.iterations,
        shifted.iterations,
    )
    print(
        "Attenuation remains explicit private storage;zero occupancy is checked independently of compiler workspace."
    )


if __name__ == "__main__":
    run_example()

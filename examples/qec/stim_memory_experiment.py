"""Report a supported original Stim circuit using existing optional SDKs.

Install the stim and pymatching extras. CPU only; no threshold or speed claim.
"""

from flagquantum.qec.execution.experiments import run_stim_experiment

SOURCE = """R 0 1
X_ERROR(0.125) 0
X_ERROR(0.25) 1
M 0 1
DETECTOR rec[-2]
DETECTOR rec[-1]
OBSERVABLE_INCLUDE(0) rec[-1]
OBSERVABLE_INCLUDE(1) rec[-2]
"""


def main() -> None:
    """Decode swapped logical columns with a final three-shot partial batch."""
    result = run_stim_experiment(
        SOURCE,
        decoder="pymatching",
        shots=35,
        batch_size=16,
        seed=53,
    )
    assert result.shots == 35
    assert result.logical_failures == 0
    assert result.observable_failures == (0, 0)
    assert result.sampling_source == "stim_circuit"
    print(result.shots, result.logical_failures, result.observable_failures)
    print(result.sampling_source)


if __name__ == "__main__":
    main()

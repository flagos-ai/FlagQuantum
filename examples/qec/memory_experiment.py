"""Run a fixed-shot CPU memory experiment with each default decoder.

Install with ``python -m pip install -e '.[stim,pymatching]'``. No GPU is used.
The explicit noise keeps every declared detector represented for PyMatching.
"""

from flagquantum.qec import PhenomenologicalNoise, RepetitionCode, build_memory_circuit
from flagquantum.qec.execution.experiments import run_memory_experiment


def main() -> None:
    """Count shot-level logical disagreements, retaining the one-shot tail."""
    memory = build_memory_circuit(RepetitionCode(distance=3), rounds=1)
    for decoder in (
        "minimum_weight_matching",
        "sliding_window_matching",
        "belief_propagation",
        "belief_propagation_osd",
        "pymatching",
    ):
        result = run_memory_experiment(
            memory,
            noise=PhenomenologicalNoise(data_flip=0.01, measurement_flip=0.01),
            decoder=decoder,
            shots=35,
            batch_size=16,
            seed=53,
        )
        print(
            decoder, result.shots, result.logical_failures, result.observable_failures
        )
        assert result.shots == 35
        assert (
            result.confidence_interval[0]
            <= result.logical_error_rate
            <= result.confidence_interval[1]
        )


if __name__ == "__main__":
    main()

"""Run a fixed-shot CPU memory experiment with each default decoder.

Install with ``python -m pip install -e '.[stim,pymatching]'``. No GPU is used.
The explicit noise keeps every declared detector represented for PyMatching.
"""

import argparse

from flagquantum.qec import (
    Pauli,
    PhenomenologicalNoise,
    RepetitionCode,
    RotatedSurfaceCode,
    build_memory_circuit,
)
from flagquantum.qec.execution.experiments import run_memory_experiment


def main() -> None:
    """Count shot-level logical disagreements, retaining the one-shot tail."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--code", choices=("repetition", "surface"), default="repetition"
    )
    parser.add_argument("--basis", choices=("x", "z"), default="z")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.code == "repetition":
        if args.basis != "z":
            parser.error("the repetition example supports only Z readout")
        memory = build_memory_circuit(RepetitionCode(distance=3), rounds=1)
        noise = PhenomenologicalNoise(data_flip=0.01, measurement_flip=0.01)
    else:
        code = RotatedSurfaceCode(distance=3)
        product = Pauli(x_qubits=code.data_qubits) if args.basis == "x" else None
        memory = build_memory_circuit(code, rounds=3, product=product)
        noise = PhenomenologicalNoise(
            data_flip=0.005, phase_flip=0.007, measurement_flip=0.009
        )
    for decoder in (
        "minimum_weight_matching",
        "sliding_window_matching",
        "belief_propagation",
        "belief_propagation_osd",
        "pymatching",
    ):
        result = run_memory_experiment(
            memory,
            noise=noise,
            decoder=decoder,
            shots=35,
            batch_size=args.batch_size,
            seed=53,
        )
        print(
            decoder, result.shots, result.logical_failures, result.observable_failures
        )
        assert result.shots == 35
        assert result.seed == 53 and result.sampling_source == "memory_circuit"
        assert result.observable_failures == (result.logical_failures,)
        assert (
            result.confidence_interval[0]
            <= result.logical_error_rate
            <= result.confidence_interval[1]
        )


if __name__ == "__main__":
    main()

"""Run a noisy repetition-code memory through the embedded stabilizer engine."""

from __future__ import annotations

import json

from flagquantum.qec import (
    PhenomenologicalNoise,
    RepetitionCode,
    build_memory_circuit,
)
from flagquantum.qec.adapters.stim import memory_to_stim_text
from flagquantum.qec.execution.experiments import run_stim_experiment


def main() -> None:
    memory = build_memory_circuit(RepetitionCode(distance=3), rounds=3)
    source = memory_to_stim_text(
        memory,
        noise=PhenomenologicalNoise(
            data_flip=0.005,
            phase_flip=0.007,
            measurement_flip=0.009,
        ),
    )
    result = run_stim_experiment(
        source,
        decoder="sparse_blossom",
        shots=64,
        batch_size=16,
        seed=5303,
    )
    print(
        json.dumps(
            {
                "shots": result.shots,
                "logical_failures": result.logical_failures,
                "sampling_source": result.sampling_source,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

"""Run the source Notebook's seeded preparation under a global width budget.

Native exact preparation uses four bits and no workspace. Its resource row is
not evidence of a depth/workspace search or approximation compression. Native
instruction-basis drawing replaces the source platform's interactive viewer;
cloud optimizer timeout, custom CX/U preferences and search seed are not claimed.
"""

import numpy as np

import flagquantum as fq
from examples.synthesis.tutorials.technology_demonstrations.auxiliary_managment.auxiliary_management.main import (
    main,
)
from flagquantum import synth
from flagquantum.drawer import draw


def source_amplitudes() -> list[float]:
    """Reproduce all sixteen RandomState(12) values without global RNG mutation."""
    amplitudes = 1 - 2 * np.random.RandomState(12).rand(16)
    return [float(value) for value in amplitudes / np.linalg.norm(amplitudes)]


def build_workflow() -> synth.Program:
    """Select the existing depth objective with the original eight-bit ceiling."""
    return synth.synthesize(
        main,
        classical_inputs={"amplitudes": source_amplitudes()},
        objective="depth",
        max_qubits=8,
        memory_limit_bytes=64 * 1024**2,
    )


def run_example() -> None:
    """Generate data, synthesize, draw, execute and report actual resources."""
    program = build_workflow()
    print(draw(program.circuit))
    print(program.report["resources"])
    print(dict(program.probabilities(fq.run(program), registers=("out",))))


if __name__ == "__main__":
    run_example()

"""Keep the frozen source-sized native eighteen-bit, four-layer quantum program."""

import math

from examples.synthesis.applications.optimization.electric_grid_optimization.main import (
    main,
)
from flagquantum import synth


def build_program(parameters: tuple[float, ...]) -> synth.Program:
    """Bind four cost and four mixer angles at synthesis time."""
    if len(parameters) != 8 or any(
        type(value) not in (int, float) or not math.isfinite(value)
        for value in parameters
    ):
        raise ValueError("four layers require eight finite real angles")
    return synth.synthesize(
        main,
        classical_inputs={"params": parameters},
        max_qubits=18,
        max_workspace=0,
        max_operations=10000,
        memory_limit_bytes=64 * 1024**2,
    )

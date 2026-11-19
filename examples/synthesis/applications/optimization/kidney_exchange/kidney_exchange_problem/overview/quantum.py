"""Retain all five native quantum layers and bind finite angles explicitly."""

import math

from examples.synthesis.applications.optimization.kidney_exchange.kidney_exchange_problem.main import (
    main,
)
from flagquantum import synth


def build_program(parameters: tuple[float, ...]) -> synth.Program:
    """Build the original nine-bit assignment circuit under the width objective."""
    if len(parameters) != 10 or any(
        type(value) not in (int, float) or not math.isfinite(value)
        for value in parameters
    ):
        raise ValueError("five layers require ten finite real angles")
    return synth.synthesize(
        main,
        classical_inputs={"params": parameters},
        objective="width",
        max_qubits=9,
        max_workspace=0,
        max_operations=5000,
        memory_limit_bytes=2 * 1024**2,
    )

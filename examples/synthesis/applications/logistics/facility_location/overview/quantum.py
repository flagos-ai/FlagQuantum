"""Reuse the complete frozen twelve-bit five-layer native facility program."""

import math

from examples.synthesis.applications.logistics.facility_location.main import main
from flagquantum import synth


def build_program(parameters: tuple[float, ...]) -> synth.Program:
    if len(parameters) != 10 or any(
        type(value) not in (int, float) or not math.isfinite(value)
        for value in parameters
    ):
        raise ValueError("five layers require ten finite real angles")
    return synth.synthesize(
        main,
        classical_inputs={"params": parameters},
        max_qubits=17,
        max_workspace=5,
        max_operations=50000,
        memory_limit_bytes=16 * 1024**2,
    )

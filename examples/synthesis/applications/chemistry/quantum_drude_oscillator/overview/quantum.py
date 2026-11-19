"""Execute both original ansatz sizes through native FlagQuantum synthesis."""

import math
from collections.abc import Sequence

from examples.synthesis.applications.chemistry.quantum_drude_oscillator.main import (
    ansatz_layer,
)
from examples.synthesis.applications.chemistry.quantum_drude_oscillator.main import (
    main as six_qubit_ansatz,
)
from flagquantum import synth
from flagquantum.synth import Output, QArray, QBit, allocate, qfunc


@qfunc
def four_qubit_ansatz(params: list[float], state: Output[QArray[QBit]]) -> None:
    """Use the source's three pair layers for two oscillators."""
    allocate(4, state)
    ansatz_layer([params[0], params[1], params[2], params[3]], state[0:2])
    ansatz_layer([params[4], params[5], params[6], params[7]], state[2:4])
    ansatz_layer([params[8], params[9], params[10], params[11]], state[1:3])


def build_program(oscillator_count: int, parameters: Sequence[float]) -> synth.Program:
    """Bind the complete finite angle array and synthesize without work bits."""
    if oscillator_count == 2:
        function, count = four_qubit_ansatz, 12
    elif oscillator_count == 3:
        function, count = six_qubit_ansatz, 40
    else:
        raise ValueError("expected two or three oscillators")
    if len(parameters) != count or not all(
        math.isfinite(value) for value in parameters
    ):
        raise ValueError(f"expected {count} finite angles")
    return synth.synthesize(
        function,
        classical_inputs={"params": list(parameters)},
        max_qubits=2 * oscillator_count,
        max_workspace=0,
        max_operations=256,
        memory_limit_bytes=1024**2,
    )

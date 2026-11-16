"""Preserve exact-original evidence separately from public native examples."""

from pathlib import Path

from examples.synthesis.applications.optimization.robust_posture_optimization.comparisons.circuit import (
    BINDINGS,
    REGISTERS,
    circuit_reference,
)
from examples.synthesis.applications.optimization.robust_posture_optimization.main import (
    main,
)
from examples.synthesis.compare_states import compare_states
from flagquantum.synth import synthesize


def run_example() -> None:
    source = Path(__file__).with_name("source.qmod")
    for bindings in BINDINGS:
        reference = circuit_reference(**bindings)
        for definition in (main, source):
            program = synthesize(definition, classical_inputs=bindings)
            compare_states(program, reference, REGISTERS)
            assert program.report["workspace_qubits"] == 0


if __name__ == "__main__":
    run_example()

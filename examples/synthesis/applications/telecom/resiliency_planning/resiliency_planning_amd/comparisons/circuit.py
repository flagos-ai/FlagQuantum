"""Independent reference shared by the byte-identical retained quantum programs."""

from examples.synthesis.applications.telecom.resiliency_planning.comparisons.circuit import (
    BINDINGS,
    cost_factors,
    reference_circuit,
    run_example,
)

__all__ = ["BINDINGS", "cost_factors", "reference_circuit", "run_example"]

if __name__ == "__main__":
    run_example()

"""Compare declared Add costs with emitted instructions, without simulation.

Run: python -m examples.algorithms.resource_model_comparison
Expected: three-bit Add estimates and emits 8 T equivalents and uses 8 wires;
the model has 2 And operations, while the emitted sequence has 2 measurements.
The second report retains symbolic widths without allocating a circuit.
"""

import json
from dataclasses import dataclass

from sympy import Symbol

import flagquantum as fq
from flagquantum.compiler.registers.arithmetic.addition import add_ir
from flagquantum.compiler.registers.arithmetic.arithmetic_resource_model import (
    source_add_cost,
)
from flagquantum.compiler.registers.resources.symbolic_resource_model import (
    source_add_resource_formula,
)
from flagquantum.compiler.resource_reporting import resource_model_report
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit


@dataclass(frozen=True)
class ThreeBitAddition:
    """Bind two three-bit words and two reusable carry wires."""

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        gates = add_ir((0, 1, 2), (3, 4, 5), (6, 7), (0, 1))
        return circuit.compose(CircuitIR(circuit.n_qubits, gates))


def main() -> None:
    circuit = DynamicCircuit(8)
    circuit.append(ThreeBitAddition())
    report = resource_model_report(source_add_cost(3, 3), emitted_ir=circuit.to_ir())
    emitted = report["emitted_ir"]
    assert report["t_equivalents"] == emitted["t_count"] == 8
    assert report["peak_logical_qubits"] == emitted["declared_qubits"] == 8
    assert report["logical_gates"]["and_bloq"] == 2
    assert emitted["unconditional"]["measure"] == 2
    assert report["execution_measurements"] is None
    print(json.dumps(report, indent=2))

    n = Symbol("n", integer=True, positive=True)
    symbolic = resource_model_report(source_add_resource_formula(n, n))
    assert symbolic["emitted_ir"] is None
    print(json.dumps(symbolic, indent=2))


if __name__ == "__main__":
    main()

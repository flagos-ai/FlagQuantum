"""Keep indexed child estimates, reserved wires and emitted gates separate.

Run python -m examples.algorithms.indexed_component_resources.
"""

import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler._component_trace.gate_budget import t_and_ccz_budget
from flagquantum.compiler._component_trace.resource_model import (
    ComponentCost,
    CostCall,
    GateCost,
)
from flagquantum.compiler.registers.arithmetic_resource_model import source_add_cost
from flagquantum.compiler.registers.controlled_addition import (
    controlled_add_constant_adjoint_ir,
    controlled_add_constant_ir,
)
from flagquantum.compiler.registers.indexed_component import indexed_component_ir
from flagquantum.compiler.registers.indexed_component_resource_model import (
    indexed_component_cost,
)
from flagquantum.core.ir import Instruction
from flagquantum.dynamic import DynamicCircuit
from flagquantum.qec.resources.estimation import estimate_ftqc_resources
from flagquantum.runtime.dynamic.execution import run_dynamic


def make_model(heavy_index: int = 2) -> ComponentCost:
    cheap = ComponentCost("ControlledXorK", 3, GateCost(clifford=1))
    # Controlled AddK controls only loading/unloading the constant. Its shared
    # Add sees the enable as one bystander, including at its internal carry peak.
    expensive = ComponentCost(
        "ControlledAddK",
        3,
        GateCost(clifford=2),
        (CostCall(source_add_cost(2, 2), bystander_qubits=1),),
    )
    children = [cheap] * 3
    children[heavy_index] = expensive
    return indexed_component_cost(
        (2,), (3,), tuple(children), target_width=2, control_count=1
    )


def make_gates(*, adjoint: bool = False) -> tuple[Instruction, ...]:
    def child(index: int, predicate: int) -> tuple[Instruction, ...]:
        if index != 2:
            return (Instruction("cx", (predicate, 4)),)
        emit = (
            controlled_add_constant_adjoint_ir
            if adjoint
            else controlled_add_constant_ir
        )
        return emit(predicate, (3, 4), 1, (7, 8), (9,), (2,))

    return indexed_component_ir(
        ((1, 2),),
        (3,),
        child,
        controls=(0,),
        workspace=(5, 6),
        classical_bits=(0, 1),
        adjoint=adjoint,
    )


def main() -> None:
    early = estimate_ftqc_resources(make_model(0))
    late = estimate_ftqc_resources(make_model())
    assert early.logical.peak_logical_qubits == 10
    assert late.logical.peak_logical_qubits == 9
    initial = torch.zeros(1024, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    amplitudes = torch.complex(torch.arange(1, 33).double(), torch.arange(32).double())
    for word in range(32):
        enable, address, value = (word >> 4) & 1, (word >> 2) & 3, word & 3
        if address == 3:
            continue
        updated = ((value + 1) % 4 if address == 2 else value ^ 1) if enable else value
        initial[word << 5] = amplitudes[word]
        expected[((word & ~3) | updated) << 5] = amplitudes[word]
    norm = initial.norm()
    initial /= norm
    expected /= norm
    circuit = DynamicCircuit(10, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(10, make_gates()))
    report = estimate_ftqc_resources(make_model(), emitted_ir=circuit.to_ir())
    assert report.emitted is not None
    assert report.emitted.t_count == report.logical.gates.total_t_count() == 12
    result = run_dynamic(circuit, shots=16, seed=43, strategy="batched")
    torch.testing.assert_close(
        result.final_states,
        expected.expand_as(result.final_states),
        atol=2e-13,
        rtol=2e-13,
    )
    print(
        "Modeled peak logical qubits: heavy child at first/last index",
        early.logical.peak_logical_qubits,
        late.logical.peak_logical_qubits,
    )
    print("Emitted circuit reserved quantum wires", report.emitted.declared_qubits)
    print(
        "Model T equivalents / emitted T instructions",
        report.logical.gates.total_t_count(),
        report.emitted.t_count,
    )
    print("Model measurements", report.logical.gates.measurement)
    print("Model magic-state budget", t_and_ccz_budget(report.logical.gates))
    print("All original inputs preserved, selected target updated and clean work zero")


if __name__ == "__main__":
    main()

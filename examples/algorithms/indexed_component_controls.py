"""Estimate and execute nested indexed controls with circuit.append(op).

Run python -m examples.algorithms.indexed_component_controls.
Add --stacked for the source repeated-single-control root.
"""

import argparse
from collections import Counter

import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler._component_trace.resource_model import ComponentCost, GateCost
from flagquantum.compiler.registers.control.control_predicate import ControlRegister
from flagquantum.compiler.registers.control.indexed_component_control import (
    conjoined_indexed_component_ir,
    specialized_indexed_component_ir,
)
from flagquantum.compiler.registers.control.indexed_component_control_resource_model import (
    conjoined_indexed_component_cost,
    specialized_indexed_component_cost,
)
from flagquantum.core.ir import Instruction
from flagquantum.dynamic import DynamicCircuit
from flagquantum.qec.resources.estimation import estimate_ftqc_resources
from flagquantum.runtime.dynamic.execution import run_dynamic


def main(*, stacked: bool = False) -> None:
    registers = (ControlRegister(2, (1,)),)

    def child(index: int, predicate: int) -> tuple[Instruction, ...]:
        return (Instruction("cx" if index == 0 else "cz", (predicate, 4)),)

    gates = (
        conjoined_indexed_component_ir(
            ((3,),),
            (2,),
            child,
            (0, 1, 2),
            (0, 1, 0),
            workspace=(5, 6, 7),
            classical_bits=(0, 1, 2),
        )
        if stacked
        else specialized_indexed_component_ir(
            ((3,),),
            (2,),
            child,
            ((0, 1),),
            registers,
            current_control=(2, 0),
            workspace=(5, 6, 7),
            classical_bits=(0, 1, 2),
        )
    )
    initial = torch.zeros(256, dtype=torch.complex128)
    amplitudes = torch.complex(torch.arange(1, 33).double(), torch.arange(32).double())
    amplitudes /= amplitudes.norm()
    initial[::8] = amplitudes
    expected = torch.zeros_like(initial)
    for word in range(32):
        active = word >> 2 == 2  # New word 01 and existing control 0.
        address = (word >> 1) & 1
        output = word ^ int(active and address == 0)
        sign = -1 if active and address and word & 1 else 1
        expected[output << 3] = sign * amplitudes[word]
    circuit = DynamicCircuit(8, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(8, gates))
    result = run_dynamic(circuit, shots=16, seed=37, strategy="batched")
    torch.testing.assert_close(
        result.final_states,
        expected.expand_as(result.final_states),
        atol=3e-13,
        rtol=3e-13,
    )
    children = (
        ComponentCost("CX", 2, GateCost(clifford=1)),
        ComponentCost("CZ", 2, GateCost(clifford=1)),
    )
    models = []
    for legacy_shims in (False, True):
        model = (
            conjoined_indexed_component_cost(
                (1,),
                (2,),
                children,
                (0, 1, 0),
                target_width=1,
                legacy_shims=legacy_shims,
            )
            if stacked
            else specialized_indexed_component_cost(
                (1,),
                (2,),
                children,
                registers,
                target_width=1,
                current_control_value=0,
                legacy_shims=legacy_shims,
            )
        )
        models.append(model)
    report = estimate_ftqc_resources(models[0], emitted_ir=circuit.to_ir())
    legacy = estimate_ftqc_resources(models[1])
    assert report.emitted is not None
    print("Modern model Clifford count", report.logical.gates.clifford)
    print("Legacy model Clifford count", legacy.logical.gates.clifford)
    print("Model T equivalents", report.logical.gates.total_t_count())
    print("Emitted T/T-dagger instructions", report.emitted.t_count)
    print("Model logical peak", report.logical.peak_logical_qubits)
    print("Reserved execution wires", 8)
    print("Emitted primitive counts", dict(Counter(g.name for g in gates)))
    print("Complex amplitudes and clean work verified")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stacked", action="store_true")
    main(stacked=parser.parse_args().stacked)

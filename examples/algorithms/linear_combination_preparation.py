"""Run coefficient PREPARE, print separate resources and restore all wires.

Run python -m examples.algorithms.linear_combination_preparation.
"""

from collections import Counter

import torch

from flagquantum.algorithms.primitives.linear_combination_preparation import (
    LinearCombinationPreparation,
)
from flagquantum.dynamic import DynamicCircuit
from flagquantum.qec.resources.estimation import estimate_ftqc_resources
from flagquantum.runtime.dynamic.execution import run_dynamic


def main() -> None:
    preparation = LinearCombinationPreparation((1.0, -3.0), (1.0, 1.0), 1)
    selection = tuple(range(preparation.prepare.selection_width))
    junk = tuple(range(len(selection), len(selection) + preparation.prepare.junk_width))
    nwork, nbits = preparation.workspace_requirements
    workspace = tuple(
        range(len(selection) + len(junk), len(selection) + len(junk) + nwork)
    )
    bits = tuple(range(nbits))
    width = len(selection) + len(junk) + nwork
    initial = torch.zeros(1 << width, dtype=torch.complex128)
    initial[0] = 1
    circuit = DynamicCircuit(width, inputs=initial, dtype=torch.complex128)
    circuit.append(
        preparation.make_on(
            selection=selection, junk=junk, workspace=workspace, classical_bits=bits
        )
    )
    ir = circuit.to_ir()
    result = run_dynamic(circuit, shots=8, seed=19, strategy="batched")
    state = result.final_states.reshape(8, 2, -1)
    probabilities = state.abs().square().sum(-1)
    torch.testing.assert_close(
        probabilities, torch.tensor([[0.25, 0.75]], dtype=torch.float64).expand(8, 2)
    )
    if nwork:
        dirty = result.final_states.reshape(8, -1, 1 << nwork)[:, :, 1:]
        torch.testing.assert_close(dirty, torch.zeros_like(dirty), atol=3e-12, rtol=0)
    circuit.append(
        preparation.make_on(
            selection=selection,
            junk=junk,
            workspace=workspace,
            classical_bits=bits,
            adjoint=True,
        )
    )
    restored = run_dynamic(circuit, shots=8, seed=23, strategy="batched")
    torch.testing.assert_close(
        restored.final_states,
        initial.expand_as(restored.final_states),
        atol=3e-12,
        rtol=0,
    )
    report = estimate_ftqc_resources(preparation, emitted_ir=ir)
    counts = Counter(g.name for g in ir.instructions)
    print("Selection probabilities", probabilities[0].tolist())
    print(
        "Model T equivalents (4 T per abstract Fredkin)",
        report.logical.gates.total_t_count(),
    )
    print("Emitted T/Tdag (7 T per exact Fredkin)", counts["t"] + counts["tdg"])
    print("Model peak logical qubits", report.logical.peak_logical_qubits)
    print("Emitted measurements", counts["measure"])
    print(
        "Private work clean; full complex inverse restored selection and logical junk"
    )


if __name__ == "__main__":
    main()

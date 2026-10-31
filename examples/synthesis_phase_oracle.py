"""Private contributor phase proof; the public output_mode keyword is proposed.

Run python -m examples.synthesis_phase_oracle from the repository root.
This expert bridge reuses Compiler selection and the existing Program factory.
"""

import torch

import flagquantum as fq
from flagquantum.compiler._construction.frontend_python import capture
from flagquantum.compiler._construction.selection import (
    ConstructionBudget,
    select_arithmetic,
)
from flagquantum.services.synthesis import _complete


def marked(x: int, y: int) -> bool:
    return x * x + y < 5


def main() -> None:
    print("private contributor evidence; public output_mode remains proposed")
    graph = capture(marked, {"x": 2, "y": 2})
    preparation = fq.Circuit(4, dtype=torch.complex128)
    for q in range(4):
        preparation.h(q)
    selection = select_arithmetic(
        (graph,),
        entry=graph.name,
        input_names=("x", "y"),
        preparation=preparation.to_ir(),
        output_mode="phase",
        budget=ConstructionBudget(qubits=22),
    )
    program = _complete(
        selection,
        {
            "function": graph.name,
            "output_mode": "phase",
            "source_format": "python",
            "scope": "actual_cpu_preparation_predicate_phase_cleanup",
        },
        memory_limit_bytes=64 * 1024**2,
        memory_limit_origin="explicit",
    )
    result = fq.run(program)
    state = result.statevector().reshape(-1)
    reference = torch.zeros_like(state)
    chosen = selection.selected
    assert chosen is not None and chosen.target is not None
    for x in range(4):
        for y in range(4):
            index = 0
            for name, value in (("x", x), ("y", y)):
                for bit, logical in enumerate(reversed(program.registers[name])):
                    index |= ((value >> bit) & 1) << (
                        program.circuit.n_wires
                        - 1
                        - chosen.target.layout.physical_of(logical)
                    )
            reference[index] = (-1) ** int(x * x + y < 5) / 4
    residual = float(torch.linalg.vector_norm(state - reference))
    assert residual <= 1e-12
    print("phase/workspace residual:", residual)
    print(
        "input probabilities:",
        dict(program.probabilities(result, registers=("x", "y"))),
    )
    print("resources:", program.report["resources"])
    print("workspace:", program.report["workspace_qubits"])
    print(
        "Uniform probabilities alone do not verify the phase; the complex reference above does."
    )


if __name__ == "__main__":
    main()

"""Reuse independent source block matrices with the Notebook register layout."""

from dataclasses import replace

import flagquantum as fq
from examples.synthesis.tutorials.technology_demonstrations.classiq_paper.qsvt.comparisons.circuit import (
    reference_circuit as source_reference,
)

from ..program import PHASES


def reference_circuit(phases: tuple[float, ...] = PHASES) -> fq.Circuit:
    """Use independent permutations and projectors, not synthesis kernels.

    The reused reference's largest dense operator is 256 by 256 (1 MiB,
    complex128), for local correctness inspection only.
    """
    original = source_reference(phases).to_ir()
    # Canonical named order is block(projection,deletion,select), data, aux.
    mapping = (1, 3, 4, 5, 6, 7, 8, 2, 0, 9)
    return fq.Circuit.from_ir(
        replace(
            original,
            instructions=tuple(
                replace(op, wires=tuple(mapping[wire] for wire in op.wires))
                for op in original.instructions
            ),
        )
    )


if __name__ == "__main__":
    from ..workflow import run_workflow

    workflow = run_workflow()
    print("Native and CX/U3 full states match the independent Circuit.")
    print(workflow.resources)

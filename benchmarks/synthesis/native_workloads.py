"""Fixed native examples and independent references for performance observation."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from flagquantum.services.synthesis import Program

EXAMPLES = {
    "arithmetic": "tutorials.arithmetic.arithmetic_baseline",
    "qaoa": "algorithms.search_and_optimization.qaoa.qaoa_max_cut",
    "qpe": "tutorials.technology_demonstrations.QPE.qpe_for_unitary_matrix",
    "pauli": "functions.hamiltonian_evolution.suzuki_trotter",
    "qdrift": "tutorials.technology_demonstrations.hamiltonian_evolution",
    "block_encoding": "community.paper_implementation_project.block_encoding.checkerboard",
}


class NativeWorkload:
    """Bind existing qfunc examples; do not replace their sizes or algorithms."""

    def __init__(self, name: str) -> None:
        from flagquantum import synth

        if name not in EXAMPLES:
            raise ValueError(f"unknown native workload: {name}")
        self.name = name
        prefix = "examples.synthesis." + EXAMPLES[name]
        self.entry = importlib.import_module(prefix + ".main")
        self.build: Callable[[], Program]
        if name == "arithmetic":
            self.build = partial(
                synth.synthesize,
                self.entry.main,
                memory_limit_bytes=64 * 1024**2,
            )
            self.reference = None
        else:
            self.build = self.entry.build_program
            self.reference = importlib.import_module(prefix + ".comparisons.circuit")

    def expected(self, program: Program) -> torch.Tensor:
        """Build the complete reference outside compilation/execution timers."""
        import torch

        import flagquantum as fq

        if self.reference is not None:
            expected = fq.run(self.reference.reference_circuit()).statevector()
            if not isinstance(expected, torch.Tensor):
                raise TypeError("CPU reference must return a Torch statevector")
            return expected.flatten()

        # Independent x*x+y support in named-register order, including clean work.
        state = torch.zeros(1 << program.circuit.n_wires, dtype=torch.complex128)
        for x in range(4):
            for y in range(4):
                index = 0
                for name, value in (("x", x), ("y", y), ("result", x * x + y)):
                    wires = program.registers[name]
                    for position, wire in enumerate(wires):
                        bit = (value >> (len(wires) - position - 1)) & 1
                        index |= bit << (program.circuit.n_wires - wire - 1)
                state[index] = 0.25
        return state

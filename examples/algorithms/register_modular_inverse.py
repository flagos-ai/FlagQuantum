"""Append the original public modular inverse, keeping logical history explicit.

Run: python -m examples.algorithms.register_modular_inverse
Constructs 38-wire IR and reports resources, without allocating a dense state.
"""

from collections import Counter
from dataclasses import dataclass

import flagquantum as fq
from flagquantum.compiler.registers.modular.modular_inverse import (
    modular_inverse_outputs,
    source_modular_inverse_t_count,
)
from flagquantum.compiler.registers.modular.modular_inverse_component import (
    modular_inverse_program,
)
from flagquantum.compiler.registers.register_binding import (
    BoundRegisterProgram,
    bind_register_program,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program import ProgramRecord


@dataclass(frozen=True)
class ModularInverse:
    """Bind the target, four-n-bit history and clean work for native append."""

    bitsize: int = 3
    modulus: int = 5
    adjoint: bool = False

    def bind(self) -> BoundRegisterProgram:
        """Bind the stored graph and follow its actual history/output ownership."""
        n = self.bitsize
        original = tuple(range(n))
        history = tuple(range(n, 5 * n))
        workspace = tuple(range(5 * n, 10 * n + 8))
        output, _ = modular_inverse_outputs(original, workspace)
        inputs = (output, history) if self.adjoint else (original,)
        outputs = (original,) if self.adjoint else (output, history)
        return _bind_layout(
            modular_inverse_program(n, self.modulus, adjoint=self.adjoint),
            inputs,
            outputs,
            10 * n + 8,
        )

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        bound = self.bind()
        return circuit.compose(CircuitIR(circuit.n_qubits, bound.circuit.instructions))


def _bind_layout(
    program: ProgramRecord,
    inputs: tuple[tuple[int, ...], ...],
    outputs: tuple[tuple[int, ...], ...],
    width: int,
) -> BoundRegisterProgram:
    """Choose a clean-pool permutation preserving the example's physical ports."""
    owned = {wire for word in inputs for wire in word}
    clean = tuple(wire for wire in range(width) if wire not in owned)
    ownership = tuple((wire, False) for wire in sorted(owned))
    first = bind_register_program(
        program,
        inputs,
        clean=clean,
        input_ownership=ownership,
        allow_workspace_outputs=True,
    )
    mapping = {wire: wire for wire in owned}
    for observed, requested in zip(first.outputs, outputs, strict=True):
        for old, new in zip(observed, requested, strict=True):
            if old in mapping and mapping[old] != new:
                raise ValueError("Output placement conflicts with an input wire")
            mapping[old] = new
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("Output placement must be a wire permutation")
    remaining = sorted(set(range(width)) - mapping.keys())
    destinations = sorted(set(range(width)) - set(mapping.values()))
    mapping.update(zip(remaining, destinations, strict=True))
    bound = bind_register_program(
        program,
        inputs,
        clean=tuple(mapping[wire] for wire in clean),
        input_ownership=ownership,
        allow_workspace_outputs=True,
    )
    assert bound.outputs == outputs
    return bound


def main() -> None:
    """Show real physical outputs and keep model costs distinct from literal gates."""
    n, p = 3, 5
    circuit = fq.Circuit(10 * n + 8)
    circuit.append(ModularInverse(n, p))
    counts = Counter(g.name for g in circuit.to_ir().instructions)
    target, history = ModularInverse(n, p).bind().outputs
    restored = ModularInverse(n, p, adjoint=True).bind().outputs[0]
    print("Output: x^-1 * 2^(2n) mod p, with zero mapped to zero.")
    print("Physical output:", target, "retained history:", history)
    print("Matching inverse physical output:", restored)
    print(
        "Retained logical history:",
        4 * n,
        "bits; supply the same history to uncompute.",
    )
    print("Measured primitive IR:", dict(counts))
    print(
        "Source models: default",
        source_modular_inverse_t_count(n),
        "T; exact seven-T swaps",
        source_modular_inverse_t_count(n, exact_swaps=True),
        "T.",
    )
    circuit.append(ModularInverse(n, p, adjoint=True))
    assert len(circuit.to_ir().instructions) == 2 * sum(counts.values())
    print("Matching inverse appended; no output-materializing SWAP or logical reset.")


if __name__ == "__main__":
    main()

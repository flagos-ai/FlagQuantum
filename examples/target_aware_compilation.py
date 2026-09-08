"""Compile a circuit for a constrained target topology and execute it."""

from __future__ import annotations

import torch

import flagquantum as fq
import flagquantum.compiler as compiler


def main() -> None:
    circuit = (
        fq.Circuit(n_qubits=5, dtype=torch.complex128)
        .h(0)
        .cx(0, 4)
        .h(4)
        .cx(0, 4)
    )
    coupling = compiler.CouplingMap.line(5)

    compiled = compiler.compile(
        circuit,
        coupling_map=coupling,
        routing_strategy="auto",
    )
    routing = compiled.metadata["routing"]

    target_edges_are_valid = all(
        len(instruction.wires) != 2 or coupling.has_edge(*instruction.wires)
        for instruction in compiled.instructions
    )
    assert target_edges_are_valid
    assert routing["mapping_restored"] is True

    options = fq.ExecutionOptions(
        mode="statevector",
        backend="pytorch",
        device="cpu",
        precision="complex128",
        allow_backend_fallback=False,
    )
    result = fq.run(compiled, options=options)
    torch.testing.assert_close(result.state, circuit.state(), atol=1e-12, rtol=0)

    print("FlagQuantum target-aware compilation check passed")
    print(f"  target topology: {coupling.edges}")
    print(f"  routing strategy: {routing['strategy']}")
    print(f"  inserted swaps: {routing['inserted_swap_count']}")
    print(f"  target edges: {'valid' if target_edges_are_valid else 'invalid'}")
    print(f"  mapping restored: {routing['mapping_restored']}")
    print(f"  execution path: {result.runtime['execution_path']}")


if __name__ == "__main__":
    main()

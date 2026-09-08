"""Optimize a circuit without changing its numerical result."""

from __future__ import annotations

import torch

import flagquantum as fq
import flagquantum.compiler as compiler


def main() -> None:
    circuit = (
        fq.Circuit(n_qubits=2, dtype=torch.complex128)
        .x(0)
        .x(0)
        .rx(1, theta=0.1)
        .rx(1, theta=0.2)
        .h(0)
        .cx(0, 1)
    )

    original = circuit.to_ir()
    optimized = compiler.optimize(original)

    assert compiler.optimize(optimized) == optimized
    assert len(original.instructions) == 6
    assert len(optimized.instructions) == 3

    options = fq.ExecutionOptions(
        mode="statevector",
        backend="pytorch",
        device="cpu",
        precision="complex128",
        allow_backend_fallback=False,
    )
    result = fq.run(optimized, options=options)
    torch.testing.assert_close(result.state, circuit.state(), atol=1e-12, rtol=0)

    print("FlagQuantum compiler optimization check passed")
    print(f"  original: {[item.name for item in original.instructions]}")
    print(f"  optimized: {[item.name for item in optimized.instructions]}")
    print(f"  instructions: {len(original.instructions)} -> {len(optimized.instructions)}")
    print(f"  execution path: {result.runtime['execution_path']}")


if __name__ == "__main__":
    main()

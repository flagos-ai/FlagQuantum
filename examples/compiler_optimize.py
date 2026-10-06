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
        .h(0)
        .rz(0, theta=0.4)
        .h(0)
        .rx(1, theta=0.2)
        .cx(0, 1)
    )

    original = circuit.to_ir()
    optimized = compiler.optimize(original)

    # `x x` cancels, the two `rx` on wire 1 add up, and the `h rz h` that mixes
    # two opcodes on wire 0 folds into one `u3`. The last of those is what makes
    # this check a guard on the fold: `u3` reproduces the run only if the emitted
    # angles carry the phase the run had, and the statevector comparison below
    # would see a difference if they did not.
    assert compiler.optimize(optimized) == optimized
    assert len(original.instructions) == 8
    assert len(optimized.instructions) == 3
    assert [item.name for item in optimized.instructions] == ["rx", "u3", "cx"]

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
    print(
        f"  instructions: {len(original.instructions)} -> {len(optimized.instructions)}"
    )
    print(f"  execution path: {result.runtime['execution_path']}")


if __name__ == "__main__":
    main()

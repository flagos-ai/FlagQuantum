"""Build, inspect, and execute a Bell-state circuit on the CPU."""

from __future__ import annotations

import torch

import flagquantum as fq


def main() -> None:
    circuit = fq.Circuit(n_qubits=2).h(0).cx(0, 1)
    options = fq.ExecutionOptions(
        mode="statevector",
        backend="pytorch",
        device="cpu",
        precision="complex128",
        allow_backend_fallback=False,
    )

    plan = fq.plan(circuit, options=options)
    result = fq.run(plan)

    expected = torch.tensor(
        [[2**-0.5, 0.0, 0.0, 2**-0.5]], dtype=torch.complex128
    )
    torch.testing.assert_close(result.state, expected, atol=1e-12, rtol=0)

    summary = plan.summary()
    print("FlagQuantum CPU statevector check passed")
    print(f"  simulator: {summary['state_mode']}")
    print(f"  execution: {summary['recommended_mode']}")
    print(f"  path: {result.runtime['execution_path']}")
    print(f"  device: {result.runtime['device']}")
    print(f"  state: {result.state}")


if __name__ == "__main__":
    main()

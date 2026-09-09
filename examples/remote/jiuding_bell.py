"""Local or Jiuding CPU Bell-state calculation; main returns a JSON result."""

import torch
import flagquantum as fq


def main():
    circuit = fq.Circuit(2).h(0).cx(0, 1)
    execution = fq.run(
        circuit, options=fq.ExecutionOptions(device="cpu", mode="statevector")
    )
    state = execution.to_statevector().reshape(-1)
    expected = torch.tensor([2**-0.5, 0, 0, 2**-0.5], dtype=state.dtype)
    torch.testing.assert_close(state, expected, atol=1e-6, rtol=1e-6)
    return {
        "framework": "FlagQuantum",
        "version": fq.__version__,
        "qubits": 2,
        "device": str(state.device),
        "dtype": str(state.dtype),
        "probabilities": state.abs().square().tolist(),
        "max_abs_error": (state - expected).abs().max().item(),
    }


if __name__ == "__main__":
    print(main())

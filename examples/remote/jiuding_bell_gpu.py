"""Single-GPU FlagQuantum Bell state; fail if CUDA execution is unavailable."""

import torch
import flagquantum as fq


def main():
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("This example requires exactly one visible CUDA GPU")
    torch.cuda.reset_peak_memory_stats()
    circuit = fq.Circuit(2).h(0).cx(0, 1)
    execution = fq.run(
        circuit, options=fq.ExecutionOptions(device="cuda:0", mode="statevector")
    )
    state = execution.to_statevector().reshape(-1)
    if state.device.type != "cuda":
        raise RuntimeError("The quantum state must remain on CUDA")
    expected = torch.tensor(
        [2**-0.5, 0, 0, 2**-0.5], dtype=state.dtype, device=state.device
    )
    torch.testing.assert_close(state, expected, atol=1e-6, rtol=1e-6)
    torch.cuda.synchronize()
    return {
        "framework": "FlagQuantum",
        "version": fq.__version__,
        "qubits": 2,
        "device": str(state.device),
        "dtype": str(state.dtype),
        "gpu_name": torch.cuda.get_device_name(0),
        "visible_gpus": torch.cuda.device_count(),
        "probabilities": state.abs().square().cpu().tolist(),
        "max_abs_error": (state - expected).abs().max().item(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }


if __name__ == "__main__":
    print(main())

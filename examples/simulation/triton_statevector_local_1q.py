"""Run a single-GPU statevector workload through the optional local-1q kernel."""

from __future__ import annotations

import os

import torch

import flagquantum as fq
from flagquantum.runtime import run_distributed


_TRUTHY = {"1", "true", "on", "yes"}


def main() -> None:
    if os.getenv("FQ_STATEVECTOR_TRITON_LOCAL_1Q", "0").strip().lower() not in _TRUTHY:
        raise RuntimeError(
            "set FQ_STATEVECTOR_TRITON_LOCAL_1Q=1 to request the Triton kernel"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("this example requires a CUDA device")

    circuit = fq.Circuit(n_qubits=3).h(0).ry(1, theta=0.3).rz(2, theta=-0.2)
    reference = fq.run(
        circuit,
        options=fq.ExecutionOptions(
            mode="statevector",
            backend="pytorch",
            device="cpu",
            precision="complex64",
            allow_backend_fallback=False,
        ),
    ).to_statevector()
    result = run_distributed(
        circuit,
        device="cuda:0",
        precision="complex64",
        distributed_profile="production",
        world_size=1,
    )
    state = result.shard_state.amplitudes
    torch.testing.assert_close(state.cpu(), reference, atol=3e-6, rtol=3e-6)

    dispatch = result.summary()["kernel_dispatch"]
    local_1q = next(
        record for record in dispatch["decisions"] if record["feature"] == "local_1q"
    )
    if local_1q["selected"] != "triton":
        raise RuntimeError(
            "local-1q Triton dispatch was not selected: "
            f"{local_1q['reason']}"
        )

    compiler = local_1q["kernel_compiler"]
    route = local_1q["kernel_route"]
    print("FlagQuantum local-1q Triton check passed")
    print(f"  device: {state.device}")
    print(f"  executions: {dispatch['triton_execution_count']}")
    print(f"  compiler distribution: {compiler['distribution']}")
    print(f"  compiler version: {compiler['version']}")
    print(f"  identity status: {compiler['identity_status']}")
    print(f"  integration path: {route['integration_path']}")


if __name__ == "__main__":
    main()

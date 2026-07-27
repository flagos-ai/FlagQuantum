"""Benchmark reusable parameter slots against rebuilding a circuit every step.

This benchmark compares two equivalent training workloads:

1. ``static``: ``fq.QuantumModule`` captures the circuit once and only updates
   parameter slots on later steps.
2. ``rebuild``: the Python builder creates a new Circuit and Instructions on
   every step, matching FlagQuantum's previous execution path.

The warm-up iteration is excluded, so import and first-compilation time do not
distort the result. Both paths execute forward + backward with the same input,
parameters, circuit topology, observable, and simulator.

Examples:

    python examples/benchmark_static_program.py --mode sv
    python examples/benchmark_static_program.py --mode mps --qubits 12 --depth 8
    python examples/benchmark_static_program.py --mode tn --steps 100
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import flagquantum as fq  # noqa: E402, I001


MODES = {
    "sv": "statevector",
    "mps": "mps",
    "tn": "tensor_network",
}


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark FlagQuantum static parameter-slot execution"
    )
    parser.add_argument("--mode", choices=MODES, default="sv")
    parser.add_argument("--qubits", type=int, default=8)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    for name in ("qubits", "depth", "batch_size", "steps", "repeats"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")

    device = torch.device(args.device)
    dtype = torch.float32
    parameter_count = args.qubits * args.depth
    generator = torch.Generator(device=device).manual_seed(7)
    inputs = torch.rand(
        args.batch_size,
        args.qubits,
        generator=generator,
        dtype=dtype,
        device=device,
    )
    initial = torch.rand(
        parameter_count, generator=generator, dtype=dtype, device=device
    )
    wires = tuple(range(args.qubits))

    def circuit_builder(parameters: torch.Tensor, encoded: torch.Tensor) -> fq.Circuit:
        circuit = fq.Circuit(args.qubits, bsz=args.batch_size, device=device)
        for layer in range(args.depth):
            offset = layer * args.qubits
            for wire in range(args.qubits):
                angle = encoded[:, wire] + parameters[offset + wire]
                circuit.ry(wire, angle)
            for wire in range(args.qubits - 1):
                circuit.cx(wire, wire + 1)
        return circuit

    static_module = fq.QuantumModule(
        circuit_builder,
        parameter_count,
        init=initial,
        device=device,
        policy=fq.RuntimePolicy(
            mode=MODES[args.mode],
            observable="z_sum",
            observable_wires=wires,
        ),
    )
    class RebuildQuantumModule(fq.QuantumModule):
        """Reference path that deliberately bypasses builder compilation."""

        def _build(self, current_inputs, current_parameters):
            return self._invoke_circuit_builder(current_parameters, current_inputs)

    rebuild_module = RebuildQuantumModule(
        circuit_builder,
        parameter_count,
        init=initial,
        device=device,
        policy=fq.RuntimePolicy(
            mode=MODES[args.mode],
            observable="z_sum",
            observable_wires=wires,
        ),
    )

    def rebuild_step() -> torch.Tensor:
        return rebuild_module(inputs).mean()

    def static_step() -> torch.Tensor:
        return static_module(inputs).mean()

    # Compile/import both paths once. Warm-up work is deliberately not timed.
    for step, parameters in (
        (static_step, tuple(static_module.parameters())),
        (rebuild_step, tuple(rebuild_module.parameters())),
    ):
        loss = step()
        loss.backward()
        for parameter in parameters:
            parameter.grad = None
    synchronize(device)

    def measure(step, parameters) -> list[float]:
        samples = []
        for _ in range(args.repeats):
            synchronize(device)
            started = time.perf_counter()
            for _ in range(args.steps):
                loss = step()
                loss.backward()
                for parameter in parameters:
                    parameter.grad = None
            synchronize(device)
            samples.append(time.perf_counter() - started)
        return samples

    static_times = measure(static_step, tuple(static_module.parameters()))
    rebuild_times = measure(rebuild_step, tuple(rebuild_module.parameters()))
    static_median = statistics.median(static_times)
    rebuild_median = statistics.median(rebuild_times)
    speedup = rebuild_median / static_median
    saved = 100.0 * (1.0 - static_median / rebuild_median)

    runtime = static_module.execute(inputs).runtime
    print("FlagQuantum static-program benchmark")
    print(f"  backend       : {MODES[args.mode]}")
    print(f"  circuit       : {args.qubits} qubits x {args.depth} layers")
    print(f"  workload      : batch={args.batch_size}, steps={args.steps}")
    print(f"  repeats       : {args.repeats} (median reported)")
    print(f"  static        : {static_median:.6f} s")
    print(f"  rebuild       : {rebuild_median:.6f} s")
    print(f"  speedup       : {speedup:.3f}x")
    print(f"  time saved    : {saved:.1f}%")
    print(f"  builder builds: {runtime['builder_compile_count']}")
    print(f"  cache hits    : {runtime['builder_cache_hits']}")
    print(f"  static reused : {runtime['static_program_reused']}")


if __name__ == "__main__":
    main()

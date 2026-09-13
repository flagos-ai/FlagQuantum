"""FlagQuantum's copy-and-adapt quantum training template.

Architecture
------------
classical data x
    -> torch.nn.Linear classical encoder
    -> fq.Module quantum layer
    -> Z expectation values
    -> classical loss
    -> PyTorch backward + optimizer

The classical layer maps one input feature to one angle per qubit. The quantum
layer applies trainable RY offsets and a CNOT chain. When the classical weights
are one and all biases/quantum offsets are zero, the exact Z expectations are
``cos(x), cos(x)^2, ..., cos(x)^n``. The target is their sum, so this hybrid
model has a known reference solution and both classical and quantum parameters
receive gradients from the same ``loss.backward()`` call.

Run the same model with a different simulator, without editing code:

    python examples/quick_start.py --mode sv
    python examples/quick_start.py --mode mps
    python examples/quick_start.py --mode tn
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import flagquantum as fq  # noqa: E402

SIMULATION_MODES = {
    "sv": "statevector",
    "mps": "mps",
    "tn": "tensor_network",
}


def main() -> None:
    # Stage 1: choose scale and simulation method from the command line.
    parser = argparse.ArgumentParser(description="FlagQuantum one-minute quick start")
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--mode", choices=SIMULATION_MODES, default="sv")
    args = parser.parse_args()
    if args.n_qubits < 1:
        parser.error("--n-qubits must be positive")

    # Stage 2: create classical data and its analytically exact quantum target.
    # Use an odd number of points so x=0 is present as a concrete reference:
    # theory(0) = sum(cos(0)^k) = n_qubits exactly.
    x = torch.linspace(-torch.pi, torch.pi, 17)
    target = sum(torch.cos(x) ** power for power in range(1, args.n_qubits + 1))

    # Stage 3: define only the quantum part. It receives angles produced by the
    # classical encoder; batch execution is handled by bsz=len(inputs).
    # Unsure what a gate needs? Try: flagquantum.operators.gate_info("ry").
    def quantum_circuit(parameters, encoded_inputs):
        q = fq.Circuit(args.n_qubits, bsz=len(encoded_inputs))
        for qubit in range(args.n_qubits):
            q.ry(qubit, encoded_inputs[:, qubit] + parameters["angles"][qubit])
        for qubit in range(args.n_qubits - 1):
            q.cx(qubit, qubit + 1)
        return q

    # Stage 4: compose an ordinary PyTorch layer with a FlagQuantum layer.
    class HybridModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.classical = nn.Linear(1, args.n_qubits)
            nn.init.ones_(self.classical.weight)
            nn.init.zeros_(self.classical.bias)
            self.quantum = fq.Module(
                quantum_circuit,
                parameters={"angles": (args.n_qubits,)},
                init="uniform",
                seed=42,
                policy=fq.RuntimePolicy(
                    execution_options=fq.ExecutionOptions(
                        mode=SIMULATION_MODES[args.mode]
                    ),
                    observable="z_sum",
                    observable_qubits=tuple(range(args.n_qubits)),
                ),
            )

        def forward(self, inputs):
            encoded = self.classical(inputs[:, None])
            return self.quantum(encoded)

    model = HybridModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # Stage 5: one standard PyTorch loop trains both sides of the hybrid model.
    losses = []
    log_interval = max(1, args.steps // 4)
    for step in range(1, args.steps + 1):
        optimizer.zero_grad()
        loss = torch.mean((model(x) - target) ** 2)
        loss.backward()  # gradients flow through quantum and classical layers
        optimizer.step()
        losses.append(float(loss.detach()))
        if step == 1 or step == args.steps or step % log_interval == 0:
            print(f"step {step}/{args.steps} loss={losses[-1]:.8g}")

    # Stage 6: inference is an ordinary module call. Compare against the known
    # analytical result so this example proves correctness, not just execution.
    prediction = model(x)
    prediction = prediction.detach()
    absolute_error = torch.abs(prediction - target)
    zero_index = len(x) // 2
    theory_at_zero = float(target[zero_index])
    prediction_at_zero = float(prediction[zero_index])
    print("\nFlagQuantum quick start complete")
    print(f"  qubits     : {args.n_qubits}")
    print(f"  simulator  : {SIMULATION_MODES[args.mode]}")
    print("  theory     : sum(cos(x)^k, k=1..n_qubits)")
    print("  reference  : classical weight=1, all biases/offsets=0")
    print(f"  initial loss: {losses[0]:.6f}")
    print(f"  final loss : {losses[-1]:.6f}")
    print("  trained    : classical + quantum parameters")
    print("\nConcrete comparison at x=0")
    print(f"  theoretical value : {theory_at_zero:.6f}")
    print(f"  trained value     : {prediction_at_zero:.6f}")
    print(f"  absolute error    : {float(absolute_error[zero_index]):.6f}")
    print(f"  max dataset error : {float(absolute_error.max()):.6f}")
    print("  theoretical loss  : 0.000000")


if __name__ == "__main__":
    main()

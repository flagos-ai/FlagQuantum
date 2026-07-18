# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os

import torch
from torch.profiler import ProfilerActivity, profile

import flagquantum as fq

# torchrun --nproc-per-node=8 scaling.py 16 20 8


class QuantumNeuralNetwork:
    """Quantum neural network for scaling experiments"""

    def __init__(self, nq, batch, world_sz, rank):
        self.nq = nq
        self.batch = batch
        self.world_sz = world_sz
        self.rank = rank
        self.qdev = None
        self.mod = None
        self._setup_device()
        self._build_network()

    def _setup_device(self):
        """Set up CUDA device for current rank"""
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        self.local_rank = local_rank
        self.device = f"cuda:{local_rank}"

        self._print_device_info()

    def _print_device_info(self):
        """Print device information for debugging"""
        print(
            f"Rank {self.rank}: local_rank {self.local_rank}, using cuda:{self.local_rank}"
        )
        print(f"Rank {self.rank}: CUDA available: {torch.cuda.is_available()}")
        print(f"Rank {self.rank}: Current device: {torch.cuda.current_device()}")

    def _build_network(self):
        """Build the quantum neural network"""
        # Initialize distributed quantum device
        self.qdev = fq.DistributedQuantumDevice(
            self.nq,
            bsz=self.batch,
            device="cuda",
            world_sz=self.world_sz,
            invertible=True,
        )

        self._print_device_mesh_info()

        # Build encoder
        func_list = [
            {"func": "ry", "wires": [i], "input_idx": [i]} for i in range(self.nq)
        ]
        encoder = fq.GeneralEncoder(func_list)

        # Build base module
        base_mod = [encoder]
        for _ in range(3):
            base_mod = (
                base_mod
                + [fq.CX(wires=[i, (i + 1) % self.nq]) for i in range(self.nq)]
                + [fq.RY(wires=[i]) for i in range(self.nq)]
            )

        self.mod = fq.invertible.InvertibleUnitary(base_mod)
        self.mod.train()

    def _print_device_mesh_info(self):
        """Print device mesh information if available"""
        if hasattr(self.qdev, "device_mesh"):
            print(f"[Rank {self.rank}] DeviceMesh shape: {self.qdev.device_mesh.shape}")
            print(f"[Rank {self.rank}] DeviceMesh size: {self.qdev.device_mesh.size()}")

    def forward(self, x):
        """Forward pass through the quantum network"""
        self.qdev.reset_states()
        self.mod(self.qdev, x)
        measurements = fq.measure_allZ(self.qdev, shots=0, training=True)
        return measurements

    def train_step(self, x, optimizer):
        """Execute a single training step"""
        optimizer.zero_grad()
        output = self.forward(x)
        loss = output.abs().sum()
        loss.backward()
        optimizer.step()
        return loss.item()


def run_training_loop(model, nq, batch, rank, n_iterations=100):
    """Run the main training loop"""
    # Initialize input parameters
    x = torch.nn.Parameter(
        torch.rand([batch, nq], device=f"cuda:{rank}") * torch.pi / 3
    )

    optimizer = torch.optim.Adam([x])
    losses = []

    for iteration in range(n_iterations):
        loss = model.train_step(x, optimizer)
        losses.append(loss)

        if rank == 0:
            print(f"Iteration {iteration + 1}/{n_iterations}")

    return losses


def scaling(batch, nq, world_sz):
    """Main scaling experiment function"""
    rank = int(os.environ.get("RANK", 0))

    print(f"world_sz passed to DistributedQuantumDevice: {world_sz}")

    # Build and train quantum neural network
    qnn = QuantumNeuralNetwork(nq, batch, world_sz, rank)
    losses = run_training_loop(qnn, nq, batch, rank)

    return losses


def parse_args():
    """Parse command line arguments"""
    arg_parser = argparse.ArgumentParser(
        description="Scaling experiments for distributed quantum neural networks"
    )
    arg_parser.add_argument("batch", type=int, help="Batch size")
    arg_parser.add_argument("nq", type=int, help="Number of qubits")
    arg_parser.add_argument(
        "world_sz", type=int, help="World size for distributed training"
    )

    return arg_parser.parse_args()


def run_with_profiling(batch, nq, world_sz):
    """Run scaling with profiling enabled"""
    rank = int(os.environ.get("RANK", 0))

    try:
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiler_logs"),
        ) as prof:
            scaling(batch, nq, world_sz)

        if rank == 0:
            print("\n=== Top 5 CUDA operators ===")
            print(prof.key_averages().table(sort_by="cpu_time", row_limit=5))
            print("Trace saved to ./profiler_logs")

    except Exception as e:
        print(f"Profiling failed: {e}")
        import traceback

        traceback.print_exc()
        # Fall back to regular execution
        scaling(batch, nq, world_sz)


def main():
    """Main entry point"""
    args = parse_args()

    # Uncomment to enable profiling
    # run_with_profiling(args.batch, args.nq, args.world_sz)

    # Regular execution
    scaling(args.batch, args.nq, args.world_sz)


if __name__ == "__main__":
    main()

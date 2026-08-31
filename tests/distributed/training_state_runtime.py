from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

import flagquantum as fq
import flagquantum.experimental.distributed as fqxd


def build(parameters: torch.Tensor) -> fq.Circuit:
    return fq.Circuit(3).ry(2, parameters[0]).cx(2, 1).ry(1, parameters[1])


def module_and_optimizer(
    plan: fqxd.HybridParallelPlan,
) -> tuple[fq.Module, torch.optim.Optimizer]:
    module = fq.Module(
        build,
        2,
        init=torch.tensor([0.23, -0.37]),
        policy=fq.RuntimePolicy(mode="distributed_statevector", observable_wires=(1,)),
    )
    module.set_parallel_context(state_process_group=dist.group.WORLD, plan=plan)
    return module, torch.optim.Adam(module.parameters(), lr=0.03)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    args = parser.parse_args()
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    try:
        plan = fq.plan_hybrid_parallel(world_size=2, state_parallel_size=2)
        module, optimizer = module_and_optimizer(plan)
        seed = fq.seed_everything(314)
        value = module().sum()
        value.backward()
        optimizer.step()
        saved_parameters = module.parameters_tensor.detach().clone()
        path = args.checkpoint_dir / f"rank-{rank}.pt"
        module.save_checkpoint(path, optimizer=optimizer, seed=seed, step=1)
        expected_random = torch.rand(3)

        restored, restored_optimizer = module_and_optimizer(plan)
        metadata = restored.load_checkpoint(path, optimizer=restored_optimizer)
        torch.testing.assert_close(restored.parameters_tensor, saved_parameters)
        torch.testing.assert_close(torch.rand(3), expected_random)
        assert metadata["topology"]["state_parallel_size"] == 2
        assert metadata["step"] == 1
        result = restored.execute()
        assert result.runtime["world_size"] == 2
        if rank == 0:
            print(
                json.dumps(
                    {
                        "status": "passed",
                        "world_size": 2,
                        "state_parallel_size": 2,
                        "checkpoint_files": [
                            str(args.checkpoint_dir / f"rank-{item}.pt")
                            for item in range(2)
                        ],
                        "seed": metadata["seed"].seed,
                        "precision": metadata["precision"].to_dict(),
                        "hostname": os.uname().nodename,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

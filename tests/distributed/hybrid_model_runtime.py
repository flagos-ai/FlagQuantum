from __future__ import annotations

import json

import torch
import torch.distributed as dist

import flagquantum as fq
import flagquantum.training as fqt

INPUTS = torch.tensor([[-0.8, -0.2], [0.7, 0.4]])
TARGETS = torch.tensor([-1.0, 1.0])


def train_step(
    model: fq.HybridQuantumClassifier, optimizer: torch.optim.Optimizer
) -> torch.Tensor:
    optimizer.zero_grad()
    loss = torch.nn.functional.mse_loss(model(INPUTS), TARGETS)
    loss.backward()
    optimizer.step()
    return loss.detach()


def main() -> None:
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    try:
        fqt.seed_everything(480)
        local = fq.HybridQuantumClassifier(
            deployment_binding={"provider": "local", "target": "simulator"}
        )
        initial = local.state_dict()
        sharded = fq.HybridQuantumClassifier(
            policy=fq.RuntimePolicy(observable_wires=(1,)),
            deployment_binding={"provider": "local", "target": "simulator"},
        )
        sharded.load_state_dict(initial)
        sharded.set_runtime_policy(fq.RuntimePolicy(observable_wires=(1,)))
        plan = fq.plan_hybrid_parallel(world_size=2, state_parallel_size=2)
        sharded.quantum.set_parallel_context(
            state_process_group=dist.group.WORLD, plan=plan
        )
        local_optimizer = torch.optim.SGD(local.parameters(), lr=0.04)
        sharded_optimizer = torch.optim.SGD(sharded.parameters(), lr=0.04)
        local_losses = []
        sharded_losses = []
        for _ in range(2):
            local_losses.append(float(train_step(local, local_optimizer)))
            sharded_losses.append(float(train_step(sharded, sharded_optimizer)))
        for (local_name, local_value), (sharded_name, sharded_value) in zip(
            local.state_dict().items(), sharded.state_dict().items()
        ):
            assert local_name == sharded_name
            if isinstance(local_value, torch.Tensor):
                torch.testing.assert_close(
                    sharded_value, local_value, atol=4e-5, rtol=4e-5
                )
        assert sharded.deployment_parameters()["binding"]["provider"] == "local"
        execution = sharded.quantum.execute()
        runtime = execution.runtime
        assert runtime["distribution_semantics"] == "sharded_across_ranks"
        assert runtime["world_size"] == 2
        ownership = runtime["rank_ownership"]
        assert (
            ownership["local_amplitudes"] * runtime["world_size"]
            == ownership["total_amplitudes"]
        )
        assert ownership["full_state_owned"] is False
        assert runtime["backward_distribution_semantics"] == "pending"
        sharded.quantum.parameters_tensor.grad = None
        execution.value.backward()
        assert runtime["backward_distribution_semantics"] == "sharded_across_ranks"
        if rank == 0:
            print(
                json.dumps(
                    {
                        "status": "passed",
                        "model": type(sharded).__name__,
                        "local_losses": local_losses,
                        "sharded_losses": sharded_losses,
                        "world_size": 2,
                        "distribution_semantics": runtime["distribution_semantics"],
                        "runtime_world_size": runtime["world_size"],
                        "rank_ownership": runtime["rank_ownership"],
                        "steps": 2,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

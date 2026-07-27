from __future__ import annotations

import json

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

import flagquantum as fq


def build(parameters: torch.Tensor, inputs: torch.Tensor | None) -> fq.Circuit:
    assert inputs is not None
    return (
        fq.Circuit(3)
        .rx(0, inputs.reshape(()))
        .ry(2, parameters[0])
        .cx(2, 1)
        .ry(1, parameters[1])
    )


def main() -> None:
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    try:
        plan = fq.plan_hybrid_parallel(
            world_size=4,
            data_parallel_size=2,
            state_parallel_size=2,
            input_batch_size=2,
        )
        state_groups = [
            dist.new_group(ranks=list(ranks)) for ranks in plan.state_groups
        ]
        data_groups = [dist.new_group(ranks=list(ranks)) for ranks in plan.data_groups]
        replica = rank // 2
        shard = rank % 2
        initial = torch.tensor([0.23, -0.37])
        module = fq.Module(
            build,
            2,
            init=initial,
            policy=fq.RuntimePolicy(
                mode="distributed_statevector", observable_wires=(1,)
            ),
        )
        module.set_parallel_context(
            state_process_group=state_groups[replica], plan=plan
        )
        ddp = DistributedDataParallel(module, process_group=data_groups[shard])
        sample = torch.tensor(0.17 if replica == 0 else -0.41)
        value = ddp(sample)
        value.sum().backward()

        reference = fq.Module(
            build,
            2,
            init=initial,
            policy=fq.RuntimePolicy(observable_wires=(1,)),
        )
        reference_values = torch.stack(
            [reference(torch.tensor(item)).sum() for item in (0.17, -0.41)]
        )
        reference_gradient = torch.autograd.grad(
            reference_values.mean(), reference.parameters_tensor
        )[0]
        torch.testing.assert_close(
            module.parameters_tensor.grad, reference_gradient, atol=3e-5, rtol=3e-5
        )
        result = module.execute(sample)
        dimensions = result.metrics["parallelism"]["parallel_dimensions"]
        assert dimensions["data_parallel"] == 2
        assert dimensions["state_parallel"] == 2
        assert result.runtime["backward_distribution_semantics"] == "pending"
        module.parameters_tensor.grad = None
        result.value.backward()
        assert (
            result.runtime["backward_distribution_semantics"] == "sharded_across_ranks"
        )
        if rank == 0:
            print(
                json.dumps(
                    {
                        "gradient": module.parameters_tensor.grad.tolist(),
                        "reference_gradient": reference_gradient.tolist(),
                        "parallelism": result.metrics["parallelism"],
                        "orchestration": "native_pytorch_process_groups_and_ddp",
                        "jax_orchestration": False,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

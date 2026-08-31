from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist

import flagquantum as fq


def build(parameters: torch.Tensor) -> fq.Circuit:
    return (
        fq.Circuit(3, device=parameters.device)
        .ry(2, parameters[0])
        .cx(2, 1)
        .ry(1, parameters[1])
    )


def main() -> None:
    dist.init_process_group("gloo")
    try:
        initial = torch.tensor([0.23, -0.37])
        local = fq.Module(
            build,
            2,
            init=initial,
            policy=fq.RuntimePolicy(observable_wires=(1,)),
        )
        expected = local()
        expected_grad = torch.autograd.grad(expected.sum(), local.parameters_tensor)[0]
        sharded = fq.Module(
            build,
            2,
            init=initial,
            policy=fq.RuntimePolicy(observable_wires=(1,)),
        )
        result = sharded.execute()
        assert result.value is not None
        result.value.backward()
        torch.testing.assert_close(result.value, expected, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(
            sharded.parameters_tensor.grad, expected_grad, atol=3e-5, rtol=3e-5
        )
        assert (
            result.runtime["backward_distribution_semantics"] == "sharded_across_ranks"
        )
        if int(os.environ["RANK"]) == 0:
            print(json.dumps(result.summary(), default=str, sort_keys=True), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

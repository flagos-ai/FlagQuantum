"""Profile the three phases of a persistent rank/local statevector transpose."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import torch
import torch.distributed as dist

from flagquantum.runtime.backends.statevector.layout import _local_bit_view


def timed(fn, device: torch.device) -> float:
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    fn()
    torch.cuda.synchronize(device)
    return time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-amplitudes", type=int, default=1 << 27)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--json-output", required=True)
    args = parser.parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    rank_bits = world.bit_length() - 1
    local_world = int(os.environ["LOCAL_WORLD_SIZE"])
    local_rank_bits = local_world.bit_length() - 1
    state = torch.zeros((1, args.local_amplitudes), dtype=torch.complex64, device=device)
    send = torch.empty((1, args.local_amplitudes // 2), dtype=state.dtype, device=device)
    recv = torch.empty_like(send)
    # Match the common persistent-layout case: the least-significant local bit.
    source = _local_bit_view(state, bit_position=0, bit_value=1)
    destination = _local_bit_view(state, bit_position=0, bit_value=1)
    results = {}
    for label, rank_bit_position in (
        ("intra_node", 0),
        ("inter_node", local_rank_bits),
    ):
        if rank_bit_position >= rank_bits:
            continue
        peer = rank ^ (1 << rank_bit_position)
        samples = {"pack": [], "p2p": [], "unpack": [], "total": []}
        for _ in range(args.repetitions + 1):
            pack = timed(lambda: send.reshape_as(source).copy_(source), device)

            def exchange() -> None:
                operations = [
                    dist.P2POp(dist.isend, send, peer),
                    dist.P2POp(dist.irecv, recv, peer),
                ]
                for request in dist.batch_isend_irecv(operations):
                    request.wait()

            p2p = timed(exchange, device)
            unpack = timed(lambda: destination.copy_(recv.reshape_as(destination)), device)
            values = torch.tensor((pack, p2p, unpack), dtype=torch.float64, device=device)
            dist.all_reduce(values, op=dist.ReduceOp.MAX)
            if _:
                samples["pack"].append(float(values[0]))
                samples["p2p"].append(float(values[1]))
                samples["unpack"].append(float(values[2]))
                samples["total"].append(float(values.sum()))
        results[label] = {
            key: {
                "median_seconds": statistics.median(values),
                "samples_seconds": values,
            }
            for key, values in samples.items()
        }
    if rank == 0:
        payload = {
            "world_size": world,
            "local_world_size": local_world,
            "local_amplitudes": args.local_amplitudes,
            "half_shard_bytes": send.numel() * send.element_size(),
            "results": results,
        }
        with open(args.json_output, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(json.dumps(payload, indent=2))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

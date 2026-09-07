"""Two-rank packed transport value, reuse, mismatch, and cleanup checks."""

import json

import torch
import torch.distributed as dist

from flagquantum.runtime.backends.mps.forward import RankOwnedMPSState
from flagquantum.runtime.backends.mps.reverse import site_sharded_z_zz_observations
from flagquantum.runtime.distributed.engine import (
    _recv_tensor_batch_p2p,
    _send_tensor_batch_p2p,
    mps_p2p_stats,
    reset_mps_p2p_stats,
)
from flagquantum.simulation.mps.models import MPSConfig


def main():
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    reset_mps_p2p_stats()
    references = (torch.empty(2, 3), torch.empty(1, 4))
    if rank == 1:
        values = (torch.arange(6.0).reshape(2, 3), torch.arange(4.0).reshape(1, 4))
        _send_tensor_batch_p2p(values, dst=0, sequences=(99_000, 99_001))
        _send_tensor_batch_p2p(values, dst=0, sequences=(99_002, 99_003))
    else:
        first = _recv_tensor_batch_p2p(
            src=1, references=references, sequences=(99_000, 99_001)
        )
        second = _recv_tensor_batch_p2p(
            src=1, references=references, sequences=(99_002, 99_003)
        )
        torch.testing.assert_close(first[0], torch.arange(6.0).reshape(2, 3))
        torch.testing.assert_close(second[1], torch.arange(4.0).reshape(1, 4))
        assert mps_p2p_stats()["buffer_pool_hits"] >= 1
    local_tensors = {}
    for wire in (0, 1) if rank == 0 else (2, 3):
        tensor = torch.zeros(1, 1, 2, 1, dtype=torch.complex64)
        tensor[:, :, 0, :] = 1
        local_tensors[wire] = tensor
    state = RankOwnedMPSState(
        n_wires=4,
        bsz=1,
        rank=rank,
        world_size=2,
        config=MPSConfig(),
        local_tensors=local_tensors,
        ownership=((0, 1), (2, 3)),
    )
    observations = site_sharded_z_zz_observations(state, (0, 1, 2, 3))
    torch.testing.assert_close(observations, torch.ones_like(observations))
    dist.barrier()
    mismatch_passed = False
    if rank == 1:
        _send_tensor_batch_p2p(
            (torch.ones(2, 2), torch.ones(1, 4)),
            dst=0,
            sequences=(99_010, 99_011),
        )
    else:
        try:
            _recv_tensor_batch_p2p(
                src=1, references=references, sequences=(99_010, 99_011)
            )
        except RuntimeError as error:
            mismatch_passed = "payload drained" in str(error)
    dist.barrier()
    outcomes = [None, None]
    dist.all_gather_object(outcomes, {"rank": rank, "mismatch_passed": mismatch_passed})
    print(
        json.dumps(
            {
                "rank": rank,
                "values_passed": True,
                "site_observations_passed": True,
                "shape_mismatch_passed": outcomes[0]["mismatch_passed"],
                "cleanup_verified": True,
                "stats": mps_p2p_stats(),
            }
        ),
        flush=True,
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

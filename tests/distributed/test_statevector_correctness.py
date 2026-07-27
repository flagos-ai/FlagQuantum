import os
import sys
from pathlib import Path

import pytest
import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from statevector_correctness import run_statevector_correctness  # noqa: E402

pytestmark = [
    pytest.mark.distributed,
    pytest.mark.distributed_cpu,
    pytest.mark.distributed_multinode,
    pytest.mark.skipif(
        "RANK" not in os.environ,
        reason="requires torchrun with distributed RANK/WORLD_SIZE environment",
    ),
]


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _backend() -> str:
    return "nccl" if torch.cuda.is_available() else "gloo"


def test_distributed_statevector_correctness_transport_pytest():
    world_size = _world_size()
    device = _device()
    backend = _backend()
    expected_distribution = (
        "qubit_address_sharded"
        if world_size > 1 and (world_size & (world_size - 1) == 0)
        else "replicated_single_rank"
    )
    expected_topology = (
        "hypercube"
        if expected_distribution == "qubit_address_sharded"
        else "single_rank"
    )

    result = run_statevector_correctness(
        world_size=world_size,
        n_wires=4,
        distribution=expected_distribution,
        topology=expected_topology,
        backend=backend,
        device=device,
    )

    plan = result["plan"]
    validation = result["validation"]
    executor = result["executor"]
    transport = result["transport"]

    assert validation.valid
    assert executor.valid
    assert transport.valid
    assert (
        validation.summary()["event_count"] == executor.summary()["trace_event_count"]
    )
    assert plan.summary()["state_mode"] == "distributed_statevector"
    assert plan.summary()["world_size"] == world_size
    if world_size > 1:
        assert transport.summary()["event_count"] > 0
        assert transport.summary()["pair_exchange_count"] > 0
        assert transport.summary()["all_to_all_count"] > 0

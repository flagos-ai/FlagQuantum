import os

import pytest
import torch
import torch.distributed as dist


@pytest.fixture(autouse=True)
def _synchronize_distributed_tests():
    yield
    if "RANK" not in os.environ:
        return
    if not dist.is_available() or not dist.is_initialized():
        return
    if torch.cuda.is_available():
        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        dist.barrier()

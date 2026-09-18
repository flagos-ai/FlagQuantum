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


_OBJECT_COLLECTIVES = (
    "all_gather_object",
    "broadcast_object_list",
    "gather_object",
    "scatter_object_list",
)


@pytest.fixture(autouse=True)
def _launched_tests_use_no_object_collective(request):
    """A launched test may not use a Python object collective.

    `all_gather_object` and its siblings route their payload through
    `Tensor.numpy()`, so they need NumPy. Nothing here depends on NumPy: the
    PyTorch-only lane installs `.[dev]`, and NumPy only arrives with the JAX
    extra -- and that is the lane the launched tests run in. So an object
    collective works everywhere except where these tests actually execute.

    That is not hypothetical: `statevector_correctness.py` gathered its
    per-rank summaries with `all_gather_object`, and the launched lane's first
    real execution failed with `RuntimeError: Numpy is not available`. It had
    been selected and skipped for as long as it existed, so nothing had ever
    run it. The runtime's own answer is `all_gather_json`, a tensor-native,
    pickle-free gather; the rest of the MPS executor already uses it.
    """
    if request.node.get_closest_marker("distributed_launch") is None:
        yield
        return
    originals = {
        name: getattr(dist, name) for name in _OBJECT_COLLECTIVES if hasattr(dist, name)
    }

    def _refuse(name: str):
        def _raise(*args: object, **kwargs: object) -> None:
            raise AssertionError(
                f"{name} moves its payload through Tensor.numpy() and so needs "
                "NumPy, which the launched lane does not install; gather tensors "
                "directly, or use flagquantum...mps.metadata_transport."
                "all_gather_json"
            )

        return _raise

    for name in originals:
        setattr(dist, name, _refuse(name))
    try:
        yield
    finally:
        for name, collective in originals.items():
            setattr(dist, name, collective)

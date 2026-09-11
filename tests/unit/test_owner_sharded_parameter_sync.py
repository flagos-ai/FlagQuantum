import torch

from flagquantum.runtime.executors.mps.compiled_training import (
    OwnerShardedParameterSynchronizer,
)


def test_owner_sharded_parameter_synchronizer_preserves_tensor_identity(monkeypatch):
    parameters = (torch.tensor(1.0), torch.tensor(9.0))
    identities = tuple(id(parameter) for parameter in parameters)

    def fake_all_reduce(packed, *, group=None):
        assert group is None
        # Rank 0 owns element 0; emulate rank 1 contributing element 1.
        packed.add_(torch.tensor([0.0, 2.0]))

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    synchronizer = OwnerShardedParameterSynchronizer(
        parameters=parameters,
        owners=(0, 1),
        rank=0,
        world_size=2,
    )
    summary = synchronizer.synchronize()

    assert tuple(id(parameter) for parameter in parameters) == identities
    assert tuple(float(parameter) for parameter in parameters) == (1.0, 2.0)
    assert summary == {
        "strategy": "owner_masked_packed_all_reduce",
        "collective_count": 1,
        "logical_payload_bytes": 8,
    }


def test_owner_sharded_parameter_synchronizer_world_one_is_noop():
    parameter = torch.tensor(3.0)
    synchronizer = OwnerShardedParameterSynchronizer(
        parameters=(parameter,),
        owners=(0,),
        rank=0,
        world_size=1,
    )
    assert synchronizer.synchronize()["collective_count"] == 0
    assert float(parameter) == 3.0

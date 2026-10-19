"""CPU torchrun check for the Jiuding distributed job result bridge."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import flagquantum as fq  # noqa: E402
from flagquantum.remote.compute._distributed_job import _response  # noqa: E402
from flagquantum.remote.compute._workspace_executor import (  # noqa: E402
    SCHEMA,
    VERSION,
)
from flagquantum.remote.compute._workspace_results import decode_tensor  # noqa: E402
from flagquantum.runtime.distributed import (  # noqa: E402
    destroy_torch_distributed,
    init_torch_distributed,
)


def main() -> None:
    context = init_torch_distributed(backend="gloo", device="cpu")
    try:
        circuit = fq.Circuit(3).h(0).cx(0, 2)
        response = _response(
            {
                "schema": SCHEMA,
                "version": VERSION,
                "operation": "statevector",
                "request_id": "batch-program",
                "target": "jiuding:gpu/test",
                "devices": 2,
                "program": circuit.to_ir().to_dict(),
            },
            device=context.device,
        )
        if context.rank == 0:
            torch.testing.assert_close(
                decode_tensor(response["state"]), circuit.state().reshape(1, -1)
            )
            evidence = response["evidence"]
            assert evidence["devices"] == 2
            assert (
                evidence["distributed_execution"]["distribution_semantics"]
                == "sharded_across_ranks"
            )
            assert (
                evidence["result_materialization"]["distribution_semantics"]
                == "replicated_per_rank"
            )
    finally:
        if dist.is_initialized():
            destroy_torch_distributed()


if __name__ == "__main__":
    main()

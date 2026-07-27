#!/usr/bin/env python
"""One-shot mechanical decomposition of the distributed compatibility engine."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "flagquantum/runtime/distributed/engine.py"
MODELS = SOURCE.with_name("models.py")
TN = SOURCE.with_name("tensor_network_execution.py")

text = SOURCE.read_text(encoding="utf-8")
model_start = text.index("@dataclass(frozen=True)\nclass TorchDistributedContext")
model_end = text.index("\n\nclass _DistributedAllReduceSum")
tn_support_start = model_end + 2
tn_support_end = text.index("\n\ndef run_distributed_mps")
tn_start = text.index("\ndef _zero_for_output")

if MODELS.exists() or TN.exists():
    raise SystemExit("distributed split targets already exist")

header = text[:model_start]
models_text = (
    header.replace(
        '"""Distributed MPS and tensor-network execution plans.',
        '"""Distributed context, plans, and rank-owned state models.',
        1,
    )
    + text[model_start:model_end]
    + "\n"
)

model_names = (
    "DistributedBoundaryProtocol",
    "DistributedBoundarySync",
    "DistributedMPSState",
    "DistributedShardPlan",
    "DistributedSliceTask",
    "DistributedTensorNetworkState",
    "ShardedMPSState",
    "TorchDistributedContext",
    "_boundary_communication_tiers",
    "_communication_tier",
    "_infer_backend",
    "_mps_shards",
    "_node_count",
    "_rank_node",
    "_rank_placement_summary",
    "_resolve_backend_policy",
    "_resolve_local_world_size",
    "_resolve_node_rank",
    "_should_use_torch_distributed",
    "_split_contiguous",
    "_tensor_slice_tasks",
    "destroy_torch_distributed",
    "init_torch_distributed",
    "torch_distributed_is_available",
)
model_import = (
    "from .models import (\n"
    + "".join(f"    {name},\n" for name in model_names)
    + ")\n"
)

tn_block = text[tn_support_start:tn_support_end] + text[tn_start:]
tn_header = header.replace(
    '"""Distributed MPS and tensor-network execution plans.',
    '"""Distributed tensor-network reduction execution.',
    1,
)
tn_header += model_import
tn_text = tn_header + tn_block + "\n"

tn_import = "from .tensor_network_execution import run_distributed_tensor_network\n"
remaining = (
    text[:model_start] + model_import + tn_import + text[tn_support_end:tn_start] + "\n"
)
SOURCE.write_text(remaining, encoding="utf-8")
MODELS.write_text(models_text, encoding="utf-8")
TN.write_text(tn_text, encoding="utf-8")

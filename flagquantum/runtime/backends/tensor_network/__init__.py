"""Tensor-network backend boundary."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = (
    "DistributedTNAdjointLayoutPlan",
    "CompiledTNForwardBucket",
    "CompiledTNForwardSchedule",
    "CompiledTNReverseBucket",
    "CompiledTNReverseSchedule",
    "DistributedTNAdjointValueLayout",
    "DistributedTNCommunicationEdge",
    "DistributedTNCheckpointPlan",
    "DistributedTNCheckpointAudit",
    "DistributedTNContractionDAG",
    "DistributedTNContractionRecord",
    "DistributedTNDynamicReverseSegment",
    "DistributedTNDynamicReverseSegmentResult",
    "DistributedTNParameterPullbackResult",
    "DistributedTNExecutionResult",
    "DistributedTNJointPlan",
    "DistributedTNWorkingSetPolicy",
    "DistributedTensorNetworkState",
    "DistributedTNMemoryEvidence",
    "DistributedTNRankMemoryMeasurement",
    "DistributedTNMultiAxisContractionResult",
    "DistributedTNMultiAxisConeResult",
    "DistributedTNMultiAxisLayout",
    "DistributedTNMultiAxisPeakPlan",
    "DistributedTNMultiAxisRedistributionPlan",
    "DistributedTNMultiAxisShard",
    "DistributedTNMeshGroupCache",
    "DistributedTNPartialMeshLayout",
    "DistributedTNPartialMeshForwardResult",
    "DistributedTNPartialMeshRedistributionPlan",
    "DistributedTNPartialMeshRedistributionResult",
    "DistributedTNPartialMeshReverseResult",
    "DistributedTNShardedDAGExecutionResult",
    "DistributedTNRedistributionBlock",
    "DistributedTNRedistributionPlan",
    "DistributedTNRedistributionResult",
    "DistributedTNReverseDAG",
    "DistributedTNReverseRecord",
    "DistributedTNReverseResult",
    "DistributedTNRematerializationProvider",
    "DistributedTNRematerializationPlan",
    "DistributedTNShard",
    "DistributedTNShardedPairResult",
    "DistributedSlicedTNReverseResult",
    "DistributedTNOptimizerStepResult",
    "DistributedTNSliceTask",
    "DistributedTNSliceTaskPlan",
    "SlicedTNExplicitReverseResult",
    "SlicedTNCheckpointMemoryPlan",
    "DistributedTNValueLayout",
    "assemble_tn_redistribution_shard",
    "combine_contracted_shards",
    "combine_output_shards",
    "clear_dynamic_tn_checkpoint_writer_lock",
    "contract_pair_for_contracted_shard",
    "contract_pair_for_output_shard",
    "execute_checkpointed_tn_reverse_dag",
    "execute_compiled_tn_forward_with_tape",
    "execute_compiled_tn_reverse_dag",
    "execute_distributed_tn_contraction_dag",
    "execute_dynamic_tn_reverse_segment",
    "execute_dynamic_tn_parameter_pullback",
    "load_dynamic_tn_reverse_checkpoint",
    "inspect_dynamic_tn_checkpoint",
    "execute_sharded_tn_contraction_dag",
    "execute_distributed_tn_redistribution",
    "execute_distributed_sliced_tn_explicit_reverse",
    "execute_rank_owned_tn_sgd_step",
    "execute_multi_axis_contracted_pair",
    "execute_multi_axis_tn_redistribution",
    "execute_multi_axis_tn_target_cone",
    "execute_partial_mesh_reverse_pair",
    "execute_partial_mesh_forward_pair",
    "execute_partial_mesh_tn_redistribution",
    "execute_pre_sharded_pair_contraction",
    "execute_explicit_tn_reverse_dag",
    "execute_sliced_tn_explicit_reverse",
    "execute_sharded_tn_dag_operation",
    "execute_tn_forward_with_checkpoint_tape",
    "execute_tn_forward_with_tape",
    "estimate_sliced_tn_full_tape_bytes",
    "build_distributed_tn_memory_evidence",
    "plan_distributed_tn_contraction_dag",
    "compile_tn_forward_schedule",
    "compile_tn_reverse_schedule",
    "plan_distributed_tn_slice_tasks",
    "plan_tn_parameter_owners",
    "partition_tn_tensor_for_shard",
    "partition_tn_tensor_for_multi_axis_shard",
    "partition_tn_tensor_by_rank_coordinates",
    "partition_tn_tensor_for_partial_mesh",
    "plan_distributed_tn_redistribution",
    "plan_dynamic_tn_reverse_segment",
    "save_dynamic_tn_reverse_checkpoint",
    "plan_multi_axis_tn_layout",
    "plan_multi_axis_tn_peak_sharding",
    "plan_multi_axis_tn_redistribution",
    "plan_partial_mesh_tn_layout",
    "plan_partial_mesh_tn_redistribution",
    "plan_partial_mesh_rematerialization",
    "plan_explicit_tn_reverse_dag",
    "plan_joint_distributed_tn_execution",
    "plan_tn_adjoint_layouts",
    "plan_tn_checkpoints",
    "plan_sliced_tn_checkpoint_memory",
    "required_local_tn_inputs",
    "require_distributed_tn_memory_evidence",
    "prepare_sharded_tn_dag_operation",
    "pack_tn_redistribution_block",
    "run_distributed_tensor_network",
    "run_tensor_network",
    "shard_distributed_tn_value_layout",
    "select_sharded_pair_mode",
    "with_sharded_tn_intermediate",
    "with_sharded_tn_input",
    "validate_tn_adjoint_tensors",
)

_EXPORTS = {
    "DistributedTensorNetworkState": (
        "flagquantum.runtime.backends.tensor_network.execution",
        "DistributedTensorNetworkState",
    ),
    "CompiledTNForwardBucket": (
        "flagquantum.runtime.backends.tensor_network.reverse_dag",
        "CompiledTNForwardBucket",
    ),
    "CompiledTNForwardSchedule": (
        "flagquantum.runtime.backends.tensor_network.reverse_dag",
        "CompiledTNForwardSchedule",
    ),
    "CompiledTNReverseBucket": (
        "flagquantum.runtime.backends.tensor_network.reverse_dag",
        "CompiledTNReverseBucket",
    ),
    "CompiledTNReverseSchedule": (
        "flagquantum.runtime.backends.tensor_network.reverse_dag",
        "CompiledTNReverseSchedule",
    ),
    "compile_tn_forward_schedule": (
        "flagquantum.runtime.backends.tensor_network.reverse_dag",
        "compile_tn_forward_schedule",
    ),
    "compile_tn_reverse_schedule": (
        "flagquantum.runtime.backends.tensor_network.reverse_dag",
        "compile_tn_reverse_schedule",
    ),
    "execute_compiled_tn_forward_with_tape": (
        "flagquantum.runtime.backends.tensor_network.reverse_dag",
        "execute_compiled_tn_forward_with_tape",
    ),
    "execute_compiled_tn_reverse_dag": (
        "flagquantum.runtime.backends.tensor_network.reverse_dag",
        "execute_compiled_tn_reverse_dag",
    ),
    "DistributedTNMemoryEvidence": (
        "flagquantum.runtime.backends.tensor_network.memory_evidence",
        "DistributedTNMemoryEvidence",
    ),
    "DistributedTNRankMemoryMeasurement": (
        "flagquantum.runtime.backends.tensor_network.memory_evidence",
        "DistributedTNRankMemoryMeasurement",
    ),
    "build_distributed_tn_memory_evidence": (
        "flagquantum.runtime.backends.tensor_network.memory_evidence",
        "build_distributed_tn_memory_evidence",
    ),
    "require_distributed_tn_memory_evidence": (
        "flagquantum.runtime.backends.tensor_network.memory_evidence",
        "require_distributed_tn_memory_evidence",
    ),
    "DistributedTNRematerializationProvider": (
        "flagquantum.runtime.backends.tensor_network.rematerialization",
        "DistributedTNRematerializationProvider",
    ),
    "DistributedTNRematerializationPlan": (
        "flagquantum.runtime.backends.tensor_network.rematerialization",
        "DistributedTNRematerializationPlan",
    ),
    "plan_partial_mesh_rematerialization": (
        "flagquantum.runtime.backends.tensor_network.rematerialization",
        "plan_partial_mesh_rematerialization",
    ),
    "DistributedTNCheckpointAudit": (
        "flagquantum.runtime.backends.tensor_network.dynamic_checkpoint",
        "DistributedTNCheckpointAudit",
    ),
    "clear_dynamic_tn_checkpoint_writer_lock": (
        "flagquantum.runtime.backends.tensor_network.dynamic_checkpoint",
        "clear_dynamic_tn_checkpoint_writer_lock",
    ),
    "inspect_dynamic_tn_checkpoint": (
        "flagquantum.runtime.backends.tensor_network.dynamic_checkpoint",
        "inspect_dynamic_tn_checkpoint",
    ),
    "DistributedTNDynamicReverseSegment": (
        "flagquantum.runtime.backends.tensor_network.dynamic_reverse",
        "DistributedTNDynamicReverseSegment",
    ),
    "DistributedTNDynamicReverseSegmentResult": (
        "flagquantum.runtime.backends.tensor_network.dynamic_reverse",
        "DistributedTNDynamicReverseSegmentResult",
    ),
    "DistributedTNParameterPullbackResult": (
        "flagquantum.runtime.backends.tensor_network.dynamic_reverse",
        "DistributedTNParameterPullbackResult",
    ),
    "execute_dynamic_tn_reverse_segment": (
        "flagquantum.runtime.backends.tensor_network.dynamic_reverse",
        "execute_dynamic_tn_reverse_segment",
    ),
    "execute_dynamic_tn_parameter_pullback": (
        "flagquantum.runtime.backends.tensor_network.dynamic_reverse",
        "execute_dynamic_tn_parameter_pullback",
    ),
    "load_dynamic_tn_reverse_checkpoint": (
        "flagquantum.runtime.backends.tensor_network.dynamic_checkpoint",
        "load_dynamic_tn_reverse_checkpoint",
    ),
    "plan_dynamic_tn_reverse_segment": (
        "flagquantum.runtime.backends.tensor_network.dynamic_reverse",
        "plan_dynamic_tn_reverse_segment",
    ),
    "save_dynamic_tn_reverse_checkpoint": (
        "flagquantum.runtime.backends.tensor_network.dynamic_checkpoint",
        "save_dynamic_tn_reverse_checkpoint",
    ),
    "DistributedTNMeshGroupCache": (
        "flagquantum.runtime.backends.tensor_network.partial_mesh",
        "DistributedTNMeshGroupCache",
    ),
    "DistributedTNPartialMeshLayout": (
        "flagquantum.runtime.backends.tensor_network.partial_mesh",
        "DistributedTNPartialMeshLayout",
    ),
    "DistributedTNPartialMeshForwardResult": (
        "flagquantum.runtime.backends.tensor_network.partial_mesh",
        "DistributedTNPartialMeshForwardResult",
    ),
    "DistributedTNPartialMeshRedistributionPlan": (
        "flagquantum.runtime.backends.tensor_network.partial_mesh",
        "DistributedTNPartialMeshRedistributionPlan",
    ),
    "DistributedTNPartialMeshRedistributionResult": (
        "flagquantum.runtime.backends.tensor_network.partial_mesh",
        "DistributedTNPartialMeshRedistributionResult",
    ),
    "DistributedTNPartialMeshReverseResult": (
        "flagquantum.runtime.backends.tensor_network.partial_mesh",
        "DistributedTNPartialMeshReverseResult",
    ),
    "execute_partial_mesh_reverse_pair": (
        "flagquantum.runtime.backends.tensor_network.partial_mesh",
        "execute_partial_mesh_reverse_pair",
    ),
    "execute_partial_mesh_forward_pair": (
        "flagquantum.runtime.backends.tensor_network.partial_mesh",
        "execute_partial_mesh_forward_pair",
    ),
    "execute_partial_mesh_tn_redistribution": (
        "flagquantum.runtime.backends.tensor_network.partial_mesh",
        "execute_partial_mesh_tn_redistribution",
    ),
    "partition_tn_tensor_for_partial_mesh": (
        "flagquantum.runtime.backends.tensor_network.partial_mesh",
        "partition_tn_tensor_for_partial_mesh",
    ),
    "plan_partial_mesh_tn_layout": (
        "flagquantum.runtime.backends.tensor_network.partial_mesh",
        "plan_partial_mesh_tn_layout",
    ),
    "plan_partial_mesh_tn_redistribution": (
        "flagquantum.runtime.backends.tensor_network.partial_mesh",
        "plan_partial_mesh_tn_redistribution",
    ),
    "DistributedTNJointPlan": (
        "flagquantum.runtime.backends.tensor_network.joint_planning",
        "DistributedTNJointPlan",
    ),
    "DistributedTNWorkingSetPolicy": (
        "flagquantum.runtime.backends.tensor_network.joint_planning",
        "DistributedTNWorkingSetPolicy",
    ),
    "plan_joint_distributed_tn_execution": (
        "flagquantum.runtime.backends.tensor_network.joint_planning",
        "plan_joint_distributed_tn_execution",
    ),
    "DistributedTNAdjointLayoutPlan": (
        "flagquantum.runtime.backends.tensor_network.reverse_dag",
        "DistributedTNAdjointLayoutPlan",
    ),
    "DistributedTNAdjointValueLayout": (
        "flagquantum.runtime.backends.tensor_network.reverse_dag",
        "DistributedTNAdjointValueLayout",
    ),
    "DistributedTNCheckpointPlan": (
        "flagquantum.runtime.backends.tensor_network.reverse_dag",
        "DistributedTNCheckpointPlan",
    ),
    "DistributedTNReverseDAG": (
        "flagquantum.runtime.backends.tensor_network.reverse_dag",
        "DistributedTNReverseDAG",
    ),
    "DistributedTNReverseRecord": (
        "flagquantum.runtime.backends.tensor_network.reverse_dag",
        "DistributedTNReverseRecord",
    ),
    "DistributedTNReverseResult": (
        "flagquantum.runtime.backends.tensor_network.reverse_dag",
        "DistributedTNReverseResult",
    ),
    "execute_explicit_tn_reverse_dag": (
        "flagquantum.runtime.backends.tensor_network.reverse_dag",
        "execute_explicit_tn_reverse_dag",
    ),
    "execute_checkpointed_tn_reverse_dag": (
        "flagquantum.runtime.backends.tensor_network.reverse_dag",
        "execute_checkpointed_tn_reverse_dag",
    ),
    "execute_tn_forward_with_checkpoint_tape": (
        "flagquantum.runtime.backends.tensor_network.reverse_dag",
        "execute_tn_forward_with_checkpoint_tape",
    ),
    "execute_tn_forward_with_tape": (
        "flagquantum.runtime.backends.tensor_network.reverse_dag",
        "execute_tn_forward_with_tape",
    ),
    "plan_explicit_tn_reverse_dag": (
        "flagquantum.runtime.backends.tensor_network.reverse_dag",
        "plan_explicit_tn_reverse_dag",
    ),
    "plan_tn_adjoint_layouts": (
        "flagquantum.runtime.backends.tensor_network.reverse_dag",
        "plan_tn_adjoint_layouts",
    ),
    "plan_tn_checkpoints": (
        "flagquantum.runtime.backends.tensor_network.reverse_dag",
        "plan_tn_checkpoints",
    ),
    "validate_tn_adjoint_tensors": (
        "flagquantum.runtime.backends.tensor_network.reverse_dag",
        "validate_tn_adjoint_tensors",
    ),
    "DistributedTNMultiAxisContractionResult": (
        "flagquantum.runtime.backends.tensor_network.multi_axis_sharding",
        "DistributedTNMultiAxisContractionResult",
    ),
    "DistributedTNMultiAxisConeResult": (
        "flagquantum.runtime.backends.tensor_network.multi_axis_sharding",
        "DistributedTNMultiAxisConeResult",
    ),
    "DistributedTNMultiAxisLayout": (
        "flagquantum.runtime.backends.tensor_network.multi_axis_sharding",
        "DistributedTNMultiAxisLayout",
    ),
    "DistributedTNMultiAxisPeakPlan": (
        "flagquantum.runtime.backends.tensor_network.multi_axis_sharding",
        "DistributedTNMultiAxisPeakPlan",
    ),
    "DistributedTNMultiAxisShard": (
        "flagquantum.runtime.backends.tensor_network.multi_axis_sharding",
        "DistributedTNMultiAxisShard",
    ),
    "execute_multi_axis_contracted_pair": (
        "flagquantum.runtime.backends.tensor_network.multi_axis_sharding",
        "execute_multi_axis_contracted_pair",
    ),
    "execute_multi_axis_tn_target_cone": (
        "flagquantum.runtime.backends.tensor_network.multi_axis_sharding",
        "execute_multi_axis_tn_target_cone",
    ),
    "partition_tn_tensor_by_rank_coordinates": (
        "flagquantum.runtime.backends.tensor_network.multi_axis_sharding",
        "partition_tn_tensor_by_rank_coordinates",
    ),
    "partition_tn_tensor_for_multi_axis_shard": (
        "flagquantum.runtime.backends.tensor_network.multi_axis_sharding",
        "partition_tn_tensor_for_multi_axis_shard",
    ),
    "plan_multi_axis_tn_layout": (
        "flagquantum.runtime.backends.tensor_network.multi_axis_sharding",
        "plan_multi_axis_tn_layout",
    ),
    "plan_multi_axis_tn_peak_sharding": (
        "flagquantum.runtime.backends.tensor_network.multi_axis_sharding",
        "plan_multi_axis_tn_peak_sharding",
    ),
    "assemble_tn_redistribution_shard": (
        "flagquantum.runtime.backends.tensor_network.redistribution",
        "assemble_tn_redistribution_shard",
    ),
    "DistributedTNRedistributionBlock": (
        "flagquantum.runtime.backends.tensor_network.redistribution",
        "DistributedTNRedistributionBlock",
    ),
    "DistributedTNMultiAxisRedistributionPlan": (
        "flagquantum.runtime.backends.tensor_network.redistribution",
        "DistributedTNMultiAxisRedistributionPlan",
    ),
    "DistributedTNRedistributionPlan": (
        "flagquantum.runtime.backends.tensor_network.redistribution",
        "DistributedTNRedistributionPlan",
    ),
    "DistributedTNRedistributionResult": (
        "flagquantum.runtime.backends.tensor_network.redistribution",
        "DistributedTNRedistributionResult",
    ),
    "execute_multi_axis_tn_redistribution": (
        "flagquantum.runtime.backends.tensor_network.redistribution",
        "execute_multi_axis_tn_redistribution",
    ),
    "plan_multi_axis_tn_redistribution": (
        "flagquantum.runtime.backends.tensor_network.redistribution",
        "plan_multi_axis_tn_redistribution",
    ),
    "combine_contracted_shards": (
        "flagquantum.runtime.backends.tensor_network.sharded_kernels",
        "combine_contracted_shards",
    ),
    "combine_output_shards": (
        "flagquantum.runtime.backends.tensor_network.sharded_kernels",
        "combine_output_shards",
    ),
    "contract_pair_for_contracted_shard": (
        "flagquantum.runtime.backends.tensor_network.sharded_kernels",
        "contract_pair_for_contracted_shard",
    ),
    "contract_pair_for_output_shard": (
        "flagquantum.runtime.backends.tensor_network.sharded_kernels",
        "contract_pair_for_output_shard",
    ),
    "DistributedTNCommunicationEdge": (
        "flagquantum.runtime.backends.tensor_network.distributed_dag",
        "DistributedTNCommunicationEdge",
    ),
    "DistributedTNContractionDAG": (
        "flagquantum.runtime.backends.tensor_network.distributed_dag",
        "DistributedTNContractionDAG",
    ),
    "DistributedTNContractionRecord": (
        "flagquantum.runtime.backends.tensor_network.distributed_dag",
        "DistributedTNContractionRecord",
    ),
    "DistributedTNExecutionResult": (
        "flagquantum.runtime.backends.tensor_network.distributed_execution",
        "DistributedTNExecutionResult",
    ),
    "DistributedTNShardedDAGExecutionResult": (
        "flagquantum.runtime.backends.tensor_network.distributed_execution",
        "DistributedTNShardedDAGExecutionResult",
    ),
    "DistributedTNShard": (
        "flagquantum.runtime.backends.tensor_network.distributed_dag",
        "DistributedTNShard",
    ),
    "DistributedTNShardedPairResult": (
        "flagquantum.runtime.backends.tensor_network.sharded_kernels",
        "DistributedTNShardedPairResult",
    ),
    "DistributedSlicedTNReverseResult": (
        "flagquantum.runtime.backends.tensor_network.distributed_sliced_reverse",
        "DistributedSlicedTNReverseResult",
    ),
    "execute_distributed_sliced_tn_explicit_reverse": (
        "flagquantum.runtime.backends.tensor_network.distributed_sliced_reverse",
        "execute_distributed_sliced_tn_explicit_reverse",
    ),
    "DistributedTNOptimizerStepResult": (
        "flagquantum.runtime.backends.tensor_network.distributed_optimizer",
        "DistributedTNOptimizerStepResult",
    ),
    "execute_rank_owned_tn_sgd_step": (
        "flagquantum.runtime.backends.tensor_network.distributed_optimizer",
        "execute_rank_owned_tn_sgd_step",
    ),
    "plan_tn_parameter_owners": (
        "flagquantum.runtime.backends.tensor_network.distributed_optimizer",
        "plan_tn_parameter_owners",
    ),
    "DistributedTNSliceTask": (
        "flagquantum.runtime.backends.tensor_network.sliced_tasks",
        "DistributedTNSliceTask",
    ),
    "DistributedTNSliceTaskPlan": (
        "flagquantum.runtime.backends.tensor_network.sliced_tasks",
        "DistributedTNSliceTaskPlan",
    ),
    "plan_distributed_tn_slice_tasks": (
        "flagquantum.runtime.backends.tensor_network.sliced_tasks",
        "plan_distributed_tn_slice_tasks",
    ),
    "SlicedTNExplicitReverseResult": (
        "flagquantum.runtime.backends.tensor_network.sliced_reverse",
        "SlicedTNExplicitReverseResult",
    ),
    "execute_sliced_tn_explicit_reverse": (
        "flagquantum.runtime.backends.tensor_network.sliced_reverse",
        "execute_sliced_tn_explicit_reverse",
    ),
    "estimate_sliced_tn_full_tape_bytes": (
        "flagquantum.runtime.backends.tensor_network.sliced_reverse",
        "estimate_sliced_tn_full_tape_bytes",
    ),
    "SlicedTNCheckpointMemoryPlan": (
        "flagquantum.runtime.backends.tensor_network.sliced_reverse",
        "SlicedTNCheckpointMemoryPlan",
    ),
    "plan_sliced_tn_checkpoint_memory": (
        "flagquantum.runtime.backends.tensor_network.sliced_reverse",
        "plan_sliced_tn_checkpoint_memory",
    ),
    "DistributedTNValueLayout": (
        "flagquantum.runtime.backends.tensor_network.distributed_dag",
        "DistributedTNValueLayout",
    ),
    "plan_distributed_tn_contraction_dag": (
        "flagquantum.runtime.backends.tensor_network.distributed_dag",
        "plan_distributed_tn_contraction_dag",
    ),
    "partition_tn_tensor_for_shard": (
        "flagquantum.runtime.backends.tensor_network.sharded_kernels",
        "partition_tn_tensor_for_shard",
    ),
    "pack_tn_redistribution_block": (
        "flagquantum.runtime.backends.tensor_network.redistribution",
        "pack_tn_redistribution_block",
    ),
    "plan_distributed_tn_redistribution": (
        "flagquantum.runtime.backends.tensor_network.redistribution",
        "plan_distributed_tn_redistribution",
    ),
    "execute_distributed_tn_contraction_dag": (
        "flagquantum.runtime.backends.tensor_network.distributed_execution",
        "execute_distributed_tn_contraction_dag",
    ),
    "execute_sharded_tn_contraction_dag": (
        "flagquantum.runtime.backends.tensor_network.distributed_execution",
        "execute_sharded_tn_contraction_dag",
    ),
    "execute_distributed_tn_redistribution": (
        "flagquantum.runtime.backends.tensor_network.redistribution",
        "execute_distributed_tn_redistribution",
    ),
    "execute_pre_sharded_pair_contraction": (
        "flagquantum.runtime.backends.tensor_network.sharded_kernels",
        "execute_pre_sharded_pair_contraction",
    ),
    "execute_sharded_tn_dag_operation": (
        "flagquantum.runtime.backends.tensor_network.distributed_execution",
        "execute_sharded_tn_dag_operation",
    ),
    "required_local_tn_inputs": (
        "flagquantum.runtime.backends.tensor_network.distributed_execution",
        "required_local_tn_inputs",
    ),
    "prepare_sharded_tn_dag_operation": (
        "flagquantum.runtime.backends.tensor_network.distributed_execution",
        "prepare_sharded_tn_dag_operation",
    ),
    "shard_distributed_tn_value_layout": (
        "flagquantum.runtime.backends.tensor_network.distributed_dag",
        "shard_distributed_tn_value_layout",
    ),
    "select_sharded_pair_mode": (
        "flagquantum.runtime.backends.tensor_network.sharded_kernels",
        "select_sharded_pair_mode",
    ),
    "with_sharded_tn_intermediate": (
        "flagquantum.runtime.backends.tensor_network.distributed_dag",
        "with_sharded_tn_intermediate",
    ),
    "with_sharded_tn_input": (
        "flagquantum.runtime.backends.tensor_network.distributed_dag",
        "with_sharded_tn_input",
    ),
    "run_distributed_tensor_network": (
        "flagquantum.runtime.backends.tensor_network.execution",
        "run_distributed_tensor_network",
    ),
    "run_tensor_network": (
        "flagquantum.simulation.tensor_network.entrypoints",
        "run_tensor_network",
    ),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, symbol = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    return getattr(import_module(module_name), symbol)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

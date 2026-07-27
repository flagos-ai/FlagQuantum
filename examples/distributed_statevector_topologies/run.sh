#!/usr/bin/env bash
set -euo pipefail

NPROC="${NPROC:-${LOCAL_WORLD_SIZE:-1}}"
ENTANGLEMENT="${ENTANGLEMENT:-ring}"
OUT="${OUT:-benchmarks/results/statevector_mlsys_current/generality/flagquantum_31q_d8_ring_${NPROC}gpu_example.json}"
PYTHON="${PYTHON:-python}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "FlagQuantum: nvidia-smi not found; this example requires CUDA GPUs." >&2
  exit 2
fi
VISIBLE="$(${PYTHON} -c 'import torch; print(torch.cuda.device_count())')"
if [[ "${VISIBLE}" -lt "${NPROC}" ]]; then
  echo "FlagQuantum: requested ${NPROC} ranks but only ${VISIBLE} CUDA devices are visible." >&2
  exit 2
fi
case "${ENTANGLEMENT}" in
  linear|ring|brickwork) ;;
  *) echo "ENTANGLEMENT must be linear, ring, or brickwork" >&2; exit 2 ;;
esac
OUT="${OUT/ring/${ENTANGLEMENT}}"

export FQ_STATEVECTOR_PERSISTENT_WIRE_LAYOUT="${FQ_STATEVECTOR_PERSISTENT_WIRE_LAYOUT:-1}"
export FQ_STATEVECTOR_TRITON_LOCAL_CX="${FQ_STATEVECTOR_TRITON_LOCAL_CX:-1}"
export FQ_STATEVECTOR_TRITON_CX_SEGMENT="${FQ_STATEVECTOR_TRITON_CX_SEGMENT:-1}"
export FQ_STATEVECTOR_LOCAL_BLOCK_FUSION="${FQ_STATEVECTOR_LOCAL_BLOCK_FUSION:-1}"
export FQ_STATEVECTOR_LOCAL_BLOCK_FUSION_WIDTH="${FQ_STATEVECTOR_LOCAL_BLOCK_FUSION_WIDTH:-4}"
export FQ_STATEVECTOR_INTER_NODE_KET_CHECKPOINTS="${FQ_STATEVECTOR_INTER_NODE_KET_CHECKPOINTS:-inter_node}"
export FQ_STATEVECTOR_CROSS_SHARD_CX_PACK="${FQ_STATEVECTOR_CROSS_SHARD_CX_PACK:-1}"

echo "FlagQuantum ${ENTANGLEMENT} example: ${NPROC} ranks, ${VISIBLE} visible CUDA devices"
echo "mode=persistent-layout cx-segment local-fusion-width=${FQ_STATEVECTOR_LOCAL_BLOCK_FUSION_WIDTH}"

exec "${PYTHON}" -m torch.distributed.run --nproc-per-node="${NPROC}" \
  benchmarks/flagquantum_statevector_value_and_grad.py \
  --n-wires=31 --layers=8 --entanglement="${ENTANGLEMENT}" --warmup=2 --repetitions=5 \
  --json-output="${OUT}"

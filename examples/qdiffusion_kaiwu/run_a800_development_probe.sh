#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 8 ]]; then
  echo "usage: $0 EXECUTION_HOST EXPECTED_HOSTNAME FLAGQUANTUM_DIR PLUGIN_DIR OUTPUT SOURCE_REVISION PLUGIN_REVISION VALIDATION_IMAGE_ID" >&2
  exit 2
fi

execution_host=$1
expected_hostname=$2
flagquantum_dir=$3
plugin_dir=$4
output_path=$5
source_revision=$6
plugin_revision=$7
validation_image_id=$8

case "$execution_host" in
  jp-a800-171|jp-a800-172) ;;
  *)
    echo "unsupported execution host: $execution_host" >&2
    exit 2
    ;;
esac

if [[ ! -d "$flagquantum_dir" || ! -d "$plugin_dir" ]]; then
  echo "FlagQuantum and plugin inputs must be existing directories" >&2
  exit 2
fi
if [[ "$flagquantum_dir" != /* || "$plugin_dir" != /* || "$output_path" != /* ]]; then
  echo "source, plugin, and output paths must be absolute" >&2
  exit 2
fi
if [[ -e "$output_path" ]]; then
  echo "refusing to overwrite existing evidence: $output_path" >&2
  exit 2
fi
if [[ ! "$validation_image_id" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "validation image ID must be a full sha256 identity" >&2
  exit 2
fi
observed_image_id=$(docker image inspect "$validation_image_id" --format '{{.Id}}')
if [[ "$observed_image_id" != "$validation_image_id" ]]; then
  echo "validation image identity mismatch: $observed_image_id" >&2
  exit 1
fi

output_dir=$(dirname "$output_path")
output_name=$(basename "$output_path")
if [[ ! -d "$output_dir" ]]; then
  echo "output directory must already exist: $output_dir" >&2
  exit 2
fi

docker run --rm \
  --gpus device=0 \
  --network none \
  --read-only \
  --hostname "$expected_hostname" \
  --user "$(id -u):$(id -g)" \
  --tmpfs /tmp:rw,noexec,nosuid,size=256m \
  --env HOME=/tmp \
  --env PYTHONNOUSERSITE=1 \
  --env PYTHONPATH=/workspace/kaiwu-plugin/src:/workspace/flagquantum \
  --volume "$flagquantum_dir:/workspace/flagquantum:ro" \
  --volume "$plugin_dir:/workspace/kaiwu-plugin:ro" \
  --volume "$output_dir:/evidence:rw" \
  "$validation_image_id" \
  python3 -s -m examples.qdiffusion_kaiwu.qdiffusion_system_development_probe \
  --device cuda:0 \
  --execution-host "$execution_host" \
  --expected-hostname "$expected_hostname" \
  --source-revision "$source_revision" \
  --plugin-revision "$plugin_revision" \
  --validation-image-id "$validation_image_id" \
  --output "/evidence/$output_name"

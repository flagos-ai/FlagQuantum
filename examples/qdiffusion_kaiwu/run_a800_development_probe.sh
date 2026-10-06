#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 8 ]]; then
  echo "usage: $0 EXECUTION_HOST EXPECTED_HOSTNAME TRANSFER_DIR SOURCE_PREFLIGHT OUTPUT SOURCE_REVISION PLUGIN_REVISION VALIDATION_IMAGE_ID" >&2
  exit 2
fi

execution_host=$1
expected_hostname=$2
transfer_dir=$3
source_preflight_path=$4
output_path=$5
source_revision=$6
plugin_revision=$7
validation_image_id=$8

case "$execution_host:$expected_hostname" in
  jp-a800-171:bm-baai-dx-zone1-lc-a800-80g-15-171) ;;
  jp-a800-172:bm-baai-dx-zone1-lc-a800-80g-15-172) ;;
  *)
    echo "execution host and observed hostname do not match the reviewed pair" >&2
    exit 2
    ;;
esac

if [[ ! "$source_revision" =~ ^[0-9a-f]{40}$ || ! "$plugin_revision" =~ ^[0-9a-f]{40}$ ]]; then
  echo "source and plugin revisions must be full lowercase Git revisions" >&2
  exit 2
fi
if [[ ! "$validation_image_id" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "validation image ID must be a full sha256 identity" >&2
  exit 2
fi
if [[ "$transfer_dir" != /* || "$source_preflight_path" != /* || "$output_path" != /* ]]; then
  echo "transfer, source-preflight, and output paths must be absolute" >&2
  exit 2
fi

source_prefix=${source_revision:0:10}
plugin_prefix=${plugin_revision:0:10}
community_prefix=b648b531c0
manifest_name="flagquantum-qboson-a800-bundle-${source_prefix}.manifest.json"
source_archive="flagquantum-qboson-${source_prefix}.tar.gz"
plugin_archive="kaiwu-plugin-${plugin_prefix}.tar.gz"
community_archive="kaiwu-community-${community_prefix}.tar.gz"
preflight_name=$(basename "$source_preflight_path")
if [[ ! "$preflight_name" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "source-preflight filename contains unsupported characters" >&2
  exit 2
fi

manifest_sha256=$(python3 -B -s -m examples.qdiffusion_kaiwu.stream_development_evidence validate-inputs \
  --execution-host "$execution_host" \
  --expected-hostname "$expected_hostname" \
  --transfer-dir "$transfer_dir" \
  --source-preflight "$source_preflight_path" \
  --output "$output_path" \
  --source-revision "$source_revision" \
  --plugin-revision "$plugin_revision")

remote_script="umask 077
mkdir -p /workspace/input /workspace/evidence
tar -xf - -C /workspace/input
tar -xzf /workspace/input/$source_archive -C /workspace
tar -xzf /workspace/input/$plugin_archive -C /workspace
tar -xzf /workspace/input/$community_archive -C /workspace
export PYTHONPATH=/workspace/FlagQuantum-$source_prefix:/workspace/kaiwu-pytorch-plugin-$plugin_prefix/src
cd /workspace/FlagQuantum-$source_prefix
python3 -B -s -m examples.qdiffusion_kaiwu.qdiffusion_system_development_probe \\
  --device cuda:0 \\
  --execution-host $execution_host \\
  --expected-hostname $expected_hostname \\
  --source-revision $source_revision \\
  --plugin-revision $plugin_revision \\
  --plugin-root /workspace/kaiwu-pytorch-plugin-$plugin_prefix \\
  --source-preflight /workspace/input/$preflight_name \\
  --validation-image-id $validation_image_id \\
  --output /workspace/evidence/development.json >&2
cat /workspace/evidence/development.json"

COPYFILE_DISABLE=1 tar -cf - \
  -C "$transfer_dir" \
  "$source_archive" "$plugin_archive" "$community_archive" "$manifest_name" \
  -C "$(dirname "$source_preflight_path")" "$preflight_name" \
| ssh "$execution_host" \
    "docker run --rm -i --gpus device=0 --network none --read-only --log-driver none --hostname $expected_hostname --tmpfs /tmp:rw,nosuid,nodev,size=512m,mode=1777 --tmpfs /workspace:rw,nosuid,nodev,size=2g,mode=0700 -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONNOUSERSITE=1 $validation_image_id sh -ceu '$remote_script'" \
| python3 -B -s -m examples.qdiffusion_kaiwu.stream_development_evidence capture \
    --execution-host "$execution_host" \
    --expected-hostname "$expected_hostname" \
    --source-revision "$source_revision" \
    --plugin-revision "$plugin_revision" \
    --source-preflight "$source_preflight_path" \
    --validation-image-id "$validation_image_id" \
    --transfer-manifest-sha256 "$manifest_sha256" \
    --output "$output_path"

echo "Private streamed development evidence written to $output_path"

#!/usr/bin/env bash
set -euo pipefail

EXPECTED_COMMUNITY_REVISION="b648b531c034bd6ae9b7a34fed994c717967cc72"
EXPECTED_PLUGIN_REVISION="f047bce7b1077449967bbe9e9fab5741542b48d4"

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 /absolute/kaiwu-community /absolute/kaiwu-pytorch-plugin [/absolute/python]" >&2
  exit 2
fi

COMMUNITY_ROOT="$1"
PLUGIN_ROOT="$2"
PYTHON_BIN="${3:-python3}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPOSITORY_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd -P)"

for source_root in "$COMMUNITY_ROOT" "$PLUGIN_ROOT"; do
  if [[ "$source_root" != /* ]]; then
    echo "source roots must be absolute paths: $source_root" >&2
    exit 2
  fi
  if [[ ! -d "$source_root/.git" || ! -d "$source_root/src" ]]; then
    echo "source root is not a Git checkout with src/: $source_root" >&2
    exit 2
  fi
  if [[ -n "$(git -C "$source_root" status --porcelain --untracked-files=all)" ]]; then
    echo "source checkout must be clean: $source_root" >&2
    exit 2
  fi
done

if [[ "$PYTHON_BIN" == */* ]]; then
  if [[ "$PYTHON_BIN" != /* || ! -x "$PYTHON_BIN" ]]; then
    echo "explicit Python path must be absolute and executable" >&2
    exit 2
  fi
elif ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python command is unavailable: $PYTHON_BIN" >&2
  exit 2
fi

COMMUNITY_REVISION="$(git -C "$COMMUNITY_ROOT" rev-parse HEAD)"
PLUGIN_REVISION="$(git -C "$PLUGIN_ROOT" rev-parse HEAD)"
if [[ "$COMMUNITY_REVISION" != "$EXPECTED_COMMUNITY_REVISION" ]]; then
  echo "Kaiwu Community revision mismatch: $COMMUNITY_REVISION" >&2
  exit 1
fi
if [[ "$PLUGIN_REVISION" != "$EXPECTED_PLUGIN_REVISION" ]]; then
  echo "Kaiwu PyTorch Plugin revision mismatch: $PLUGIN_REVISION" >&2
  exit 1
fi

unset QBOSON_USER_ID QBOSON_SDK_CODE QBOSON_PROJECT_NO
export PYTHONNOUSERSITE=1
export PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/flagquantum-kaiwu-pycache-$$"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export FLAGQUANTUM_TEST_KAIWU_SOURCE=1
export PYTHONPATH="$COMMUNITY_ROOT/src:$PLUGIN_ROOT/src:$REPOSITORY_ROOT"

cd -- "$REPOSITORY_ROOT"
"$PYTHON_BIN" -B -m pytest -q \
  tests/team/ecosystem/test_kaiwu_matrix_boundary.py \
  tests/team/ecosystem/test_kaiwu_community_conformance.py \
  tests/team/ecosystem/test_kaiwu_sampler.py \
  tests/team/ecosystem/test_kaiwu_qdiffusion_binding.py \
  tests/team/ecosystem/test_kaiwu_pytorch_plugin_conformance.py \
  tests/team/ecosystem/test_qdiffusion_live_system_probe.py \
  tests/team/ecosystem/test_qdiffusion_environment_lock.py \
  tests/team/remote/test_kaiwu_credentials.py \
  tests/team/remote/test_kaiwu_jobs.py \
  tests/team/remote/test_kaiwu_client.py \
  tests/team/remote/test_kaiwu_sdk.py

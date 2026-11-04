#!/usr/bin/env bash
# Source this in the SDK job to use a private environment for the checkout.
set -euo pipefail

fq_attach_preinstalled_sdk() {
  local profile="$1" seed minor packages private
  seed="/opt/flagquantum-runners/tools/ci-envs/sdk-$profile"
  test -r "$seed/.ready" || { echo "Missing preinstalled SDK profile: $profile" >&2; return 1; }
  minor="$("$seed/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  packages="$seed/lib/python$minor/site-packages"
  private="$(mktemp -d "$RUNNER_TEMP/fq-sdk.XXXXXXXX")"
  "$seed/bin/python" -m venv --without-pip "$private"
  {
    printf '%s\n' "$packages"
    cat "$packages/fq-core.pth"
  } > "$private/lib/python$minor/site-packages/fq-sdk-seed.pth"
  export PATH="$private/bin:$PATH" VIRTUAL_ENV="$private" PYTHONNOUSERSITE=1
  export PYTHONPATH= PYTHONHOME= PYTHONDONTWRITEBYTECODE=1
  python -m pip --isolated install --no-index --no-deps --no-build-isolation -e .
}

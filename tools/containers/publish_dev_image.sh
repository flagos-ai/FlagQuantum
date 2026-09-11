#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 <cpu|cuda|cpu-no-jax|cuda-no-jax> [version]" >&2
  echo "environment: GHCR_OWNER, GHCR_IMAGE, FLAGQUANTUM_BASE_IMAGE" >&2
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
  exit 2
fi

variant="$1"

case "$variant" in
  cpu | cuda | cpu-no-jax | cuda-no-jax) ;;
  *)
    usage
    exit 2
    ;;
esac

owner="${GHCR_OWNER:-flagos-ai}"
owner="$(printf '%s' "$owner" | tr '[:upper:]' '[:lower:]')"
image="${GHCR_IMAGE:-ghcr.io/${owner}/flagquantum-dev}"

if [[ $# -eq 2 ]]; then
  version="$2"
elif command -v git >/dev/null 2>&1 && git rev-parse --verify HEAD >/dev/null 2>&1; then
  version="$(git rev-parse --short=12 HEAD)"
else
  echo "unable to derive a version; pass one explicitly" >&2
  exit 2
fi

case "$version" in
  *[!a-zA-Z0-9_.-]* | "")
    echo "version must contain only letters, digits, dot, underscore, or dash" >&2
    exit 2
    ;;
esac

if ! docker info >/dev/null 2>&1; then
  echo "Docker is unavailable; start Docker and log in to ghcr.io first" >&2
  exit 1
fi

source_url="https://github.com/flagos-ai/FlagQuantum"
target=full
extras=dev,jax,viz,examples,interop-all
suffix=""
if [[ "$variant" == *-no-jax ]]; then
  target=no-jax
  extras=dev,viz,examples,quafu
  suffix=-no-jax
  variant="${variant%-no-jax}"
fi

case "$variant" in
  cpu)
    docker buildx build \
      --platform linux/amd64 \
      --file docker/dev/Dockerfile \
      --target "$target" \
      --build-arg BASE_IMAGE=python:3.12-slim-bookworm \
      --build-arg EXTRAS="$extras" \
      --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu \
      --build-arg SOURCE_URL="$source_url" \
      --build-arg SOURCE_REVISION="$version" \
      --tag "$image:cpu${suffix}" \
      --tag "$image:cpu${suffix}-$version" \
      --provenance=mode=max \
      --sbom=true \
      --push \
      .
    ;;
  cuda)
    base_image="${FLAGQUANTUM_BASE_IMAGE:-python:3.12-slim-bookworm}"
    version_tag="$image:cuda-$version"
    current_tag="$image:cuda-amd64"
    if [[ -n "$suffix" ]]; then
      version_tag="$image:cuda${suffix}-$version"
      current_tag="$image:cuda-amd64${suffix}"
    fi

    docker build \
      --platform linux/amd64 \
      --file docker/dev/Dockerfile \
      --target "$target" \
      --build-arg BASE_IMAGE="$base_image" \
      --build-arg EXTRAS="$extras,cuda" \
      --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 \
      --build-arg 'JAX_SPEC=jax[cuda12]>=0.10,<0.11' \
      --build-arg SOURCE_URL="$source_url" \
      --build-arg SOURCE_REVISION="$version" \
      --tag "$version_tag" \
      --tag "$current_tag" \
      .

    docker push "$version_tag"
    docker push "$current_tag"
    ;;
esac

echo "published $variant development image to $image"

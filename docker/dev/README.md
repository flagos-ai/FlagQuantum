# FlagQuantum development container

This image is for daily editable development and remote GPU testing. It is not
an SC27 frozen evidence image and must not be used to promote benchmark claims.

## Local CPU development

Build the default Python 3.12 image and run the daily test tier:

```bash
docker compose -f compose.dev.yaml build dev
docker compose -f compose.dev.yaml run --rm dev \
  python tools/ci_tier.py pr-default
```

The default build installs PyTorch from its CPU wheel index, avoiding the
multi-gigabyte CUDA dependency closure on a non-NVIDIA laptop.

Open an interactive shell with the live checkout mounted at `/workspace`:

```bash
docker compose -f compose.dev.yaml run --rm dev
```

## Remote NVIDIA GPU development

The remote host needs the NVIDIA Container Toolkit. Use a CUDA/PyTorch base
that is already validated on that host; for example, the existing FlagQuantum
development base can be selected without changing this Dockerfile:

```bash
export FLAGQUANTUM_BASE_IMAGE='tovx/flagquantum@sha256:bce47a929a36ed60a3a199183aead552b8466818839325ac149c28be9f02f2b5'
export FLAGQUANTUM_DEV_EXTRAS='dev,jax,cuda'
export FLAGQUANTUM_DEV_IMAGE='flagquantum-dev:cuda-local'

docker compose -f compose.dev.yaml build dev-gpu
docker compose -f compose.dev.yaml run --rm dev-gpu \
  python -c 'import torch; print(torch.__version__, torch.cuda.device_count())'
docker compose -f compose.dev.yaml run --rm dev-gpu \
  python tools/ci_tier.py gpu-scheduled
```

The repository is bind-mounted, so code changes do not require an image
rebuild. Rebuild only when dependencies or the selected base image change.

## Private GHCR publishing

The GitHub Actions workflow `.github/workflows/publish-dev-container.yml`
publishes a multi-architecture CPU image on changes to `main`, or when manually
dispatched:

```text
ghcr.io/flagquantum/flagquantum-dev:cpu
ghcr.io/flagquantum/flagquantum-dev:cpu-sha-<commit>
```

It authenticates with `GITHUB_TOKEN`, attaches OCI provenance and an SBOM, and
fails if the resulting organization package is not private. The image is
explicitly labelled `development_only` and cannot serve as SC27 evidence.

The CUDA base is available on the A800 development host rather than a public
registry, so publish CUDA images from that AMD64 host after logging in to GHCR:

```bash
read -s GHCR_TOKEN
printf '%s' "$GHCR_TOKEN" | docker login ghcr.io \
  -u '<github-user>' --password-stdin
unset GHCR_TOKEN

tools/containers/publish_dev_image.sh cuda
```

This publishes both a rolling and immutable tag:

```text
ghcr.io/flagquantum/flagquantum-dev:cuda-amd64
ghcr.io/flagquantum/flagquantum-dev:cuda-<commit>
```

The login token needs `write:packages` on the publishing host. A test host only
needs `read:packages` to pull the private image.

## Push once, pull over SSH

After `docker login`, choose a registry path and push the image:

```bash
export FLAGQUANTUM_DEV_IMAGE='ghcr.io/flagquantum/flagquantum-dev:cuda-amd64'
docker compose -f compose.dev.yaml build dev-gpu
docker push "$FLAGQUANTUM_DEV_IMAGE"
```

On each remote node, pull the same tag (or preferably its immutable digest),
keep a synchronized source checkout, and use the `dev-gpu` service. Never put
registry passwords or tokens in this repository or in Docker build arguments.

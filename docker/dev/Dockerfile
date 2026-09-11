ARG BASE_IMAGE=python:3.12-slim-bookworm
FROM ${BASE_IMAGE}

ARG EXTRAS=dev,jax,viz,examples,interop-all
ARG TORCH_VERSION=2.10.0
ARG JAX_SPEC=jax>=0.10,<0.11
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG SOURCE_URL=https://github.com/flagos-ai/FlagQuantum
ARG SOURCE_REVISION=development

LABEL org.opencontainers.image.source="${SOURCE_URL}" \
      org.opencontainers.image.revision="${SOURCE_REVISION}" \
      org.opencontainers.image.title="FlagQuantum development environment" \
      org.opencontainers.image.description="FlagQuantum development environment with JAX, QSteed and ecosystem tools" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.flagquantum.image.class="development_only" \
      org.flagquantum.sc27.release_evidence="false"

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        iproute2 \
        openssh-client \
        procps \
        tini \
    && rm -rf /var/lib/apt/lists/*

# Install an image-contained checkout first so dependency installation is
# cached. The compose workflow bind-mounts the live checkout at /workspace and
# puts it first on PYTHONPATH, so edits are visible without rebuilding.
COPY . /opt/FlagQuantum
WORKDIR /opt/FlagQuantum

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX_URL}" \
    && python -m pip install --no-build-isolation "torch==${TORCH_VERSION}" "${JAX_SPEC}" -e ".[${EXTRAS}]" \
    && python -m pip check \
    && JAX_PLATFORMS=cpu python docker/dev/smoke.py

# QSteed/pyquafu require NumPy 1.x, while JAX requires NumPy 2.x.
# Keep the compiler in an isolated environment; never bypass dependency checks.
RUN python -m venv /opt/qsteed \
    && /opt/qsteed/bin/python -m pip install --upgrade pip setuptools wheel \
    && /opt/qsteed/bin/python -m pip install "torch==${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/cpu \
    && /opt/qsteed/bin/python -m pip install --no-build-isolation "torch==${TORCH_VERSION}" -e ".[quafu]" -r docker/dev/requirements-qsteed.txt \
    && /opt/qsteed/bin/python -m pip check \
    && /opt/qsteed/bin/python docker/dev/smoke.py --qsteed \
    && install -m 755 docker/dev/flagquantum-qsteed-python /usr/local/bin/flagquantum-qsteed-python

ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
ENV PYTHONPATH=/workspace
WORKDIR /workspace

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash"]

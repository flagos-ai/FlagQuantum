ARG BASE_IMAGE=python:3.12-slim-bookworm
FROM ${BASE_IMAGE}

ARG EXTRAS=dev,jax
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG SOURCE_URL=https://github.com/FlagQuantum/FlagQuantum
ARG SOURCE_REVISION=development

LABEL org.opencontainers.image.source="${SOURCE_URL}" \
      org.opencontainers.image.revision="${SOURCE_REVISION}" \
      org.opencontainers.image.title="FlagQuantum development environment" \
      org.opencontainers.image.description="Editable CPU and NVIDIA GPU development environment for FlagQuantum" \
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
    && python -m pip install "torch>=2.5,<2.14" --index-url "${TORCH_INDEX_URL}" \
    && python -m pip install --no-build-isolation -e ".[${EXTRAS}]" \
    && python -m pip check \
    && python -c "import flagquantum as fq; print('FlagQuantum', fq.__version__)"

ENV PYTHONPATH=/workspace
WORKDIR /workspace

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash"]

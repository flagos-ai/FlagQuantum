ARG BASE_IMAGE=pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
FROM ${BASE_IMAGE}

ARG FLAGQUANTUM_WHEEL=flagquantum-0.2.0-py3-none-any.whl

LABEL org.opencontainers.image.title="FlagQuantum runtime" \
      org.opencontainers.image.description="Minimal NVIDIA GPU job runtime for FlagQuantum" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.flagquantum.image.class="runtime_only" \
      org.flagquantum.sc27.release_evidence="false"

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY dist/${FLAGQUANTUM_WHEEL} /tmp/${FLAGQUANTUM_WHEEL}

RUN apt-get update \
    && apt-get install -yq --no-install-recommends openssh-server tzdata \
    && mkdir -p /run/sshd \
    && python -m pip install --no-deps /tmp/${FLAGQUANTUM_WHEEL} \
    && rm -f /tmp/${FLAGQUANTUM_WHEEL} \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && python -c "import flagquantum, torch; assert torch.version.cuda"

WORKDIR /workspace

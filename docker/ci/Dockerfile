ARG PYTHON_VERSION
FROM python:${PYTHON_VERSION}-bookworm
ARG PROFILE
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
LABEL org.opencontainers.image.source=https://github.com/flagos-ai/FlagQuantum
COPY pyproject.toml docker/ci/profiles.json docker/ci/requirements.py /opt/flagquantum/
RUN python -m venv /opt/ci
ENV PATH="/opt/ci/bin:${PATH}"
RUN python -m pip install --only-binary=:all: --no-deps \
        --index-url https://download.pytorch.org/whl/cpu 'torch>=2.13,<2.14' \
    && python -m pip install --only-binary=:all: 'tomli>=2,<3; python_version < "3.11"' \
    && python /opt/flagquantum/requirements.py "$PROFILE" > /opt/flagquantum/requirements.in
RUN python -m pip freeze | grep '^torch==' > /opt/flagquantum/torch.lock \
    && python -m pip install --only-binary=:all: -c /opt/flagquantum/torch.lock \
        -r /opt/flagquantum/requirements.in \
    && python -m pip check \
    && python -c 'import torch; assert torch.version.cuda is None and "+cpu" in torch.__version__' \
    && python -m pip freeze --all > /opt/flagquantum/requirements.lock
ENV PIP_NO_INDEX=1 PYTHONNOUSERSITE=1

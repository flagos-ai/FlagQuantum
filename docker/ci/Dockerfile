ARG PYTHON_VERSION
FROM python:${PYTHON_VERSION}-bookworm
ARG PROFILE
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
COPY pyproject.toml /opt/flagquantum/pyproject.toml
RUN python -m venv /opt/ci
ENV PATH="/opt/ci/bin:${PATH}"
RUN python -m pip install --only-binary=:all: --no-deps \
        --index-url https://download.pytorch.org/whl/cpu 'torch>=2.13,<2.14' \
    && python -m pip install --only-binary=:all: 'tomli>=2,<3; python_version < "3.11"' \
    && python -c 'import sys; import importlib; tomllib=importlib.import_module("tomli" if sys.version_info < (3,11) else "tomllib"); from pathlib import Path; p=tomllib.loads(Path("/opt/flagquantum/pyproject.toml").read_text()); extras="dev,jax,viz,braket,cirq,pennylane,cotengra,stim,pymatching,kaiwu" if sys.argv[1]=="coverage" else "dev"; print("\n".join(p["build-system"]["requires"]+p["project"]["dependencies"]+[r for x in extras.split(",") for r in p["project"]["optional-dependencies"][x]]))' "$PROFILE" > /opt/flagquantum/requirements.in
RUN python -m pip install --only-binary=:all: -r /opt/flagquantum/requirements.in \
    && python -m pip check \
    && python -c 'import torch; assert torch.version.cuda is None and "+cpu" in torch.__version__' \
    && python -m pip freeze --all > /opt/flagquantum/requirements.lock
ENV PIP_NO_INDEX=1 PYTHONNOUSERSITE=1

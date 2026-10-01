ARG BASE_IMAGE=swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/vllm/vllm-openai@sha256:50509e700235cea487715cedeb501d20a1cd15fa6a54ce93688284bd0d96995d
FROM ${BASE_IMAGE}

ARG FLAGTREE_VERSION=0.7.0
ARG FLAGTREE_INDEX_URL=https://resource.flagos.net/repository/flagos-pypi-nvidia/simple

RUN /usr/bin/python3 -m pip uninstall -y triton \
    && /usr/bin/python3 -m pip install --no-deps \
       "flagtree==${FLAGTREE_VERSION}" \
       --index-url "${FLAGTREE_INDEX_URL}" \
    && /usr/bin/python3 -c \
       'from importlib import metadata; assert metadata.packages_distributions().get("triton") == ["flagtree"]'

ENV PYTHONDONTWRITEBYTECODE=1
ENTRYPOINT []

"""Historical deployment import facade."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compute.remote.validation import (
    ArtifactDeploymentDryRun as ArtifactDeploymentDryRun,
)
from flagquantum.compute.remote.validation import (
    prepare_artifact_deployment as prepare_artifact_deployment,
)

_sys.modules[__name__] = _import_module("flagquantum.compute.remote.validation")

"""Historical deployment import facade."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compiler.artifacts.routing_evidence import (
    DeploymentRoutingEvidenceError as DeploymentRoutingEvidenceError,
)
from flagquantum.compiler.artifacts.routing_evidence import (
    _accepted_direction_semantics as _accepted_direction_semantics,
)
from flagquantum.compiler.artifacts.routing_evidence import (
    _non_negative_int as _non_negative_int,
)
from flagquantum.compiler.artifacts.routing_evidence import (
    _undirected_couplings as _undirected_couplings,
)
from flagquantum.compiler.artifacts.routing_evidence import (
    build_deployment_routing_evidence as build_deployment_routing_evidence,
)
from flagquantum.compiler.artifacts.routing_evidence import (
    deployment_artifact_sha256 as deployment_artifact_sha256,
)
from flagquantum.compiler.artifacts.routing_evidence import (
    same_undirected_device as same_undirected_device,
)
from flagquantum.compiler.artifacts.routing_evidence import (
    stable_payload_sha256 as stable_payload_sha256,
)
from flagquantum.compiler.artifacts.routing_evidence import (
    validate_deployment_routing_plan as validate_deployment_routing_plan,
)

_sys.modules[__name__] = _import_module(
    "flagquantum.compiler.artifacts.routing_evidence"
)

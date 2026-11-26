"""Historical deployment import facade."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.runtime.dynamic.routing_rematch import (
    _core_requirement as _core_requirement,
)
from flagquantum.runtime.dynamic.routing_rematch import (
    _DynamicRoutingRematchDecision as _DynamicRoutingRematchDecision,
)
from flagquantum.runtime.dynamic.routing_rematch import _fact_state as _fact_state
from flagquantum.runtime.dynamic.routing_rematch import (
    _route_fact_states as _route_fact_states,
)
from flagquantum.runtime.dynamic.routing_rematch import (
    _route_requirements as _route_requirements,
)
from flagquantum.runtime.dynamic.routing_rematch import (
    evaluate_dynamic_routing_rematch as evaluate_dynamic_routing_rematch,
)

_sys.modules[__name__] = _import_module("flagquantum.runtime.dynamic.routing_rematch")

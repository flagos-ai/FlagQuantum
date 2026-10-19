"""Stable expert compiler interface.

Normal execution compiles internally through :func:`flagquantum.plan`.

The optimization pipeline is a named sequence of registered passes, and this
interface exposes both halves of that: the entry points that run it, and the
registry a caller extends it with. A caller-supplied transformation is admitted by
name through ``optimize(..., passes=...)`` and ``compile(..., passes=...)``, or
registered directly on ``default_pass_registry()`` when it needs to declare an
analysis, and then run by ``PassManager`` over ``OPTIMIZATION_PIPELINE``. The
modules hold the authority; these names are the supported way to reach it.
"""

from .layout import Layout, apply_layout, final_layout, remove_layout_restore
from .layout_planning import plan_dense_layout, plan_trivial_layout
from .noise import lower_noise_model
from .pass_manager import (
    OPTIMIZATION_PIPELINE,
    RESERVED_PASS_NAMES,
    AnalysisSpec,
    PassManager,
    PassRegistry,
    PassRegistryError,
    extension_pass_name,
)
from .pipeline import (
    CouplingMap,
    compile,
    default_pass_registry,
    optimize,
    route_to_topology,
    schedule_layers,
)
from .resource_estimation import ESTIMATE_BASIS, ResourceEstimate, estimate_resources
from .translate import TRANSLATION_FORMATS, translate

__all__ = (
    "CouplingMap",
    "ESTIMATE_BASIS",
    "Layout",
    "OPTIMIZATION_PIPELINE",
    "RESERVED_PASS_NAMES",
    "ResourceEstimate",
    "TRANSLATION_FORMATS",
    "AnalysisSpec",
    "PassManager",
    "PassRegistry",
    "PassRegistryError",
    "apply_layout",
    "compile",
    "default_pass_registry",
    "estimate_resources",
    "extension_pass_name",
    "final_layout",
    "lower_noise_model",
    "optimize",
    "plan_dense_layout",
    "plan_trivial_layout",
    "remove_layout_restore",
    "route_to_topology",
    "schedule_layers",
    "translate",
)

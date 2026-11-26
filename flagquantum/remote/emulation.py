"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compute.remote.emulation import (
    _APPROXIMATE_REPRESENTATION as _APPROXIMATE_REPRESENTATION,
)
from flagquantum.compute.remote.emulation import (
    _EMULATED_DEFAULT_MAXIMUM_OPERATIONS as _EMULATED_DEFAULT_MAXIMUM_OPERATIONS,
)
from flagquantum.compute.remote.emulation import (
    _EMULATED_DEFAULT_MAXIMUM_SHOTS as _EMULATED_DEFAULT_MAXIMUM_SHOTS,
)
from flagquantum.compute.remote.emulation import (
    _EMULATED_RESULT_FORMATS as _EMULATED_RESULT_FORMATS,
)
from flagquantum.compute.remote.emulation import (
    _SUPPORTED_REQUEST_KINDS as _SUPPORTED_REQUEST_KINDS,
)
from flagquantum.compute.remote.emulation import (
    EMULATED_TARGET_ADAPTER_VERSION as EMULATED_TARGET_ADAPTER_VERSION,
)
from flagquantum.compute.remote.emulation import (
    EMULATED_TARGET_CLASS as EMULATED_TARGET_CLASS,
)
from flagquantum.compute.remote.emulation import (
    EMULATED_TARGET_DECLARATION_SOURCE_KIND as EMULATED_TARGET_DECLARATION_SOURCE_KIND,
)
from flagquantum.compute.remote.emulation import (
    EMULATED_TARGET_PROVIDER as EMULATED_TARGET_PROVIDER,
)
from flagquantum.compute.remote.emulation import (
    LOCAL_TARGET_EMULATION_SCHEMA as LOCAL_TARGET_EMULATION_SCHEMA,
)
from flagquantum.compute.remote.emulation import (
    NOISE_SOURCE_CALLER_MODEL as NOISE_SOURCE_CALLER_MODEL,
)
from flagquantum.compute.remote.emulation import (
    NOISE_SOURCE_DEVICE_PROFILE as NOISE_SOURCE_DEVICE_PROFILE,
)
from flagquantum.compute.remote.emulation import (
    NOISE_SOURCE_NONE as NOISE_SOURCE_NONE,
)
from flagquantum.compute.remote.emulation import (
    EmulatedExecutionRecord as EmulatedExecutionRecord,
)
from flagquantum.compute.remote.emulation import (
    LocalTargetEmulation as LocalTargetEmulation,
)
from flagquantum.compute.remote.emulation import (
    TargetCompilationDiagnostic as TargetCompilationDiagnostic,
)
from flagquantum.compute.remote.emulation import (
    TargetEmulationError as TargetEmulationError,
)
from flagquantum.compute.remote.emulation import __all__ as __all__
from flagquantum.compute.remote.emulation import (
    _declaration_digest as _declaration_digest,
)
from flagquantum.compute.remote.emulation import (
    _declared_emission_profile as _declared_emission_profile,
)
from flagquantum.compute.remote.emulation import (
    _declared_limits as _declared_limits,
)
from flagquantum.compute.remote.emulation import (
    _declared_native_gates as _declared_native_gates,
)
from flagquantum.compute.remote.emulation import (
    _declared_target_profile as _declared_target_profile,
)
from flagquantum.compute.remote.emulation import (
    _execute as _execute,
)
from flagquantum.compute.remote.emulation import (
    _local_counts as _local_counts,
)
from flagquantum.compute.remote.emulation import (
    _prepare_program as _prepare_program,
)
from flagquantum.compute.remote.emulation import (
    _record_payload as _record_payload,
)
from flagquantum.compute.remote.emulation import (
    _requested_kind as _requested_kind,
)
from flagquantum.compute.remote.emulation import (
    _require_declared_request_kinds as _require_declared_request_kinds,
)
from flagquantum.compute.remote.emulation import (
    _require_profile_coverage as _require_profile_coverage,
)
from flagquantum.compute.remote.emulation import (
    _select_declared_device as _select_declared_device,
)
from flagquantum.compute.remote.emulation import (
    _select_emission_profile as _select_emission_profile,
)
from flagquantum.compute.remote.emulation import (
    _target_snapshot as _target_snapshot,
)
from flagquantum.compute.remote.emulation import (
    emulate as emulate,
)
from flagquantum.compute.remote.emulation import (
    emulate_declared_target as emulate_declared_target,
)

_implementation = _import_module("flagquantum.compute.remote.emulation")
_sys.modules[__name__] = _implementation

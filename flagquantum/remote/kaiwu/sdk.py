"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compute.remote.kaiwu.sdk import (
    KaiwuLicenseInitializationError as KaiwuLicenseInitializationError,
)
from flagquantum.compute.remote.kaiwu.sdk import (
    KaiwuSDKEnvironment as KaiwuSDKEnvironment,
)
from flagquantum.compute.remote.kaiwu.sdk import (
    KaiwuSDKError as KaiwuSDKError,
)
from flagquantum.compute.remote.kaiwu.sdk import (
    KaiwuSDKUnavailableError as KaiwuSDKUnavailableError,
)
from flagquantum.compute.remote.kaiwu.sdk import (
    KaiwuSDKVersionError as KaiwuSDKVersionError,
)
from flagquantum.compute.remote.kaiwu.sdk import (
    _initialize_preflighted_kaiwu_license as _initialize_preflighted_kaiwu_license,
)
from flagquantum.compute.remote.kaiwu.sdk import (
    _load_kaiwu_module as _load_kaiwu_module,
)
from flagquantum.compute.remote.kaiwu.sdk import (
    _preflight_kaiwu_sdk as _preflight_kaiwu_sdk,
)
from flagquantum.compute.remote.kaiwu.sdk import (
    initialize_kaiwu_license as initialize_kaiwu_license,
)

_implementation = _import_module("flagquantum.compute.remote.kaiwu.sdk")
_sys.modules[__name__] = _implementation

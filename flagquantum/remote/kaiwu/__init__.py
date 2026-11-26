"""Historical import facade for the relocated implementation."""

import sys as _alias_sys
from importlib import import_module as _import_module
from typing import Any as _Any

from flagquantum.compute.remote.kaiwu import (
    KaiwuCredentials as KaiwuCredentials,
)
from flagquantum.compute.remote.kaiwu import (
    KaiwuJobStatus as KaiwuJobStatus,
)
from flagquantum.compute.remote.kaiwu import (
    KaiwuLicenseInitializationError as KaiwuLicenseInitializationError,
)
from flagquantum.compute.remote.kaiwu import (
    KaiwuRemoteJob as KaiwuRemoteJob,
)
from flagquantum.compute.remote.kaiwu import (
    KaiwuSDKClient as KaiwuSDKClient,
)
from flagquantum.compute.remote.kaiwu import (
    KaiwuSDKEnvironment as KaiwuSDKEnvironment,
)
from flagquantum.compute.remote.kaiwu import (
    KaiwuSDKError as KaiwuSDKError,
)
from flagquantum.compute.remote.kaiwu import (
    KaiwuSDKUnavailableError as KaiwuSDKUnavailableError,
)
from flagquantum.compute.remote.kaiwu import (
    KaiwuSDKVersionError as KaiwuSDKVersionError,
)
from flagquantum.compute.remote.kaiwu import (
    KaiwuTaskClient as KaiwuTaskClient,
)
from flagquantum.compute.remote.kaiwu import (
    KaiwuTaskMode as KaiwuTaskMode,
)
from flagquantum.compute.remote.kaiwu import (
    KaiwuTaskReceipt as KaiwuTaskReceipt,
)
from flagquantum.compute.remote.kaiwu import (
    KaiwuTaskResult as KaiwuTaskResult,
)
from flagquantum.compute.remote.kaiwu import __all__ as __all__
from flagquantum.compute.remote.kaiwu import (
    initialize_kaiwu_license as initialize_kaiwu_license,
)
from flagquantum.compute.remote.kaiwu import (
    new_receipt as new_receipt,
)
from flagquantum.compute.remote.kaiwu import (
    resolve_kaiwu_credentials as resolve_kaiwu_credentials,
)
from flagquantum.compute.remote.kaiwu import (
    restore_kaiwu_job as restore_kaiwu_job,
)
from flagquantum.compute.remote.kaiwu import (
    submit_kaiwu_task as submit_kaiwu_task,
)

_implementation = _import_module("flagquantum.compute.remote.kaiwu")
__doc__ = _implementation.__doc__ or ""


def __getattr__(name: str) -> _Any:
    cached = _alias_sys.modules.get(f"{__name__}.{name}")
    if cached is not None:
        return cached
    return getattr(_implementation, name)

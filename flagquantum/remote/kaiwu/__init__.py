"""Remote control-plane support for QBoson Kaiwu services.

The module is intentionally not re-exported from :mod:`flagquantum.remote`
while its provider contract is under review.
"""

from ._credentials import KaiwuCredentials, resolve_kaiwu_credentials
from .client import KaiwuSDKClient
from .contracts import (
    KaiwuJobStatus,
    KaiwuTaskClient,
    KaiwuTaskMode,
    KaiwuTaskReceipt,
    KaiwuTaskResult,
)
from .jobs import (
    KaiwuRemoteJob,
    new_receipt,
    restore_kaiwu_job,
    submit_kaiwu_task,
)
from .sdk import (
    KaiwuLicenseInitializationError,
    KaiwuSDKEnvironment,
    KaiwuSDKError,
    KaiwuSDKUnavailableError,
    KaiwuSDKVersionError,
    initialize_kaiwu_license,
)

__all__ = (
    "KaiwuCredentials",
    "KaiwuJobStatus",
    "KaiwuLicenseInitializationError",
    "KaiwuRemoteJob",
    "KaiwuSDKClient",
    "KaiwuSDKEnvironment",
    "KaiwuSDKError",
    "KaiwuSDKUnavailableError",
    "KaiwuSDKVersionError",
    "KaiwuTaskClient",
    "KaiwuTaskMode",
    "KaiwuTaskReceipt",
    "KaiwuTaskResult",
    "new_receipt",
    "initialize_kaiwu_license",
    "resolve_kaiwu_credentials",
    "restore_kaiwu_job",
    "submit_kaiwu_task",
)

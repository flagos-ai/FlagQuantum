"""Remote control-plane support for QBoson Kaiwu services.

The module is intentionally not re-exported from :mod:`flagquantum.remote`
while its provider contract is under review.
"""

from ._credentials import KaiwuCredentials, resolve_kaiwu_credentials
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

__all__ = (
    "KaiwuCredentials",
    "KaiwuJobStatus",
    "KaiwuRemoteJob",
    "KaiwuTaskClient",
    "KaiwuTaskMode",
    "KaiwuTaskReceipt",
    "KaiwuTaskResult",
    "new_receipt",
    "resolve_kaiwu_credentials",
    "restore_kaiwu_job",
    "submit_kaiwu_task",
)

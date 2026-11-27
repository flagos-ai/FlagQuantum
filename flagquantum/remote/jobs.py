"""Historical import facade for the relocated implementation."""

from flagquantum.compute.remote.jobs import (
    JobStatus as JobStatus,
)
from flagquantum.compute.remote.jobs import (
    MultiTargetSubmission as MultiTargetSubmission,
)
from flagquantum.compute.remote.jobs import (
    RemoteJob as RemoteJob,
)
from flagquantum.compute.remote.jobs import (
    TargetSubmission as TargetSubmission,
)
from flagquantum.compute.remote.jobs import __all__ as __all__
from flagquantum.compute.remote.jobs import (
    restore_job as restore_job,
)
from flagquantum.compute.remote.jobs import (
    submit as submit,
)
from flagquantum.compute.remote.jobs import (
    submit_to_targets as submit_to_targets,
)

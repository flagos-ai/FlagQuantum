"""Frozen version admission from the v2 reader at fedb6f2f754555eaa063e6dbc1dce02b0f4f27d0.

Only the version gate is retained. It precedes workload validation and every
restore mutation; this fixture does not implement a historical state loader.
"""

from collections.abc import Mapping
from typing import Any

from flagquantum.runtime.checkpointing import TrainingStateError

_LEGACY_TRAINING_STATE_VERSION = "flagquantum.training_state.v1"
TRAINING_STATE_VERSION = "flagquantum.training_state.v2"


def admit_version(payload: Mapping[str, Any]) -> None:
    version = payload.get("version")
    if version not in (_LEGACY_TRAINING_STATE_VERSION, TRAINING_STATE_VERSION):
        raise TrainingStateError(
            f"unsupported training state {payload.get('version')!r}"
        )

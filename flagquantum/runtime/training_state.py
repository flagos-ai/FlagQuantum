"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.runtime.checkpointing import (
    _CAPTURED_TRAINING_STATE_VERSION as _CAPTURED_TRAINING_STATE_VERSION,
)
from flagquantum.runtime.checkpointing import (
    _LEGACY_TRAINING_STATE_VERSION as _LEGACY_TRAINING_STATE_VERSION,
)
from flagquantum.runtime.checkpointing import (
    TRAINING_RESTORE_SCHEMA as TRAINING_RESTORE_SCHEMA,
)
from flagquantum.runtime.checkpointing import (
    TRAINING_RESTORE_VERSION as TRAINING_RESTORE_VERSION,
)
from flagquantum.runtime.checkpointing import (
    TRAINING_STATE_VERSION as TRAINING_STATE_VERSION,
)
from flagquantum.runtime.checkpointing import (
    NonFiniteTrainingError as NonFiniteTrainingError,
)
from flagquantum.runtime.checkpointing import (
    PrecisionPolicy as PrecisionPolicy,
)
from flagquantum.runtime.checkpointing import (
    PrecisionPolicyError as PrecisionPolicyError,
)
from flagquantum.runtime.checkpointing import (
    SeedContract as SeedContract,
)
from flagquantum.runtime.checkpointing import (
    TopologyMismatchError as TopologyMismatchError,
)
from flagquantum.runtime.checkpointing import (
    TrainingCheckpointRestore as TrainingCheckpointRestore,
)
from flagquantum.runtime.checkpointing import (
    TrainingStateError as TrainingStateError,
)
from flagquantum.runtime.checkpointing import __all__ as __all__
from flagquantum.runtime.checkpointing import (
    _captured_workload_signature as _captured_workload_signature,
)
from flagquantum.runtime.checkpointing import (
    _current_workload_signature as _current_workload_signature,
)
from flagquantum.runtime.checkpointing import (
    _has_custom_checkpoint_methods as _has_custom_checkpoint_methods,
)
from flagquantum.runtime.checkpointing import (
    _load_training_checkpoint_payload as _load_training_checkpoint_payload,
)
from flagquantum.runtime.checkpointing import (
    _optimizer_contract as _optimizer_contract,
)
from flagquantum.runtime.checkpointing import (
    _parameter_training as _parameter_training,
)
from flagquantum.runtime.checkpointing import (
    _quantum_owner as _quantum_owner,
)
from flagquantum.runtime.checkpointing import (
    _restore_rng as _restore_rng,
)
from flagquantum.runtime.checkpointing import (
    _restore_training_checkpoint as _restore_training_checkpoint,
)
from flagquantum.runtime.checkpointing import (
    _rng_state as _rng_state,
)
from flagquantum.runtime.checkpointing import (
    _topology as _topology,
)
from flagquantum.runtime.checkpointing import (
    _validate_checkpoint_target as _validate_checkpoint_target,
)
from flagquantum.runtime.checkpointing import (
    _validate_parameter_training as _validate_parameter_training,
)
from flagquantum.runtime.checkpointing import (
    _validate_training_checkpoint as _validate_training_checkpoint,
)
from flagquantum.runtime.checkpointing import (
    _workload_signature as _workload_signature,
)
from flagquantum.runtime.checkpointing import (
    assert_finite_training as assert_finite_training,
)
from flagquantum.runtime.checkpointing import (
    load_training_checkpoint as load_training_checkpoint,
)
from flagquantum.runtime.checkpointing import (
    save_training_checkpoint as save_training_checkpoint,
)
from flagquantum.runtime.checkpointing import (
    seed_everything as seed_everything,
)

_implementation = _import_module("flagquantum.runtime.checkpointing")
_sys.modules[__name__] = _implementation

import pytest

from flagquantum.runtime.backends.mps import checkpointing, training_engine

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "name",
    (
        "_checkpoint_path",
        "_checkpoint_manifest_path",
        "_checkpoint_checksum_path",
        "_commit_checkpoint_generation",
        "_preflight_checkpoint_generation",
        "_checkpoint_start_policy_error",
        "_validate_checkpoint_start_policy_collective",
        "_checkpoint_writer_lease_path",
        "_acquire_checkpoint_writer_lease",
        "_refresh_checkpoint_writer_lease",
        "_release_checkpoint_writer_lease",
        "_prune_checkpoint_generations",
        "_prune_checkpoint_generations_collective",
        "_checkpoint_capacity_error",
        "_checkpoint_storage_preflight",
        "_validate_shared_checkpoint_root",
        "_save_checkpoint",
        "_load_checkpoint",
    ),
)
def test_training_engine_checkpoint_names_preserve_function_identity(name):
    assert getattr(training_engine, name) is getattr(checkpointing, name)

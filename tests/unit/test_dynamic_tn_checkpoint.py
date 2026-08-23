import hashlib
import json

import pytest

from flagquantum.runtime.backends.tensor_network import (
    clear_dynamic_tn_checkpoint_writer_lock,
    inspect_dynamic_tn_checkpoint,
)


def test_checkpoint_writer_lock_requires_audited_identity(tmp_path):
    lock = tmp_path / ".checkpoint.lock"
    payload = {
        "pid": 31415,
        "segment_identity": "segment-a",
        "next_record_index": 7,
    }
    encoded = json.dumps(payload, sort_keys=True).encode()
    lock.write_bytes(encoded)

    audit = inspect_dynamic_tn_checkpoint(tmp_path)

    assert audit.writer_lock_exists is True
    assert audit.writer_lock_identity == hashlib.sha256(encoded).hexdigest()
    assert audit.writer_pid == 31415
    assert audit.writer_segment_identity == "segment-a"
    assert audit.writer_next_record_index == 7
    assert audit.loadable is False
    with pytest.raises(RuntimeError, match="identity changed"):
        clear_dynamic_tn_checkpoint_writer_lock(
            tmp_path,
            expected_lock_identity="0" * 64,
        )
    assert lock.exists()

    clear_dynamic_tn_checkpoint_writer_lock(
        tmp_path,
        expected_lock_identity=audit.writer_lock_identity,
    )

    assert not lock.exists()
    assert inspect_dynamic_tn_checkpoint(tmp_path).writer_lock_exists is False

import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from flagquantum.runtime.executors.tensor_network import dynamic_checkpoint
from flagquantum.runtime.executors.tensor_network.dynamic_checkpoint import (
    clear_dynamic_tn_checkpoint_writer_lock,
    inspect_dynamic_tn_checkpoint,
)
from flagquantum.runtime.executors.tensor_network.dynamic_reverse import (
    DistributedTNDynamicReverseSegment,
)


@pytest.mark.parametrize("manifest", [None, [], "invalid"])
def test_checkpoint_rejects_non_object_manifest_collectively(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, manifest: object
) -> None:
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "COMMITTED").write_text("identity", encoding="utf-8")
    monkeypatch.setattr(dynamic_checkpoint.dist, "is_initialized", lambda: True)
    invalid_flags: list[bool] = []

    def collective_invalid(invalid: bool, device: torch.device) -> bool:
        invalid_flags.append(invalid)
        return invalid

    monkeypatch.setattr(dynamic_checkpoint, "_collective_invalid", collective_invalid)
    with pytest.raises(RuntimeError, match="not durably committed"):
        dynamic_checkpoint.load_dynamic_tn_reverse_checkpoint(
            Mock(spec=DistributedTNDynamicReverseSegment),
            tmp_path,
            device=torch.device("cpu"),
        )
    assert invalid_flags == [True]


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

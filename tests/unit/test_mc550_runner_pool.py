"""Physical CPU assignments leave reserved business cores on both sockets."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def test_mc550_slots_do_not_overlap_or_consume_reserved_business_cores() -> None:
    spec = importlib.util.spec_from_file_location(
        "mc550_provision", ROOT / "tools/ci/provision_mc550_runners.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reserved = set(range(8)) | set(range(40, 48))
    reserved |= {cpu + 80 for cpu in reserved}
    used: set[int] = set()
    for slot in range(1, 17):
        cpus = module.cpus_for_slot(slot)
        assert len(set(cpus)) == 8
        assert cpus[4:] == [cpu + 80 for cpu in cpus[:4]]
        assert len({cpu // 40 for cpu in cpus[:4]}) == 1
        assert not used.intersection(cpus)
        assert not reserved.intersection(cpus)
        used.update(cpus)
    assert used | reserved == set(range(160))
    for invalid in (0, 17):
        with pytest.raises(ValueError):
            module.cpus_for_slot(invalid)


def test_mc550_validation_targets_every_exact_slot_only_when_requested() -> None:
    document = yaml.safe_load((ROOT / ".github/workflows/local-gpu.yml").read_text())
    job = document["jobs"]["validate-mc550-pool"]
    assert job["if"] == "${{ inputs.validate_mc550_pool }}"
    assert job["strategy"]["matrix"] == {
        "host": ["mc550-111", "mc550-112", "mc550-121", "mc550-122"],
        "slot": [f"cpu{index:02d}" for index in range(1, 17)],
    }
    assert "${{ matrix.host }}" in job["runs-on"]
    assert "${{ matrix.slot }}" in job["runs-on"]
    assert job["needs"] == "prepare-source"
    assert job["steps"][0]["with"]["digest-mismatch"] == "error"
    restore = job["steps"][1]["run"]
    assert 'test "$sha" = "$GITHUB_SHA"' in restore
    assert 'test "$(git rev-parse HEAD)" = "$GITHUB_SHA"' in restore
    events = document.get("on", document.get(True))
    assert "pull_request_target" not in events
    assert (
        events["workflow_dispatch"]["inputs"]["validate_mc550_pool"]["default"] is False
    )


def test_cached_archives_are_verified_before_the_installer_uses_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = importlib.util.spec_from_file_location(
        "mc550_download", ROOT / "tools/ci/provision_mc550_runners.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module.os, "chown", lambda *_: None)
    directory = tmp_path / "packages"
    directory.mkdir()
    target = directory / "runtime.tar.gz"
    target.write_bytes(b"verified public archive")
    row = {
        "filename": target.name,
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }
    assert module.download(row) == target
    target.write_bytes(b"corrupted relay or cache content")
    with pytest.raises(RuntimeError, match="Official SHA256 mismatch"):
        module.download(row)

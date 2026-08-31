from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[2] / "paper" / "sc27" / "seal_run.py"
SPEC = importlib.util.spec_from_file_location("sc27_seal_run", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_seal_binds_payload_to_log(tmp_path: Path) -> None:
    raw_log = tmp_path / "run.log"
    raw_log.write_text("rank0 complete\n", encoding="utf-8")
    payload = {"source_identity": {"raw_log_sha256": None}}
    sealed = MODULE.seal(payload, raw_log=raw_log)
    assert sealed["source_identity"]["raw_log_sha256"] == MODULE.file_sha256(raw_log)
    assert sealed["source_identity"]["raw_log_path_hint"] == "run.log"


def test_seal_rejects_conflicting_existing_digest(tmp_path: Path) -> None:
    raw_log = tmp_path / "run.log"
    raw_log.write_text("rank0 complete\n", encoding="utf-8")
    with pytest.raises(ValueError, match="conflicts"):
        MODULE.seal(
            {"source_identity": {"raw_log_sha256": "0" * 64}},
            raw_log=raw_log,
        )

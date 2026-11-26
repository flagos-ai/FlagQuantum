"""Shared receipt and result files for submitted Jiuding jobs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from flagquantum._import_compat import alias_module as _alias_module


def save_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_job_result(path: Path, run_id: str) -> object:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise RuntimeError("Jiuding job result must be a JSON object")
    if result.get("run_id") != run_id:
        raise RuntimeError("Result does not belong to this submission")
    if "value" not in result:
        raise RuntimeError("Jiuding job result is missing its value")
    return result["value"]


# Preserve historical import identities for pickles and public introspection.
for _identity_export in (
    save_receipt,
    read_job_result,
):
    _identity_export.__module__ = "flagquantum.remote.compute._job_results"
del _identity_export

_alias_module("flagquantum.remote.compute._job_results", __name__)

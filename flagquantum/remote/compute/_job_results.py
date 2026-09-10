"""Shared receipt and result files for submitted Jiuding jobs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


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

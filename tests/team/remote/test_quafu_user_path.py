"""The Quafu golden path uses only the public execution API."""

from __future__ import annotations

import runpy
from pathlib import Path

import flagquantum as fq


def test_quafu_bell_example_uses_complete_public_path(monkeypatch, capsys) -> None:
    captured = {}

    class Result:
        provenance = {"task_id": "task-17"}
        counts = [{"00": 513, "11": 511}]

    def run(program, **options):
        captured.update(program=program, options=options)
        return Result()

    monkeypatch.setenv("QUAFU_API_TOKEN", "test-token")
    monkeypatch.setattr(fq, "run", run)

    root = Path(__file__).resolve().parents[3]
    runpy.run_path(root / "examples" / "remote" / "quafu_bell.py", run_name="__main__")

    assert captured["program"].n_qubits == 2
    assert captured["options"] == {
        "compiler": "qsteed",
        "target": "quafu:ScQ-P10",
        "shots": 1024,
        "name": "flagquantum bell",
    }
    assert capsys.readouterr().out == (
        "task: task-17\ncounts: {'00': 513, '11': 511}\n"
    )

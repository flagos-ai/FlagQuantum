import json

import pytest

from tools.hardware_run_manifest import main

pytestmark = pytest.mark.unit


def test_hardware_manifest_records_command_rank_logs_and_skip_reason(
    tmp_path, monkeypatch
):
    log = tmp_path / "run.log"
    log.write_text("passed\n", encoding="utf-8")
    output = tmp_path / "run.json"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("WORLD_SIZE", "2")

    assert (
        main(
            [
                "--output",
                str(output),
                "--command",
                "torchrun test.py",
                "--log",
                str(log),
                "--world-size",
                "2",
                "--evidence-scope",
                "two_gpu_semantic_regression",
                "--skip-reason",
                "maintenance",
            ]
        )
        == 0
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == "flagquantum_hardware_run_v1"
    assert payload["artifact_class"] == "development_run"
    assert payload["evidence_scope"] == "two_gpu_semantic_regression"
    assert payload["command"] == "torchrun test.py"
    assert payload["rank_placement"]["WORLD_SIZE"] == "2"
    assert payload["rank_placement"]["expected_world_size"] == 2
    assert payload["rank_placement"]["visible_device_mapping"] == [
        {"local_rank": 0, "visible_device": "0"},
        {"local_rank": 1, "visible_device": "1"},
    ]
    assert payload["skip_reason"] == "maintenance"
    assert payload["logs"][0]["sha256"]

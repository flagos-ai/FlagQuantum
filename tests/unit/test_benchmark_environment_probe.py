from benchmarks.runners.environment_probe import build_payload


def test_environment_probe_has_runner_schema(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
    payload = build_payload()
    assert payload["runner"] == "environment_probe"
    assert payload["schema"] == "flagquantum.benchmark.environment.v1"
    assert payload["world_size"] == 8
    assert payload["local_world_size"] == 4

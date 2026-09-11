import json

from flagquantum.benchmarking.statevector_weak_scaling import main


def test_weak_scaling_adapter_emits_runner_contract(tmp_path, monkeypatch):
    def artifact(world, wires):
        return {
            "world_size": world,
            "schema_version": "flagquantum.statevector.strong_scaling.v1",
            "artifact_class": "measured_development_run",
            "backend": "nccl",
            "correctness": {"passed": True},
            "workload": {
                "n_wires": wires,
                "depth": 2,
                "gate_count": 4,
                "dtype": "complex64",
            },
            "rank_timings": [{"local_state_bytes": 64}] * world,
            "rank_peak_memory_bytes": [128] * world,
            "timing": {
                "samples_seconds": [1.0, 1.1, 0.9],
                "coefficient_of_variation": 0.1,
            },
            "node_count": 1,
            "communication_tiers": {"route_classification": "intra_node_collective"},
        }

    first = tmp_path / "one.json"
    second = tmp_path / "two.json"
    first.write_text(json.dumps(artifact(1, 4)))
    second.write_text(json.dumps(artifact(2, 5)))
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        "sys.argv",
        ["runner", str(first), str(second), "--json-output", str(output)],
    )
    assert main() == 0
    payload = json.loads(output.read_text())
    assert payload["runner"] == "statevector_weak_scaling"
    assert payload["report"]["schema_version"].endswith("weak_scaling_report.v1")

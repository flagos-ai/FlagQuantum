from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).parents[2]


def load(name: str, filename: str):
    path = ROOT / "paper" / "sc27" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MANIFEST = load("sc27_trajectory_manifest", "build_correctness_trajectory_manifest.py")
CATALOG = load("sc27_trajectory_catalog", "build_correctness_trajectory_catalog.py")
MATRIX = load("sc27_correctness_matrix", "build_correctness_matrix_manifest.py")


def test_manifest_freezes_all_72_single_gpu_references() -> None:
    result = MANIFEST.build_manifest(
        source_commit="a" * 40,
        core_container_digest="sha256:" + "b" * 64,
        output_root="/shared/sc27/correctness",
    )
    assert result["run_count"] == 72
    assert len({run["workload_key"] for run in result["runs"]}) == 72
    for run in result["runs"]:
        command = run["command"]
        assert command[:3] == ["torchrun", "--standalone", "--nproc-per-node=1"]
        assert command[command.index("--warmup") + 1] == "0"
        assert command[command.index("--repetitions") + 1] == "10"


def test_catalog_accepts_exact_trajectory_matrix(tmp_path: Path) -> None:
    paths = []
    for n_wires in (12, 20, 28):
        for depth in (2, 8, 16):
            for dtype in ("complex64", "complex128"):
                for topology, name in (
                    ("nearest_neighbor_linear", "full_width_linear_hea"),
                    ("ring_boundary_stress", "full_width_ring_hea"),
                ):
                    for optimizer in ("sgd", "adam"):
                        count = n_wires * depth
                        trajectory = [
                            {
                                "step": step,
                                "parameters_before": [0.1] * count,
                                "value": 0.2,
                                "gradients": [0.0] * count,
                                "parameters_after": [0.1] * count,
                            }
                            for step in range(10)
                        ]
                        payload = {
                            "world_size": 1,
                            "node_count": 1,
                            "rank_placement": [
                                {
                                    "rank": 0,
                                    "gpu_uuid": "GPU-test",
                                    "pci_bus_id": "0000:00:00.0",
                                    "identity_complete": True,
                                    "topology": {
                                        "captured": True,
                                        "nvidia_smi_topology_sha256": "d" * 64,
                                    },
                                }
                            ],
                            "workload": {
                                "name": name,
                                "n_wires": n_wires,
                                "layers": depth,
                                "dtype": dtype,
                                "seed": 41,
                                "optimizer": optimizer,
                                "learning_rate": 0.01,
                            },
                            "protocol": {"warmup": 0, "repetitions": 10},
                            "correctness": {"passed": True, "trajectory": trajectory},
                            "fallback_events": [],
                            "source_identity": {
                                "commit": "a" * 40,
                                "container_digest": "sha256:" + "b" * 64,
                                "source_dirty": False,
                                "raw_log_sha256": "c" * 64,
                            },
                        }
                        key = MANIFEST.key(n_wires, depth, dtype, topology, optimizer)
                        path = tmp_path / f"{key}.json"
                        path.write_text(json.dumps(payload))
                        paths.append(path)
    result = CATALOG.build(paths)
    assert len(result["references"]) == 72


def test_distributed_manifest_binds_216_replays_to_references() -> None:
    references = {
        MANIFEST.key(n, depth, dtype, topology, optimizer): {
            "path": f"/refs/{MANIFEST.key(n, depth, dtype, topology, optimizer)}.json",
            "sha256": f"{index + 1:064x}",
        }
        for index, (n, depth, dtype, topology, optimizer) in enumerate(
            (n, depth, dtype, topology, optimizer)
            for n in (12, 20, 28)
            for depth in (2, 8, 16)
            for dtype in ("complex64", "complex128")
            for topology in ("nearest_neighbor_linear", "ring_boundary_stress")
            for optimizer in ("sgd", "adam")
        )
    }
    catalog = {
        "schema": "flagquantum.sc27.correctness_trajectory_catalog.v1",
        "source_commit": "a" * 40,
        "container_digest": "sha256:" + "b" * 64,
        "reference_world_size": 1,
        "trajectory_steps": 10,
        "references": references,
    }
    result = MATRIX.build_manifest(
        source_commit="a" * 40,
        core_container_digest="sha256:" + "b" * 64,
        trajectory_catalog=catalog,
        output_root="/shared/sc27/correctness-distributed",
    )
    assert result["run_count"] == 216
    assert {run["world_size"] for run in result["runs"]} == {2, 4, 8}
    for run in result["runs"]:
        reference_path = run["command"][run["command"].index("--reference-json") + 1]
        assert reference_path == run["reference"]["path"]

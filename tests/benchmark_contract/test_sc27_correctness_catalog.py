from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[2] / "paper" / "sc27" / "build_correctness_catalog.py"
SPEC = importlib.util.spec_from_file_location("sc27_correctness_catalog", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def payload(n_wires: int, seed: int) -> dict:
    world = 1 << (n_wires - 28)
    placement = [
        {
            "rank": rank,
            "hostname": "node0" if rank < 8 else "node1",
            "gpu_uuid": f"GPU-{n_wires}-{seed}-{rank}",
            "pci_bus_id": f"0000:{rank:02x}:00.0",
            "identity_complete": True,
            "topology": {
                "captured": True,
                "nvidia_smi_topology_sha256": "d" * 64,
            },
        }
        for rank in range(world)
    ]
    return {
        "benchmark": "pennylane_lightning_gpu_adjoint_training",
        "world_size": world,
        "node_count": 2 if n_wires == 32 else 1,
        "rank_placement": placement,
        "distribution_semantics": (
            "single_device_fast_path"
            if n_wires == 28
            else "sharded_across_mpi_ranks"
        ),
        "workload": {
            "n_wires": n_wires,
            "name": "full_width_linear_hea",
            "layers": 8,
            "parameter_count": 8 * n_wires,
            "observable": f"Z({n_wires // 2})",
            "dtype": "complex64",
            "seed": seed,
        },
        "protocol": {
            "gradient_method": "adjoint",
            "optimizer": "none",
            "optimizer_included": False,
            "independent_run_index": {41: 1, 42: 2, 43: 3}[seed],
        },
        "correctness": {
            "initial_value": 0.25,
            "initial_gradients": [0.0] * (8 * n_wires),
            "values_and_gradients_finite": True,
        },
        "source_identity": {
            "commit": "a" * 40,
            "container_digest": "sha256:" + "b" * 64,
            "source_dirty": False,
            "raw_log_sha256": "c" * 64,
        },
    }


def write_matrix(tmp_path: Path) -> list[Path]:
    paths = []
    for n_wires in range(28, 33):
        for seed in (41, 42, 43):
            path = tmp_path / f"n{n_wires}-seed{seed}.json"
            path.write_text(json.dumps(payload(n_wires, seed)))
            paths.append(path)
    return paths


def test_builds_complete_workload_keyed_catalog(tmp_path: Path) -> None:
    result = MODULE.build(write_matrix(tmp_path))
    assert len(result["references"]) == 15
    assert set(result["references"]) >= {"n28-seed41", "n32-seed43"}


def test_rejects_missing_reference(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="catalog:missing:n32-seed43"):
        MODULE.build(write_matrix(tmp_path)[:-1])

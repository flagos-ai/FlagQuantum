from __future__ import annotations

import importlib.util
from pathlib import Path

MODULE_PATH = (
    Path(__file__).parents[2]
    / "paper"
    / "sc27"
    / "build_correctness_reference_manifest.py"
)
SPEC = importlib.util.spec_from_file_location("sc27_reference_manifest", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def manifest() -> dict:
    return MODULE.build_manifest(
        source_commit="a" * 40,
        pennylane_container_digest="sha256:" + "b" * 64,
        output_root="/shared/sc27/references",
    )


def test_compiles_complete_reference_matrix() -> None:
    result = manifest()
    assert result["run_count"] == 15
    assert len({run["workload_key"] for run in result["runs"]}) == 15
    assert len(result["catalog_command"]) == 19


def test_freezes_weak_scaling_placement_and_reference_semantics() -> None:
    by_key = {run["workload_key"]: run for run in manifest()["runs"]}
    assert by_key["n28-seed41"]["command"][:3] == ["mpirun", "-np", "1"]
    largest = by_key["n32-seed43"]
    assert largest["resources"] == {
        "nodes": 2,
        "gpus_per_node": 8,
        "launch_mode": "mpi_all_nodes",
    }
    assert "{hostfile}" in largest["command"]
    assert largest["command"][largest["command"].index("--optimizer") + 1] == "none"
    for run in by_key.values():
        command = run["command"]
        assert command[command.index("--warmup") + 1] == "5"
        assert command[command.index("--repetitions") + 1] == "20"
        assert run["depends_on"] == ["pennylane-custatevec-mpi-preflight"]


def test_requires_two_gpu_cuda_aware_mpi_preflight() -> None:
    preflight = manifest()["preflight"]
    assert preflight["resources"] == {
        "nodes": 1,
        "gpus_per_node": 2,
        "launch_mode": "mpi_all_nodes",
    }
    assert preflight["command"][:3] == ["mpirun", "-np", "2"]
    assert preflight["command"][preflight["command"].index("--repetitions") + 1] == "3"
    assert preflight["command"][preflight["command"].index("--mpi-buf-size") + 1] == "1"
    assert preflight["purpose"] == (
        "fail_closed_custatevec_cuda_aware_mpi_communicator_probe"
    )

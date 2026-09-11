from tools.check_mps_single_node_certification_environment import (
    build_payload,
    certification_commands,
    readiness_errors,
)


def test_mps_certification_preflight_accepts_complete_environment() -> None:
    names = ("NVIDIA A800-SXM4-80GB",) * 8
    payload = build_payload(
        cuda_available=True,
        nccl_available=True,
        device_names=names,
        source_tree_dirty=False,
        signing_key_present=True,
        source_commit="a" * 40,
    )

    assert payload["ready"] is True
    assert payload["blockers"] == ()
    assert payload["required_world_sizes"] == (1, 2, 4, 8)
    assert payload["scalability_claim_allowed"] is False
    assert payload["release_gate_allowed"] is False


def test_mps_certification_preflight_reports_every_environment_blocker() -> None:
    errors = readiness_errors(
        cuda_available=False,
        nccl_available=False,
        device_names=("GPU-A", "GPU-B"),
        source_tree_dirty=True,
        signing_key_present=False,
    )

    assert errors == (
        "CUDA runtime is unavailable",
        "PyTorch NCCL backend is unavailable",
        "MPS certification requires 8 visible GPUs; detected 2",
        "the first 8 visible GPUs must use one accelerator model",
        "MPS certification requires a clean source tree",
        "FQ_EVIDENCE_SIGNING_KEY is not configured",
    )


def test_mps_certification_matrix_is_frozen_to_one_two_four_eight_gpus() -> None:
    commands = certification_commands(output_root="candidate")

    assert len(commands) == 4
    for world_size, command in zip((1, 2, 4, 8), commands, strict=True):
        assert f"--nproc-per-node={world_size}" in command
        assert f"candidate/{world_size}gpu-adam.json" in command
        assert "--gradient-policy exact" in command

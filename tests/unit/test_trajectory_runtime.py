import pytest
import torch

import flagquantum as fq
import flagquantum.backends as fqb
import flagquantum.noise as fqn
from flagquantum.runtime.trajectories import (
    TensorWelford,
    TrajectoryCheckpoint,
    TrajectoryFailure,
    derive_trajectory_seed,
    load_trajectory_checkpoint,
    merge_trajectory_checkpoints,
    owned_trajectory_ids,
    save_trajectory_checkpoint,
)

pytestmark = pytest.mark.unit


def test_noisy_mps_wrapper_delegates_runtime_lifecycle(monkeypatch):
    import flagquantum.runtime.trajectories.mps as mps_runtime

    expected = object()
    calls = []

    def replacement(circuit_or_ir, noise_model, **options):
        calls.append((circuit_or_ir, noise_model, options))
        return expected

    monkeypatch.setattr(mps_runtime, "run_noisy_mps_runtime", replacement)
    circuit = fq.Circuit(1).x(0)
    noise_model = fqn.NoiseModel().add("x", fq.bit_flip_channel(0.25))

    assert fqb.run_noisy_mps(circuit, noise_model, trajectories=3, seed=7) is expected
    assert calls[0][0] is circuit
    assert calls[0][1] is noise_model
    assert calls[0][2]["trajectories"] == 3
    assert calls[0][2]["seed"] == 7
    assert callable(calls[0][2]["trajectory_executor"])
    assert calls[0][2]["result_factory"] is fq.MPSMonteCarloResult


def test_run_native_uses_lowered_mps_entrypoints(monkeypatch):
    import flagquantum.simulation.mps_execution as mps_execution

    def legacy_entrypoint(*args, **kwargs):
        raise AssertionError("run_native must not re-enter legacy noise lowering")

    monkeypatch.setattr(mps_execution, "run_noisy_mps_trajectory", legacy_entrypoint)
    monkeypatch.setattr(mps_execution, "run_noisy_mps", legacy_entrypoint)
    circuit = fq.Circuit(1).x(0)
    noise_model = fqn.NoiseModel().add("x", fq.bit_flip_channel(1.0))

    single = fqb.run_native(circuit, noise_model=noise_model, mode="mps_trajectory")
    sampled = fqb.run_native(
        circuit,
        noise_model=noise_model,
        mode="noisy_mps",
        trajectories=2,
        seed=5,
        retain_trajectories=False,
    )

    assert torch.allclose(single.expectation_z(0), torch.ones(1, 1))
    assert sampled.n_trajectories == 2


@pytest.mark.parametrize(
    ("mode", "options"),
    (
        ("mps_trajectory", {}),
        ("noisy_mps", {"trajectories": 2, "seed": 5}),
    ),
)
def test_planned_mps_noise_execution_preserves_lowering_and_plan_identity(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    options: dict[str, int],
):
    circuit = fq.Circuit(1).x(0)
    noise_model = fqn.NoiseModel().add("x", fq.bit_flip_channel(1.0))
    expected, plan = fqb.run_native(
        circuit,
        noise_model=noise_model,
        mode=mode,
        return_plan=True,
        dtype=torch.complex64,
        **options,
    )
    lowered = fq.lower_noise_model(circuit, noise_model)

    def forbidden(*args, **kwargs):
        raise AssertionError("planned MPS noise execution must not lower noise again")

    monkeypatch.setattr("flagquantum.runtime.execution.lower_noise_model", forbidden)
    actual, returned_plan = fqb.run_native(
        lowered,
        noise_model=noise_model,
        mode=mode,
        return_plan=True,
        _execution_plan=plan,
        dtype=torch.complex64,
        **options,
    )

    assert returned_plan is plan
    if mode == "mps_trajectory":
        torch.testing.assert_close(actual.expectation_z(0), expected.expectation_z(0))
    else:
        torch.testing.assert_close(
            actual.expectation_z_mean,
            expected.expectation_z_mean,
        )
        assert actual.n_trajectories == expected.n_trajectories


def test_trajectory_ownership_preserves_global_ids_across_world_sizes():
    expected = tuple(range(17))

    for world_size in (1, 2, 4, 8):
        owned = tuple(
            sorted(
                trajectory_id
                for rank in range(world_size)
                for trajectory_id in owned_trajectory_ids(
                    len(expected),
                    rank=rank,
                    world_size=world_size,
                )
            )
        )
        assert owned == expected


def test_trajectory_seeds_depend_only_on_base_seed_and_global_id():
    expected = tuple(derive_trajectory_seed(23, item) for item in range(17))

    for world_size in (1, 3, 8):
        observed = {
            trajectory_id: derive_trajectory_seed(23, trajectory_id)
            for rank in range(world_size)
            for trajectory_id in owned_trajectory_ids(
                len(expected),
                rank=rank,
                world_size=world_size,
            )
        }
        assert tuple(observed[item] for item in range(17)) == expected
    assert len(set(expected)) == len(expected)


def test_tensor_welford_matches_population_statistics():
    values = tuple(torch.tensor([float(item), float(item * item)]) for item in range(5))
    accumulator = TensorWelford()
    for value in values:
        accumulator.update(value)

    statistics = accumulator.finalize()
    stacked = torch.stack(values)

    assert statistics.count == len(values)
    assert torch.allclose(statistics.mean, stacked.mean(dim=0))
    assert torch.allclose(
        statistics.variance,
        torch.var(stacked, dim=0, unbiased=False),
    )
    assert torch.allclose(
        statistics.standard_error,
        torch.sqrt(statistics.variance / len(values)),
    )


def test_seeded_noisy_mps_is_reproducible_by_global_trajectory_id():
    circuit = fq.Circuit(1).x(0)
    model = fqn.NoiseModel().add("x", fq.bit_flip_channel(0.37))

    first = fqb.run_noisy_mps(circuit, model, trajectories=12, seed=101)
    second = fqb.run_noisy_mps(circuit, model, trajectories=12, seed=101)

    assert first.trajectory_ids == tuple(range(12))
    assert first.trajectory_seeds == second.trajectory_seeds
    assert torch.equal(first.expectation_z_mean, second.expectation_z_mean)
    assert torch.equal(first.expectation_z_variance, second.expectation_z_variance)
    assert first.statistics is not None
    assert first.statistics.count == 12


def test_adaptive_noisy_mps_stops_at_minimum_for_deterministic_channel():
    circuit = fq.Circuit(1).x(0)
    model = fqn.NoiseModel().add("x", fq.bit_flip_channel(1.0))

    result = fqb.run_noisy_mps(
        circuit,
        model,
        trajectories=100,
        min_trajectories=7,
        target_standard_error=1e-4,
        seed=17,
        retain_trajectories=False,
    )

    assert result.n_trajectories == 7
    assert result.converged is True
    assert result.stopped_early is True
    assert result.summary()["target_standard_error"] == 1e-4


def test_adaptive_noisy_mps_reports_unmet_target_at_maximum():
    circuit = fq.Circuit(1).x(0)
    model = fqn.NoiseModel().add("x", fq.bit_flip_channel(0.5))

    result = fqb.run_noisy_mps(
        circuit,
        model,
        trajectories=8,
        min_trajectories=4,
        target_standard_error=1e-12,
        seed=29,
        retain_trajectories=False,
    )

    assert result.n_trajectories == 8
    assert result.converged is False
    assert result.stopped_early is False


def test_adaptive_noisy_mps_distributed_path_fails_closed():
    with pytest.raises(ValueError, match="collective statistics reduction"):
        fqb.run_noisy_mps(
            fq.Circuit(1).x(0),
            fqn.NoiseModel().add("x", fq.bit_flip_channel(0.5)),
            trajectories=8,
            min_trajectories=4,
            target_standard_error=0.1,
            seed=3,
            rank=0,
            world_size=2,
        )


def test_adaptive_checkpoint_resume_does_not_run_after_target_is_met(tmp_path):
    checkpoint = tmp_path / "adaptive.pt"
    options = {
        "trajectories": 50,
        "min_trajectories": 4,
        "target_standard_error": 1e-4,
        "seed": 41,
        "checkpoint_path": checkpoint,
        "retain_trajectories": False,
    }
    circuit = fq.Circuit(1).x(0)
    model = fqn.NoiseModel().add("x", fq.bit_flip_channel(1.0))

    first = fqb.run_noisy_mps(circuit, model, **options)
    resumed = fqb.run_noisy_mps(circuit, model, resume=True, **options)

    assert first.trajectory_ids == tuple(range(4))
    assert resumed.trajectory_ids == first.trajectory_ids
    assert resumed.n_trajectories == 4
    assert resumed.converged is True


def test_noisy_checkpoint_rejects_changed_noise_model(tmp_path):
    checkpoint = tmp_path / "model-bound.pt"
    circuit = fq.Circuit(1).x(0)
    fqb.run_noisy_mps(
        circuit,
        fqn.NoiseModel().add("x", fq.bit_flip_channel(0.2)),
        trajectories=4,
        seed=43,
        checkpoint_path=checkpoint,
        max_trajectories_per_run=2,
        retain_trajectories=False,
    )

    with pytest.raises(ValueError, match="noise model identity"):
        fqb.run_noisy_mps(
            circuit,
            fqn.NoiseModel().add("x", fq.bit_flip_channel(0.3)),
            trajectories=4,
            seed=43,
            checkpoint_path=checkpoint,
            resume=True,
            retain_trajectories=False,
        )


def test_run_native_exposes_adaptive_trajectory_controls_in_plan():
    result, plan = fqb.run_native(
        fq.Circuit(1).x(0),
        noise_model=fqn.NoiseModel().add("x", fq.bit_flip_channel(1.0)),
        mode="noisy_mps",
        trajectories=20,
        min_trajectories=5,
        target_standard_error=1e-3,
        seed=13,
        retain_trajectories=False,
        return_plan=True,
    )

    assert result.n_trajectories == 5
    assert plan.noisy_execution_plan.trajectory.min_count == 5
    assert plan.noisy_execution_plan.trajectory.target_standard_error == 1e-3
    assert plan.noisy_execution_plan.noise_model_identity is not None
    assert result.noise_model_identity == plan.noisy_execution_plan.noise_model_identity


def test_noisy_mps_rejects_ambiguous_random_stream_configuration():
    with pytest.raises(ValueError, match="either generator or seed"):
        fqb.run_noisy_mps(
            fq.Circuit(1).x(0),
            fqn.NoiseModel().add("x", fq.bit_flip_channel(0.5)),
            trajectories=2,
            generator=torch.Generator(),
            seed=3,
        )


def _checkpoint_for_values(
    *,
    requested_count: int,
    base_seed: int,
    trajectory_ids: tuple[int, ...],
    values: tuple[torch.Tensor, ...],
) -> TrajectoryCheckpoint:
    accumulator = TensorWelford()
    for value in values:
        accumulator.update(value)
    return TrajectoryCheckpoint(
        requested_count=requested_count,
        base_seed=base_seed,
        completed_ids=trajectory_ids,
        statistics_state=accumulator.state_dict(),
    )


def test_rank_local_checkpoints_merge_to_single_pass_statistics():
    values = tuple(torch.tensor([float(item), float(item * item)]) for item in range(8))
    even = _checkpoint_for_values(
        requested_count=8,
        base_seed=31,
        trajectory_ids=(0, 2, 4, 6),
        values=values[0::2],
    )
    odd = _checkpoint_for_values(
        requested_count=8,
        base_seed=31,
        trajectory_ids=(1, 3, 5, 7),
        values=values[1::2],
    )

    merged = merge_trajectory_checkpoints((even, odd))
    statistics = merged.accumulator().finalize()
    stacked = torch.stack(values)

    assert merged.completed_ids == tuple(range(8))
    assert merged.pending_ids() == ()
    assert torch.allclose(statistics.mean, stacked.mean(dim=0))
    assert torch.allclose(
        statistics.variance,
        torch.var(stacked, dim=0, unbiased=False),
    )


def test_checkpoint_pending_ids_respect_rank_and_failure_policy():
    accumulator = TensorWelford()
    accumulator.update(torch.tensor([1.0]))
    checkpoint = TrajectoryCheckpoint(
        requested_count=6,
        base_seed=17,
        completed_ids=(0,),
        failures=(
            TrajectoryFailure(2, "RuntimeError", "transient", retryable=True),
            TrajectoryFailure(4, "ValueError", "invalid", retryable=False),
        ),
        statistics_state=accumulator.state_dict(),
    )

    assert checkpoint.pending_ids(rank=0, world_size=2) == (2,)
    assert checkpoint.pending_ids(rank=0, world_size=2, retry_failed=False) == ()
    assert checkpoint.pending_ids(rank=1, world_size=2) == (1, 3, 5)


def test_trajectory_checkpoint_round_trip(tmp_path):
    checkpoint = _checkpoint_for_values(
        requested_count=4,
        base_seed=99,
        trajectory_ids=(0, 1),
        values=(torch.tensor([1.0]), torch.tensor([-1.0])),
    )
    path = save_trajectory_checkpoint(checkpoint, tmp_path / "trajectory.pt")

    restored = load_trajectory_checkpoint(path)

    assert restored.requested_count == checkpoint.requested_count
    assert restored.base_seed == checkpoint.base_seed
    assert restored.completed_ids == checkpoint.completed_ids
    assert restored.pending_ids() == (2, 3)
    assert torch.equal(
        restored.accumulator().finalize().mean,
        checkpoint.accumulator().finalize().mean,
    )


def test_checkpoint_rejects_statistics_count_mismatch():
    accumulator = TensorWelford()
    accumulator.update(torch.tensor([1.0]))

    with pytest.raises(ValueError, match="statistics count"):
        TrajectoryCheckpoint(
            requested_count=3,
            base_seed=5,
            completed_ids=(0, 1),
            statistics_state=accumulator.state_dict(),
        )


def test_noisy_mps_checkpoint_resume_matches_single_pass(tmp_path):
    circuit = fq.Circuit(1).x(0)
    model = fqn.NoiseModel().add("x", fq.bit_flip_channel(0.37))
    checkpoint_path = tmp_path / "mps-trajectories.pt"

    reference = fqb.run_noisy_mps(
        circuit,
        model,
        trajectories=8,
        seed=73,
        retain_trajectories=False,
    )
    first = fqb.run_noisy_mps(
        circuit,
        model,
        trajectories=8,
        seed=73,
        checkpoint_path=checkpoint_path,
        max_trajectories_per_run=3,
        retain_trajectories=False,
    )
    second = fqb.run_noisy_mps(
        circuit,
        model,
        trajectories=8,
        seed=73,
        checkpoint_path=checkpoint_path,
        resume=True,
        max_trajectories_per_run=2,
        retain_trajectories=False,
    )
    final = fqb.run_noisy_mps(
        circuit,
        model,
        trajectories=8,
        seed=73,
        checkpoint_path=checkpoint_path,
        resume=True,
        retain_trajectories=False,
    )

    assert first.n_trajectories == 3
    assert second.n_trajectories == 5
    assert final.n_trajectories == 8
    assert final.requested_trajectories == 8
    assert final.trajectory_ids == tuple(range(8))
    assert final.trajectories == ()
    assert final.retained_trajectory_ids == ()
    assert torch.equal(final.expectation_z_mean, reference.expectation_z_mean)
    assert torch.equal(
        final.expectation_z_variance,
        reference.expectation_z_variance,
    )
    assert load_trajectory_checkpoint(checkpoint_path).pending_ids() == ()


def test_resumed_noisy_mps_rejects_state_retention(tmp_path):
    circuit = fq.Circuit(1).x(0)
    model = fqn.NoiseModel().add("x", fq.bit_flip_channel(0.5))
    checkpoint_path = tmp_path / "mps-trajectories.pt"
    fqb.run_noisy_mps(
        circuit,
        model,
        trajectories=2,
        seed=9,
        checkpoint_path=checkpoint_path,
        max_trajectories_per_run=1,
        retain_trajectories=False,
    )

    with pytest.raises(ValueError, match="retain_trajectories=False"):
        fqb.run_noisy_mps(
            circuit,
            model,
            trajectories=2,
            seed=9,
            checkpoint_path=checkpoint_path,
            resume=True,
        )


def test_rank_local_noisy_mps_merge_matches_single_rank():
    circuit = fq.Circuit(1).x(0)
    model = fqn.NoiseModel().add("x", fq.bit_flip_channel(0.37))

    reference = fqb.run_noisy_mps(circuit, model, trajectories=12, seed=113)
    shards = tuple(
        fqb.run_noisy_mps(
            circuit,
            model,
            trajectories=12,
            seed=113,
            rank=rank,
            world_size=3,
        )
        for rank in range(3)
    )
    merged = fq.merge_noisy_mps_results(shards)

    assert tuple(item.trajectory_ids for item in shards) == (
        (0, 3, 6, 9),
        (1, 4, 7, 10),
        (2, 5, 8, 11),
    )
    assert merged.trajectory_ids == tuple(range(12))
    assert merged.trajectory_seeds == reference.trajectory_seeds
    assert merged.retained_trajectory_ids == tuple(range(12))
    assert len(merged.trajectories) == 12
    assert torch.allclose(
        merged.expectation_z_mean,
        reference.expectation_z_mean,
        atol=1e-7,
    )
    assert torch.allclose(
        merged.expectation_z_variance,
        reference.expectation_z_variance,
        atol=1e-7,
    )


def test_rank_local_checkpoint_paths_must_be_isolated(tmp_path):
    circuit = fq.Circuit(1).x(0)
    model = fqn.NoiseModel().add("x", fq.bit_flip_channel(0.5))

    with pytest.raises(ValueError, match=r"\{rank\}"):
        fqb.run_noisy_mps(
            circuit,
            model,
            trajectories=4,
            seed=11,
            rank=0,
            world_size=2,
            checkpoint_path=tmp_path / "shared.pt",
            retain_trajectories=False,
        )

    result = fqb.run_noisy_mps(
        circuit,
        model,
        trajectories=4,
        seed=11,
        rank=1,
        world_size=2,
        checkpoint_path=tmp_path / "rank-{rank}.pt",
        retain_trajectories=False,
    )
    assert result.trajectory_ids == (1, 3)
    assert (tmp_path / "rank-1.pt").exists()


def test_run_native_exposes_rank_local_parallel_noisy_plan():
    circuit = fq.Circuit(1).x(0)
    model = fqn.NoiseModel().add("x", fq.bit_flip_channel(0.5))

    result, plan = fqb.run_native(
        circuit,
        noise_model=model,
        mode="noisy_mps",
        trajectories=4,
        seed=19,
        rank=0,
        world_size=2,
        return_plan=True,
    )

    assert result.trajectory_ids == (0, 2)
    assert plan.world_size == 2
    assert plan.noisy_execution_plan.parallel.world_size == 2

from __future__ import annotations

import pytest
import torch

import flagquantum as fq
import flagquantum.training as fqt

pytestmark = pytest.mark.unit


def build(parameters: torch.Tensor) -> fq.Circuit:
    dtype = torch.complex128 if parameters.dtype == torch.float64 else torch.complex64
    return (
        fq.Circuit(2, device=parameters.device, dtype=dtype)
        .ry(0, parameters[0])
        .cx(0, 1)
        .ry(1, parameters[1])
    )


def step(module: fq.Module, optimizer: torch.optim.Optimizer) -> torch.Tensor:
    optimizer.zero_grad()
    value = module().sum()
    value.backward()
    optimizer.step()
    return value.detach()


def test_checkpoint_restores_module_optimizer_rng_and_next_step(tmp_path) -> None:
    seed = fqt.seed_everything(123)
    source = fq.Module(build, 2, init=torch.tensor([0.2, -0.3]))
    source_optimizer = torch.optim.Adam(source.parameters(), lr=0.05)
    step(source, source_optimizer)
    path = source.save_checkpoint(
        tmp_path / "training.pt",
        optimizer=source_optimizer,
        seed=seed.seed,
        step=1,
    )
    expected_random = torch.rand(4)
    expected_loss = step(source, source_optimizer)
    expected_parameters = source.parameters_tensor.detach().clone()

    target = fq.Module(build, 2)
    target_optimizer = torch.optim.Adam(target.parameters(), lr=0.05)
    restored = target.load_checkpoint(path, optimizer=target_optimizer)
    torch.testing.assert_close(torch.rand(4), expected_random)
    actual_loss = step(target, target_optimizer)
    torch.testing.assert_close(actual_loss, expected_loss)
    torch.testing.assert_close(target.parameters_tensor, expected_parameters)
    assert restored["step"] == 1 and restored["seed"].seed == 123
    assert restored["schema"] == "flagquantum.training_checkpoint_restore"
    assert restored["version"] == "1.0"


def test_checkpoint_rejects_non_equivalent_topology(tmp_path, monkeypatch) -> None:
    import flagquantum.runtime.training_state as training_state

    seed = fqt.seed_everything(7)
    source = fq.Module(build, 2)
    source.execute()
    path = source.save_checkpoint(
        tmp_path / "topology.pt", optimizer=None, seed=seed.seed
    )
    target = fq.Module(build, 2)
    active = training_state._topology(target)
    monkeypatch.setattr(
        training_state,
        "_topology",
        lambda module: {**active, "world_size": 2, "state_parallel_size": 2},
    )
    with pytest.raises(fqt.TopologyMismatchError):
        target.load_checkpoint(path, optimizer=None)


def test_default_world_is_recorded_as_distributed_state_group(monkeypatch) -> None:
    import flagquantum.runtime.training_state as training_state

    module = fq.Module(build, 2)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 4)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 2)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda group=None: "gloo")
    topology = training_state._topology(module)
    assert topology["world_size"] == 4
    assert topology["state_parallel_size"] == 4
    assert topology["state_rank"] == 2


def test_checkpoint_rejects_different_workload_and_optimizer_presence(tmp_path) -> None:
    seed = fqt.seed_everything(11)
    source = fq.Module(build, 2)
    optimizer = torch.optim.Adam(source.parameters(), lr=0.03)
    step(source, optimizer)
    path = source.save_checkpoint(
        tmp_path / "contract.pt", optimizer=optimizer, seed=seed.seed
    )

    def different(parameters: torch.Tensor) -> fq.Circuit:
        return fq.Circuit(2).rx(0, parameters[0]).cx(0, 1).rz(1, parameters[1])

    with pytest.raises(fqt.TrainingStateError, match="workload"):
        fq.Module(different, 2).load_checkpoint(path, optimizer=None)
    with pytest.raises(fqt.TrainingStateError, match="optimizer presence"):
        fq.Module(build, 2).load_checkpoint(path, optimizer=None)
    wrong_module = fq.Module(build, 2)
    wrong_optimizer = torch.optim.SGD(wrong_module.parameters(), lr=0.03)
    with pytest.raises(fqt.TrainingStateError, match="optimizer type"):
        wrong_module.load_checkpoint(path, optimizer=wrong_optimizer)


def test_checkpoint_signature_preserves_fixed_tensor_gate_values(tmp_path) -> None:
    def fixed_one(parameters: torch.Tensor) -> fq.Circuit:
        return fq.Circuit(1).ry(0, parameters[0]).rz(0, torch.tensor(0.2))

    def fixed_two(parameters: torch.Tensor) -> fq.Circuit:
        return fq.Circuit(1).ry(0, parameters[0]).rz(0, torch.tensor(0.7))

    source = fq.Module(fixed_one, 1)
    source.execute()
    path = source.save_checkpoint(tmp_path / "fixed.pt", optimizer=None, seed=17)
    with pytest.raises(fqt.TrainingStateError, match="workload"):
        fq.Module(fixed_two, 1).load_checkpoint(path, optimizer=None)


def test_public_loader_installs_saved_precision_policy(tmp_path) -> None:
    policy = fqt.PrecisionPolicy(
        accumulator_dtype="float64", mode="mixed", atol=3e-5, rtol=4e-5
    )
    source = fq.Module(build, 2, precision=policy)
    source.execute()
    path = source.save_checkpoint(tmp_path / "precision.pt", optimizer=None, seed=13)
    target = fq.Module(build, 2, precision=policy)
    restored = target.load_checkpoint(path, optimizer=None)
    assert restored["precision"] == policy
    assert target.precision == policy


def test_failed_optimizer_restore_rolls_back_module_precision_and_rng(
    tmp_path, monkeypatch
) -> None:
    source = fq.Module(build, 2, init=torch.tensor([0.2, -0.3]))
    source_optimizer = torch.optim.Adam(source.parameters(), lr=0.03)
    step(source, source_optimizer)
    path = source.save_checkpoint(
        tmp_path / "rollback.pt",
        optimizer=source_optimizer,
        seed=23,
    )
    target = fq.Module(build, 2, init=torch.tensor([0.7, 0.8]))
    target_optimizer = torch.optim.Adam(target.parameters(), lr=0.09)
    original_parameters = target.parameters_tensor.detach().clone()
    original_precision = target.precision
    rng_before = torch.get_rng_state()
    expected_random = torch.rand(3)
    torch.set_rng_state(rng_before)
    original_load = target_optimizer.load_state_dict
    calls = 0

    def fail(state):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("injected optimizer restore failure")
        return original_load(state)

    monkeypatch.setattr(target_optimizer, "load_state_dict", fail)
    with pytest.raises(RuntimeError, match="injected optimizer"):
        target.load_checkpoint(path, optimizer=target_optimizer)
    torch.testing.assert_close(target.parameters_tensor, original_parameters)
    assert target.precision == original_precision
    torch.testing.assert_close(torch.rand(3), expected_random)


def test_precision_policy_is_explicit_and_forbids_silent_downcast() -> None:
    high = fq.Module(build, 2, dtype=torch.float64)
    with pytest.raises(fqt.PrecisionPolicyError, match="downcast"):
        fqt.PrecisionPolicy(parameter_dtype="float32").apply(high)
    policy = fqt.PrecisionPolicy(
        complex_dtype="complex64",
        parameter_dtype="float32",
        accumulator_dtype="float64",
        mode="mixed",
        allow_parameter_downcast=True,
        atol=3e-5,
        rtol=3e-5,
    )
    policy.apply(high)
    assert high.parameters_tensor.dtype == torch.float32
    double = fq.Module(build, 2, dtype=torch.float64)
    assert double().dtype == torch.float64


@pytest.mark.parametrize(
    "values",
    (
        {"complex_dtype": "complex64", "parameter_dtype": "float64"},
        {"complex_dtype": "complex64", "accumulator_dtype": "float64"},
        {
            "complex_dtype": "complex128",
            "parameter_dtype": "float64",
            "accumulator_dtype": "float32",
        },
    ),
)
def test_full_precision_policy_rejects_inconsistent_real_dtypes(values) -> None:
    with pytest.raises(fqt.PrecisionPolicyError, match="full .* precision requires"):
        fqt.PrecisionPolicy(**values)


def test_mixed_precision_policy_requires_an_explicit_mode() -> None:
    policy = fqt.PrecisionPolicy(accumulator_dtype="float64", mode="mixed")

    assert policy.parameter_dtype == "float32"
    assert policy.accumulator_dtype == "float64"


def test_seed_streams_and_correctness_debug_are_reproducible() -> None:
    first = fqt.seed_everything(99)
    torch_values = torch.rand(5)
    sample_values = torch.rand(5, generator=first.torch_generator())
    second = fqt.seed_everything(99)
    torch.testing.assert_close(torch.rand(5), torch_values)
    torch.testing.assert_close(
        torch.rand(5, generator=second.torch_generator()), sample_values
    )
    assert first.jax_key_words == second.jax_key_words == (0, 99)

    debug = fq.Module(
        build,
        2,
        policy=fq.RuntimePolicy(correctness_debug=True),
    )
    debug.parameters_tensor.data[0] = torch.nan
    with pytest.raises(fqt.NonFiniteTrainingError, match="parameter"):
        debug.execute()


def test_checkpoint_restore_installs_correctness_debug_gradient_hook(tmp_path) -> None:
    source = fq.Module(build, 2, policy=fq.RuntimePolicy(correctness_debug=True))
    source.execute()
    path = source.save_checkpoint(tmp_path / "debug.pt", optimizer=None, seed=31)
    target = fq.Module(build, 2)
    target.load_checkpoint(path, optimizer=None)
    assert target.policy.correctness_debug is True
    assert target._correctness_debug_hook_handle is not None
    with pytest.raises(fqt.NonFiniteTrainingError, match="gradient"):
        (target().sum() * torch.tensor(float("inf"))).backward()

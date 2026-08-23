from __future__ import annotations

import io

import pytest
import torch

import flagquantum as fq

pytestmark = pytest.mark.unit


def build_circuit(
    parameters: torch.Tensor, inputs: torch.Tensor | None = None
) -> fq.Circuit:
    circuit = fq.Circuit(2, device=parameters.device)
    if inputs is not None:
        circuit.rx(0, inputs.reshape(()))
    return circuit.ry(0, parameters[0]).cx(0, 1).ry(1, parameters[1])


def test_module_is_normal_trainable_pytorch_module() -> None:
    module = fq.Module(build_circuit, 2, init=torch.tensor([0.2, -0.3]))
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    before = module.parameters_tensor.detach().clone()
    loss = module(torch.tensor(0.1)).sum()
    loss.backward()
    optimizer.step()
    assert isinstance(module, torch.nn.Module)
    assert not hasattr(fq, "QuantumModule")
    assert module.parameters_tensor.grad is not None
    assert not torch.equal(before, module.parameters_tensor)
    module.eval()
    assert not module.training
    module.train()
    assert module.training


def test_quantum_module_supports_named_parameter_groups() -> None:
    def build_named(parameters):
        return (
            fq.Circuit(2)
            .ry(0, parameters["encoder"][0])
            .rx(1, parameters["readout"])
            .cx(0, 1)
        )

    module = fq.Module(
        build_named,
        parameters={"encoder": (2,), "readout": ()},
        init={"encoder": [0.2, -0.1], "readout": 0.3},
    )
    value = module().sum()
    value.backward()

    assert tuple(module.parameter_groups) == ("encoder", "readout")
    assert module.parameter_groups["encoder"].shape == (2,)
    assert module.parameter_groups["readout"].shape == ()
    assert all(
        parameter.grad is not None for parameter in module.parameter_groups.values()
    )
    assert "named_parameter_groups.encoder" in module.state_dict()


def test_quantum_module_mps_hamiltonian_uses_policy_and_reports_bonds() -> None:
    hamiltonian = fq.zz_chain_hamiltonian(4, coupling=-1.0, field=0.1)

    def build(parameters):
        circuit = fq.Circuit(4)
        for wire in range(4):
            circuit.ry(wire, parameters[wire])
        return circuit.cx(0, 1).cx(2, 3)

    module = fq.Module(
        build,
        4,
        init=torch.linspace(0.1, 0.4, 4),
        hamiltonian=hamiltonian,
        policy=fq.RuntimePolicy(
            mode="mps",
            observable="hamiltonian",
            mps_max_bond=4,
            mps_cutoff=1e-6,
        ),
    )
    result = module.execute()
    result.value.sum().backward()

    assert result.runtime["mode"] == "mps"
    assert result.runtime["max_bond"] <= 4
    assert result.runtime["triton_mps_two_site_regions"] == 0
    assert result.runtime["eager_mps_two_site_regions"] == 2
    assert module.parameters_tensor.grad is not None


def test_quantum_module_automatically_binds_symbolic_circuit_parameters() -> None:
    theta = fq.Parameter("theta")
    phi = fq.Parameter("phi")
    template = fq.Circuit(2).ry(0, theta).rx(1, 2 * phi).cx(0, 1)

    module = fq.Module(template, init={"theta": 0.2, "phi": -0.1})
    value = module().sum()
    value.backward()

    assert tuple(module.parameter_groups) == ("phi", "theta")
    assert all(
        parameter.grad is not None for parameter in module.parameter_groups.values()
    )
    expected = template.bind_parameters({"theta": 0.2, "phi": -0.1})
    torch.testing.assert_close(module.execute().state, expected.state())


@pytest.mark.parametrize("strategy", ("uniform", "normal"))
def test_quantum_module_random_initialization_is_seeded(strategy: str) -> None:
    first = fq.Module(build_circuit, 2, init=strategy, seed=17)
    second = fq.Module(build_circuit, 2, init=strategy, seed=17)
    different = fq.Module(build_circuit, 2, init=strategy, seed=18)

    torch.testing.assert_close(first.parameters_tensor, second.parameters_tensor)
    assert not torch.equal(first.parameters_tensor, different.parameters_tensor)
    if strategy == "uniform":
        assert torch.all(first.parameters_tensor >= 0)
        assert torch.all(first.parameters_tensor < 2 * torch.pi)


def test_named_groups_accept_global_and_per_group_init_strategies() -> None:
    global_init = fq.Module(
        lambda p: fq.Circuit(1).ry(0, p["angles"][0]),
        parameters={"angles": (3,), "bias": ()},
        init="uniform",
        seed=9,
    )
    mixed_init = fq.Module(
        lambda p: fq.Circuit(1).ry(0, p["angles"][0] + p["bias"]),
        parameters={"angles": (3,), "bias": ()},
        init={"angles": "normal", "bias": 0.25},
        seed=9,
    )

    assert torch.all(global_init.parameter_groups["angles"] >= 0)
    torch.testing.assert_close(mixed_init.parameter_groups["bias"], torch.tensor(0.25))
    with pytest.raises(ValueError, match="init strategy"):
        fq.Module(build_circuit, 2, init="unknown")


def test_execution_result_fields_and_backend_selection_are_stable() -> None:
    module = fq.Module(
        build_circuit,
        2,
        policy=fq.RuntimePolicy(backend="jax", observable_wires=(1,)),
    )
    result = module.execute()
    assert isinstance(result, fq.ExecutionResult)
    assert result.value is not None
    assert result.samples is None and result.plan is None
    assert result.compatibility["selected_backend"] in {"jax", "pytorch"}
    if result.compatibility["selected_backend"] == "jax":
        assert result.state is None
    else:
        assert result.state is not None
    assert result.detach().value is not None
    with pytest.raises(ValueError, match="unsupported fq.Module backend"):
        fq.Module(
            build_circuit,
            2,
            policy=fq.RuntimePolicy(backend="unknown"),
        ).execute()


def test_circuit_run_matches_uniform_execution_entry_point() -> None:
    circuit = fq.Circuit(2).h(0).cx(0, 1)
    result = circuit.run(mode="statevector")
    assert isinstance(result, fq.ExecutionResult)
    assert result.plan is not None
    assert result.compatibility["legacy_return_normalized"] is True

    with pytest.warns(DeprecationWarning, match="fq.run_native"):
        legacy = circuit.run(mode="statevector", result=False)
    assert isinstance(legacy, torch.Tensor)
    torch.testing.assert_close(result.state, legacy)

    with pytest.raises(TypeError, match="always includes the execution plan"):
        circuit.run(return_plan=True)


def test_fq_run_executes_ordered_measurement_requests() -> None:
    circuit = fq.Circuit(2).x(0)
    result = fq.run(
        circuit,
        mode="statevector",
        measurements=(
            fq.MeasurementNode("expectation_z", (0, 1)),
            fq.MeasurementNode("sample", (1,), shots=4, metadata={"seed": 7}),
            fq.MeasurementNode(
                "counts",
                (0,),
                shots=4,
                metadata={"seed": 7, "format": "int"},
            ),
        ),
    )

    assert all(isinstance(item, fq.MeasurementResult) for item in result.measurements)
    assert tuple(item.kind for item in result.measurements) == (
        "expectation_z",
        "sample",
        "counts",
    )
    torch.testing.assert_close(
        result.measurements[0].value,
        torch.tensor([[-1.0, 1.0]]),
    )
    assert result.measurements[1].value.tolist() == [[[0], [0], [0], [0]]]
    assert result.measurements[2].value == [{1: 4}]
    assert result.samples is result.measurements[1].value
    assert result.summary()["measurement_count"] == 3


def test_fq_run_is_the_uniform_execution_entry_point() -> None:
    circuit = fq.Circuit(2).h(0).cx(0, 1)

    result = fq.run(circuit)

    assert isinstance(result, fq.ExecutionResult)
    assert result.state is not None
    assert result.plan is not None
    assert result.runtime["mode"] == "statevector"
    torch.testing.assert_close(result.state, circuit.state())

    with pytest.raises(TypeError, match="always includes the execution plan"):
        fq.run(circuit, return_plan=True)
    with pytest.raises(TypeError, match="always returns ExecutionResult"):
        fq.run(circuit, result=False)


def test_quantum_module_tracks_static_topology_cache_hits() -> None:
    module = fq.Module(build_circuit, 2)

    first = module.execute()
    second = module.execute()

    assert first.runtime["compiled"] is True
    assert first.runtime["compile_count"] == 1
    assert first.runtime["program_cache_hits"] == 0
    assert second.runtime["compile_count"] == 1
    assert second.runtime["program_cache_hits"] == 1
    assert second.runtime["topology_hash"] == first.runtime["topology_hash"]


@pytest.mark.parametrize("mode", ("statevector", "mps", "tensor_network"))
def test_compiled_builder_runs_once_and_rebinds_dynamic_tensor_slots(
    mode: str,
) -> None:
    calls = 0

    def counted_builder(parameters, inputs):
        nonlocal calls
        calls += 1
        return fq.Circuit(1, bsz=len(inputs)).ry(0, inputs + parameters["angle"])

    module = fq.Module(
        counted_builder,
        parameters={"angle": ()},
        init={"angle": 0.2},
        policy=fq.RuntimePolicy(mode=mode),
    )
    first_inputs = torch.tensor([0.1, 0.3], requires_grad=True)
    first = module.execute(first_inputs)
    static_circuit = module._builder_bound_circuit
    assert static_circuit is not None
    static_instructions = tuple(map(id, static_circuit._instructions))
    first.value.sum().backward()
    first_gradient = first_inputs.grad.detach().clone()
    module.zero_grad(set_to_none=True)

    second_inputs = torch.tensor([-0.4, 0.7], requires_grad=True)
    second = module.execute(second_inputs)
    second.value.sum().backward()

    assert calls == 1
    assert module._builder_bound_circuit is static_circuit
    assert tuple(map(id, static_circuit._instructions)) == static_instructions
    assert second.runtime["builder_compiled"] is True
    assert second.runtime["builder_compile_count"] == 1
    assert second.runtime["builder_cache_hits"] == 1
    assert second.runtime["static_program_reused"] is True
    assert second_inputs.grad is not None
    assert not torch.equal(first_gradient, second_inputs.grad)
    assert not any("builder_program" in key for key in module.state_dict())
    with pytest.raises(RuntimeError, match="parameter slots are not bound"):
        static_circuit._instructions[0].params["theta"]


def test_compiled_builder_rejects_tensor_dependent_topology() -> None:
    def dynamic_builder(parameters):
        circuit = fq.Circuit(1)
        if parameters[0].item() > 0:
            circuit.x(0)
        return circuit.ry(0, parameters[0])

    module = fq.Module(dynamic_builder, 1, init=torch.tensor([0.2]))
    with pytest.raises(RuntimeError, match="requires static circuit topology"):
        module()


@pytest.mark.parametrize("mode", ("statevector", "mps", "tensor_network"))
def test_fq_run_normalizes_local_backend_result_types(mode: str) -> None:
    result = fq.run(fq.Circuit(2).h(0).cx(0, 1), mode=mode)

    assert isinstance(result, fq.ExecutionResult)
    assert result.plan is not None
    assert result.runtime["mode"] == mode


def test_fq_train_owns_the_optimizer_loop_and_returns_training_result() -> None:
    module = fq.Module(build_circuit, 2, init=torch.tensor([0.2, -0.3]))
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    before = module.parameters_tensor.detach().clone()

    result = fq.train(
        module,
        optimizer=optimizer,
        objective=lambda value: value.sum(),
        steps=2,
        inputs=lambda step: torch.tensor(0.1 + step * 0.05),
    )

    assert isinstance(result, fq.TrainingResult)
    assert result.completed_steps == 2
    assert len(result.losses) == 2
    assert isinstance(result.last_execution, fq.ExecutionResult)
    assert result.last_execution.value is not None
    assert not result.last_execution.value.requires_grad
    assert not torch.equal(before, module.parameters_tensor)


def test_fq_train_log_interval_and_callback_are_unambiguous(capsys) -> None:
    module = fq.Module(build_circuit, 2)
    optimizer = torch.optim.SGD(module.parameters(), lr=0.01)
    events = []

    fq.train(
        module,
        optimizer=optimizer,
        objective=lambda value: value.sum(),
        steps=5,
        log_interval=2,
        callback=lambda step, loss, execution: events.append((step, loss, execution)),
    )

    lines = capsys.readouterr().out.strip().splitlines()
    assert [line.split()[1] for line in lines] == ["1/5", "2/5", "4/5", "5/5"]
    assert [event[0] for event in events] == [1, 2, 3, 4, 5]
    assert all(isinstance(event[1], float) for event in events)
    assert all(not event[2].value.requires_grad for event in events)


def test_fq_train_is_silent_by_default_and_validates_logging(capsys) -> None:
    module = fq.Module(build_circuit, 2)
    optimizer = torch.optim.SGD(module.parameters(), lr=0.01)
    options = dict(
        module=module,
        optimizer=optimizer,
        objective=lambda value: value.sum(),
        steps=1,
    )

    fq.train(**options)
    assert capsys.readouterr().out == ""
    for interval in (0, -1, True):
        with pytest.raises(ValueError, match="positive integer or None"):
            fq.train(**options, log_interval=interval)
    with pytest.raises(TypeError, match="callback must be callable"):
        fq.train(**options, callback=object())


def test_fq_train_rejects_ambiguous_or_invalid_training_inputs() -> None:
    module = fq.Module(build_circuit, 2)
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)

    with pytest.raises(TypeError, match="requires an fq.Module"):
        fq.train(
            fq.Circuit(1), optimizer=optimizer, objective=lambda x: x.sum(), steps=1
        )
    with pytest.raises(ValueError, match="positive integer"):
        fq.train(module, optimizer=optimizer, objective=lambda x: x.sum(), steps=0)
    with pytest.raises(ValueError, match="scalar tensor"):
        fq.train(module, optimizer=optimizer, objective=lambda value: value, steps=1)


def test_module_state_dict_round_trip_includes_policy_and_deployment() -> None:
    source = fq.Module(
        build_circuit,
        2,
        init=torch.tensor([0.4, -0.6]),
        policy=fq.RuntimePolicy(observable="z_sum", observable_wires=(0, 1)),
        deployment_binding={"provider": "local", "target": "simulator"},
    )
    buffer = io.BytesIO()
    torch.save(source.state_dict(), buffer)
    buffer.seek(0)
    target = fq.Module(build_circuit, 2)
    target.load_state_dict(torch.load(buffer, weights_only=True))
    torch.testing.assert_close(target.parameters_tensor, source.parameters_tensor)
    assert target.policy == source.policy
    assert target.deployment_binding == source.deployment_binding
    torch.testing.assert_close(target(), source())


def test_module_has_no_legacy_layer_adapter() -> None:
    assert not hasattr(fq.Module, "from_quantum_torch_layer")


def test_module_to_updates_precision_and_invalidates_jax_cache() -> None:
    module = fq.Module(build_circuit, 2)
    module._jax_kernel = object()
    module.to(dtype=torch.float64)
    assert module.parameters_tensor.dtype == torch.float64
    assert module.precision.parameter_dtype == "float64"
    assert module.precision.complex_dtype == "complex128"
    assert module._jax_kernel is None


def test_module_to_rejects_invalid_dtype_before_mutating_parameters() -> None:
    module = fq.Module(build_circuit, 2)
    with pytest.raises(TypeError, match="float32 or float64"):
        module.to(dtype=torch.complex64)
    assert module.parameters_tensor.dtype == torch.float32


@pytest.mark.parametrize(
    ("mode", "executor"),
    [
        ("mps", "pytorch_native_mps"),
        ("tensor_network", "pytorch_native_tensor_network"),
    ],
)
def test_structured_modes_execute_the_selected_backend(mode, executor) -> None:
    module = fq.Module(build_circuit, 2, policy=fq.RuntimePolicy(mode=mode))
    result = module.execute()
    reference = build_circuit(module.parameters_tensor).expectation_z((0,))[..., 0]
    torch.testing.assert_close(result.value, reference, atol=1e-5, rtol=1e-5)
    assert result.state is None
    assert result.runtime["executor"] == executor


def test_policy_rejects_unknown_modes() -> None:
    with pytest.raises(ValueError, match="unsupported fq.Module mode"):
        fq.RuntimePolicy(mode="bogus")


@pytest.mark.parametrize("mode", ("statevector", "mps", "tensor_network"))
def test_module_returns_multiple_z_observables_in_one_execution(mode) -> None:
    parameters = torch.tensor([0.2, -0.3], requires_grad=True)
    module = fq.Module(
        build_circuit,
        2,
        init=parameters,
        policy=fq.RuntimePolicy(
            mode=mode,
            observable="z",
            observable_wires=(0, 1),
        ),
    )

    value = module()
    reference_parameters = parameters.detach().clone().requires_grad_(True)
    reference = build_circuit(reference_parameters).expectation_z((0, 1))

    assert value.shape == (1, 2)
    torch.testing.assert_close(value, reference, atol=1e-5, rtol=1e-5)
    value.square().sum().backward()
    reference.square().sum().backward()
    torch.testing.assert_close(
        module.parameters_tensor.grad,
        reference_parameters.grad,
        atol=1e-5,
        rtol=1e-5,
    )


def test_batched_module_returns_batch_by_observable_shape() -> None:
    def batched(parameters, inputs):
        circuit = fq.Circuit(2, bsz=inputs.shape[0], device=inputs.device)
        return (
            circuit.ry(0, inputs[:, 0] + parameters[0])
            .cx(0, 1)
            .ry(1, inputs[:, 1] + parameters[1])
        )

    module = fq.Module(
        batched,
        2,
        policy=fq.RuntimePolicy(observable="z", observable_wires=(0, 1)),
    )
    inputs = torch.randn(5, 2, requires_grad=True)
    value = module(inputs)

    assert value.shape == (5, 2)
    value.sum().backward()
    assert inputs.grad is not None
    assert module.parameters_tensor.grad is not None


def test_local_forward_uses_tensor_only_fast_path(monkeypatch) -> None:
    module = fq.Module(
        build_circuit,
        2,
        policy=fq.RuntimePolicy(observable="z", observable_wires=(0, 1)),
    )

    def reject_execute(*args, **kwargs):
        raise AssertionError("forward should not construct an ExecutionResult")

    monkeypatch.setattr(module, "execute", reject_execute)
    value = module()
    assert value.shape == (1, 2)
    value.sum().backward()
    assert module.parameters_tensor.grad is not None


def test_single_rank_distributed_policy_reports_local_parallel_semantics() -> None:
    module = fq.Module(
        build_circuit,
        2,
        policy=fq.RuntimePolicy(mode="distributed_statevector"),
    )
    result = module.execute()
    assert result.runtime["forward_distribution_semantics"] == "single_device_fast_path"
    assert (
        result.metrics["parallelism"]["distribution_semantics"]
        == "single_device_fast_path"
    )


def test_jax_module_supports_input_values_and_gradients() -> None:
    pytest.importorskip("jax")
    parameters = torch.tensor([0.2, -0.3])
    module = fq.Module(
        build_circuit,
        2,
        init=parameters,
        policy=fq.RuntimePolicy(backend="jax", allow_backend_fallback=False),
    )
    inputs = torch.tensor(0.17, requires_grad=True)
    value = module(inputs)
    parameter_grad, input_grad = torch.autograd.grad(
        value.sum(), (module.parameters_tensor, inputs)
    )

    reference_parameters = parameters.clone().requires_grad_(True)
    reference_inputs = inputs.detach().clone().requires_grad_(True)
    reference = build_circuit(reference_parameters, reference_inputs).expectation_z(
        (0,)
    )[..., 0]
    expected_parameter_grad, expected_input_grad = torch.autograd.grad(
        reference.sum(), (reference_parameters, reference_inputs)
    )
    torch.testing.assert_close(value, reference, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        parameter_grad, expected_parameter_grad, atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(input_grad, expected_input_grad, atol=1e-5, rtol=1e-5)

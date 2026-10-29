"""One ordinary synthesis product, real execution and routed named reading."""

import math
from dataclasses import replace

import pytest
import torch

import flagquantum as fq
from examples.algorithms.synthesis_trial import polynomial, pressure
from flagquantum.core.ir import Instruction
from flagquantum.runtime.execution_plan import ExecutionPlan
from flagquantum.services import _synthesis as synth
from flagquantum.services import _synthesis_readout as readout

pytestmark = pytest.mark.integration


def _program(**kwargs):
    return synth.synthesize(
        pressure,
        inputs={"y": synth.UInt(1), "x": synth.UInt(1)},
        input_state=synth.InputState.uniform(),
        **kwargs,
    )


@pytest.mark.parametrize("topology", [None, "line"])
@pytest.mark.parametrize("objective", ["width", "depth"])
@pytest.mark.parametrize("state", ["basis", "uniform"])
def test_high_level_inputs_and_names_need_no_gates_maps_or_decoding(
    topology, objective, state
) -> None:
    program = synth.synthesize(
        pressure,
        inputs={"y": synth.UInt(1), "x": synth.UInt(1)},
        input_state=(
            synth.InputState.uniform()
            if state == "uniform"
            else synth.InputState.basis(x=1, y=1)
        ),
        topology=topology,
        objective=objective,
    )
    result = fq.run(program.execution)
    observed = program.read_distribution(result, registers=("y", "result", "x"))
    expected = (
        {(y, pressure(x, y), x): 0.25 for x in range(2) for y in range(2)}
        if state == "uniform"
        else {(1, 2, 1): 1.0}
    )
    assert dict(observed) == pytest.approx(expected, abs=1e-10)
    assert program.report["candidate"] == (
        "recompute" if objective == "width" else "retain"
    )
    assert program.report["resources"] == program._checked.verify_result(result)
    row = program.selection.selected
    assert row.source_identities
    if topology:
        assert any(g.name == "swap" for g in row.target.executable.instructions)
        assert row.target.layout.logical_to_physical != tuple(
            range(program.circuit.n_wires)
        )
    # Actual mapped workspace, not a logical low-bit mask after routing.
    statevector = result.statevector().reshape(-1)
    indices = torch.arange(statevector.numel())
    dirty = torch.zeros_like(indices, dtype=torch.bool)
    for logical in range(4, program.circuit.n_wires):
        physical = row.target.layout.physical_of(logical)
        dirty |= ((indices >> (program.circuit.n_wires - 1 - physical)) & 1).bool()
    assert float(statevector[dirty].abs().square().sum()) <= 1e-12


def test_final_constraints_change_schedule_without_user_selecting_one() -> None:
    narrow = _program(max_qubits=9)
    shallow = _program(max_operations=42)
    assert narrow.report["candidate"] == "recompute"
    assert shallow.report["candidate"] == "retain"
    for program in (narrow, shallow):
        assert sum(
            program.read_distribution(fq.run(program.execution)).values()
        ) == pytest.approx(1.0)


def test_every_user_repair_is_a_real_complete_retry() -> None:
    with pytest.raises(synth.SynthesisError) as caught:
        _program(max_qubits=8, max_operations=39, max_depth=34, max_workspace=4)
    error = caught.value
    assert error.selection.complete and len(error.repairs) == 2
    assert "not a global lower bound" in str(error)
    for repair in error.repairs:
        program = _program(**repair)
        assert sum(
            program.read_distribution(fq.run(program.execution)).values()
        ) == pytest.approx(1.0)
        with pytest.raises(TypeError):
            repair["max_qubits"] = 100


def test_budget_exhaustion_is_not_a_partial_success() -> None:
    with pytest.raises(synth.SynthesisError) as caught:
        _program(construction_budget=synth.ConstructionBudget(candidate_checks=1))
    assert not caught.value.selection.complete and not caught.value.repairs
    assert caught.value.selection.diagnostics[0].code == "selection.budget"
    assert _program().execution is not None


def test_synthesis_plans_once_and_execution_property_and_reading_do_not_replan(
    monkeypatch,
) -> None:
    import flagquantum._api as api
    import flagquantum.compiler as compiler

    original = api.plan
    calls = []

    def count(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("this operation must not execute, compile or replan")

    monkeypatch.setattr(api, "plan", count)
    monkeypatch.setattr(fq, "run", forbidden)
    program = _program()
    assert len(calls) == 1
    monkeypatch.undo()
    result = fq.run(program.execution)
    monkeypatch.setattr(api, "plan", forbidden)
    monkeypatch.setattr(fq, "plan", forbidden)
    monkeypatch.setattr(fq, "run", forbidden)
    monkeypatch.setattr(compiler, "compile", forbidden)
    assert program.execution is result.plan
    assert len(program.read_distribution(result)) == 2


def test_reader_budget_precedes_norm_scan_and_marginal_allocation(monkeypatch) -> None:
    program = _program(topology="line")
    result = fq.run(program.execution)

    def forbidden(*args, **kwargs):
        raise AssertionError("read budget must precede state scanning/allocation")

    monkeypatch.setattr(torch.linalg, "vector_norm", forbidden)
    monkeypatch.setattr(readout, "execute_measurements", forbidden)
    with pytest.raises(ValueError, match="readout budget"):
        program.read_distribution(result, registers=("x", "y", "result"), max_keys=2)


@pytest.mark.parametrize("topology", [None, "line"])
def test_checked_plan_serialization_preserves_named_reading(topology) -> None:
    program = _program(topology=topology)
    restored = ExecutionPlan.from_json(program.execution.to_json())
    observed = program.read_distribution(
        fq.run(restored), registers=("x", "y", "result")
    )
    assert dict(observed) == pytest.approx(
        {(x, y, pressure(x, y)): 0.25 for x in range(2) for y in range(2)}, abs=1e-10
    )


def test_changed_options_same_size_program_and_logical_bindings_are_refused() -> None:
    program = _program()
    other = replace(
        program.circuit,
        instructions=(*program.circuit.instructions, Instruction("x", (0,))),
    )
    options = fq.ExecutionOptions(
        mode="statevector",
        backend="torch",
        device="cpu",
        precision="complex128",
        require_gradients=False,
        allow_approximate=False,
        allow_backend_fallback=False,
    )
    for source, controls in (
        (other, options),
        (program.circuit, replace(options, seed=17)),
    ):
        with pytest.raises(ValueError, match="execution.identity"):
            program.read_distribution(fq.run(source, options=controls))
    altered = replace(program, registers={"result": (0, 1)})
    with pytest.raises(ValueError, match="registers changed"):
        altered.read_distribution(fq.run(program.execution))


def test_compiler_prediction_is_bound_to_actual_runtime_even_if_fake_claims_success(
    monkeypatch,
) -> None:
    original = synth.select_arithmetic

    def wrong_prediction(*args, **kwargs):
        selection = original(*args, **kwargs)
        row = selection.selected
        wrong = replace(
            row.target.executable,
            instructions=(*row.target.executable.instructions, Instruction("x", (0,))),
        )
        target = replace(
            row.target,
            executable=wrong,
            stages=(*row.target.stages[:-1], ("executable", wrong)),
        )
        target.validate()
        return replace(selection, selected=replace(row, target=target))

    monkeypatch.setattr(synth, "select_arithmetic", wrong_prediction)
    with pytest.raises(ValueError, match="execution identity"):
        _program()


def test_unsupported_basis_topology_and_target_fail_without_execution() -> None:
    with pytest.raises(synth.SynthesisError) as caught:
        _program(basis=("u3", "cx"))
    assert (
        not caught.value.repairs
        and caught.value.selection.diagnostics[0].code == "selection.unsupported_target"
    )
    with pytest.raises(ValueError, match="topology"):
        _program(topology="ring")
    with pytest.raises(ValueError, match="input-only"):
        synth.synthesize(
            polynomial,
            inputs={"x": synth.UInt(1), "y": synth.UInt(1)},
            preparation=fq.Circuit(2, bsz=2).to_ir(),
            preparation_map={"x": (0,), "y": (1,)},
        )


def test_physical_port_substitution_and_mutated_compiler_stages_are_refused() -> None:
    from flagquantum.services._synthesis_execution import inspect_execution

    program = _program()
    forged = inspect_execution(
        program.execution,
        limits=program.selection.constraints.limits,
        registers={"x": (2,), "y": (1,), "result": (0, 3)},
    )
    with pytest.raises(ValueError, match="named ports"):
        replace(program, _checked=forged).read_distribution(fq.run(program.execution))
    program.selection.selected.target.stages[0][1].metadata["stale"] = True
    with pytest.raises(ValueError, match="circuit changed"):
        _ = program.execution


def test_routed_filtering_reports_omitted_mass_and_never_renormalizes() -> None:
    program = synth.synthesize(
        polynomial,
        inputs={"x": synth.UInt(1), "y": synth.UInt(1)},
        preparation=fq.Circuit(2).ry(0, 0.4).to_ir(),
        preparation_map={"x": (0,), "y": (1,)},
        topology="line",
    )
    observed = program.read_distribution(fq.run(program.execution), threshold=0.05)
    assert dict(observed) == pytest.approx({(0,): math.cos(0.2) ** 2}, abs=1e-10)
    assert observed.diagnostics["omitted_probability"] == pytest.approx(
        math.sin(0.2) ** 2, abs=1e-10
    )
    assert observed.diagnostics["probability_mass"] == pytest.approx(1.0, abs=1e-10)


@pytest.mark.parametrize("topology", [None, "line"])
@pytest.mark.parametrize("theta", [0.37, 1.1, 2.0])
def test_bound_preparation_gradient_survives_single_user_product(
    topology, theta
) -> None:
    parameter = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
    prep = fq.Circuit(2, dtype=torch.complex128).ry(0, parameter).to_ir()
    program = synth.synthesize(
        polynomial,
        inputs={"x": synth.UInt(1), "y": synth.UInt(1)},
        preparation=prep,
        preparation_map={"x": (0,), "y": (1,)},
        topology=topology,
    )
    result = fq.run(program.execution)
    marginal = program.read_distribution(result)
    assert marginal[(1,)] == pytest.approx(math.sin(theta / 2) ** 2, abs=1e-10)
    state = result.statevector().reshape(-1)
    indices = torch.arange(state.numel())
    physical = program._checked.registers["result"][-1]
    signs = 1 - 2 * ((indices >> (program.circuit.n_wires - 1 - physical)) & 1)
    value = (state.abs().square() * signs).sum()
    gradient = torch.autograd.grad(value, parameter)[0]
    torch.testing.assert_close(value, torch.cos(parameter), atol=1e-10, rtol=0)
    torch.testing.assert_close(gradient, -torch.sin(parameter), atol=1e-8, rtol=0)
    assert float(gradient.abs()) > 0.1

"""User arithmetic workflows, named real results and bounded failure recovery."""

import math
from collections import Counter
from dataclasses import replace
from typing import Any

import pytest
import torch

import flagquantum as fq
from flagquantum.compiler import compile as compile_program
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.core.ir import CircuitIR, Instruction, MeasurementNode
from flagquantum.core.parameters import Parameter
from flagquantum.services import _synthesis as synth
from flagquantum.services import _synthesis_readout as readout

pytestmark = pytest.mark.integration


def polynomial(x: int, y: int) -> int:
    return x * x + y


@synth.qfunc
def declared(x: int, y: int) -> int:
    return x**2 + y


def identity(x: int) -> int:
    return x


def configured(x: int, scale: int) -> int:
    return scale * x


def _options(mode: str = "statevector") -> fq.ExecutionOptions:
    return fq.ExecutionOptions(
        mode=mode,
        backend="torch",
        device="cpu",
        precision="complex128",
        require_gradients=False,
        allow_approximate=False,
        allow_backend_fallback=False,
    )


def _small() -> synth.Program:
    return synth.synthesize(
        identity, inputs={"x": synth.UInt(1)}, input_state=synth.InputState.uniform()
    )


def test_every_basis_input_uses_only_high_level_logic_and_named_readout() -> None:
    assert polynomial(3, 2) == 11  # Ordinary Python remains useful for host tests.
    for x in range(4):
        for y in range(4):
            program = synth.synthesize(
                polynomial,
                inputs={"y": synth.UInt(2), "x": synth.UInt(2)},
                input_state=synth.InputState.basis(x=x, y=y),
                max_qubits=18,
            )
            distribution = program.read_distribution(
                fq.run(program.circuit, options=_options()),
                registers=("x", "y", "result"),
            )
            assert set(distribution) == {(x, y, x * x + y)}
            assert math.isclose(distribution[(x, y, x * x + y)], 1, abs_tol=1e-10)
            assert program.report["workspace_qubits"] == 10
            assert program.report["result_width"] == 4


def test_uniform_joint_marginal_and_reordered_registers_read_real_probabilities() -> (
    None
):
    program = synth.synthesize(
        polynomial,
        inputs={"x": synth.UInt(2), "y": synth.UInt(2)},
        input_state=synth.InputState.uniform(),
        max_qubits=18,
    )
    result = fq.run(program.circuit, options=_options())
    joint = program.read_distribution(result, registers=("y", "result", "x"))
    expected = {(y, x * x + y, x) for x in range(4) for y in range(4)}
    assert set(joint) == expected
    assert all(abs(joint[key] - 1 / 16) < 1e-10 for key in expected)
    marginal = program.read_distribution(result)
    counts = Counter(x * x + y for x in range(4) for y in range(4))
    assert set(marginal) == {(value,) for value in counts}
    assert all(
        abs(marginal[(value,)] - count / 16) < 1e-10 for value, count in counts.items()
    )
    assert marginal.diagnostics["scanned_amplitudes"] == 1 << 18


def test_plain_and_qfunc_forms_generate_identical_circuits() -> None:
    kwargs = {
        "inputs": {"x": synth.UInt(2), "y": synth.UInt(2)},
        "input_state": synth.InputState.uniform(),
        "max_qubits": 18,
    }
    plain = synth.synthesize(polynomial, **kwargs)
    explicit = synth.synthesize(declared, **kwargs)
    assert plain.circuit.content_hash == explicit.circuit.content_hash
    assert plain.report["resources"] == explicit.report["resources"]
    with pytest.raises(TypeError, match="captured quantum definition"):
        declared(3, 2)


def test_classical_configuration_and_named_helper_recovery() -> None:
    program = synth.synthesize(
        configured,
        inputs={"x": synth.UInt(2)},
        constants={"scale": 3},
        input_state=synth.InputState.basis(x=3),
    )
    assert dict(
        program.read_distribution(fq.run(program.circuit, options=_options()))
    ) == {(9,): 1.0}

    def square(x: int) -> int:
        return x * x

    def main(x: int) -> int:
        return square(x) + x

    with pytest.raises(HybridVerificationError, match="register it explicitly"):
        synth.synthesize(main, inputs={"x": synth.UInt(2)})
    fixed = synth.synthesize(
        main,
        inputs={"x": synth.UInt(2)},
        functions=(square,),
        input_state=synth.InputState.basis(x=3),
    )
    assert dict(fixed.read_distribution(fq.run(fixed.circuit, options=_options()))) == {
        (12,): 1.0
    }


def test_omitted_input_state_is_explicitly_reported_as_default_zero() -> None:
    program = synth.synthesize(identity, inputs={"x": synth.UInt(2)})
    assert program.report["input_state"] == "basis_default_zero"
    assert program.report["workspace_qubits"] == 0
    assert dict(
        program.read_distribution(fq.run(program.circuit, options=_options()))
    ) == {(0,): 1.0}


def test_wrong_same_size_execution_missing_plan_and_changed_source_are_refused() -> (
    None
):
    program = _small()
    other = replace(
        program.circuit,
        instructions=(*program.circuit.instructions, Instruction("x", (0,))),
    )
    with pytest.raises(ValueError, match="execution identity"):
        program.read_distribution(fq.run(other, options=_options()))
    result = fq.run(program.circuit, options=_options())
    with pytest.raises(ValueError, match="execution plan"):
        program.read_distribution(replace(result, plan=None))
    program.circuit.metadata["changed"] = True
    with pytest.raises(ValueError, match="circuit changed"):
        program.read_distribution(result)


def test_content_identity_survives_serialization_without_an_object_registry() -> None:
    program = _small()
    restored = CircuitIR.from_json(program.circuit.to_json())
    assert dict(
        program.read_distribution(fq.run(restored, options=_options()))
    ) == pytest.approx({(0,): 0.5, (1,): 0.5})


def test_routed_recompiled_result_requires_explicit_revalidation() -> None:
    program = synth.synthesize(
        identity, inputs={"x": synth.UInt(2)}, input_state=synth.InputState.basis(x=1)
    )
    routed = compile_program(
        program.circuit, coupling_map=((0, 1), (1, 2), (2, 3)), optimize=False
    )
    assert any(g.name == "swap" for g in routed.instructions)
    with pytest.raises(ValueError, match="routing|identity"):
        program.read_distribution(fq.run(routed, options=_options()))


def test_reading_never_runs_replans_or_recompiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    program = _small()
    result = fq.run(program.circuit, options=_options())

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("named reading must not execute or replan")

    monkeypatch.setattr(fq, "run", forbidden)
    monkeypatch.setattr(fq, "plan", forbidden)
    monkeypatch.setattr(synth, "compile_program", forbidden)
    assert len(program.read_distribution(result)) == 2


@pytest.mark.parametrize(
    "limits", [{"max_bits": 1}, {"max_keys": 2}, {"max_amplitudes": 3}]
)
def test_readout_limits_precede_marginal_allocation(
    limits: dict[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    program = _small()
    result = fq.run(program.circuit, options=_options())

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("budget must be checked before the marginal")

    monkeypatch.setattr(readout, "execute_measurements", forbidden)
    with pytest.raises(ValueError, match="readout budget"):
        program.read_distribution(result, registers=("x", "result"), **limits)


def test_filtering_reports_omitted_mass_without_renormalizing_or_editing_report() -> (
    None
):
    program = _small()
    report = dict(program.report)
    observed = program.read_distribution(
        fq.run(program.circuit, options=_options()), threshold=0.6
    )
    assert not observed
    assert abs(observed.diagnostics["omitted_probability"] - 1) < 1e-10
    assert dict(program.report) == report


@pytest.mark.parametrize("names", [(), ("missing",), ("result", "result"), "result"])
def test_invalid_named_readout_is_actionable(names: tuple[str, ...]) -> None:
    program = _small()
    result = fq.run(program.circuit, options=_options())
    with pytest.raises(ValueError, match="register names"):
        program.read_distribution(result, registers=names)


@pytest.mark.parametrize(
    "state",
    [
        synth.InputState.basis(x=True, y=0),
        synth.InputState.basis(x=4, y=0),
        synth.InputState.basis(x=-1, y=0),
        synth.InputState.basis(x=1),
    ],
)
def test_bad_preparation_names_and_ranges_are_rejected(state: synth.InputState) -> None:
    with pytest.raises(ValueError, match="basis"):
        synth.synthesize(
            polynomial,
            inputs={"x": synth.UInt(2), "y": synth.UInt(2)},
            input_state=state,
        )


def test_resource_failure_can_be_repaired_without_changing_the_function() -> None:
    with pytest.raises(HybridVerificationError, match="increase max_qubits"):
        synth.synthesize(
            polynomial, inputs={"x": synth.UInt(2), "y": synth.UInt(2)}, max_qubits=17
        )
    program = synth.synthesize(
        polynomial, inputs={"x": synth.UInt(2), "y": synth.UInt(2)}, max_qubits=18
    )
    assert program.circuit.n_wires == 18
    with pytest.raises(HybridVerificationError, match="increase max_operations"):
        synth.synthesize(
            polynomial,
            inputs={"x": synth.UInt(2), "y": synth.UInt(2)},
            max_operations=161,
        )


def test_existing_preparation_ir_and_parameterized_circuit_composition() -> None:
    preparation = fq.Circuit(2).h(1).to_ir()
    program = synth.synthesize(
        identity,
        inputs={"x": synth.UInt(2)},
        preparation=preparation,
        preparation_map={"x": (1, 0)},
    )
    observed = program.read_distribution(
        fq.run(program.circuit, options=_options()), registers=("x", "result")
    )
    assert dict(observed) == pytest.approx({(0, 0): 0.5, (2, 2): 0.5})
    theta = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
    circuit = (
        fq.Circuit(program.circuit.n_wires, dtype=torch.complex128)
        .ry(1, theta)
        .compose(program.arithmetic)
    )
    value = circuit.expectation_ps(z=[program.registers["result"][-1]])
    gradient = torch.autograd.grad(value.sum(), theta)[0]
    torch.testing.assert_close(gradient, -torch.sin(theta), atol=1e-8, rtol=0)


def test_preparation_roles_and_symbolic_angles_are_not_silently_ignored() -> None:
    options: dict[str, Any] = {"inputs": {"x": synth.UInt(1)}}
    with pytest.raises(ValueError, match="not both"):
        synth.synthesize(
            identity,
            **options,
            input_state=synth.InputState.uniform(),
            preparation=fq.Circuit(1).to_ir(),
        )
    with pytest.raises(ValueError, match="preparation_map requires"):
        synth.synthesize(identity, **options, preparation_map={"x": (0,)})
    reset = CircuitIR(
        n_wires=1,
        instructions=(Instruction("reset", (0,), metadata={"is_dynamic": True}),),
    )
    with pytest.raises(ValueError, match="unitary"):
        synth.synthesize(
            identity, **options, preparation=reset, preparation_map={"x": (0,)}
        )
    symbolic = fq.Circuit(1).ry(0, Parameter("theta")).to_ir()
    with pytest.raises(ValueError, match="bind symbolic"):
        synth.synthesize(
            identity, **options, preparation=symbolic, preparation_map={"x": (0,)}
        )
    measured = replace(
        fq.Circuit(1).to_ir(), measurements=(MeasurementNode("probabilities", (0,)),)
    )
    with pytest.raises(ValueError, match="without measurements"):
        synth.synthesize(
            identity, **options, preparation=measured, preparation_map={"x": (0,)}
        )


def test_reader_refuses_density_and_invalid_state_data() -> None:
    program = _small()
    with pytest.raises(ValueError, match="statevectors"):
        program.read_distribution(
            fq.run(program.circuit, options=_options("density_matrix"))
        )
    result = fq.run(program.circuit, options=_options())
    assert result.state is not None
    with pytest.raises(ValueError, match="not normalized"):
        program.read_distribution(replace(result, state=result.state * 2))
    with pytest.raises(ValueError, match="matching size"):
        program.read_distribution(
            replace(result, state=torch.cat((result.state, result.state)))
        )


def test_partial_filtering_preserves_retained_probability() -> None:
    program = synth.synthesize(
        identity,
        inputs={"x": synth.UInt(1)},
        preparation=fq.Circuit(1).ry(0, 0.4).to_ir(),
        preparation_map={"x": (0,)},
    )
    observed = program.read_distribution(
        fq.run(program.circuit, options=_options()), threshold=0.05
    )
    assert set(observed) == {(0,)}
    assert abs(observed[(0,)] - math.cos(0.2) ** 2) < 1e-10
    assert abs(observed.diagnostics["omitted_probability"] - math.sin(0.2) ** 2) < 1e-10


def test_named_reader_uses_supplied_state_data_instead_of_predicting_the_function() -> (
    None
):
    program = _small()
    result = fq.run(program.circuit, options=_options())
    assert result.state is not None
    supplied = torch.zeros_like(result.state)
    supplied[..., 0] = 1
    observed = program.read_distribution(replace(result, state=supplied))
    assert dict(observed) == {(0,): 1.0}


def test_synthesis_is_construction_without_implicit_runtime_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("synthesis must not execute or plan a runtime")

    monkeypatch.setattr(fq, "run", forbidden)
    monkeypatch.setattr(fq, "plan", forbidden)
    assert _small().report["result_width"] == 1


@pytest.mark.parametrize("width", [True, 1.5, 0, -1, 257])
def test_structural_domains_reject_invalid_widths(width: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        synth.UInt(width)


def test_workflow_limits_and_preparation_cost_are_enforced() -> None:
    options: dict[str, Any] = {"inputs": {"x": synth.UInt(1)}}
    for limits in ({"max_qubits": 257}, {"max_operations": 100001}):
        with pytest.raises(ValueError, match="safety caps"):
            synth.synthesize(identity, **options, **limits)
    with pytest.raises(ValueError, match="exact result need"):
        synth.synthesize(identity, **options, max_qubits=1)
    with pytest.raises(ValueError, match="preparation plus arithmetic"):
        synth.synthesize(
            identity,
            **options,
            input_state=synth.InputState.uniform(),
            max_operations=1,
        )
    for invalid in (
        {"inputs": []},
        {"inputs": {"x": 1}},
        {"inputs": {"x": synth.UInt(1)}, "constants": []},
    ):
        with pytest.raises(TypeError, match="mapping|UInt"):
            synth.synthesize(identity, **invalid)


@pytest.mark.parametrize("mapping", [{"x": ()}, {"x": (1,)}, {"x": (True,)}, {}])
def test_existing_preparation_requires_a_complete_injective_map(
    mapping: dict[str, tuple[int, ...]],
) -> None:
    with pytest.raises(ValueError, match="map"):
        synth.synthesize(
            identity,
            inputs={"x": synth.UInt(1)},
            preparation=fq.Circuit(1).to_ir(),
            preparation_map=mapping,
        )


@pytest.mark.parametrize(
    "limits",
    [
        {"max_bits": 0},
        {"max_keys": True},
        {"max_amplitudes": 1.5},
        {"threshold": float("nan")},
        {"threshold": True},
        {"threshold": -1},
    ],
)
def test_invalid_read_limits_fail_before_state_reduction(
    limits: dict[str, Any],
) -> None:
    program = _small()
    result = fq.run(program.circuit, options=_options())
    with pytest.raises(ValueError, match="positive|threshold"):
        program.read_distribution(result, **limits)

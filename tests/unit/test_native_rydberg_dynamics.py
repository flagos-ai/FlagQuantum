"""Native input consistency, source-aware replay and existing engine reuse."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import replace

import pytest
import torch

import flagquantum as fq
import flagquantum.lindblad as fql
from examples.rydberg_native_dynamics import native_pair_plan
from flagquantum.errors import CapabilityError, SerializationError
from flagquantum.experimental.simulation import RydbergProgram
from flagquantum.lindblad._plan import _identity
from flagquantum.simulation.lindblad import EvolutionValidationError

pytestmark = pytest.mark.unit


def atom() -> RydbergProgram:
    return RydbergProgram(
        atoms=(("a", "trap"),),
        sites={"trap": (0.0, 0.0)},
        length_unit="um",
        time_unit="us",
        c6=0.0,
        loss_rate=0.0,
        minimum_separation=1.0,
        drive_channels=("laser",),
    )


def pulse() -> RydbergProgram:
    p = atom()
    p.pulse("a", start=0.0, duration=0.5, amplitude=0.7, channel="laser")
    return p


@pytest.mark.parametrize("raw_record", (False, True))
def test_native_inputs_preserve_plan_source_and_product_result(
    raw_record: bool,
) -> None:
    p = pulse()
    plan = fql.plan(p.to_ir() if raw_record else p, p.basis_state(("g",)), [0.0, 0.5])
    payload = plan.to_dict()
    assert set(payload) == {
        "schema",
        "version",
        "request",
        "decision",
        "program",
        "bindings",
        "identity",
    }
    assert payload["version"] == "1.7" and payload["bindings"] is None
    assert payload["program"]["schema"] == "flagquantum.program_ir"
    assert plan.dimensions == (3,)
    assert fql.run(plan).populations.shape == (2, 3)
    with pytest.raises(CapabilityError):
        _ = plan.n_qubits


@pytest.mark.parametrize(
    "times", [[], [0.0], [-1.0, 0.5], [0.1, 0.5], [0.0, 0.25], [0.0, 0.5, 0.5]]
)
def test_native_grids_refuse_inconsistency_with_the_existing_error_category(
    times: list[float],
) -> None:
    p = pulse()
    with pytest.raises(EvolutionValidationError) as caught:
        fql.plan(p, p.basis_state(("g",)), times)
    assert (
        caught.value.code == "invalid_native_request" and caught.value.field == "times"
    )


@pytest.mark.parametrize(
    "name,value",
    [
        ("n_qubits", 1),
        ("dimensions", (2,)),
        ("basis_labels", (("g",), ("r",), ("lost",))),
        ("collapse_operators", []),
    ],
)
def test_native_semantic_overrides_fail_explicitly(name: str, value: object) -> None:
    p = pulse()
    with pytest.raises(EvolutionValidationError) as caught:
        fql.plan(p, p.basis_state(("g",)), [0.0, 0.5], **{name: value})
    assert caught.value.code == "invalid_native_request" and caught.value.field == name


def test_matching_dimensions_and_float32_boundaries_use_the_existing_precision() -> (
    None
):
    p = atom()
    p.pulse("a", start=0.1, duration=0.1, amplitude=0.7, channel="laser")
    result = fql.run(
        p,
        p.basis_state(("g",)),
        [0.0, 0.1, 0.2],
        dimensions=(3,),
        options=fq.ExecutionOptions(precision="complex64"),
    )
    assert result.populations.dtype == torch.float32
    assert result.populations[-1, 1].item() == pytest.approx(
        math.sin(0.035) ** 2, abs=1e-6
    )


def test_distinct_source_boundaries_cannot_collapse_in_float32() -> None:
    p = atom()
    p.pulse("a", start=0.0, duration=0.1, amplitude=0.7, channel="laser")
    p.pulse("a", start=0.100000001, duration=0.1, amplitude=0.7, channel="laser")
    with pytest.raises(EvolutionValidationError) as caught:
        fql.plan(
            p,
            p.basis_state(("g",)),
            [0.0, 0.1, 0.200000001],
            options=fq.ExecutionOptions(precision="complex64"),
        )
    assert caught.value.code == "invalid_native_request"


def test_a_boundary_cannot_ambiguously_match_two_output_points() -> None:
    p = atom()
    p.pulse("a", start=0.1, duration=0.1, amplitude=0.7, channel="laser")
    t = torch.tensor(0.1, dtype=torch.float32)
    near = torch.nextafter(t, torch.tensor(math.inf)).item()
    with pytest.raises(EvolutionValidationError):
        fql.plan(
            p,
            p.basis_state(("g",)),
            [0.0, t.item(), near, 0.2],
            options=fq.ExecutionOptions(precision="complex64"),
        )


def test_native_time_gradients_are_refused_before_detachment() -> None:
    p = pulse()
    times = torch.tensor([0.0, 0.5], dtype=torch.float64, requires_grad=True)
    with pytest.raises(EvolutionValidationError) as caught:
        fql.plan(p, p.basis_state(("g",)), times)
    assert caught.value.code == "unsupported_time_gradient"


@pytest.mark.parametrize(
    "options",
    [
        fq.ExecutionOptions(shots=10),
        fq.ExecutionOptions(seed=1),
        fq.ExecutionOptions(memory_limit_bytes=1000),
        fq.ExecutionOptions(allow_approximate=False),
        fq.ExecutionOptions(allow_backend_fallback=False),
    ],
)
def test_native_dispatch_retains_the_existing_option_refusals(
    options: fq.ExecutionOptions,
) -> None:
    p = pulse()
    with pytest.raises(CapabilityError):
        fql.plan(p, p.basis_state(("g",)), [0.0, 0.5], options=options)


def test_larger_sources_refuse_before_operator_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    p = RydbergProgram(
        atoms=(("a", "s0"), ("b", "s1"), ("c", "s2")),
        sites={"s0": (0.0, 0.0), "s1": (2.0, 0.0), "s2": (4.0, 0.0)},
        length_unit="um",
        time_unit="us",
        c6=1.0,
        loss_rate=0.0,
        minimum_separation=1.0,
    )

    def no_allocation(*args: object, **kwargs: object) -> None:
        raise AssertionError("exponential operator allocation")

    monkeypatch.setattr(torch, "kron", no_allocation)
    with pytest.raises(CapabilityError, match="one or two"):
        fql.plan(p, [1.0], [0.0, 1.0])


@pytest.mark.parametrize(
    "change",
    [
        "top",
        "source",
        "hamiltonian",
        "jump",
        "tensor_field",
        "decision",
        "boolean_time",
    ],
)
def test_rehashed_inconsistent_or_unknown_native_payloads_are_refused(
    change: str,
) -> None:
    payload = native_pair_plan(intervals=4).to_dict()
    if change == "top":
        payload["unknown"] = True
    elif change == "source":
        payload["program"]["physical_context"]["c6"] *= 2
    elif change == "hamiltonian":
        payload["request"]["hamiltonians"][0]["values"][4][4][0] += 0.01
    elif change == "jump":
        payload["request"]["collapse_operators"][0]["rate"] += 0.1
    elif change == "tensor_field":
        payload["request"]["hamiltonians"][0]["hidden"] = 1
    elif change == "decision":
        payload["decision"]["numerics"]["unknown"] = 1
    else:
        payload["request"]["times"][0] = False
    payload["identity"] = _identity(payload)
    with pytest.raises(SerializationError):
        fql.LindbladPlan.from_dict(payload)


def test_bindings_are_real_snapshots_and_rehashed_changes_cannot_reuse_matrices() -> (
    None
):
    p = atom()
    p.pulse(
        "a", start=0.0, duration=0.5, amplitude=fq.Parameter("omega"), channel="laser"
    )
    values = [{fq.Parameter("omega"): torch.tensor(0.7, dtype=torch.float64)}]
    plan = fql.plan(p, p.basis_state(("g",)), [0.0, 0.5], parameter_values=values)
    assert plan.to_dict()["bindings"] == [{"omega": 0.7}]
    values[0][fq.Parameter("omega")] = torch.tensor(0.9)
    payload = plan.to_dict()
    payload["bindings"][0]["omega"] = 0.9
    payload["identity"] = _identity(payload)
    with pytest.raises(SerializationError):
        fql.LindbladPlan.from_dict(payload)
    assert plan.to_dict()["bindings"] == [{"omega": 0.7}]


def test_sealed_live_and_json_execution_do_not_recompile_the_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = native_pair_plan(intervals=4)
    restored = fql.LindbladPlan.from_json(plan.to_json())
    from flagquantum.lindblad import _native_plan

    def no_recompile(*args: object, **kwargs: object) -> None:
        raise AssertionError("source recompiled during sealed execution")

    monkeypatch.setattr(_native_plan, "native_model", no_recompile)
    a, b = fql.run(plan), fql.run(restored)
    assert torch.equal(a.populations, b.populations)
    assert torch.equal(a.density_matrices, b.density_matrices)
    changed = deepcopy(plan.to_dict())
    changed["program"]["physical_context"]["c6"] = 0.0
    assert plan.to_dict() != changed


@pytest.mark.parametrize("method", fql.INTEGRATOR_NAMES)
def test_all_three_controls_match_an_independent_dense_unitary_and_gradient(
    method: str,
) -> None:
    p = atom()
    w, d, phase = [
        torch.tensor(x, dtype=torch.float64, requires_grad=True)
        for x in (0.6, 0.2, 0.4)
    ]
    p.pulse(
        "a",
        start=0.0,
        duration=0.5,
        amplitude=fq.Parameter("w") + 0.1,
        detuning=fq.Parameter("d"),
        phase=fq.Parameter("phase"),
        channel="laser",
    )
    x = torch.zeros((3, 3), dtype=torch.complex128)
    x[0, 1] = x[1, 0] = 0.5
    result = fql.run(
        p,
        p.basis_state(("g",)),
        torch.linspace(0.0, 0.5, 201, dtype=torch.float64),
        parameter_values=[{"w": w, "d": d, "phase": phase}] * 200,
        outputs=(
            fq.expectation(x, name="x"),
            fq.expectation(p.projector({"a": "r"}), name="r"),
        ),
        options=fq.ExecutionOptions(require_gradients=True),
        method=method,
    )
    actual = result.expectation("x")[-1] + 0.4 * result.expectation("r")[-1]
    gradient = torch.autograd.grad(actual, (w, d, phase))
    wr, dr, pr = [v.detach().requires_grad_(True) for v in (w, d, phase)]
    drive = (wr + 0.1) * torch.exp(1j * pr) / 2
    zero = torch.zeros((), dtype=torch.complex128)
    h = torch.stack(
        (
            torch.stack((zero, drive, zero)),
            torch.stack((drive.conj(), -dr.to(torch.complex128), zero)),
            torch.stack((zero, zero, zero)),
        )
    )
    state = torch.matrix_exp(-0.5j * h)[:, 0]
    expected = (state[0] * state[1].conj()).real + 0.4 * state[1].abs().square()
    reference = torch.autograd.grad(expected, (wr, dr, pr))
    tolerance = 1e-6 if method == "crank-nicolson" else 1e-10
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=0)
    for a, b in zip(gradient, reference, strict=True):
        torch.testing.assert_close(a, b, atol=tolerance, rtol=0)


def test_float32_binding_capture_has_no_implicit_float64_contract() -> None:
    p = atom()
    p.pulse(
        "a", start=0.0, duration=0.5, amplitude=fq.Parameter("omega"), channel="laser"
    )
    plan = fql.plan(
        p,
        p.basis_state(("g",)),
        [0.0, 0.5],
        parameter_values=[{"omega": 0.7}],
        options=fq.ExecutionOptions(precision="complex64"),
    )
    assert plan.to_dict()["bindings"] == [
        {"omega": torch.tensor(0.7, dtype=torch.float32).item()}
    ]
    restored = fql.LindbladPlan.from_json(plan.to_json())
    assert torch.equal(fql.run(plan).populations, fql.run(restored).populations)


def test_low_level_mathematical_api_does_not_gain_native_dispatch() -> None:
    from flagquantum.simulation.lindblad import evolve_density_matrix

    p = pulse()
    with pytest.raises(EvolutionValidationError):
        evolve_density_matrix(
            p, p.basis_state(("g",)), None, [0.0, 0.5], dimensions=(3,)
        )


@pytest.mark.parametrize(
    "mapping,code",
    [
        ({}, "invalid_parameter_binding"),
        ({"omega": 0.7, "unused": 1.0}, "invalid_parameter_binding"),
        ({"omega": 0.7, fq.Parameter("omega"): 0.8}, "invalid_parameter_binding"),
        ({"omega": [0.7]}, "invalid_coefficient"),
        ({"omega": True}, "invalid_coefficient"),
        ({"omega": complex(0.7, 0.0)}, "invalid_coefficient"),
        ({"omega": float("inf")}, "invalid_coefficient"),
    ],
)
def test_native_bindings_reuse_existing_validation_categories(
    mapping: dict[object, object], code: str
) -> None:
    p = atom()
    p.pulse(
        "a", start=0.0, duration=0.5, amplitude=fq.Parameter("omega"), channel="laser"
    )
    with pytest.raises(EvolutionValidationError) as caught:
        fql.plan(p, p.basis_state(("g",)), [0.0, 0.5], parameter_values=[mapping])
    assert caught.value.code == code


def test_live_plan_snapshots_source_and_binding_but_json_restores_no_graph() -> None:
    p = atom()
    p.pulse(
        "a", start=0.0, duration=0.5, amplitude=fq.Parameter("omega"), channel="laser"
    )
    omega = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    plan = fql.plan(
        p,
        p.basis_state(("g",)),
        torch.linspace(0.0, 0.5, 201, dtype=torch.float64),
        parameter_values=[{"omega": omega}] * 200,
        method="runge-kutta",
        options=fq.ExecutionOptions(require_gradients=True),
    )
    sealed = plan.to_json()
    with torch.no_grad():
        omega.add_(0.2)
    p.pulse("a", start=0.5, duration=0.5, amplitude=0.1, channel="laser")
    assert plan.to_json() == sealed
    population = fql.run(plan).populations[-1, 1]
    assert population.item() == pytest.approx(math.sin(0.175) ** 2, abs=1e-6)
    assert torch.autograd.grad(population, omega)[0].item() == pytest.approx(
        0.25 * math.sin(0.35), abs=1e-6
    )
    with pytest.raises(EvolutionValidationError) as caught:
        fql.run(fql.LindbladPlan.from_json(sealed))
    assert caught.value.code == "gradients_not_available"


def test_native_reconstruction_budget_is_encoding_not_a_physics_tolerance() -> None:
    payload = native_pair_plan(intervals=4).to_dict()
    epsilon = torch.finfo(torch.float64).eps
    payload["request"]["hamiltonians"][0]["values"][0][0][0] += 8 * epsilon
    payload["identity"] = _identity(payload)
    fql.LindbladPlan.from_dict(payload)
    payload["request"]["hamiltonians"][0]["values"][0][0][0] += 32 * epsilon
    payload["identity"] = _identity(payload)
    with pytest.raises(SerializationError):
        fql.LindbladPlan.from_dict(payload)


def test_real_bindings_are_classified_without_default_integer_or_float_coercion() -> (
    None
):
    p = atom()
    p.pulse(
        "a", start=0.0, duration=0.5, amplitude=fq.Parameter("omega"), channel="laser"
    )
    # Finite at the requested real precision, outside the default int64 range.
    value = 2**70
    plan = fql.plan(
        p, p.basis_state(("g",)), [0.0, 0.5], parameter_values=[{"omega": value}]
    )
    assert plan.to_dict()["bindings"] == [{"omega": float(value)}]
    assert plan.to_dict()["request"]["hamiltonians"][0]["values"][0][1][0] == value / 2


def test_numeric_static_source_keeps_existing_empty_mapping_semantics() -> None:
    p = atom()
    initial = p.basis_state(("g",))
    plain = fql.plan(p, initial, [0.0, 0.5])
    empty = fql.plan(p, initial, [0.0, 0.5], parameter_values=[{}])
    assert plain.to_json() == empty.to_json()
    assert empty.to_dict()["version"] == "1.7"
    assert empty.to_dict()["bindings"] is None
    with pytest.raises(EvolutionValidationError) as caught:
        fql.plan(p, initial, [0.0, 0.5], parameter_values=[{"unused": 0.7}])
    assert caught.value.code == "invalid_parameter_binding"


def test_simultaneous_pair_drive_matches_dense_unitary_and_rydberg_blockade() -> None:
    double_excitation = []
    for c6 in (0.0, 640.0):
        p = RydbergProgram(
            atoms=(("a", "left"), ("b", "right")),
            sites={"left": (0.0, 0.0), "right": (2.0, 0.0)},
            length_unit="um",
            time_unit="us",
            c6=c6,
            loss_rate=0.0,
            minimum_separation=1.0,
            drive_channels=("laser",),
        )
        p.pulse(("a", "b"), start=0.0, duration=2.0, amplitude=1.0, channel="laser")
        result = fql.run(
            p,
            p.basis_state(("g", "g")),
            torch.linspace(0.0, 2.0, 81, dtype=torch.float64),
            method="krylov-exponential",
            return_density_matrices=True,
        )
        # Independent product Hamiltonian in the declared g/r/lost coordinates.
        x = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.complex128
        )
        n = torch.diag(torch.tensor([0.0, 1.0, 0.0], dtype=torch.complex128))
        eye = torch.eye(3, dtype=torch.complex128)
        h = 0.5 * (torch.kron(x, eye) + torch.kron(eye, x)) + c6 / 2**6 * torch.kron(
            n, n
        )
        state = torch.matrix_exp(-2j * h)[:, 0]
        assert result.density_matrices is not None
        torch.testing.assert_close(
            result.density_matrices[-1],
            state[:, None] * state.conj()[None, :],
            atol=1e-10,
            rtol=0,
        )
        double_excitation.append(result.populations[-1, 4].item())
    assert double_excitation[0] == pytest.approx(math.sin(1.0) ** 4, abs=1e-10)
    assert double_excitation[1] < double_excitation[0] / 10


@pytest.mark.parametrize("opcode", ["rydberg.measure_presence", "rydberg.unknown"])
def test_unconditional_assembly_refuses_observations_before_grid_work(
    opcode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from flagquantum.simulation import rydberg_model

    record = pulse().to_ir()
    function = record.functions[0]
    block = function.blocks[0]
    observation = replace(block.operations[0], opcode=opcode)
    changed = replace(
        record,
        functions=(
            replace(
                function,
                blocks=(replace(block, operations=(*block.operations, observation)),),
            ),
        ),
    )

    def forbidden_grid(*args: object, **kwargs: object) -> None:
        pytest.fail("an unadmitted observation reached numerical grid assembly")

    monkeypatch.setattr(rydberg_model, "_grid", forbidden_grid)
    with pytest.raises(CapabilityError, match="conditional trajectory"):
        rydberg_model.native_model(
            changed, [0.0, 0.5], dtype=torch.complex128, device="cpu"
        )

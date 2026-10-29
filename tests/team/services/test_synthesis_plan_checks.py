"""Conformance against live and recorded plans, plus fail-closed consistency."""

from dataclasses import replace
from typing import Any

import pytest
import torch

import flagquantum as fq
from flagquantum.compiler import compile as compile_program
from flagquantum.compiler._construction.constraints import CircuitLimits
from flagquantum.compiler.coupling import CouplingMap
from flagquantum.compiler.layout import remove_layout_restore
from flagquantum.core.ir import CircuitIR, MeasurementNode
from flagquantum.runtime.execution_plan import ExecutionPlan
from flagquantum.runtime.execution_plan_contract import plan_execution_program
from flagquantum.services._synthesis_execution import (
    ExecutionContractError,
    inspect_execution,
)

pytestmark = pytest.mark.unit

TOPOLOGY = CouplingMap(3, ((0, 1), (1, 2)))


def _options(**changes: Any) -> fq.ExecutionOptions:
    return replace(
        fq.ExecutionOptions(
            mode="statevector",
            backend="torch",
            device="cpu",
            precision="complex128",
            require_gradients=False,
            allow_approximate=False,
            allow_backend_fallback=False,
        ),
        **changes,
    )


def _routed() -> CircuitIR:
    source = fq.Circuit(3, dtype=torch.complex128).h(0).cx(0, 2)
    return remove_layout_restore(
        compile_program(
            source,
            coupling_map=TOPOLOGY,
            routing_strategy="persistent_layout",
            optimize=False,
        )
    )


@pytest.mark.parametrize("provider", ["runtime", "recorded_replay_fake"])
def test_same_consumer_accepts_live_and_recorded_plan_conformance(
    provider: str,
) -> None:
    source = _routed()
    live = fq.plan(source, options=_options())
    recorded_json = live.to_json()

    def replay_fake(program: CircuitIR) -> ExecutionPlan:
        # A fixed recorded transport fake, not a second planner or physics oracle.
        assert program.content_hash == source.content_hash
        return ExecutionPlan.from_json(recorded_json)

    plan = (
        fq.plan(source, options=_options())
        if provider == "runtime"
        else replay_fake(source)
    )
    checked = inspect_execution(
        plan,
        limits=CircuitLimits(max_qubits=3, max_depth=3, max_operations=3),
        basis=("h", "swap", "cx"),
        coupling=TOPOLOGY,
        registers={"x": (0,), "result": (2,)},
    )
    assert checked.registers == {"x": (1,), "result": (2,)}
    assert checked.resources.n_operations == 3
    assert checked.resources.depth == 3
    assert checked.plan.identity == live.identity


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_qubits", True),
        ("max_qubits", 0),
        ("max_qubits", 1.5),
        ("max_depth", -1),
        ("max_operations", torch.tensor(3.0, requires_grad=True)),
    ],
)
def test_limits_preserve_discrete_categories(field: str, value: object) -> None:
    with pytest.raises(ValueError, match="plain integer"):
        CircuitLimits(**{field: value})


def test_final_limits_report_every_exceeded_resource() -> None:
    plan = fq.plan(_routed(), options=_options())
    with pytest.raises(ExecutionContractError) as error:
        inspect_execution(
            plan,
            coupling=TOPOLOGY,
            limits=CircuitLimits(max_qubits=2, max_operations=2, max_depth=2),
        )
    assert error.value.code == "execution.constraints"
    assert all(
        name in str(error.value)
        for name in ("max_qubits=2", "max_operations=2", "max_depth=2")
    )
    checked = inspect_execution(
        plan,
        coupling=TOPOLOGY,
        limits=CircuitLimits(max_qubits=3, max_operations=3, max_depth=3),
    )
    assert checked.resources.n_qubits == 3


def test_zero_operation_and_depth_limits_are_valid_for_an_empty_program() -> None:
    plan = fq.plan(fq.Circuit(1), options=_options())
    checked = inspect_execution(
        plan, limits=CircuitLimits(max_operations=0, max_depth=0)
    )
    assert checked.resources.n_operations == checked.resources.depth == 0


def test_resources_are_fresh_and_named_ports_are_immutable() -> None:
    plan = fq.plan(fq.Circuit(1).h(0), options=_options())
    checked = inspect_execution(plan, registers={"x": (0,)})
    checked.resources.operation_counts["h"] = 999
    assert checked.resources.operation_counts == {"h": 1}
    with pytest.raises(TypeError):
        checked.registers["x"] = (1,)


def test_stricter_policy_cannot_reuse_an_old_resource_or_target_claim() -> None:
    checked = inspect_execution(
        fq.plan(_routed(), options=_options()), coupling=TOPOLOGY
    )
    for changed in (
        replace(checked, limits=CircuitLimits(max_operations=2)),
        replace(checked, basis=("h", "cx")),
        replace(checked, coupling=CouplingMap(3, ((0, 1),))),
    ):
        with pytest.raises(ExecutionContractError):
            _ = changed.resources


def test_allocated_width_is_not_replaced_by_used_qubit_count() -> None:
    plan = fq.plan(fq.Circuit(4).h(0), options=_options())
    with pytest.raises(ExecutionContractError, match="max_qubits=1, actual=4"):
        inspect_execution(plan, limits=CircuitLimits(max_qubits=1))
    assert inspect_execution(plan).resources.used_qubits == 1


@pytest.mark.parametrize("damage", ["missing", "backend", "precision", "schema"])
def test_raw_runtime_configuration_must_agree_with_the_sealed_decision(
    damage: str,
) -> None:
    plan = fq.plan(fq.Circuit(1), options=_options())
    config = dict(plan.runtime_config)
    if damage == "backend":
        config["backend"] = "jax"
    elif damage == "precision":
        config.update(
            complex_dtype="complex64", real_dtype="float32", jax_enable_x64=False
        )
    elif damage == "schema":
        config["schema"] = "unrecognized"
    with pytest.raises(ExecutionContractError, match="execution.controls"):
        inspect_execution(
            replace(plan, runtime_config=None if damage == "missing" else config)
        )


@pytest.mark.parametrize(
    "registers",
    [
        {"": (0,)},
        {"x": ()},
        {"x": (-1,)},
        {"x": (True,)},
        {"x": (3,)},
        {"x": "0"},
        {"x": (0,), "y": (0,)},
    ],
)
def test_named_ports_require_disjoint_bounded_integer_bindings(registers: Any) -> None:
    plan = fq.plan(_routed(), options=_options())
    with pytest.raises(ExecutionContractError, match="execution.registers"):
        inspect_execution(plan, registers=registers, coupling=TOPOLOGY)


def test_unsupported_arguments_fail_before_treating_them_as_contracts() -> None:
    plan = fq.plan(fq.Circuit(1), options=_options())
    with pytest.raises(TypeError, match="ExecutionPlan"):
        inspect_execution(object())
    with pytest.raises(TypeError, match="CircuitLimits"):
        inspect_execution(plan, limits={})
    with pytest.raises(TypeError, match="registers"):
        inspect_execution(plan, registers=[])
    with pytest.raises(TypeError):
        inspect_execution(plan, max_workspace=0)


@pytest.mark.parametrize(
    "changes",
    [
        {"mode": "density_matrix"},
        {"allow_approximate": True},
        {"allow_backend_fallback": True},
        {"batch_size": 2},
    ],
)
def test_unadmitted_execution_profiles_are_not_silently_accepted(
    changes: dict[str, Any],
) -> None:
    plan = fq.plan(
        fq.Circuit(1, bsz=changes.get("batch_size", 1)).h(0),
        options=_options(**changes),
    )
    with pytest.raises(ExecutionContractError, match="execution.profile"):
        inspect_execution(plan)


def test_output_projections_are_outside_this_full_state_contract() -> None:
    source = replace(
        fq.Circuit(1).h(0).to_ir(),
        measurements=(MeasurementNode("probabilities", (0,)),),
    )
    plan = fq.plan(source, options=_options())
    with pytest.raises(ExecutionContractError, match="projections"):
        inspect_execution(plan)


def test_changed_raw_controls_cannot_hide_behind_the_sealed_payload() -> None:
    plan = fq.plan(fq.Circuit(1).h(0), options=_options())
    checked = inspect_execution(plan)
    assert plan.runtime_config is not None
    plan.runtime_config["drawing_style"] = "altered"
    with pytest.raises(ExecutionContractError, match="execution.identity"):
        _ = checked.resources
    with pytest.raises(ExecutionContractError, match="allocation"):
        inspect_execution(replace(plan, state_bytes=plan.state_bytes + 1))
    with pytest.raises(ExecutionContractError, match="analysis dimensions"):
        inspect_execution(replace(plan, analysis=replace(plan.analysis, n_wires=2)))
    with pytest.raises(ExecutionContractError, match="execution.profile"):
        inspect_execution(replace(plan, state_mode="density_matrix"))


def test_layer_order_and_gate_content_must_match_the_canonical_payload() -> None:
    plan = fq.plan(fq.Circuit(2).h(0).cx(0, 1), options=_options())
    assert len(plan.layers) == 2
    with pytest.raises(ExecutionContractError, match="layer order"):
        inspect_execution(replace(plan, layers=tuple(reversed(plan.layers))))
    layer = plan.layers[0]
    with pytest.raises(ExecutionContractError, match="qubit summary"):
        inspect_execution(
            replace(plan, layers=(replace(layer, wires=()), *plan.layers[1:]))
        )
    changed_gate = replace(layer.instructions[0], name="x")
    with pytest.raises(ExecutionContractError, match="instruction layers"):
        inspect_execution(
            replace(
                plan,
                layers=(replace(layer, instructions=(changed_gate,)), *plan.layers[1:]),
            )
        )


@pytest.mark.parametrize(
    "damage",
    [
        "final",
        "pre_restore",
        "initial",
        "count",
        "schema",
        "strategy",
        "restored",
        "edges",
    ],
)
def test_corrupted_routing_records_do_not_authorize_named_mapping(damage: str) -> None:
    source = _routed()
    metadata = dict(source.metadata)
    routing = dict(metadata["routing"])
    changes = {
        "final": {"final_logical_to_physical": (0, 2, 1)},
        "pre_restore": {"pre_restore_logical_to_physical": (0, 1, 2)},
        "initial": {"initial_logical_to_physical": (1, 0, 2)},
        "count": {"inserted_swap_count": 7, "post_optimization_inserted_swap_count": 7},
        "schema": {"schema": "unrecognized-route"},
        "strategy": {"strategy": "unrecognized-strategy"},
        "restored": {"mapping_restored": True},
        "edges": {"coupling_edges": ((0, 2), (1, 2))},
    }
    routing.update(changes[damage])
    metadata["routing"] = routing
    plan = fq.plan(replace(source, metadata=metadata), options=_options())
    with pytest.raises(
        ExecutionContractError, match="execution.layout|execution.topology"
    ):
        inspect_execution(plan, coupling=TOPOLOGY)


def test_missing_routing_witness_and_topology_are_refused() -> None:
    source = _routed()
    plan = fq.plan(source, options=_options())
    with pytest.raises(ExecutionContractError, match="explicit topology"):
        inspect_execution(plan)
    metadata = dict(source.metadata)
    del metadata["routing"]
    missing = fq.plan(replace(source, metadata=metadata), options=_options())
    with pytest.raises(ExecutionContractError, match="no corresponding"):
        inspect_execution(missing, coupling=TOPOLOGY)
    with pytest.raises(ExecutionContractError, match="exactly"):
        inspect_execution(plan, coupling=CouplingMap(2, ((0, 1),)))
    unrouted = fq.plan(fq.Circuit(3).cx(0, 2), options=_options())
    with pytest.raises(ExecutionContractError, match="no admitted"):
        inspect_execution(unrouted, coupling=TOPOLOGY)


def test_runtime_routing_controls_cannot_disagree_with_program_evidence() -> None:
    plan = fq.plan(_routed(), options=_options())
    assert plan.routing_plan is not None
    plan.routing_plan["mapping_restored"] = True
    with pytest.raises(ExecutionContractError, match="controls differ"):
        inspect_execution(plan, coupling=TOPOLOGY)


def test_inspection_never_plans_compiles_or_executes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = fq.plan(_routed(), options=_options())

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("inspection cannot invoke a planner, compiler or executor")

    monkeypatch.setattr(fq, "plan", forbidden)
    monkeypatch.setattr(fq, "run", forbidden)
    import flagquantum.compiler as compiler

    monkeypatch.setattr(compiler, "compile", forbidden)
    checked = inspect_execution(plan, coupling=TOPOLOGY)
    assert checked.resources == compiler.estimate_resources(
        plan_execution_program(plan)
    )

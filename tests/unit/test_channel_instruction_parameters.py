"""A channel written into a circuit declares the parameter its factory needs.

`fq.Circuit(1).depolarizing(0, 0.1)` used to raise
``TypeError: depolarizing accepts 1 qubit(s) and 0 parameter(s)``: the channel
schemas declared no parameters, so the public gate method had nowhere to put the
probability, even though `flagquantum.noise` already had the factory, the
`KrausChannel` already carried the value under its own name, and the serialized
plan already recorded it.

The defect was in the entry point, not the machinery, so these tests pin the
entry point: the declared name, the refusal for a name or a value that does not
belong to the channel, the marker the rest of the runtime keys on, the round trip
through the IR, the route the planner picks, and the number the channel actually
produces. The numerics are checked against the analytic value of the channel
rather than against a recorded run, so a wrong Kraus operator cannot pass by
matching an earlier wrong answer.
"""

from __future__ import annotations

import json

import pytest
import torch

import flagquantum as fq
from flagquantum.compiler import lower_noise_model
from flagquantum.core.ir import CircuitIR, Instruction, IRValidationError
from flagquantum.core.operator_schema import OPERATOR_SCHEMAS
from flagquantum.errors import ValidationError
from flagquantum.noise import NoiseModel, channel_from_parameters, depolarizing_channel
from flagquantum.runtime.execution_plan_contract import plan_from_dict, plan_to_json

pytestmark = pytest.mark.unit

# A channel opcode, the name its factory takes, and a value inside its range.
CHANNELS = (
    ("bit_flip", "probability", 0.25),
    ("phase_flip", "probability", 0.25),
    ("depolarizing", "probability", 0.1),
    ("amplitude_damping", "gamma", 0.2),
)


def _ghz(n_qubits: int) -> fq.Circuit:
    """A chain of CNOTs on a Hadamard: the state whose correlations a channel
    breaks, and one an entangled bond of dimension two already represents."""

    circuit = fq.Circuit(n_qubits).h(0)
    for qubit in range(n_qubits - 1):
        circuit.cx(qubit, qubit + 1)
    return circuit


@pytest.mark.parametrize(("opcode", "parameter", "value"), CHANNELS)
def test_the_schema_declares_the_name_the_factory_takes(
    opcode: str, parameter: str, value: float
) -> None:
    """The declared name is the keyword a user writes, so the two must agree."""

    assert OPERATOR_SCHEMAS[opcode].parameters == (parameter,)

    channel = channel_from_parameters(opcode, {parameter: value})

    assert dict(channel.parameters)[parameter] == pytest.approx(value, abs=1e-6)


@pytest.mark.parametrize(("opcode", "parameter", "value"), CHANNELS)
def test_the_gate_method_accepts_the_value_by_position_and_by_name(
    opcode: str, parameter: str, value: float
) -> None:
    """Both spellings must reach the same instruction, which is what makes the
    declared name real rather than documentation."""

    from_position = getattr(fq.Circuit(1), opcode)(0, value)
    from_keyword = getattr(fq.Circuit(1), opcode)(0, **{parameter: value})

    assert from_position.to_ir().instructions[0].params == pytest.approx(
        from_keyword.to_ir().instructions[0].params, abs=1e-6
    )


def test_a_channel_value_is_refused_under_another_channel_name() -> None:
    """`probability` names nothing on the damping channel, which takes `gamma`."""

    with pytest.raises(
        ValidationError,
        match=r"amplitude_damping does not accept parameter\(s\): probability; "
        r"accepted: gamma",
    ):
        fq.Circuit(1).amplitude_damping(0, probability=0.1)


def test_a_missing_channel_value_is_refused() -> None:
    """A channel with no probability is not a noiseless channel; it is unusable."""

    with pytest.raises(
        ValidationError, match=r"depolarizing requires parameter\(s\): probability"
    ):
        fq.Circuit(1).depolarizing(0)


def test_an_undeclared_channel_keyword_is_refused() -> None:
    with pytest.raises(
        ValidationError,
        match=r"depolarizing does not accept parameter\(s\): rate; "
        r"accepted: probability",
    ):
        fq.Circuit(1).depolarizing(0, rate=0.1)


def test_an_unknown_channel_opcode_names_the_ones_that_exist() -> None:
    with pytest.raises(ValidationError, match=r"unknown channel 'nope'"):
        channel_from_parameters("nope", {})


@pytest.mark.parametrize("value", [-0.1, 1.5])
def test_a_probability_outside_the_unit_interval_is_refused(value: float) -> None:
    """A number with no physical meaning is refused where its meaning is known.

    The registry rule only asks for a finite real number, because a rotation
    angle may be any real number; the ``[0, 1]`` limit belongs to the channel.
    """

    with pytest.raises(ValidationError, match=r"probability must be between 0 and 1"):
        fq.Circuit(1).depolarizing(0, value)


@pytest.mark.parametrize("value", ["0.1", True])
def test_a_value_that_is_not_a_real_number_is_refused(value: object) -> None:
    """A string reached the tensor constructor naming neither the channel nor the
    parameter, and ``True`` was read as the probability one."""

    with pytest.raises(
        TypeError, match=r"parameter 'probability' must be a real number"
    ):
        fq.Circuit(1).depolarizing(0, value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_a_non_finite_probability_is_refused(value: float) -> None:
    with pytest.raises(IRValidationError, match=r"must be a finite real number"):
        fq.Circuit(1).depolarizing(0, value)


def test_a_matrix_for_a_channel_is_refused_rather_than_ignored() -> None:
    """The parameters define the operators, so a second definition is a conflict."""

    with pytest.raises(
        ValidationError,
        match=r"derives its Kraus operators from its declared parameters",
    ):
        fq.Circuit(1).gate("depolarizing", (0,), matrix=depolarizing_channel(0.1).kraus)


def test_a_second_positional_value_is_refused() -> None:
    """One qubit and one probability is the whole signature."""

    with pytest.raises(TypeError, match=r"accepts 1 qubit\(s\) and 1 parameter\(s\)"):
        fq.Circuit(1).depolarizing(0, 0.1, 0.2)


def test_the_instruction_carries_the_value_and_the_channel_marker() -> None:
    """The runtime keys on both: the value to build the operators, the marker to
    know they are operators rather than a unitary."""

    instruction = fq.Circuit(1).depolarizing(0, 0.1).to_ir().instructions[0]

    assert instruction.params["probability"] == pytest.approx(0.1, abs=1e-6)
    assert instruction.metadata["is_channel"] is True
    assert len(instruction.matrix) == 4


def test_the_value_survives_the_ir_json_round_trip() -> None:
    """The IR is the single source of truth, so the number must be in it."""

    ir = fq.Circuit(1).h(0).amplitude_damping(0, gamma=0.2).to_ir()

    restored = CircuitIR.from_json(ir.to_json())

    assert restored.to_json() == ir.to_json()
    assert restored.instructions[-1].params["gamma"] == pytest.approx(0.2, abs=1e-6)


def test_the_parameter_reaches_the_instruction_a_noise_model_lowers_to() -> None:
    """Both routes must write the same instruction, or a plan differs by route."""

    lowered = lower_noise_model(
        fq.Circuit(1).h(0),
        NoiseModel().add(
            "h", channel_from_parameters("depolarizing", {"probability": 0.1})
        ),
    )
    instruction = lowered.instructions[-1]

    assert instruction.name == "depolarizing"
    assert instruction.metadata["is_channel"] is True
    assert instruction.params["probability"] == pytest.approx(0.1, abs=1e-6)


def test_an_inline_channel_is_planned_as_a_noisy_program() -> None:
    """A channel in the program is noise, whether or not a model was passed.

    Before this, only `noise_model=` made the planner treat the program as noisy,
    so the inline form selected the statevector route and then failed inside a
    kernel, and the MPS route returned the noiseless number instead.
    """

    plan = fq.plan(fq.Circuit(1).h(0).depolarizing(0, 0.1))

    assert plan.analysis.channel_count == 1
    assert plan.analysis.has_noise is True
    assert plan.to_dict()["decision"]["mode"] == "density_matrix"
    assert plan.to_dict()["decision"]["recommended_mode"] == "density"


@pytest.mark.parametrize("mode", ["mps", "tensor_network", "statevector"])
def test_a_representation_that_holds_amplitudes_refuses_a_channel(mode: str) -> None:
    """Refusing is the point: the alternative was returning the noiseless number."""

    with pytest.raises(
        ValidationError,
        match=r"stable noisy execution supports mode='auto' or mode='density_matrix'",
    ):
        fq.run(
            fq.Circuit(1).h(0).depolarizing(0, 0.1),
            options=fq.ExecutionOptions(mode=mode),
            outputs=[fq.probabilities()],
        )


def test_an_inline_channel_reaches_the_approximate_mps_route() -> None:
    """``auto`` may put a noisy program on MPS, and an inline channel must too.

    ``allow_approximate`` with a memory limit the density matrix cannot meet is
    the one route that admits a channel without ``mode="density_matrix"``. It
    used to be entered only when the caller passed a ``noise_model=``, so a
    channel written into the circuit stayed on a plain MPS representation and
    the run returned the noiseless number. The plan is the assertion because it
    is where the route is decided; the distributed seed makes the counts below
    reproducible without making this test depend on how the draw lands.

    The bond dimension is 2 here, the smallest that represents the state, so a
    truncation artefact cannot be mistaken for the channel.
    """

    circuit = _ghz(6).depolarizing(0, 1.0)
    options = fq.ExecutionOptions(
        mode="auto",
        memory_limit_bytes=16384,
        allow_approximate=True,
        seed=7,
        shots=200,
    )

    plan = fq.plan(circuit, options=options)
    noisy = plan.noisy_execution_plan

    assert plan.to_dict()["decision"]["mode"] == "mps"
    assert noisy is not None
    assert noisy.representation == "mps"
    assert noisy.evolution == "quantum_trajectory"

    # p=1.0 leaves no identity branch, so the pair of GHZ outcomes is the one
    # thing the channel cannot produce and the noiseless route is the only way
    # to see it alone.
    counts = fq.run(circuit, options=options, outputs=fq.counts()).counts[0]

    assert set(counts) != {"0" * 6, "1" * 6}


def test_an_inline_channel_survives_a_saved_and_restored_plan() -> None:
    """A plan is a portable artifact, so its noisy route has to survive the file.

    ``plan_from_dict`` rebuilds the noisy execution plan from the recorded
    extension, and an inline channel has none: it is in the program. Restoring
    therefore dropped the noisy plan and executing the restored object ran the
    noiseless program, which is why the condition reads the restored program as
    well as the extension list.
    """

    circuit = _ghz(6).depolarizing(0, 0.4)
    options = fq.ExecutionOptions(
        mode="auto",
        memory_limit_bytes=16384,
        allow_approximate=True,
        seed=7,
    )

    restored = plan_from_dict(
        json.loads(plan_to_json(fq.plan(circuit, options=options)))
    )

    assert restored.noisy_execution_plan is not None
    assert restored.noisy_execution_plan.representation == "mps"
    assert restored.noisy_execution_plan.evolution == "quantum_trajectory"
    assert restored.to_dict() == fq.plan(circuit, options=options).to_dict()


def test_the_population_is_the_analytic_value_of_the_channel() -> None:
    """``depolarizing(p)`` on ``|0>`` gives ``1 - 2p/3`` and ``2p/3``.

    The channel is ``(1-p) rho + p/3 (X rho X + Y rho Y + Z rho Z)``. ``I`` and
    ``Z`` leave the population alone and ``X`` and ``Y`` each move it, so ``|1>``
    collects ``2p/3``. An implementation that dropped a branch, applied the
    channel twice, or applied it to the wrong state would miss by at least
    ``p/3``, three orders of magnitude above the 1e-6 tolerance.
    """

    probability = 0.1
    result = fq.run(
        fq.Circuit(1).depolarizing(0, probability), outputs=[fq.probabilities()]
    )
    populations = result.measurements[0].value.reshape(-1)

    assert float(populations[0]) == pytest.approx(1 - 2 * probability / 3, abs=1e-6)
    assert float(populations[1]) == pytest.approx(2 * probability / 3, abs=1e-6)


def test_amplitude_damping_moves_the_excited_population_to_the_ground_state() -> None:
    """On ``|+>`` the damping channel gives ``P(0) = (1 + gamma) / 2``.

    With ``E0 = diag(1, sqrt(1 - gamma))`` and ``E1 = sqrt(gamma) |0><1|``, the
    coherence term contracts to ``sqrt(1 - gamma) / 2`` and the populations become
    ``(1 + gamma) / 2`` and ``(1 - gamma) / 2``.
    """

    gamma = 0.2
    result = fq.run(
        fq.Circuit(1).h(0).amplitude_damping(0, gamma), outputs=[fq.probabilities()]
    )
    populations = result.measurements[0].value.reshape(-1)

    assert float(populations[0]) == pytest.approx((1 + gamma) / 2, abs=1e-5)
    assert float(populations[1]) == pytest.approx((1 - gamma) / 2, abs=1e-5)


def test_two_depolarized_wires_of_a_bell_state_compose() -> None:
    """Two independent channels compose, and on an entangled state they mix it.

    ``depolarizing(p)`` sends ``P(0) -> (1 - 2p/3) P(0) + (2p/3) P(1)``. Applied
    to wire 0 of ``(|00> + |11>)/2`` that gives ``P(00) = 0.5 (1 - 2p/3)``;
    applied to both wires it gives ``P(00) = 0.5 ((1 - 2p/3)**2 + (2p/3)**2)``
    and a cross population ``P(01) = (2p/3)(1 - 2p/3)``, because the ``|11>``
    branch of the first channel is what the second channel moves into ``|01>``.

    A hand computation with the analytic Kraus operators reproduces both
    numbers, so this is a second anchor on the composition rather than a
    restatement of the single-channel test. ``p = 0.3`` separates the two-wire
    result (0.34) from the one-wire result (0.40) by 0.06, and the ``P(01)``
    population is zero unless both channels act, so a channel dropped, applied
    twice to one wire, or pointed at the wrong wire misses by at least 0.04.
    """

    probability = 0.3
    circuit = (
        fq.Circuit(2)
        .h(0)
        .cx(0, 1)
        .depolarizing(0, probability)
        .depolarizing(1, probability)
    )
    populations = (
        fq.run(circuit, outputs=[fq.probabilities()]).measurements[0].value.reshape(-1)
    )
    kept = 1 - 2 * probability / 3
    flipped = 2 * probability / 3

    assert float(populations[0]) == pytest.approx(
        0.5 * (kept**2 + flipped**2), abs=1e-5
    )
    assert float(populations[3]) == pytest.approx(
        0.5 * (kept**2 + flipped**2), abs=1e-5
    )
    assert float(populations[1]) == pytest.approx(kept * flipped, abs=1e-5)
    assert float(populations[2]) == pytest.approx(kept * flipped, abs=1e-5)
    assert float(populations.sum()) == pytest.approx(1.0, abs=1e-6)


def test_one_depolarized_wire_of_a_bell_state_puts_a_third_of_the_rate_crosswise() -> (
    None
):
    """The one-wire contrast the two-wire assertion is measured against.

    One channel on wire 0 of a Bell state gives ``P(00) = P(11) =
    0.5 (1 - 2p/3)`` and ``P(01) = P(10) = p/3``: the ``X`` and ``Y`` branches
    move the wire-0 bit to ``|1>``, which is a disagreement only for the
    ``|00>`` half of the state. The two-wire test above asserts 0.34 and 0.16
    where this one asserts 0.40 and 0.10 at the same probability, so neither
    can be satisfied by the other's implementation.
    """

    probability = 0.3
    circuit = fq.Circuit(2).h(0).cx(0, 1).depolarizing(0, probability)
    populations = (
        fq.run(circuit, outputs=[fq.probabilities()]).measurements[0].value.reshape(-1)
    )
    kept = 1 - 2 * probability / 3

    assert float(populations[0]) == pytest.approx(0.5 * kept, abs=1e-5)
    assert float(populations[3]) == pytest.approx(0.5 * kept, abs=1e-5)
    assert float(populations[1]) == pytest.approx(probability / 3, abs=1e-5)
    assert float(populations[2]) == pytest.approx(probability / 3, abs=1e-5)
    assert float(populations.sum()) == pytest.approx(1.0, abs=1e-6)


def test_the_two_ways_to_write_a_channel_agree() -> None:
    """`NoiseModel.add` after a gate and an inline channel are the same operation."""

    inline = fq.run(
        fq.Circuit(1).x(0).depolarizing(0, 0.1), outputs=[fq.probabilities()]
    )
    modelled = fq.run(
        fq.Circuit(1).x(0),
        noise_model=NoiseModel().add("x", depolarizing_channel(0.1)),
        outputs=[fq.probabilities()],
    )

    assert torch.allclose(
        inline.measurements[0].value, modelled.measurements[0].value, atol=1e-6
    )


def test_a_materialized_channel_without_declared_parameters_still_loads() -> None:
    """An IR written before the opcodes declared parameters must stay readable.

    ``IR_VERSION`` is an exact-match pin, so a payload that carries the Kraus
    operators but no parameter names cannot be rewritten: the operators are the
    channel, and demanding names the document never had would make an accepted
    artifact unreadable.
    """

    instruction = Instruction(
        name="bit_flip", wires=(0,), matrix=depolarizing_channel(0.5).kraus
    )

    assert instruction.params == {}
    assert len(instruction.matrix) == 4


def test_a_channel_without_parameters_and_without_operators_is_refused() -> None:
    """Nothing defines it, so it cannot silently become a no-op."""

    with pytest.raises(
        IRValidationError, match=r"opcode 'depolarizing' is missing parameter\(s\)"
    ):
        Instruction(name="depolarizing", wires=(0,))


def test_every_channel_opcode_can_be_written_from_a_circuit() -> None:
    """No channel opcode may be reachable only from the internals."""

    for opcode, parameter, _ in CHANNELS:
        circuit = getattr(fq.Circuit(1), opcode)(0, **{parameter: 0.1})
        assert circuit.to_ir().instructions[0].name == opcode

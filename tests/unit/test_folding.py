"""Folding: a scale factor realized by lengthening the program.

The premise this file rests on is that a fold is an *identity* on the ideal
signal and a *noise multiplier* on the measured one, and it is checked in that
order. Every ideal-signal assertion takes the body's own state through the
repository's own simulator and compares it with the folded program's state, so a
fold that changed the ideal signal -- the one failure mode a fold can have --
fails those and no other assertion has to know. Every noise assertion is taken
against the closed form the channel implies rather than against a second
implementation of the fold, so a fold that merely produced *some* longer program
would not satisfy it.

The adjoint is not restated here either. The test of the inverse is that
``Circuit.adjoint``, this module's source for it, returns exactly the sequence
the fold inserts, and that the fold refuses the same instructions ``adjoint``
refuses.

The assertions are deliberately of two shapes. Where a quantity is a *claim*
that could be improved (a realized factor, an amplified population), the
assertion is an inequality, a ratio, or a closed form, so it cannot go stale by
being beaten. Where a quantity is the *evidence* itself (the length table, the
instruction that carries the first pair), the value is pinned, because a test
that did not pin it would not be reporting the behaviour it exists to report.
"""

from __future__ import annotations

import json
import math

import pytest
import torch

import flagquantum as fq
from flagquantum.algorithms import (
    FOLD_NOISE_SCHEMA,
    FOLDING_ASSUMPTIONS,
    FOLDING_PLACEMENTS,
    FOLDING_SCHEMA,
    FOLDING_STRATEGIES,
    FoldNoiseAmplification,
    Hamiltonian,
    HamiltonianTerm,
    fold_noise_amplification,
    fold_program,
    run_zne,
)
from flagquantum.algorithms.folding import _pair_count
from flagquantum.algorithms.folding import fold_program as module_fold_program
from flagquantum.core.ir import (
    CircuitIR,
    IRValidationError,
    MeasurementNode,
    ObservableNode,
    ensure_circuit_ir,
)
from flagquantum.errors import CapabilityError
from flagquantum.noise import (
    NoiseModel,
    ReadoutError,
    ReadoutRule,
    bit_flip_channel,
    depolarizing_channel,
)

pytestmark = pytest.mark.unit

ZZ = Hamiltonian([HamiltonianTerm(1.0, "zz", (0, 1))])
Z = Hamiltonian([HamiltonianTerm(1.0, "z", (0,))])
BODY_INSTRUCTIONS = 5

#: The absolute tolerance a complex128 density-matrix run repeats at. These pins
#: are expectation values measured through a state propagation, and two machines
#: that contract the same density matrix in a different order land within 5e-07
#: of each other: measured as 0.9147288742971 here against 0.914728760419 on the
#: x86 runner, 4.3e-07 apart at the widest. This is twenty times that, so a pin
#: still fails when the fold's arithmetic moves and not when the BLAS kernel does.
_EVIDENCE_TOLERANCE = 1e-5


def body() -> fq.Circuit:
    """A five-instruction two-qubit program: one ``h``, then four others."""

    circuit = fq.Circuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.rz(0, 0.3)
    circuit.ry(1, 0.2)
    circuit.t(0)
    return circuit


def bell() -> fq.Circuit:
    circuit = fq.Circuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    return circuit


def excited() -> fq.Circuit:
    circuit = fq.Circuit(1)
    circuit.x(0)
    return circuit


def state_of(program: object, *, dtype: str = "complex128") -> torch.Tensor:
    """The program's own statevector, flat, simulated at ``dtype``."""

    ir = ensure_circuit_ir(program)
    exact = CircuitIR(n_wires=ir.n_wires, instructions=ir.instructions, dtype=dtype)
    return torch.as_tensor(fq.Circuit.from_ir(exact).state()).reshape(-1)


def names(program: object) -> list[str]:
    return [instruction.name for instruction in ensure_circuit_ir(program).instructions]


def noisy(model: NoiseModel, program: object) -> float:
    """``Tr(ZZ rho)`` for a program under a model, exactly."""

    from flagquantum.runtime.noise_registry import noisy_density_matrix

    density = noisy_density_matrix(program, model, dtype=torch.complex128)
    return float(ZZ.expectation(density))


def model_for(gate_names: tuple[str, ...], probability: float) -> NoiseModel:
    model = NoiseModel()
    model.add(gate_names, bit_flip_channel(probability))
    return model


# --------------------------------------------------------------------------
# The fold is an identity on the ideal signal.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", FOLDING_STRATEGIES)
@pytest.mark.parametrize("factor", [1.0, 1.4, 2.2, 3.0, 5.0])
def test_the_folded_program_acts_on_the_state_exactly_as_the_body_did(
    strategy: str, factor: float
) -> None:
    """The whole point of a fold: the extra instructions cancel among themselves."""

    reference = state_of(body())
    plan = fold_program(body(), scale_factor=factor, strategy=strategy)
    folded = state_of(plan.program)
    deviation = float((reference - folded).abs().max())
    # 6.7e-08 is complex64's rounding floor at these amplitudes, which is the
    # dtype the body's own IR declares; the fold introduces no error of its own.
    assert deviation < 1.0e-6


def test_the_only_deviation_is_the_dtype_the_body_declares() -> None:
    """The fold adds no error of its own, so the residual is the dtype's own rounding."""

    ir = body().to_ir()
    at_single = CircuitIR(
        n_wires=ir.n_wires, instructions=ir.instructions, dtype="complex64"
    )
    at_double = CircuitIR(
        n_wires=ir.n_wires, instructions=ir.instructions, dtype="complex128"
    )
    deviations: dict[str, float] = {}
    for name, source in (("complex64", at_single), ("complex128", at_double)):
        plan = fold_program(source, scale_factor=3.0)
        # The folded program keeps the body's declared dtype rather than
        # quietly promoting it.
        assert plan.program.dtype == source.dtype
        deviations[name] = float(
            (state_of(source, dtype=name) - state_of(plan.program, dtype=name))
            .abs()
            .max()
        )
    assert deviations["complex64"] < 1.0e-6
    assert deviations["complex128"] < 1.0e-14
    # The single-precision residual is real rather than absent: it is the
    # rounding of the dtype the body asked for, and it shrinks with the dtype.
    assert deviations["complex64"] > 0.0
    assert deviations["complex64"] > deviations["complex128"]


def test_the_fold_inserts_the_bodys_own_adjoint_and_not_a_second_table() -> None:
    """The inverses the fold appends are ``Circuit.adjoint``'s, read back in order."""

    program = body()
    plan = fold_program(program, scale_factor=3.0)
    # One pair per instruction at this factor, so instruction ``i``'s inverse
    # sits at index ``3 * i + 1``: the instruction, its inverse, the instruction.
    folded = names(plan.program)
    inserted = [folded[3 * index + 1] for index in range(BODY_INSTRUCTIONS)]
    declared = [
        instruction.name
        for instruction in reversed(program.adjoint().to_ir().instructions)
    ]
    assert inserted == declared
    assert declared == ["h", "cx", "rz", "ry", "tdg"]
    # And the body's own instructions survive in order, once each.
    assert [folded[3 * index] for index in range(BODY_INSTRUCTIONS)] == names(program)


def test_a_factor_of_one_is_the_body_itself() -> None:
    """Nothing to realize, so nothing is added; the record says so rather than lying."""

    program = body()
    plan = fold_program(program, scale_factor=1.0)
    assert plan.folded_instructions == BODY_INSTRUCTIONS
    assert plan.added_pairs == 0
    assert plan.scale_factor == 1.0
    assert plan.exact
    assert plan.program.content_hash == program.to_ir().content_hash


# --------------------------------------------------------------------------
# The realized factor, which is what the noise saw.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("factor", "gate_length", "gate_realized", "circuit_realized"),
    [
        (1.0, 5, 1.0, 1.0),
        (1.4, 7, 1.4, 3.0),
        (1.9, 11, 2.2, 3.0),
        (2.0, 11, 2.2, 3.0),
        (2.5, 13, 2.6, 3.0),
        (3.0, 15, 3.0, 3.0),
        (5.0, 25, 5.0, 5.0),
        (7.0, 35, 7.0, 7.0),
    ],
)
def test_the_realized_factor_is_the_length_ratio_and_is_reported(
    factor: float, gate_length: int, gate_realized: float, circuit_realized: float
) -> None:
    """Pinned: a fold adds two instructions at a time, so a request is not a promise."""

    gate = fold_program(body(), scale_factor=factor, strategy="gate")
    assert gate.folded_instructions == gate_length
    assert gate.scale_factor == pytest.approx(gate_realized)
    circuit = fold_program(body(), scale_factor=factor, strategy="circuit")
    assert circuit.scale_factor == pytest.approx(circuit_realized)
    # A request is always met or overshot, never undershot: the fold adds.
    for plan in (gate, circuit):
        assert plan.scale_factor >= plan.requested_scale_factor
        assert plan.overshoot == pytest.approx(
            plan.scale_factor - plan.requested_scale_factor
        )
        assert plan.exact == (plan.scale_factor == plan.requested_scale_factor)


def test_a_request_that_lands_on_a_reachable_length_realizes_exactly_that_length() -> (
    None
):
    """``1.4 * 5`` is 7.000000000000001 in binary floating point, and is not rounded up."""

    for factor in (1.4, 2.2, 2.6, 3.0, 4.6):
        plan = fold_program(body(), scale_factor=factor, strategy="gate")
        assert plan.exact, (factor, plan.folded_instructions, plan.scale_factor)
    assert math.ceil(math.nextafter(1.4 * 5, math.inf)) == 8


def test_the_gate_fold_concentrates_its_pairs_leftmost_first() -> None:
    """One pair on a five-instruction body lengthens the first instruction, not the last."""

    plan = fold_program(body(), scale_factor=1.4, strategy="gate")
    assert names(plan.program) == ["h", "h", "h", "cx", "rz", "ry", "t"]


def test_the_gate_fold_spends_one_pair_per_instruction_leftmost_first() -> None:
    """With more than one pair the distribution is per instruction, not all on the first."""

    plan = fold_program(body(), scale_factor=2.2, strategy="gate")
    # Three pairs over five instructions: the first three instructions carry one
    # each, so those three names are tripled and the last two stand alone.
    # Spending every pair on the first instruction would triple ``h`` alone and
    # leave the body's other four instructions unfolded.
    assert names(plan.program) == [
        "h",
        "h",
        "h",
        "cx",
        "cx",
        "cx",
        "rz",
        "rz",
        "rz",
        "ry",
        "t",
    ]
    assert plan.added_pairs == 3
    assert plan.folded_instructions == BODY_INSTRUCTIONS + 2 * plan.added_pairs


def test_the_circuit_fold_repeats_the_whole_body_and_never_lands_between_odd_integers() -> (
    None
):
    """Its realized factor is ``1 + 2m`` whatever was asked for, so the overshoot can be large."""

    plan = fold_program(body(), scale_factor=1.4, strategy="circuit")
    # ``adjoint`` emits the body's inverses back to front, which is exactly the
    # order ``U (U^dagger U)^m`` wants them in, so the fold reads it unreversed.
    inverses = [
        instruction.name for instruction in body().adjoint().to_ir().instructions
    ]
    assert inverses == ["tdg", "ry", "rz", "cx", "h"]
    # The body, the body's adjoint, and the body again, and never a body with
    # its own adjoint interleaved through it.
    assert names(plan.program) == names(body()) + inverses + names(body())
    assert plan.folded_instructions == 3 * BODY_INSTRUCTIONS
    assert plan.overshoot == pytest.approx(1.6)


def test_a_body_of_one_instruction_makes_the_two_strategies_agree() -> None:
    """With one instruction there is nowhere to distribute a pair, so only integer factors exist."""

    for strategy in FOLDING_STRATEGIES:
        assert (
            fold_program(excited(), scale_factor=3.0, strategy=strategy).scale_factor
            == 3.0
        )
        assert (
            fold_program(excited(), scale_factor=2.0, strategy=strategy).scale_factor
            == 3.0
        )


def test_the_pair_count_is_the_smallest_that_reaches_the_requested_length() -> None:
    """Pinned arithmetic: the count, the length it produces, and the request it meets."""

    pinned = {
        1.0: (0, 0),
        1.2: (1, 1),
        1.4: (1, 1),
        1.6: (2, 1),
        1.9: (3, 1),
        2.0: (3, 1),
        2.4: (4, 1),
        3.0: (5, 1),
        5.0: (10, 2),
        7.0: (15, 3),
    }
    for factor, (gate_pairs, circuit_pairs) in pinned.items():
        assert (
            _pair_count(factor * BODY_INSTRUCTIONS, BODY_INSTRUCTIONS, strategy="gate")
            == gate_pairs
        )
        assert (
            _pair_count(
                factor * BODY_INSTRUCTIONS, BODY_INSTRUCTIONS, strategy="circuit"
            )
            == circuit_pairs
        )
    # The pair count is what the record reports and what the length is built from.
    for factor in (1.4, 2.5, 5.0):
        plan = fold_program(body(), scale_factor=factor)
        assert plan.added_pairs == _pair_count(
            factor * BODY_INSTRUCTIONS, BODY_INSTRUCTIONS, strategy="gate"
        )
        assert plan.folded_instructions == BODY_INSTRUCTIONS + 2 * plan.added_pairs


def test_a_larger_request_never_produces_a_shorter_program() -> None:
    """Monotone: the fold is a function of the requested length and of nothing else."""

    for strategy in FOLDING_STRATEGIES:
        lengths = [
            fold_program(
                body(), scale_factor=factor, strategy=strategy
            ).folded_instructions
            for factor in (
                1.0,
                1.1,
                1.3,
                1.5,
                1.7,
                1.9,
                2.1,
                2.3,
                2.5,
                2.7,
                3.0,
                3.5,
                4.0,
                5.0,
            )
        ]
        assert lengths == sorted(lengths)


# --------------------------------------------------------------------------
# The fold multiplies the noise, which is why it is a scale factor at all.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("factor", "population"),
    [
        (1.0, 0.8999999663496001),
        (3.0, 0.7559999353912377),
        (5.0, 0.6638399310839926),
        (7.0, 0.6048575382512626),
    ],
)
def test_folding_one_bit_flip_multiplies_that_bit_flip_by_the_pair_count(
    factor: float, population: float
) -> None:
    """Pinned against the closed form ``(1 + (1 - 2p)**m) / 2`` at ``p = 0.1``.

    The body is one ``x``, so a gate fold of factor ``k`` realizes exactly ``k``
    instructions and the fold count is ``k``. ``x`` is its own inverse, so the
    folded program is still ``x`` and the qubit stays excited going in; the
    bit-flip channel then runs ``k`` times, so the qubit is measured excited
    exactly when it flipped an even number of times.

    The ``population`` column is this repository's measurement of that closed
    form and is compared at ``_EVIDENCE_TOLERANCE`` rather than bit for bit; the
    closed form itself is compared at ``rel=1e-6`` and is the stronger of the
    two, because it is derived here rather than carried in a table.
    """

    from flagquantum.runtime.noise_registry import noisy_density_matrix

    model = model_for(("x",), 0.1)
    plan = fold_program(excited(), scale_factor=factor)
    assert plan.folded_instructions == int(factor)
    density = noisy_density_matrix(plan.program, model, dtype=torch.complex128)
    measured = 0.5 * (1.0 - float(Z.expectation(density)))
    expected = (1.0 + (1.0 - 2.0 * 0.1) ** int(factor)) / 2.0
    assert measured == pytest.approx(population, abs=_EVIDENCE_TOLERANCE)
    assert measured == pytest.approx(expected, rel=1e-6)
    # Folding strictly increases the noise; it cannot leave it where it was.
    if factor > 1.0:
        assert measured != pytest.approx(0.8999999663496001)


def test_a_folded_run_reaches_a_factor_the_scaling_path_has_to_refuse() -> None:
    """Scaling a probability is bounded by one; lengthening a program is not."""

    model = model_for(("x",), 0.4)
    with pytest.raises(ValueError, match="which is above one"):
        run_zne(excited(), Z, noise_model=model, scale_factors=(1.0, 3.0, 5.0), order=2)
    folded = run_zne(
        excited(),
        Z,
        noise_model=model,
        scale_factors=(1.0, 3.0, 5.0),
        order=2,
        fold="gate",
        dtype=torch.complex128,
    )
    assert [item.expectation for item in folded.measurements] == pytest.approx(
        [-0.2, -0.008, -0.00032], rel=1e-6
    )
    assert folded.estimate == pytest.approx(-0.3651200679, rel=1e-6)


def test_the_two_strategies_agree_when_the_model_covers_every_instruction_alike() -> (
    None
):
    """Uniform coverage makes the fold a pure count, so which instructions repeat stops mattering."""

    model = NoiseModel()
    model.add(("h", "cx"), depolarizing_channel(0.03))
    factors = (1.0, 3.0, 5.0, 7.0)
    gate = run_zne(
        bell(),
        ZZ,
        noise_model=model,
        scale_factors=factors,
        order=2,
        fold="gate",
        dtype=torch.complex128,
    )
    circuit = run_zne(
        bell(),
        ZZ,
        noise_model=model,
        scale_factors=factors,
        order=2,
        fold="circuit",
        dtype=torch.complex128,
    )
    assert [item.expectation for item in gate.measurements] == pytest.approx(
        [item.expectation for item in circuit.measurements], rel=1e-12
    )
    assert gate.estimate == pytest.approx(circuit.estimate, rel=1e-12)
    # And the fold is not a no-op: the population really moves with the factor.
    values = [item.expectation for item in gate.measurements]
    assert values == sorted(values, reverse=True)
    assert values[0] - values[-1] > 0.1


def test_a_model_that_covers_part_of_the_body_scales_only_that_part() -> None:
    """The honest limit of a fold: an instruction the model does not name gains no noise.

    ``gate`` folding repeats the two instructions ``rz`` and ``ry`` once each at
    a realized factor of 3.0, so ``rz`` carries its noise twice as often as the
    folding path implies from the length ratio alone. The measurement records
    that, and it differs from what scaling the channel parameter gives.
    """

    model = NoiseModel()
    model.add(("rz",), depolarizing_channel(0.05))
    factors = (1.0, 3.0, 5.0)
    folded = run_zne(
        body(),
        ZZ,
        noise_model=model,
        scale_factors=factors,
        order=1,
        fold="gate",
        dtype=torch.complex128,
    )
    scaled = run_zne(
        body(),
        ZZ,
        noise_model=model,
        scale_factors=factors,
        order=1,
        dtype=torch.complex128,
    )
    assert [item.expectation for item in folded.measurements] == pytest.approx(
        [0.914728760419, 0.796830307476, 0.694127665363], abs=_EVIDENCE_TOLERANCE
    )
    assert [item.expectation for item in scaled.measurements] == pytest.approx(
        [0.914728760419, 0.78405327657, 0.653377695107], abs=_EVIDENCE_TOLERANCE
    )
    assert folded.estimate != pytest.approx(scaled.estimate)
    assert FOLDING_ASSUMPTIONS[1] in folded.assumptions


# --------------------------------------------------------------------------
# The folded program is a program: its requests and its identity survive.
# --------------------------------------------------------------------------


def test_the_fold_carries_the_declared_requests_into_the_folded_program() -> None:
    """A fold lengthens a program; it does not change what is asked of its state.

    The round trip through ``Circuit.from_ir`` drops both declared requests, so
    the fold is built as IR directly and this is the assertion that keeps it so.
    """

    ir = CircuitIR(
        n_wires=2,
        instructions=bell().to_ir().instructions,
        observables=(ObservableNode("zz", (0, 1)),),
        measurements=(MeasurementNode("counts", (0, 1), shots=512),),
    )
    plan = fold_program(ir, scale_factor=3.0)
    assert plan.program.observables == ir.observables
    assert plan.program.measurements == ir.measurements
    assert plan.program.n_wires == ir.n_wires
    # The round trip the fold deliberately avoids does drop them.
    assert fq.Circuit.from_ir(ir).to_ir().measurements == ()
    assert fq.Circuit.from_ir(ir).to_ir().observables == ()


def test_the_plan_identity_separates_every_input_that_can_change_the_program() -> None:
    """Two plans differ in identity exactly when they differ in an input or an output."""

    base = fold_program(body(), scale_factor=3.0, strategy="gate")
    same = fold_program(body(), scale_factor=3.0, strategy="gate")
    assert base.identity == same.identity
    assert base.identity != fold_program(body(), scale_factor=5.0).identity
    assert (
        base.identity
        != fold_program(body(), scale_factor=3.0, strategy="circuit").identity
    )
    assert base.identity != fold_program(bell(), scale_factor=3.0).identity
    # The placement is an input even where a circuit fold refuses it: at one
    # factor the two rules build one program only when the model is uniform, and
    # they still carry different identities, because a plan names the rule it
    # applied rather than only the program it produced.
    placed = fold_program(
        body(),
        scale_factor=3.0,
        placement="model_weighted",
        noise_model=model_for(("t",), 0.05),
    )
    assert base.identity != placed.identity
    # A different request that realizes the same length is a different record.
    assert (
        fold_program(body(), scale_factor=1.9).identity
        != fold_program(body(), scale_factor=2.0).identity
    )
    # Every identity is the program's own content hash, chained into the plan's.
    assert base.program_identity == base.program.content_hash
    assert base.program_identity != base.identity


def test_the_plan_serializes_to_the_schema_it_declares_and_back() -> None:
    plan = fold_program(body(), scale_factor=2.5)
    record = plan.to_dict()
    assert record["schema"] == FOLDING_SCHEMA
    assert set(record) == {
        "schema",
        "strategy",
        "placement",
        "requested_scale_factor",
        "scale_factor",
        "body_instructions",
        "folded_instructions",
        "added_pairs",
        "program_identity",
        "identity",
    }
    assert json.loads(json.dumps(record)) == record
    rebuilt = module_fold_program(
        body(),
        scale_factor=record["requested_scale_factor"],
        strategy=record["strategy"],
        placement=record["placement"],
    )
    assert rebuilt.identity == record["identity"]
    assert rebuilt.to_dict() == record


def test_the_plan_is_frozen_and_rejects_a_record_that_describes_no_fold() -> None:
    """Every invariant the record states is checked where the record is built."""

    plan = fold_program(body(), scale_factor=3.0)
    with pytest.raises(Exception):
        plan.scale_factor = 1.0  # type: ignore[misc]
    good = plan.to_dict()
    fields = {
        key: good[key]
        for key in (
            "strategy",
            "placement",
            "requested_scale_factor",
            "scale_factor",
            "body_instructions",
            "folded_instructions",
            "added_pairs",
        )
    }
    from flagquantum.algorithms.folding import FoldingPlan

    broken = dict(fields, strategy="linear", program=plan.program, identity="x")
    with pytest.raises(ValueError, match="unsupported folding strategy"):
        FoldingPlan(**broken)
    broken = dict(fields, scale_factor=9.0, program=plan.program, identity="x")
    with pytest.raises(ValueError, match="length"):
        FoldingPlan(**broken)
    broken = dict(fields, added_pairs=99, program=plan.program, identity="x")
    with pytest.raises(ValueError, match="in pairs"):
        FoldingPlan(**broken)
    # A record whose realized factor sits below its own request is unreachable
    # through the fold, and is refused rather than reported as a real scale.
    short = fold_program(body(), scale_factor=3.0)
    with pytest.raises(ValueError, match="at or above the one asked for"):
        FoldingPlan(
            strategy=short.strategy,
            placement=short.placement,
            requested_scale_factor=5.0,
            scale_factor=short.scale_factor,
            body_instructions=short.body_instructions,
            folded_instructions=short.folded_instructions,
            added_pairs=short.added_pairs,
            program=short.program,
            identity="x",
        )
    broken = dict(fields, identity="")
    with pytest.raises(ValueError, match="identity"):
        FoldingPlan(program=plan.program, **broken)
    broken = dict(fields, body_instructions=0, program=plan.program, identity="x")
    with pytest.raises(ValueError, match="at least one instruction"):
        FoldingPlan(**broken)
    broken = dict(
        fields, folded_instructions=1, added_pairs=0, program=plan.program, identity="x"
    )
    with pytest.raises(ValueError, match="lengthen"):
        FoldingPlan(**broken)
    # A circuit fold repeats the whole body, so a record of one that claims a
    # placement describes a rule the fold did not apply.
    circuit = fold_program(body(), scale_factor=3.0, strategy="circuit")
    with pytest.raises(ValueError, match="has no placement to report"):
        FoldingPlan(
            strategy=circuit.strategy,
            placement="model_weighted",
            requested_scale_factor=circuit.requested_scale_factor,
            scale_factor=circuit.scale_factor,
            body_instructions=circuit.body_instructions,
            folded_instructions=circuit.folded_instructions,
            added_pairs=circuit.added_pairs,
            program=circuit.program,
            identity="x",
        )


# --------------------------------------------------------------------------
# Refusals: all of them before a program is built, in the same spirit as run_zne.
# --------------------------------------------------------------------------


def test_a_scale_factor_below_one_is_refused_because_a_fold_cannot_remove() -> None:
    for strategy in FOLDING_STRATEGIES:
        with pytest.raises(ValueError, match="can only add instructions"):
            fold_program(body(), scale_factor=0.5, strategy=strategy)
        with pytest.raises(ValueError, match="can only add instructions"):
            fold_program(body(), scale_factor=0.0, strategy=strategy)
    with pytest.raises(ValueError, match="must be finite"):
        fold_program(body(), scale_factor=math.inf)
    with pytest.raises(ValueError, match="must be finite"):
        fold_program(body(), scale_factor=math.nan)
    with pytest.raises(ValueError, match="must be a number"):
        fold_program(body(), scale_factor="three")  # type: ignore[arg-type]


def test_an_unknown_strategy_and_an_unfoldable_program_are_refused() -> None:
    with pytest.raises(ValueError, match="unsupported folding strategy"):
        fold_program(body(), scale_factor=2.0, strategy="linear")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="needs a Circuit or a CircuitIR"):
        fold_program("h(0)", scale_factor=2.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="no instruction to fold"):
        fold_program(fq.Circuit(2), scale_factor=2.0)


def test_an_instruction_with_no_inverse_is_refused_by_name_before_anything_is_built() -> (
    None
):
    """A channel has no unitary inverse, and the refusal is the adjoint's own."""

    channel = fq.Circuit(2)
    channel.h(0)
    channel.depolarizing(0, 0.1)
    with pytest.raises(
        CapabilityError, match="noise channel, which has no unitary inverse"
    ):
        fold_program(channel, scale_factor=2.0)
    # The same program's adjoint refuses it identically, so there is one rule.
    with pytest.raises(
        CapabilityError, match="noise channel, which has no unitary inverse"
    ):
        channel.adjoint()


def test_a_custom_operation_whose_matrix_is_not_unitary_is_refused() -> None:
    """The IR accepts it and ``adjoint`` inverts it, so the fold would be a different program."""

    shear = fq.Circuit(1)
    shear.any(
        0,
        unitary=torch.tensor([[1.0, 1.0], [0.0, 1.0]], dtype=torch.complex128),
        name="shear",
    )
    with pytest.raises(ValueError, match="is not unitary"):
        fold_program(shear, scale_factor=2.0)
    # A genuinely unitary custom operation is folded, so the check is not a ban
    # on custom operations.
    flip = fq.Circuit(1)
    flip.any(
        0,
        unitary=torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex128),
        name="flip",
    )
    assert names(fold_program(flip, scale_factor=3.0).program) == ["flip"] * 3
    # A non-square matrix has no conjugate transpose to fold with. The operand is
    # spelled as a nested list so that it reaches this guard: a tensor publishes a
    # shape, and `Instruction.__post_init__` refuses a wrong entry count from it, so
    # a tensor would be refused an entire layer earlier.
    wide = fq.Circuit(1)
    wide.any(
        0,
        unitary=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        name="wide",
    )
    with pytest.raises(ValueError, match="not square"):
        fold_program(wide, scale_factor=2.0)
    # And the same operand in the spelling that carries its shape is refused where
    # the instruction is built, which is what `test_instruction_operand_shape.py`
    # pins in full.
    with pytest.raises(IRValidationError):
        fq.Circuit(1).any(
            0,
            unitary=torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.complex128
            ),
            name="wide",
        )


# --------------------------------------------------------------------------
# The extrapolation path.
# --------------------------------------------------------------------------


def test_the_fit_is_given_the_realized_factors_rather_than_the_requests() -> None:
    """The abscissa is the ratio the noise saw; a length ratio 1.9 was never realized."""

    model = model_for(("h", "cx", "rz", "ry", "t"), 0.05)
    folded = run_zne(
        body(),
        ZZ,
        noise_model=model,
        scale_factors=(1.0, 1.9, 3.0),
        order=2,
        fold="gate",
        dtype=torch.complex128,
    )
    assert [item.scale_factor for item in folded.measurements] == pytest.approx(
        [1.0, 2.2, 3.0]
    )
    assert list(folded.fit.scale_factors) == pytest.approx([1.0, 2.2, 3.0])
    # Every abscissa is a ratio the fold really built, and 1.9 is not among them.
    assert 1.9 not in [item.scale_factor for item in folded.measurements]
    for measurement, request in zip(folded.measurements, (1.0, 1.9, 3.0), strict=True):
        plan = fold_program(body(), scale_factor=request, strategy="gate")
        assert measurement.scale_factor == pytest.approx(plan.scale_factor)
        assert measurement.scale_factor >= request


def test_a_folded_run_reports_the_foldings_own_assumptions_not_a_scaling_claim() -> (
    None
):
    """No channel parameter moved, so neither scaling assumption is true of this run."""

    model = model_for(("x",), 0.05)
    folded = run_zne(
        excited(),
        Z,
        noise_model=model,
        scale_factors=(1.0, 3.0, 5.0),
        order=2,
        fold="gate",
        dtype=torch.complex128,
    )
    assert folded.assumptions[-len(FOLDING_ASSUMPTIONS) :] == FOLDING_ASSUMPTIONS
    assert len(folded.assumptions) == len(FOLDING_ASSUMPTIONS) + 3
    text = " ".join(folded.assumptions)
    assert "length ratio rather than a channel parameter" in text
    assert "The declared scaling multiplies" not in text
    scaled = run_zne(
        excited(),
        Z,
        noise_model=model,
        scale_factors=(1.0, 3.0, 5.0),
        order=2,
        dtype=torch.complex128,
    )
    assert FOLDING_ASSUMPTIONS[0] not in " ".join(scaled.assumptions)
    assert "The declared scaling multiplies" in " ".join(scaled.assumptions)


def test_the_unmitigated_point_is_the_factor_of_one_and_the_estimate_is_the_extrapolation() -> (
    None
):
    model = model_for(("x",), 0.05)
    folded = run_zne(
        excited(),
        Z,
        noise_model=model,
        scale_factors=(1.0, 3.0, 5.0, 7.0),
        order=2,
        fold="gate",
        dtype=torch.complex128,
    )
    assert folded.unmitigated == pytest.approx(-0.8999999544, abs=_EVIDENCE_TOLERANCE)
    assert folded.unmitigated == folded.measurements[0].expectation
    assert folded.estimate == pytest.approx(-0.995600323065, abs=_EVIDENCE_TOLERANCE)
    # The fold's curve is a polynomial in the realized factor at these points
    # only to a tolerance, and the diagnostic says so rather than hiding it.
    assert folded.fit.max_residual > 0.0
    assert folded.fit.max_residual < 1.0e-2
    assert folded.applied_correction == pytest.approx(
        folded.estimate - folded.unmitigated
    )


def test_two_requests_that_fold_to_one_length_are_refused_rather_than_fit() -> None:
    """A repeated abscissa makes the system singular, and the refusal names the cause."""

    model = model_for(("x",), 0.05)
    with pytest.raises(ValueError, match="realize the same one more than once"):
        run_zne(
            excited(),
            Z,
            noise_model=model,
            scale_factors=(1.0, 1.1, 3.0),
            order=2,
            fold="gate",
            dtype=torch.complex128,
        )
    with pytest.raises(ValueError, match=r"\[3\.0\]"):
        run_zne(
            excited(),
            Z,
            noise_model=model,
            scale_factors=(1.0, 2.0, 3.0),
            order=2,
            fold="gate",
            dtype=torch.complex128,
        )


def test_folding_and_a_caller_supplied_scaling_are_refused_together() -> None:
    """A length ratio times a channel parameter names neither unit."""

    model = model_for(("x",), 0.05)

    def caller(model: NoiseModel, factor: float) -> NoiseModel:
        return model

    with pytest.raises(ValueError, match="different units of scale"):
        run_zne(
            excited(),
            Z,
            noise_model=model,
            scale_factors=(1.0, 3.0, 5.0),
            order=2,
            fold="gate",
            scaling=caller,
        )
    # The declared scaling is the default and is not a caller's, so folding with
    # it is the supported path.
    assert (
        run_zne(
            excited(),
            Z,
            noise_model=model,
            scale_factors=(1.0, 3.0, 5.0),
            order=2,
            fold="gate",
            dtype=torch.complex128,
        ).estimate
        is not None
    )


def test_an_unknown_folding_strategy_is_refused_by_the_extrapolation_too() -> None:
    """One vocabulary: the extrapolation's refusal names the same strategies and the default."""

    model = model_for(("x",), 0.05)
    with pytest.raises(ValueError) as refusal:
        run_zne(
            excited(),
            Z,
            noise_model=model,
            scale_factors=(1.0, 3.0, 5.0),
            order=2,
            fold="linear",  # type: ignore[arg-type]
        )
    message = str(refusal.value)
    assert "unsupported folding strategy 'linear'" in message
    for strategy in FOLDING_STRATEGIES:
        assert strategy in message
    assert "or None to scale the noise model" in message


def test_the_readout_refusal_and_the_point_count_refusal_still_apply_when_folding() -> (
    None
):
    """Folding changes the abscissa and nothing else about the extrapolation's checks."""

    model = model_for(("x",), 0.05)
    model.readout_rules.append(
        ReadoutRule((0,), ReadoutError(((0.95, 0.05), (0.05, 0.95))))
    )
    with pytest.raises(ValueError, match="declares a readout rule"):
        run_zne(
            excited(),
            Z,
            noise_model=model,
            scale_factors=(1.0, 3.0, 5.0),
            order=2,
            fold="gate",
        )
    clean = model_for(("x",), 0.05)
    with pytest.raises(ValueError):
        run_zne(
            excited(),
            Z,
            noise_model=clean,
            scale_factors=(1.0, 3.0),
            order=3,
            fold="gate",
        )


def test_a_folded_run_is_reproducible_and_does_not_depend_on_the_request_beyond_the_abscissa() -> (
    None
):
    model = model_for(("x",), 0.05)
    first = run_zne(
        excited(),
        Z,
        noise_model=model,
        scale_factors=(1.0, 3.0, 5.0),
        order=2,
        fold="gate",
        dtype=torch.complex128,
    )
    second = run_zne(
        excited(),
        Z,
        noise_model=model,
        scale_factors=(1.0, 3.0, 5.0),
        order=2,
        fold="gate",
        dtype=torch.complex128,
    )
    assert first.estimate == second.estimate
    assert [item.expectation for item in first.measurements] == [
        item.expectation for item in second.measurements
    ]
    assert first.fit == second.fit


# --------------------------------------------------------------------------
# The fold's noise amplification, measured against a model.
#
# A plan reports a length ratio. These tests are about the other quantity, the
# ratio of channel *applications*, which is what a curve actually sees and which
# the plan does not carry. The assertions are of two shapes for the same reason
# as above: a ratio that could be improved is an inequality or a closed form, and
# a count that is the evidence itself is pinned.
# --------------------------------------------------------------------------


def phase_only_body() -> fq.Circuit:
    """Five instructions whose only population change is the leading ``x``."""

    circuit = fq.Circuit(1)
    circuit.x(0)
    circuit.rz(0, 0.3)
    circuit.t(0)
    circuit.sdg(0)
    circuit.rz(0, 0.2)
    return circuit


def test_the_reading_separates_the_length_ratio_from_the_strength() -> None:
    """A gate fold reaches 1.4 in length and 2.0 in noise on one qubit, and says both."""

    model = NoiseModel()
    model.add(("h", "cx"), depolarizing_channel(0.03))
    plan = fold_program(body(), scale_factor=1.4)
    reading = fold_noise_amplification(body(), plan, model)

    assert plan.scale_factor == 1.4
    # The pair lands on the leading h, which is on qubit 0, so qubit 0 gains it
    # and qubit 1 does not. This is the whole point: one plan, two strengths.
    assert reading.per_qubit == ((0, 2, 4), (1, 1, 1))
    assert reading.amplification_on((0,)) == pytest.approx(2.0)
    assert reading.amplification_on((1,)) == pytest.approx(1.0)
    # The whole-program ratio is between the two and equals neither the ratio nor
    # either qubit's, which is why a caller with an observable has to ask by qubit.
    assert reading.amplification == pytest.approx(5 / 3)
    assert reading.amplification != pytest.approx(plan.scale_factor)
    assert reading.reached_uniformly is False


def test_a_circuit_fold_repeats_the_whole_body_so_it_cannot_be_uneven() -> None:
    """The two strategies differ in where the added noise lands, not only how much."""

    model = NoiseModel()
    model.add(("h", "cx"), depolarizing_channel(0.03))
    plan = fold_program(body(), scale_factor=1.4, strategy="circuit")
    reading = fold_noise_amplification(body(), plan, model)

    # A request of 1.4 costs a circuit fold three whole bodies, so every covered
    # instruction repeats three times and the ratio is recovered exactly.
    assert plan.scale_factor == 3.0
    assert reading.reached_uniformly is True
    assert reading.amplification == pytest.approx(3.0)
    assert reading.amplification == pytest.approx(plan.scale_factor)
    assert reading.amplification_on((0,)) == pytest.approx(3.0)
    assert reading.amplification_on((1,)) == pytest.approx(3.0)


def test_a_model_that_names_the_gate_and_not_its_inverse_reads_below_the_ratio() -> (
    None
):
    """A fold of ``t`` inserts ``t``, ``tdg``, ``t``; a model naming ``t`` reaches two."""

    model = NoiseModel()
    model.add("t", depolarizing_channel(0.02))
    program = fq.Circuit(1).t(0)
    plan = fold_program(program, scale_factor=3.0)
    assert names(plan.program) == ["t", "tdg", "t"]

    reading = fold_noise_amplification(program, plan, model)
    assert reading.body_locations == 1
    assert reading.folded_locations == 2
    assert reading.amplification == pytest.approx(2.0)
    assert reading.amplification < plan.scale_factor
    assert reading.covered_instructions == 1
    assert reading.uncovered_instructions == 0


def test_full_coverage_of_the_gate_names_is_still_not_uniform_coverage() -> None:
    """Two ways a fold drops below its ratio, and only one of them is a body the model skips."""

    model = NoiseModel()
    model.add(("h", "cx", "rz", "ry", "t"), depolarizing_channel(0.03))
    body5 = fq.Circuit(2).h(0).cx(0, 1).rz(0, 0.4).ry(1, 0.6).t(0)

    gate_plan = fold_program(body5, scale_factor=1.4)
    gate = fold_noise_amplification(body5, gate_plan, model)
    # Every instruction carries a channel, and the gate fold is still uneven: the
    # pair it spends is two instructions on qubit 0 alone, so qubit 0 gains half
    # its applications again while qubit 1 gains none.
    assert (gate.covered_instructions, gate.uncovered_instructions) == (5, 0)
    assert gate.per_qubit == ((0, 4, 6), (1, 2, 2))
    assert gate.amplification == pytest.approx(4 / 3)
    assert gate.reached_uniformly is False

    circuit_plan = fold_program(body5, scale_factor=1.4, strategy="circuit")
    circuit = fold_noise_amplification(body5, circuit_plan, model)
    # The circuit fold repeats whole bodies, so its own pairs are uniform, and it
    # is *still* uneven: the adjoint body it inserts carries ``tdg``, which this
    # model does not name, so the qubit holding the ``t`` loses one application
    # per repeat while the other qubit loses none.
    assert circuit_plan.scale_factor == 3.0
    assert (circuit.covered_instructions, circuit.uncovered_instructions) == (5, 0)
    assert circuit.per_qubit == ((0, 4, 11), (1, 2, 6))
    assert circuit.amplification == pytest.approx(17 / 6)
    assert circuit.amplification < circuit_plan.scale_factor
    assert circuit.amplification_on((1,)) == pytest.approx(3.0)
    assert circuit.reached_uniformly is False


def test_a_qubit_the_model_never_touches_is_named_rather_than_reported_as_one() -> None:
    """A silent qubit's flat curve is the model's absence, not a measurement of the fold."""

    model = NoiseModel()
    model.add("h", depolarizing_channel(0.03))
    plan = fold_program(body(), scale_factor=3.0)
    reading = fold_noise_amplification(body(), plan, model)

    assert reading.per_qubit == ((0, 1, 3),)
    assert reading.silent_qubits == (1,)
    # No channel on qubit 1 means no denominator, so the honest answer is that
    # there is no ratio rather than a ratio of one.
    assert reading.amplification_on((1,)) is None
    assert reading.amplification_on((0,)) == pytest.approx(3.0)
    # Coverage is reported rather than averaged into the ratio: four of the five
    # instructions carry no channel, so the fold repeats them without adding error.
    assert (reading.covered_instructions, reading.uncovered_instructions) == (1, 4)


def test_the_reading_is_the_count_the_closed_form_is_evaluated_at() -> None:
    """The measured population follows the reading's ratio, not the plan's."""

    from flagquantum.runtime.noise_registry import noisy_density_matrix

    model = model_for(("x",), 0.1)
    plan = fold_program(phase_only_body(), scale_factor=1.4)
    reading = fold_noise_amplification(phase_only_body(), plan, model)
    # The pair lands on the leading x, so the channel the population is measured
    # through runs three times while the program is only 1.4 times as long.
    assert reading.amplification_on((0,)) == pytest.approx(3.0)
    assert plan.scale_factor == 1.4

    density = noisy_density_matrix(plan.program, model, dtype=torch.complex128)
    measured = 0.5 * (1.0 - float(Z.expectation(density)))
    at_reading = (1.0 + (1.0 - 2.0 * 0.1) ** 3) / 2.0
    at_ratio = (1.0 + (1.0 - 2.0 * 0.1) ** 1.4) / 2.0
    assert measured == pytest.approx(at_reading, abs=_EVIDENCE_TOLERANCE)
    # The two candidate abscissas are far enough apart that the assertion above
    # is a discrimination rather than a coincidence.
    assert measured != pytest.approx(at_ratio, abs=1e-3)


def test_the_reading_refuses_a_plan_built_from_another_program() -> None:
    """A plan records counts a same-length sibling would agree with, so the program is compared."""

    model = NoiseModel()
    model.add(("h", "cx"), depolarizing_channel(0.03))
    plan = fold_program(body(), scale_factor=3.0)
    # Five instructions, so every count this sibling records matches the plan's
    # except the identity, which covers the program itself.
    other = fq.Circuit(2).x(0).cx(0, 1).rz(0, 0.3).ry(1, 0.2).t(0)
    assert len(ensure_circuit_ir(other).instructions) == plan.body_instructions

    with pytest.raises(ValueError, match="the plan is not this body's fold"):
        fold_noise_amplification(other, plan, model)


def test_a_model_that_covers_nothing_is_refused_rather_than_read_as_a_ratio_of_zero() -> (
    None
):
    """An empty denominator is a refusal, and the message names what was counted."""

    model = NoiseModel()
    model.add("zz", depolarizing_channel(0.03))
    plan = fold_program(body(), scale_factor=3.0)
    with pytest.raises(ValueError, match="applies a channel to no instruction"):
        fold_noise_amplification(body(), plan, model)

    empty = fq.Circuit(2)
    with pytest.raises(ValueError, match="an empty program applies no channel"):
        fold_noise_amplification(empty, fold_program(body(), scale_factor=3.0), model)
    with pytest.raises(TypeError, match="needs the built body as a Circuit"):
        fold_noise_amplification("h(0)", plan, model)  # type: ignore[arg-type]


def test_the_reading_serializes_to_the_schema_it_declares_and_back() -> None:
    """The record is data: the schema names it and a JSON round trip keeps the identity."""

    model = NoiseModel()
    model.add(("h", "cx"), depolarizing_channel(0.03))
    plan = fold_program(body(), scale_factor=1.4)
    reading = fold_noise_amplification(body(), plan, model)
    payload = reading.to_dict()

    assert payload["schema"] == FOLD_NOISE_SCHEMA
    assert payload["per_qubit"] == [[0, 2, 4], [1, 1, 1]]
    assert payload["amplification"] == pytest.approx(5 / 3)
    assert payload["reached_uniformly"] is False
    assert json.loads(json.dumps(payload)) == payload
    # The same body, plan and model produce the same identity, and a different
    # model produces a different one, so the reading is named by its inputs.
    assert fold_noise_amplification(body(), plan, model).identity == reading.identity
    other = NoiseModel()
    other.add(("h", "cx"), depolarizing_channel(0.04))
    assert fold_noise_amplification(body(), plan, other).identity != reading.identity


def test_the_reading_is_frozen_and_rejects_a_record_that_describes_no_fold() -> None:
    """The record's own arithmetic is checked, so a pasted count fails at construction."""

    import dataclasses

    model = NoiseModel()
    model.add(("h", "cx"), depolarizing_channel(0.03))
    plan = fold_program(body(), scale_factor=1.4)
    reading = fold_noise_amplification(body(), plan, model)
    with pytest.raises(dataclasses.FrozenInstanceError):
        reading.body_locations = 99  # type: ignore[misc]

    def made(**overrides: object) -> object:
        fields = {
            "strategy": reading.strategy,
            "placement": reading.placement,
            "scale_factor": reading.scale_factor,
            "body_locations": reading.body_locations,
            "folded_locations": reading.folded_locations,
            "covered_instructions": reading.covered_instructions,
            "uncovered_instructions": reading.uncovered_instructions,
            "per_qubit": reading.per_qubit,
            "silent_qubits": reading.silent_qubits,
            "noise_identity": reading.noise_identity,
            "identity": reading.identity,
        }
        fields.update(overrides)
        return FoldNoiseAmplification(**fields)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="per-qubit counts must add up"):
        made(per_qubit=((0, 2, 1), (1, 1, 1)))
    with pytest.raises(ValueError, match="needs at least one channel application"):
        made(body_locations=0)
    with pytest.raises(ValueError, match="cannot apply a channel fewer times"):
        made(folded_locations=1)
    with pytest.raises(ValueError, match="must carry the identity of its inputs"):
        made(identity="")
    with pytest.raises(ValueError, match="must name the noise model it was taken"):
        made(noise_identity="")
    with pytest.raises(ValueError, match="unsupported folding strategy"):
        made(strategy="random")
    with pytest.raises(ValueError, match="unsupported folding placement"):
        made(placement="random")


# --------------------------------------------------------------------------
# Placement: the pair count is the fold's, and where the pairs go is the
# model's. The two rules realize one scale factor and two different programs.
# --------------------------------------------------------------------------


def test_the_two_placements_realize_one_ratio_and_spend_one_pair_count() -> None:
    """Placement decides which instructions carry the repeats, never how many there are.

    The pair count is ``_pair_count``'s, which reads the requested factor and the
    body's length and nothing else, so a placement cannot change the cost of a
    fold in instructions. It changes the program, and through the program the
    identity: a caller who stores an identity is storing which rule was applied.
    """

    model = model_for(("t",), 0.05)
    for factor in (1.4, 2.2, 3.0):
        left = fold_program(body(), scale_factor=factor)
        right = fold_program(
            body(), scale_factor=factor, placement="model_weighted", noise_model=model
        )
        assert left.scale_factor == right.scale_factor
        assert left.added_pairs == right.added_pairs
        assert left.body_instructions == right.body_instructions
        assert left.folded_instructions == right.folded_instructions
        assert left.placement == "body_order"
        assert right.placement == "model_weighted"
        # Same cost, different program, and therefore a different identity.
        assert left.program.content_hash != right.program.content_hash
        assert left.identity != right.identity
        # The folded program is still the requested length, under both rules.
        assert len(right.program.instructions) == right.folded_instructions


def test_body_order_spends_the_pairs_leftmost_and_the_placed_rule_spends_them_first() -> (
    None
):
    """One pair at a factor of 1.4 lands on the body's first instruction or on the noisy one.

    The default rule knows only the body's order, so its single pair goes to
    ``h``. The placed rule reads the model, which names only ``t``, so its single
    pair goes to ``t``. Neither is a longer program than the other.
    """

    model = model_for(("t",), 0.05)
    left = fold_program(body(), scale_factor=1.4)
    right = fold_program(
        body(), scale_factor=1.4, placement="model_weighted", noise_model=model
    )
    assert names(left.program) == ["h", "h", "h", "cx", "rz", "ry", "t"]
    assert names(right.program) == ["h", "cx", "rz", "ry", "t", "tdg", "t"]
    # The pair that the placed rule added sits on the only instruction the model
    # names, and the pair the default rule added sits on one it does not.
    assert names(right.program)[4:6] == ["t", "tdg"]
    assert "t" not in names(left.program)[:2]


def test_a_placed_fold_reaches_noise_the_body_order_rule_cannot() -> None:
    """The capability, measured: the default fold's curve is exactly flat and the placed one is not.

    The model names only ``t``, which is the body's last instruction. One pair
    spent on ``h`` adds no noise of the model's, so every abscissa the default
    rule can reach reads the body's own value and the extrapolation returns the
    unmitigated measurement unchanged: the fold paid instructions for nothing.
    The placed pair is spent on ``t``, so the same factor and the same pair count
    move the observable.
    """

    model = model_for(("t",), 0.05)
    factors = (1.0, 1.4, 1.8)
    plain = run_zne(
        body(),
        ZZ,
        noise_model=model,
        scale_factors=factors,
        order=1,
        fold="gate",
        dtype=torch.complex128,
    )
    placed = run_zne(
        body(),
        ZZ,
        noise_model=model,
        scale_factors=factors,
        order=1,
        fold="gate",
        fold_placement="model_weighted",
        dtype=torch.complex128,
    )
    # Flat to the dtype's own rounding and not to a tolerance chosen here: the
    # fold adds instructions that the model applies no channel to, so the density
    # matrix it propagates is the body's up to the last bit of two contractions
    # taken in a different order. The spread is measured rather than asserted
    # away, because a curve that moved by a real amount would break it.
    spread = max(item.expectation for item in plain.measurements) - min(
        item.expectation for item in plain.measurements
    )
    assert 0.0 <= spread < 64 * float(torch.finfo(torch.float64).eps)
    assert plain.estimate == pytest.approx(plain.unmitigated, abs=1e-15)
    assert [item.expectation for item in placed.measurements] == pytest.approx(
        [0.882059875334, 0.793853847550, 0.714468426569], abs=_EVIDENCE_TOLERANCE
    )
    assert placed.estimate != pytest.approx(plain.estimate)
    # And the reading is what says so, per qubit, before any simulation runs.
    plain_reading = fold_noise_amplification(
        body(), fold_program(body(), scale_factor=1.4), model
    )
    placed_reading = fold_noise_amplification(
        body(),
        fold_program(
            body(), scale_factor=1.4, placement="model_weighted", noise_model=model
        ),
        model,
    )
    assert plain_reading.body_locations == placed_reading.body_locations == 1
    assert plain_reading.folded_locations == 1
    assert placed_reading.folded_locations == 2
    assert plain_reading.per_qubit == ((0, 1, 1),)
    assert placed_reading.per_qubit == ((0, 1, 2),)


def test_a_placed_fold_is_not_a_more_accurate_fold() -> None:
    """What placement buys is reach, and this is the estimate saying it did not buy accuracy.

    One pair on the only noisy instruction doubles that instruction's noise while
    the abscissa moves by 1.4, so the curve is far from linear in the length ratio
    it is fitted against and the linear extrapolation overshoots the ideal value
    further than the flat curve's did. Both estimates are pinned against the ideal,
    which is measured with an empty model rather than assumed to be one.
    """

    from flagquantum.runtime.noise_registry import noisy_density_matrix

    ideal = float(
        ZZ.expectation(
            noisy_density_matrix(body(), NoiseModel(), dtype=torch.complex128)
        )
    )
    assert ideal == pytest.approx(0.980066577841, abs=_EVIDENCE_TOLERANCE)
    model = model_for(("t",), 0.05)
    placed = run_zne(
        body(),
        ZZ,
        noise_model=model,
        scale_factors=(1.0, 1.4, 1.8),
        order=1,
        fold="gate",
        fold_placement="model_weighted",
        dtype=torch.complex128,
    )
    assert abs(placed.estimate - ideal) > abs(placed.unmitigated - ideal)


def test_the_reading_names_the_plan_on_the_qubits_the_noise_landed_on() -> None:
    """A placed fold's amplification is per qubit, and the rule that placed it is recorded.

    The reading carries the placement because it is the reading that says where
    the repeats went, and a record that named only the strategy would leave a
    caller unable to tell a placed reading from a leftmost one.
    """

    model = NoiseModel()
    model.add(("t",), bit_flip_channel(0.05))
    plan = fold_program(
        body(), scale_factor=2.2, placement="model_weighted", noise_model=model
    )
    reading = fold_noise_amplification(body(), plan, model)
    assert reading.placement == "model_weighted"
    assert reading.strategy == "gate"
    # Three pairs, all spent on the one instruction the model names, so the
    # amplification is the same on every qubit that instruction touches and the
    # qubits it does not touch are silent rather than counted as one.
    assert reading.body_locations == 1
    assert reading.folded_locations == 4
    assert reading.per_qubit == ((0, 1, 4),)
    assert reading.amplification == pytest.approx(4.0)
    assert reading.silent_qubits == (1,)
    assert reading.to_dict()["schema"] == FOLD_NOISE_SCHEMA


def test_a_uniform_model_makes_the_two_rules_build_one_program() -> None:
    """One location per instruction leaves nothing to reorder, so the rules agree.

    This is why the placed rule is a second placement rather than a second fold:
    it is the same cycle over a sequence, and the sequence is the body's own when
    the model declares the same noise everywhere. The identity still separates the
    two, because it names the rule and not only the program.
    """

    uniform = NoiseModel()
    uniform.add(("h", "cx", "rz", "ry", "t"), bit_flip_channel(0.01))
    left = fold_program(body(), scale_factor=2.2)
    right = fold_program(
        body(), scale_factor=2.2, placement="model_weighted", noise_model=uniform
    )
    assert left.program.content_hash == right.program.content_hash
    assert left.identity != right.identity


def test_a_placement_with_nowhere_to_go_and_a_circuit_fold_are_refused() -> None:
    """Every refusal names the rule and the argument that would make it legal."""

    model = model_for(("t",), 0.05)
    with pytest.raises(ValueError, match="unsupported folding placement"):
        fold_program(body(), scale_factor=1.4, placement="linear")  # type: ignore[arg-type]
    with pytest.raises(
        ValueError, match="cannot honour the .model_weighted. placement"
    ):
        fold_program(
            body(),
            scale_factor=1.4,
            strategy="circuit",
            placement="model_weighted",
            noise_model=model,
        )
    with pytest.raises(ValueError, match="it needs the model to read"):
        fold_program(body(), scale_factor=1.4, placement="model_weighted")
    with pytest.raises(ValueError, match="applies a channel to no instruction"):
        fold_program(
            body(),
            scale_factor=1.4,
            placement="model_weighted",
            noise_model=NoiseModel(),
        )
    # A body the model names nothing at is refused even when the model is not
    # empty, because the model's rules match none of the body's instructions.
    with pytest.raises(ValueError, match="applies a channel to no instruction"):
        fold_program(
            body(),
            scale_factor=1.4,
            placement="model_weighted",
            noise_model=model_for(("zz",), 0.05),
        )


def test_the_extrapolation_refuses_a_placement_when_it_is_not_folding() -> None:
    """A placement names where a pair goes, so a scaling run has none to name."""

    model = model_for(("t",), 0.05)
    with pytest.raises(ValueError, match="it has no pair to place"):
        run_zne(
            body(),
            ZZ,
            noise_model=model,
            scale_factors=(1.0, 2.0, 3.0),
            order=1,
            fold_placement="model_weighted",
            dtype=torch.complex128,
        )
    with pytest.raises(ValueError, match="unsupported folding placement"):
        run_zne(
            body(),
            ZZ,
            noise_model=model,
            scale_factors=(1.0, 2.0, 3.0),
            order=1,
            fold="gate",
            fold_placement="linear",  # type: ignore[arg-type]
            dtype=torch.complex128,
        )


def test_a_reading_refuses_a_plan_placed_by_a_different_model() -> None:
    """Placement makes the re-derivation depend on the model, and that is checked.

    The reading re-derives the fold to decide that the plan is this body's. A plan
    placed by one model and read against another re-derives to a different
    program -- the pairs sit where the other model's noise is -- so it is refused
    rather than reported with this model's distribution attached to it.
    """

    left = model_for(("t",), 0.05)
    right = model_for(("h",), 0.05)
    plan = fold_program(
        body(), scale_factor=2.2, placement="model_weighted", noise_model=left
    )
    assert fold_noise_amplification(body(), plan, left).placement == "model_weighted"
    with pytest.raises(ValueError, match="the plan is not this body's fold"):
        fold_noise_amplification(body(), plan, right)


def test_both_placements_are_listed_in_the_one_vocabulary() -> None:
    """The values a caller passes are the values a record reports."""

    assert FOLDING_PLACEMENTS == ("body_order", "model_weighted")
    for placement in FOLDING_PLACEMENTS:
        plan = fold_program(
            body(),
            scale_factor=1.4,
            placement=placement,
            noise_model=model_for(("t",), 0.05),
        )
        assert plan.placement == placement
        assert plan.to_dict()["placement"] == placement

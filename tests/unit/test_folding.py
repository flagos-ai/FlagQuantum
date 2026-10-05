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
    FOLDING_ASSUMPTIONS,
    FOLDING_SCHEMA,
    FOLDING_STRATEGIES,
    Hamiltonian,
    HamiltonianTerm,
    fold_program,
    run_zne,
)
from flagquantum.algorithms.folding import _pair_count
from flagquantum.algorithms.folding import fold_program as module_fold_program
from flagquantum.core.ir import (
    CircuitIR,
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
    """

    from flagquantum.runtime.noise_registry import noisy_density_matrix

    model = model_for(("x",), 0.1)
    plan = fold_program(excited(), scale_factor=factor)
    assert plan.folded_instructions == int(factor)
    density = noisy_density_matrix(plan.program, model, dtype=torch.complex128)
    measured = 0.5 * (1.0 - float(Z.expectation(density)))
    expected = (1.0 + (1.0 - 2.0 * 0.1) ** int(factor)) / 2.0
    assert measured == pytest.approx(population, rel=1e-9)
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
        [0.914728760419, 0.796830307476, 0.694127665363], rel=1e-9
    )
    assert [item.expectation for item in scaled.measurements] == pytest.approx(
        [0.914728760419, 0.78405327657, 0.653377695107], rel=1e-9
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
    # A non-square matrix has no conjugate transpose to fold with.
    wide = fq.Circuit(1)
    wide.any(
        0,
        unitary=torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.complex128
        ),
        name="wide",
    )
    with pytest.raises(ValueError, match="not square"):
        fold_program(wide, scale_factor=2.0)


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
    assert folded.assumptions[-2:] == FOLDING_ASSUMPTIONS
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
    assert folded.unmitigated == pytest.approx(-0.8999999544, rel=1e-9)
    assert folded.unmitigated == folded.measurements[0].expectation
    assert folded.estimate == pytest.approx(-0.995600323065, rel=1e-9)
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

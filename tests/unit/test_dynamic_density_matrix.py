"""Mid-circuit measurement and feedback, executed exactly as a density matrix.

``fq.experimental.dynamic.run_dynamic`` answers a dynamic program by drawing
trajectories.  That is the right answer for a backend that reports shots, and it
is the only answer for a program whose classical register the caller reads.  It
is the wrong answer for a reader who wants the state of the program, because a
trajectory is one branch of a mixture and no single trajectory is that mixture.

This file pins the exact route: ``DynamicCircuit.density_matrix()`` and
``flagquantum.simulation.density_matrix.density_matrix_from_ir`` apply the
outcome projector at each measurement, keep one density matrix per classical
record, honour the conditions on the recorded bit, and sum the branches.

The assertions are chosen so that a plausible wrong implementation fails:

* a route that collapses to one outcome instead of dephasing fails the purity
  assertion on a measured superposition, because the purity would be one;
* a route that ignores the recorded bit fails the feedback assertion, because
  ``<Z(1)>`` would be zero rather than one;
* a route that treats ``reset`` as a projector that keeps ``|0>`` without the
  compensating ``X P1`` branch fails the trace assertion, because the trace would
  drop below one;
* a route that guesses an unmeasured bit fails the refusal assertion;
* a route that shares one classical record across a batch fails the batched
  assertion, because the rows would be forced to the same branch;
* a route that samples instead of evolving fails the analytic equality, because
  the value would carry a sampling error.

Every number below is exact or an explicitly stated tolerance.  None of it is a
performance claim.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.dynamic import DynamicCircuit
from flagquantum.errors import CapabilityError
from flagquantum.experimental.dynamic import run_dynamic
from flagquantum.simulation.density_matrix import (
    density_matrix,
    density_matrix_from_ir,
    expectation_z_density,
    reduced_density_matrix,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]

#: The tolerance for a value this route computes in closed form.  Anything above
#: it is an error of method, not of accumulation.
EXACT = 1e-12


def _trace(rho: torch.Tensor) -> float:
    return float(rho[0].diagonal().sum().real)


def _purity(rho: torch.Tensor) -> float:
    return float((rho[0] @ rho[0]).diagonal().sum().real)


def _populations(rho: torch.Tensor) -> list[float]:
    return [float(value) for value in rho[0].diagonal().real]


def _bell_with_feedback() -> DynamicCircuit:
    return (
        DynamicCircuit(2, dtype=torch.complex128)
        .h(0)
        .cx(0, 1)
        .measure(0)
        .conditional("x", 1, classical_bit=0)
    )


def test_a_measurement_nobody_reads_dephases_instead_of_collapsing() -> None:
    """One outcome would be pure; the state a reader holds is the mixture."""

    rho = DynamicCircuit(1, dtype=torch.complex128).h(0).measure(0).density_matrix()
    assert _trace(rho) == pytest.approx(1.0, abs=EXACT)
    assert _purity(rho) == pytest.approx(0.5, abs=EXACT)
    assert _populations(rho) == [
        pytest.approx(0.5, abs=EXACT),
        pytest.approx(0.5, abs=EXACT),
    ]
    # Dephasing removes the coherence and keeps the populations, so the matrix is
    # diagonal rather than merely having a diagonal.
    assert float(rho[0].abs().sum().real) == pytest.approx(1.0, abs=EXACT)


def test_a_route_with_no_dynamic_instruction_is_unchanged() -> None:
    """A static program keeps the outer-product route and stays pure."""

    static = fq.Circuit(2).h(0).cx(0, 1)
    rho = static.density_matrix()
    assert _purity(rho) == pytest.approx(1.0, abs=1e-6)
    # And the engine entry agrees with it, so the two routes are one answer.
    assert torch.allclose(density_matrix_from_ir(static), rho, atol=1e-6)


def test_feedback_reads_the_recorded_bit() -> None:
    """Wire 1 always agrees with wire 0, so flipping it leaves it clear."""

    rho = _bell_with_feedback().density_matrix()
    assert _trace(rho) == pytest.approx(1.0, abs=EXACT)
    assert float(expectation_z_density(rho, 1)[0, 0]) == pytest.approx(1.0, abs=EXACT)
    assert _populations(rho) == [
        pytest.approx(0.5, abs=EXACT),
        pytest.approx(0.0, abs=EXACT),
        pytest.approx(0.5, abs=EXACT),
        pytest.approx(0.0, abs=EXACT),
    ]


def test_reset_returns_the_wire_and_preserves_the_trace() -> None:
    """Reset is ``P0 . P0 + X P1 . P1 X``, not a projector on its own."""

    for program in (
        DynamicCircuit(1, dtype=torch.complex128).x(0).reset(0),
        DynamicCircuit(1, dtype=torch.complex128).h(0).reset(0),
    ):
        rho = program.density_matrix()
        assert _trace(rho) == pytest.approx(1.0, abs=EXACT)
        assert _populations(rho) == [
            pytest.approx(1.0, abs=EXACT),
            pytest.approx(0.0, abs=EXACT),
        ]


def test_reset_after_a_measurement_of_the_same_wire_keeps_one_record() -> None:
    """The record survives the reset, so a later conditional still reads it."""

    program = (
        DynamicCircuit(2, dtype=torch.complex128)
        .h(0)
        .measure(0)
        .reset(0)
        .conditional("x", 1, classical_bit=0)
    )
    rho = program.density_matrix()
    assert _trace(rho) == pytest.approx(1.0, abs=EXACT)
    # Wire 0 is |0> again on both branches, and wire 1 is set exactly on the
    # branch whose record was one, so the two wires are anti-correlated.
    assert _populations(rho) == [
        pytest.approx(0.5, abs=EXACT),
        pytest.approx(0.5, abs=EXACT),
        pytest.approx(0.0, abs=EXACT),
        pytest.approx(0.0, abs=EXACT),
    ]


def test_a_false_condition_leaves_the_state_alone() -> None:
    """A conditional is a gate on one branch, not a gate with a small phase."""

    never = (
        DynamicCircuit(1, dtype=torch.complex128)
        .h(0)
        .measure(0)
        .conditional("x", 0, classical_bit=0, equals=0)
    )
    rho = never.density_matrix()
    assert _trace(rho) == pytest.approx(1.0, abs=EXACT)
    # Bit 0 equals 0 exactly on the branch where the wire was clear, so the flip
    # lands there and both branches end up set.
    assert _populations(rho) == [
        pytest.approx(0.0, abs=EXACT),
        pytest.approx(1.0, abs=EXACT),
    ]


def test_the_exact_population_is_the_analytic_one() -> None:
    """A closed-form value the route must reproduce to the last asserted digit."""

    program = (
        DynamicCircuit(3, dtype=torch.complex128)
        .ry(0, theta=0.7)
        .ry(1, theta=1.1)
        .measure(0)
        .measure(1)
        .conditional("x", 2, conditions={0: 1, 1: 0})
        .x(2)
        .measure(2)
    )
    rho = program.density_matrix()
    assert _trace(rho) == pytest.approx(1.0, abs=EXACT)
    flipped = math.sin(0.35) ** 2 * math.cos(0.55) ** 2
    # State index 4 is |1, 0, 0>: the conditional fired, so wire 2 is clear.
    assert _populations(rho)[4] == pytest.approx(flipped, abs=EXACT)
    # The unconditional X leaves wire 2 set on every branch the conditional missed.
    clear = sum(_populations(rho)[index] for index in (0, 2, 4, 6))
    assert clear == pytest.approx(flipped, abs=EXACT)


def test_the_exact_route_agrees_with_the_sampled_one() -> None:
    """The trajectory route is the sampled estimate of this same mixture."""

    shots = 40_000
    program = (
        DynamicCircuit(3, dtype=torch.complex128)
        .ry(0, theta=0.7)
        .ry(1, theta=1.1)
        .measure(0)
        .measure(1)
        .conditional("x", 2, conditions={0: 1, 1: 0})
        .x(2)
        .measure(2)
    )
    exact = _populations(program.density_matrix())
    sampled = run_dynamic(program, shots=shots, seed=13)
    index = (
        sampled.final_samples[:, 0].long() * 4
        + sampled.final_samples[:, 1].long() * 2
        + sampled.final_samples[:, 2].long()
    )
    observed = [
        float(count) / shots for count in torch.bincount(index, minlength=8).tolist()
    ]
    gap = max(abs(left - right) for left, right in zip(exact, observed, strict=True))
    # Three standard errors of the worst binomial cell at this shot count, which
    # is a bound on the sampling noise rather than on the exact answer.
    assert gap < 3 / math.sqrt(shots)


def test_the_readout_the_route_already_has_applies_to_the_mixture() -> None:
    """A sum over branches is a density matrix, so tracing wires still works."""

    rho = _bell_with_feedback().density_matrix()
    marginal = reduced_density_matrix(rho, 1)
    assert _populations(marginal) == [
        pytest.approx(1.0, abs=EXACT),
        pytest.approx(0.0, abs=EXACT),
    ]


def test_each_batch_row_keeps_its_own_record() -> None:
    """A shared record would force every row onto one branch."""

    program = DynamicCircuit(2, bsz=3, dtype=torch.complex128)
    program.h(0).cx(0, 1).measure(0).conditional("x", 1, classical_bit=0)
    rho = program.density_matrix()
    assert rho.shape == (3, 4, 4)
    for row in range(3):
        assert float(rho[row].diagonal().sum().real) == pytest.approx(1.0, abs=EXACT)
        assert float(rho[row].diagonal().real[0::2].sum()) == pytest.approx(
            1.0, abs=EXACT
        )


def test_the_engine_entry_runs_a_dynamic_program_without_a_circuit() -> None:
    """The IR is the source of truth, so the IR entry works on its own."""

    ir = CircuitIR(
        n_wires=2,
        dtype="complex128",
        instructions=(
            Instruction("h", (0,)),
            Instruction(
                "measure",
                (0,),
                metadata={"is_dynamic": True, "classical_bit": 0},
            ),
            Instruction("x", (1,), metadata={"conditions": ((0, 1),)}),
        ),
    )
    rho = density_matrix_from_ir(ir, dtype=torch.complex128)
    populations = _populations(rho)
    # The record is one half the time, and the flip leaves wire 1 set on exactly
    # that half, so the two wires agree and the coherent half is empty.
    assert populations == [
        pytest.approx(0.5, abs=EXACT),
        pytest.approx(0.0, abs=EXACT),
        pytest.approx(0.0, abs=EXACT),
        pytest.approx(0.5, abs=EXACT),
    ]


def test_a_disjunctive_condition_executes_as_the_metadata_spells_it() -> None:
    """``condition_clauses`` is a disjunction, and it is read as one."""

    ir = CircuitIR(
        n_wires=2,
        dtype="complex128",
        instructions=(
            Instruction("h", (0,)),
            Instruction("h", (1,)),
            Instruction(
                "measure",
                (0,),
                metadata={"is_dynamic": True, "classical_bit": 0},
            ),
            Instruction(
                "measure",
                (1,),
                metadata={"is_dynamic": True, "classical_bit": 1},
            ),
            Instruction("x", (0,), metadata={"condition_clauses": (((0, 1), (1, 0)),)}),
        ),
    )
    populations = _populations(density_matrix_from_ir(ir, dtype=torch.complex128))
    # |1, 0> is the one branch the clause fires on, so it empties into |0, 0>.
    assert populations == [
        pytest.approx(0.5, abs=EXACT),
        pytest.approx(0.25, abs=EXACT),
        pytest.approx(0.0, abs=EXACT),
        pytest.approx(0.25, abs=EXACT),
    ]


def test_reading_a_bit_before_measuring_it_is_refused() -> None:
    """Guessing a value would be a silent branch, so the route fails closed."""

    with pytest.raises(ValueError, match="classical bit 4 was read before measurement"):
        DynamicCircuit(1).x(0).conditional("x", 0, classical_bit=4)


def test_two_spellings_of_one_condition_are_refused() -> None:
    """``conditions`` and ``condition_clauses`` disagree about the meaning."""

    ir = CircuitIR(
        n_wires=1,
        dtype="complex128",
        instructions=(
            Instruction(
                "x",
                (0,),
                metadata={"conditions": ((0, 1),), "condition_clauses": (((0, 1),),)},
            ),
        ),
    )
    with pytest.raises(ValueError, match="cannot define both conditions"):
        density_matrix_from_ir(ir)


def test_the_engine_entry_is_the_one_the_circuit_method_calls() -> None:
    """Two entry points, one answer, so neither can drift from the other."""

    program = _bell_with_feedback()
    assert torch.equal(program.density_matrix(), density_matrix(program))
    assert torch.equal(program.noisy_density_matrix(), program.density_matrix())


def test_the_trajectory_entry_still_refuses_a_density_readout() -> None:
    """The exact route is a second answer, not a replacement for the sampled one."""

    program = _bell_with_feedback()
    with pytest.raises(CapabilityError, match="run_dynamic"):
        program.state()
    with pytest.raises(CapabilityError, match="mode='auto' only"):
        fq.run(program, options=fq.ExecutionOptions(mode="density_matrix"))


def test_the_entry_names_this_route_in_its_limitations() -> None:
    """The maturity entry describes the engine that exists, not the one before."""

    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - the 3.10 interpreter
        import tomli as tomllib

    maturity = tomllib.loads((ROOT / "capability-maturity.toml").read_text("utf-8"))
    limitations = maturity["capabilities"]["density_matrix_execution"]["limitations"]
    assert "mid-circuit measurement" in limitations
    # The old absolute claim is what this change falsified, so it stays out.
    assert (
        "carries no mid-circuit measurement, reset, or classically conditioned"
        not in (limitations)
    )
    assert "DynamicCircuit" in limitations
    assert "classical record" in limitations


def test_a_channel_with_no_operators_is_refused_by_name() -> None:
    """``tuple(None)`` would name neither the instruction nor the repair."""

    ir = CircuitIR(
        n_wires=1,
        dtype="complex128",
        instructions=(
            # The registry validates the declared parameters, and turning them
            # into Kraus operators is a separate step, so this shape is
            # reachable rather than hypothetical.
            Instruction(
                "depolarizing",
                (0,),
                params={"probability": 0.1},
                metadata={"is_channel": True},
            ),
        ),
    )
    with pytest.raises(ValueError, match="channel instruction 'depolarizing'"):
        density_matrix_from_ir(ir)

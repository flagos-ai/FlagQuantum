"""Lengthen a program by an exact identity so its noise scales, and pay in gates.

`flagquantum.algorithms.folding` is reachable through the subpackage surface
only -- `from flagquantum.algorithms import fold_program` -- because the
algorithms package adds no root-level `fq.` name.

The premise the unit rests on, and cannot check: the noise is local to the
instruction that carries it, so repeating an instruction with its own inverse
repeats its noise and nothing else. The fold itself is an exact identity --
``U (U^dagger U)^m = U`` -- so every extra instruction is ideal, and the only
thing the longer program buys is more of the same channel. That is an assumption
about the device rather than about the program, and a fold cannot check it: a
crosstalk term coupling two gates is spread over a pair of instructions rather
than attached to one, so repeating a gate does not repeat that term the way the
device applies it. The plan therefore reports a length ratio, and the strength a
curve sees is a different number: the fold repeats instructions, so it repeats the
error of exactly the instructions the model names. `fold_noise_amplification` is
the reading that separates the two. It counts channel *applications* -- one channel
the model applies at one instruction -- on the body and on the folded program,
which needs no arithmetic on a channel parameter and so holds for a depolarizing
channel, a damping channel and a mixture alike, and it reports the count per qubit
because a gate fold spends its pairs where the strategy puts them rather than where
the observable is sensitive. The count is below the ratio for two distinct reasons,
and a caller who reads only the plan cannot tell which one applies: a pair lands on
one qubit of a multi-qubit body, and the inverse a fold inserts is a different name --
a ``t`` is followed by ``tdg`` -- so a model that lists the gate and not its inverse
stops counting at the inverse even in a fold that repeats whole bodies.

Two strategies are offered and they realize different factors. `circuit` repeats
whole bodies, so its length ratio is ``1 + 2m`` and a request between two odd
integers is met by overshooting. `gate` distributes the same pairs one
instruction at a time, so it reaches a requested factor exactly whenever the
parity allows, and reaches a nearby one in fewer instructions than the circuit
strategy needs for its next odd ratio. Both produce an exact identity; they
differ only in how many instructions they spend to reach a factor, which is the
cost a fold is read as. The realized ratio, not the requested one, is the
abscissa the extrapolation is given, because it is the ratio the noise saw.

Sizes, and why: a five-instruction body on two wires, an ``xx`` readout, and a
depolarizing channel on every gate. The readout is ``xx`` rather than the ``zz``
this repository's Bell-state examples use because a fold amplifies one
instruction at a time, and a ``zz`` readout is exactly blind to the first
instruction a gate fold touches -- the pair lands on ``h`` before the ``cx``,
and depolarizing qubit 0 there moves the state to a mixture of ``|+>`` and
``|->``, both of which give the same ``zz`` after the ``cx``. That blindness is
shown rather than asserted, because the honest reading of a flat folded curve is
that the readout could not see the noise rather than that the noise was not
there. The reference is the same observable measured with an empty model through
``run_zne`` itself, so the distance is the extrapolation's error and not the
simulator's rounding.

Run it with:

    python -m examples.algorithms.folding
"""

from __future__ import annotations

from typing import Any

import torch

from flagquantum.algorithms import (
    FOLDING_PLACEMENTS,
    Hamiltonian,
    HamiltonianTerm,
    fold_noise_amplification,
    fold_program,
    run_zne,
)
from flagquantum.circuit import Circuit
from flagquantum.core.ir import CircuitIR, ensure_circuit_ir
from flagquantum.errors import CapabilityError
from flagquantum.noise import NoiseModel, depolarizing_channel

#: The factor asked of both strategies, chosen because it is one the ``gate``
#: strategy reaches exactly and the ``circuit`` strategy cannot reach at all.
REQUESTED = 1.4
#: One abscissa set per strategy, four points so a degree-two fit is not square
#: and can report a residual. Every factor in the gate set is reached exactly;
#: every factor in the circuit set is an odd ratio, which is all it can reach.
GATE_SCALE_FACTORS = (1.0, 1.4, 1.8, 2.2)
CIRCUIT_SCALE_FACTORS = (1.0, 3.0, 5.0, 7.0)
#: A request set the circuit strategy lands on one length twice for.
COLLIDING_SCALE_FACTORS = (1.0, 1.4, 2.6)
REFERENCE_SCALE_FACTORS = (1.0, 3.0)
DEPOLARIZING_PROBABILITY = 0.03
GATE_NAMES = ("h", "cx", "rz", "ry", "t")
DTYPE = torch.complex128
LABEL_WIDTH = 30


def body() -> Circuit:
    """A five-instruction body whose first gate is the one a small fold repeats."""

    return Circuit(2).h(0).cx(0, 1).rz(0, 0.4).ry(1, 0.6).t(0)


def xx_observable() -> Hamiltonian:
    return Hamiltonian([HamiltonianTerm(1.0, "xx", (0, 1))])


def zz_observable() -> Hamiltonian:
    return Hamiltonian([HamiltonianTerm(1.0, "zz", (0, 1))])


def model() -> NoiseModel:
    return NoiseModel().add(
        GATE_NAMES, depolarizing_channel(DEPOLARIZING_PROBABILITY, dtype=DTYPE)
    )


def report(label: str, value: object) -> None:
    print(f"  {label:<{LABEL_WIDTH}}: {value}")


def state_gap(left: object, right: object) -> float:
    """The largest amplitude difference between two programs' own states.

    Both sides are read at ``complex128``, so the number is the fold's exactness
    rather than the runtime's default complex64 rounding floor.
    """

    def state_of(program: object) -> torch.Tensor:
        ir = ensure_circuit_ir(program)
        exact = CircuitIR(
            n_wires=ir.n_wires, instructions=ir.instructions, dtype="complex128"
        )
        return torch.as_tensor(Circuit.from_ir(exact).state()).reshape(-1)

    return float((state_of(left) - state_of(right)).abs().max())


def measured(
    program: object, observable: Hamiltonian, noise: NoiseModel, *, fold: str | None
) -> float:
    """One exact expectation of ``observable`` under ``noise``, folded or not."""

    factors = GATE_SCALE_FACTORS if fold == "gate" else REFERENCE_SCALE_FACTORS
    return float(
        run_zne(
            program,
            observable,
            noise_model=noise,
            scale_factors=factors,
            order=1,
            fold=fold,  # type: ignore[arg-type]
            dtype=DTYPE,
        )
        .measurements[0]
        .expectation
    )


def refuse(description: str, action: Any) -> None:
    """Run ``action``, print the refusal it must raise, and fail if it returns."""

    try:
        action()
    except (ValueError, CapabilityError) as error:
        report(description, f"{error}")
    else:
        raise SystemExit(
            f"{description} was accepted, and the example exists to refuse it"
        )


def main() -> None:
    circuit = body()
    observable = xx_observable()
    noise = model()

    noiseless = measured(circuit, observable, NoiseModel(), fold=None)

    print("=" * 72)
    print("gate and circuit folding -- flagquantum.algorithms.folding")
    print("=" * 72)
    report("task", "lengthen a program by an exact identity")
    report("premise", "the noise is local to the instruction that carries it, so")
    print(f"  {'':<{LABEL_WIDTH}}  repeating an instruction repeats its noise and")
    print(f"  {'':<{LABEL_WIDTH}}  nothing else; that is an assumption about the")
    print(f"  {'':<{LABEL_WIDTH}}  device rather than about the program, and it is the")
    print(f"  {'':<{LABEL_WIDTH}}  one thing here no program and no model can check")
    print()

    print("inputs")
    report("circuit", "h(0); cx(0, 1); rz(0, 0.4); ry(1, 0.6); t(0)")
    report("observable", "1.0 * xx(0, 1)")
    report("noiseless value", f"{noiseless:.17g}")
    report(
        "channel",
        f"depolarizing at p = {DEPOLARIZING_PROBABILITY} on {', '.join(GATE_NAMES)}",
    )
    report("extrapolation dtype", DTYPE)
    print()

    print(f"the same request of {REQUESTED}, folded both ways")
    plans = {
        strategy: fold_program(circuit, scale_factor=REQUESTED, strategy=strategy)
        for strategy in ("gate", "circuit")
    }
    report("strategy", "requested  realized  body  folded  added  exact")
    for strategy, plan in plans.items():
        report(
            f"  fold {strategy}",
            f"{plan.requested_scale_factor:>8.2f}  {plan.scale_factor:>8.2f}  "
            f"{plan.body_instructions:>4}  {plan.folded_instructions:>6}  "
            f"{plan.added_pairs:>5}  {str(plan.exact):>5}",
        )
    print(f"  {'':<{LABEL_WIDTH}}  -- a fold adds instructions in pairs, so the")
    print(f"  {'':<{LABEL_WIDTH}}  reachable ratios are the body length plus an even")
    print(f"  {'':<{LABEL_WIDTH}}  count; the gate strategy spends the pairs one at a")
    print(f"  {'':<{LABEL_WIDTH}}  time and lands on the request, and the circuit")
    print(f"  {'':<{LABEL_WIDTH}}  strategy repeats whole bodies and overshoots")
    print()

    print("the fold is an identity, so the extra instructions are ideal")
    for strategy, plan in plans.items():
        report(
            f"{strategy} max amplitude gap", f"{state_gap(circuit, plan.program):.3e}"
        )
    print(f"  {'':<{LABEL_WIDTH}}  -- both gaps are complex128 rounding rather than")
    print(f"  {'':<{LABEL_WIDTH}}  the fold's, so the longer program computes the same")
    print(f"  {'':<{LABEL_WIDTH}}  state and every added gate is noise")
    print()

    print("the pair lands on the first instruction, and the readout decides")
    folded = plans["gate"].program
    report(
        "folded program",
        " ".join(i.name for i in ensure_circuit_ir(folded).instructions),
    )
    for name, obs in (("zz(0, 1)", zz_observable()), ("xx(0, 1)", xx_observable())):
        plain = measured(circuit, obs, noise, fold=None)
        small = measured(folded, obs, noise, fold=None)
        report(f"{name} at 1.0 then 1.4", f"{plain:.12f}  {small:.12f}")
    print(f"  {'':<{LABEL_WIDTH}}  -- the fold put its pair on ``h``, which sits")
    print(f"  {'':<{LABEL_WIDTH}}  before the ``cx``; depolarizing qubit 0 there turns")
    print(f"  {'':<{LABEL_WIDTH}}  ``|+>`` into a mixture of ``|+>`` and ``|->``, and")
    print(f"  {'':<{LABEL_WIDTH}}  both give the same ``zz`` after a ``cx``, so the")
    print(f"  {'':<{LABEL_WIDTH}}  ``zz`` curve is flat where the ``xx`` curve is not")
    print()

    print("how much noise the fold added, counted against the model")
    for strategy in ("gate", "circuit"):
        plan = fold_program(circuit, scale_factor=REQUESTED, strategy=strategy)
        reading = fold_noise_amplification(circuit, plan, noise)
        report(
            f"{strategy} length ratio",
            f"{plan.scale_factor:.2f}  in {plan.folded_instructions} instructions",
        )
        report(
            f"{strategy} channel applications",
            f"{reading.body_locations} -> {reading.folded_locations}"
            f"  (x{reading.amplification:.4f})",
        )
        report(
            f"{strategy} per qubit",
            "  ".join(
                f"q{qubit}: {body_count} -> {folded_count}"
                for qubit, body_count, folded_count in reading.per_qubit
            ),
        )
        report(
            f"{strategy} uniform per qubit",
            f"{reading.reached_uniformly}"
            f"  covered {reading.covered_instructions}/"
            f"{reading.covered_instructions + reading.uncovered_instructions}",
        )
    print(f"  {'':<{LABEL_WIDTH}}  -- the ratio is a length statement and this is a")
    print(
        f"  {'':<{LABEL_WIDTH}}  noise statement, and the two differ in the direction"
    )
    print(f"  {'':<{LABEL_WIDTH}}  a caller cannot read off the plan. The gate fold")
    print(f"  {'':<{LABEL_WIDTH}}  spent its pair on the leading ``h``, which sits on")
    print(f"  {'':<{LABEL_WIDTH}}  qubit 0 alone: qubit 0 gains two of its four")
    print(f"  {'':<{LABEL_WIDTH}}  applications and qubit 1 gains none, so the count")
    print(f"  {'':<{LABEL_WIDTH}}  rises by 1.3333 where the length rose by 1.4 and no")
    print(
        f"  {'':<{LABEL_WIDTH}}  single number is the strength either qubit sees. The"
    )
    print(
        f"  {'':<{LABEL_WIDTH}}  circuit fold repeated the whole body and still lands"
    )
    print(f"  {'':<{LABEL_WIDTH}}  at 2.8333 rather than 3.0, because the adjoint it")
    print(f"  {'':<{LABEL_WIDTH}}  inserts carries ``tdg``, a name this model does not")
    print(
        f"  {'':<{LABEL_WIDTH}}  list: qubit 1 reaches 3.0 exactly and qubit 0, which"
    )
    print(f"  {'':<{LABEL_WIDTH}}  carries the ``t``, reaches 2.75")
    print()

    print("where a pair goes is a choice, and the model is what decides it")
    #: A second model, naming one instruction where the first named all five.
    partial = NoiseModel().add(
        ("t",), depolarizing_channel(DEPOLARIZING_PROBABILITY, dtype=DTYPE)
    )
    report("model names", "t only, so four of the five instructions carry no noise")
    placed = {
        placement: fold_program(
            circuit,
            scale_factor=REQUESTED,
            placement=placement,
            noise_model=partial,
        )
        for placement in FOLDING_PLACEMENTS
    }
    report("placement", "requested  realized  body  folded  added")
    for placement, plan in placed.items():
        report(
            f"  place {placement}",
            f"{plan.requested_scale_factor:>8.2f}  {plan.scale_factor:>8.2f}  "
            f"{plan.body_instructions:>4}  {plan.folded_instructions:>6}  "
            f"{plan.added_pairs:>5}",
        )
    for placement, plan in placed.items():
        report(
            f"{placement} program",
            " ".join(i.name for i in ensure_circuit_ir(plan.program).instructions),
        )
    for placement, plan in placed.items():
        reading = fold_noise_amplification(circuit, plan, partial)
        report(
            f"{placement} applications",
            f"{reading.body_locations} -> {reading.folded_locations}"
            f"  (x{reading.amplification:.4f})",
        )
    for placement in FOLDING_PLACEMENTS:
        result = run_zne(
            circuit,
            observable,
            noise_model=partial,
            scale_factors=GATE_SCALE_FACTORS,
            order=2,
            fold="gate",
            fold_placement=placement,
            dtype=DTYPE,
        )
        report(
            f"{placement} curve",
            "  ".join(f"{m.expectation:.9f}" for m in result.measurements),
        )
        report(
            f"{placement} estimate",
            f"{result.estimate:.12f}, {abs(result.estimate - noiseless):.4e} "
            f"from the noiseless value",
        )
    print(f"  {'':<{LABEL_WIDTH}}  -- the two rules realize one ratio and one pair")
    print(f"  {'':<{LABEL_WIDTH}}  count, because both are fixed by the request and")
    print(f"  {'':<{LABEL_WIDTH}}  the body's length. What they do not share is where")
    print(f"  {'':<{LABEL_WIDTH}}  the pair goes. The default rule knows the body's")
    print(f"  {'':<{LABEL_WIDTH}}  order and nothing else, so its pair lands on the")
    print(f"  {'':<{LABEL_WIDTH}}  first ``h``, which this model does not name: the")
    print(f"  {'':<{LABEL_WIDTH}}  curve is flat to the last bit and the extrapolation")
    print(f"  {'':<{LABEL_WIDTH}}  returns the unmitigated point, having paid two")
    print(f"  {'':<{LABEL_WIDTH}}  instructions for no added noise at all. The placed")
    print(f"  {'':<{LABEL_WIDTH}}  rule reads the model and spends the pair on the")
    print(f"  {'':<{LABEL_WIDTH}}  ``t``, so the same cost moves the observable. The")
    print(f"  {'':<{LABEL_WIDTH}}  placed estimate is not the closer one, and that is")
    print(f"  {'':<{LABEL_WIDTH}}  the point: placement buys reach, not accuracy")
    print()

    print("measured curves, one per strategy, at the abscissas each can reach")
    results = {
        "gate": run_zne(
            circuit,
            observable,
            noise_model=noise,
            scale_factors=GATE_SCALE_FACTORS,
            order=2,
            fold="gate",
            dtype=DTYPE,
        ),
        "circuit": run_zne(
            circuit,
            observable,
            noise_model=noise,
            scale_factors=CIRCUIT_SCALE_FACTORS,
            order=2,
            fold="circuit",
            dtype=DTYPE,
        ),
    }
    for strategy, result in results.items():
        plan_rows = [
            fold_program(circuit, scale_factor=factor, strategy=strategy)
            for factor in (
                GATE_SCALE_FACTORS if strategy == "gate" else CIRCUIT_SCALE_FACTORS
            )
        ]
        for measurement, plan in zip(result.measurements, plan_rows, strict=True):
            report(
                f"  {strategy} scale {measurement.scale_factor:g}",
                f"{measurement.expectation:.12f}  "
                f"({plan.folded_instructions} instructions)",
            )
    print(f"  {'':<{LABEL_WIDTH}}  -- the abscissa is the ratio the fold realized")
    print(f"  {'':<{LABEL_WIDTH}}  rather than the factor requested. The gate strategy")
    print(
        f"  {'':<{LABEL_WIDTH}}  reaches a request of 2.2 in 11 instructions, and the"
    )
    print(f"  {'':<{LABEL_WIDTH}}  circuit strategy's next reachable ratio, 3.0, costs")
    print(f"  {'':<{LABEL_WIDTH}}  it 15, so the exact factor is also the cheaper one")
    print()

    print("what folding buys, against the unmitigated point")
    for strategy, result in results.items():
        unmitigated = result.unmitigated
        if unmitigated is None:
            raise SystemExit(
                "no scale factor of 1.0 was given, so there is no unmitigated point "
                "to compare against; every strategy's abscissas start at 1.0"
            )
        report(f"{strategy} unmitigated", f"{unmitigated:.12f}")
        report(f"{strategy} estimate", f"{result.estimate:.12f}")
        report(
            f"{strategy} distance",
            f"{abs(result.estimate - noiseless):.4e} against "
            f"{abs(unmitigated - noiseless):.4e}, residual "
            f"{result.fit.max_residual:.4e}",
        )
    print(f"  {'':<{LABEL_WIDTH}}  -- both estimates land nearer the noiseless value")
    print(f"  {'':<{LABEL_WIDTH}}  than the unmitigated point, and the gate curve")
    print(f"  {'':<{LABEL_WIDTH}}  spans less noise, so it amplifies less of the")
    print(f"  {'':<{LABEL_WIDTH}}  measurement's own error")
    print()

    print("what is refused")
    refuse(
        "fold with scaling",
        lambda: run_zne(
            circuit,
            observable,
            noise_model=noise,
            scale_factors=GATE_SCALE_FACTORS,
            order=2,
            fold="gate",
            scaling=lambda declared, factor: declared,
            dtype=DTYPE,
        ),
    )
    refuse(
        "factor below one",
        lambda: fold_program(circuit, scale_factor=0.5),
    )
    refuse(
        "two requests, one length",
        lambda: run_zne(
            circuit,
            observable,
            noise_model=noise,
            scale_factors=COLLIDING_SCALE_FACTORS,
            order=2,
            fold="circuit",
            dtype=DTYPE,
        ),
    )
    refuse(
        "placement without a fold",
        lambda: run_zne(
            circuit,
            observable,
            noise_model=noise,
            scale_factors=GATE_SCALE_FACTORS,
            order=2,
            fold_placement="model_weighted",
            dtype=DTYPE,
        ),
    )
    refuse(
        "placement of a circuit fold",
        lambda: fold_program(
            circuit,
            scale_factor=REQUESTED,
            strategy="circuit",
            placement="model_weighted",
            noise_model=noise,
        ),
    )
    refuse(
        "a channel in the body",
        lambda: fold_program(
            Circuit(1).x(0).depolarizing(0, DEPOLARIZING_PROBABILITY).x(0),
            scale_factor=3.0,
        ),
    )
    print()

    print("take away")
    print("  folding needs no noise model and no channel parameter: it needs a")
    print("  program, an inverse for every instruction in it, and a device whose")
    print("  noise is local enough that repeating a gate repeats its error. What")
    print("  it returns is a length ratio, and the ratio is realized rather than")
    print("  requested, so a run handed abscissas it did not ask for is reporting")
    print("  what happened rather than what was intended. The two strategies are")
    print("  not interchangeable for that reason: at one request the gate strategy")
    print("  reaches the factor in fewer instructions, and the circuit strategy")
    print("  refuses a set of requests that would land on one length twice. A fold")
    print("  is only as good as the locality it assumes -- and its curve is only as")
    print("  readable as its readout is sensitive, which is why the flat ``zz``")
    print("  curve above is a statement about the observable rather than about the")
    print("  noise. The ratio a plan reports is a length, not a strength: ask")
    print("  fold_noise_amplification for the strength, and ask it per qubit,")
    print("  because a gate fold's pairs land on a subset of the qubits. Where")
    print("  those pairs go is a second choice, and it is the model's rather than")
    print("  the program's: the default rule spends them on the body's leading")
    print("  instructions and can miss every instruction the model names, while the")
    print("  placed rule spends them where the noise was declared. Neither reaches")
    print("  a strength the model does not declare, and neither is the more")
    print("  accurate estimate -- a pair on one loud instruction amplifies that")
    print("  instruction far past the length ratio, so the placed curve is the")
    print("  steeper one and not the better-fitted one.")


if __name__ == "__main__":
    main()

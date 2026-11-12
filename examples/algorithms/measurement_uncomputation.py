# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Composition adapted from Qualtran v0.7.0 simulation/MBUC.ipynb.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/simulation/MBUC.ipynb

"""Check random phases and exact branches of measurement-assisted AND erasure.

Run ``python -m examples.algorithms.measurement_uncomputation``. The numerical
trace uses the existing phased SSA owner; physical append uses the existing
exact AND adjoint, with explicit caller-owned consumed-target reset.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING

import numpy as np

import flagquantum as fq
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_classical import discard_word_function
from flagquantum.compiler.registers.component_feedback import classical_cz_word_function
from flagquantum.compiler.registers.component_graph import iter_component_calls
from flagquantum.compiler.registers.component_measurement import measure_x_word_function
from flagquantum.compiler.registers.primitives.and_compute import and_uncompute_ir
from flagquantum.core.classical_control import instruction_condition_clauses
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import FunctionRecord, ProgramRecord
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.classical_and import and_classical_values
from flagquantum.simulation.classical_controlled_values import controlled_basis_phase
from flagquantum.simulation.classical_measurement import (
    MeasurementOutcomePhase,
    measure_x_basis_phase,
    measure_x_classical_values,
)
from flagquantum.simulation.classical_pauli_values import cz_basis_phase
from flagquantum.simulation.component_classical_state import ComponentClassicalState

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def uncomputation_program(*, feedback: bool) -> ProgramRecord:
    """Keep q1/q2, consume target, and discard the measured classical record.

    All ports are scalar bits in q1,q2,target MSB-first order. The graph owns
    the measurement result until discard; it allocates no quantum scratch.
    """
    bit = quantum_bit_type()
    builder = ComponentProgramBuilder(
        Signature(
            [
                Register("q1", bit),
                Register("q2", bit),
                Register("target", bit, side=RegisterSide.LEFT),
            ]
        ),
        (
            measure_x_word_function("measure_x"),
            classical_cz_word_function("feedback_cz"),
            discard_word_function("discard"),
        ),
    )
    q1, q2 = builder.inputs["q1"], builder.inputs["q2"]
    measured = builder.call_dict("measure_x", {"q": builder.inputs["target"]})["c"]
    if feedback:
        outputs = builder.call_dict(
            "feedback_cz", {"ctrl": measured, "q1": q1, "q2": q2}
        )
        measured, q1, q2 = outputs["ctrl"], outputs["q1"], outputs["q2"]
    builder.call_dict("discard", {"c": measured})
    return builder.finalize({"q1": q1, "q2": q2})


def _values(function: FunctionRecord, values: dict[str, object]) -> object:
    if function.name == "measure_x":
        return measure_x_classical_values(values["q"])
    if function.name == "feedback_cz":
        return dict(values)
    if function.name == "discard":
        return {}
    raise ValueError(f"Unbound measurement-uncomputation recipe {function.name}")


def _cz_phase(**values: object) -> complex:
    q1, q2 = values["q1"], values["q2"]
    if not isinstance(q1, (int, np.integer)) or not isinstance(q2, (int, np.integer)):
        raise ValueError("CZ numerical phase requires integer basis bits")
    return complex(cz_basis_phase(q1, q2))


def _phase(
    function: FunctionRecord, values: dict[str, object]
) -> complex | MeasurementOutcomePhase | None:
    if function.name == "measure_x":
        return measure_x_basis_phase(values["q"])
    if function.name == "feedback_cz":
        return controlled_basis_phase(
            values,
            ("ctrl",),
            ("q1", "q2"),
            (np.asarray(1),),
            _cz_phase,
        )
    return None


def uncomputation_state(
    values: Mapping[str, object],
    *,
    feedback: bool = True,
    rng: np.random.Generator | None = None,
    fixed_values: Mapping[int, object] | None = None,
) -> ComponentClassicalState:
    """Bind numerical actions explicitly; measurement occurrence ID is zero.

    Fixed outcomes override RNG through the existing state owner. This models
    phases on computational basis inputs, not their projection probabilities.
    """
    record = uncomputation_program(feedback=feedback)
    order = tuple(i for i, _, _ in iter_component_calls(record))
    return ComponentClassicalState(
        record,
        order,
        values,
        _values,
        _phase,
        phased=True,
        instance_ids=order,
        rng=rng,
        fixed_values=fixed_values,
    )


def checked_and_adjoint(q1: int, q2: int, target: int) -> dict[str, object]:
    """Reuse the source-equivalent guard for a consistent AND predicate."""
    return dict(
        and_classical_values(
            np.asarray([q1, q2], dtype=np.uint8), target=target, uncompute=True
        )
    )


def phase_trace(state: ComponentClassicalState) -> tuple[complex, ...]:
    """Observe self-owned state after each actual call, never a global variable."""
    trace: list[complex] = []
    while True:
        try:
            state.step()
        except StopIteration:
            return tuple(trace)
        trace.append(state.phase)


def seeded_trials(*, feedback: bool) -> tuple[complex, ...]:
    rng = np.random.default_rng(123)
    phases = []
    for _ in range(10):
        state = uncomputation_state(
            {"q1": 1, "q2": 1, "target": 1}, feedback=feedback, rng=rng
        )
        assert state.simulate() == {"q1": 1, "q2": 1}
        phases.append(state.phase)
    return tuple(phases)


@dataclass(frozen=True)
class EraseAnd:
    """Erase caller-owned wire2; preserve controls0/1 and any spectator3.

    Corrected form is the existing exact AND adjoint. The intentionally naive
    example omits only conditional CZ to demonstrate a relative-phase failure.
    """

    feedback: bool = True

    def instructions(self) -> tuple[Instruction, ...]:
        gates = and_uncompute_ir(0, 1, 2, 0)
        return gates if self.feedback else tuple(g for g in gates if g.name != "cz")

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        return circuit.compose(CircuitIR(circuit.n_qubits, self.instructions()))


def draw_erasure_steps(circuit: fq.Circuit) -> "Figure":
    """Display actual instruction order and feedback instead of hiding conditions."""
    from matplotlib import pyplot as plt

    rows = []
    for i, gate in enumerate(circuit.to_ir().instructions):
        clauses = instruction_condition_clauses(gate)
        condition = " OR ".join(
            " AND ".join(f"c{bit}={value}" for bit, value in clause)
            for clause in clauses
        )
        if gate.name == "measure":
            condition = f"writes c{gate.metadata['classical_bit']}"
        rows.append(
            [str(i), gate.name, ", ".join(map(str, gate.wires)), condition or "always"]
        )
    figure, axes = plt.subplots(figsize=(8, 3))
    axes.axis("off")
    table = axes.table(
        cellText=rows,
        colLabels=["Step", "Primitive", "Quantum wires", "Classical condition"],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    axes.set_title("Ordered measurement and feedback instructions")
    figure.tight_layout()
    return figure


def main() -> None:
    print("Naive seeded phases:", seeded_trials(feedback=False))
    print("Corrected seeded phases:", seeded_trials(feedback=True))
    for q1, q2, outcome in product(range(2), repeat=3):
        state = uncomputation_state(
            {"q1": q1, "q2": q2, "target": q1 & q2}, fixed_values={0: outcome}
        )
        trace = phase_trace(state)
        assert state.finalize() == {"q1": q1, "q2": q2} and state.phase == 1
        print((q1, q2, outcome), trace)


if __name__ == "__main__":
    main()

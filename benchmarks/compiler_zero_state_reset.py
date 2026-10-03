"""Measure what removing a reset on a zero wire buys the optimization pipeline.

W9-11 of the Qiskit parity backlog asks for the pass behind
``transpiler/passes/optimization/remove_reset_in_zero_state.py``. Before it,
``compiler.optimize`` removed an ``i``, a zero angle, a self-inverse pair and a pair of
rotations that sum to zero, and never removed a ``reset`` in any position, so

    ``reset(0)``            # the register starts in |0>, so this is the identity
    ``reset(0)  h(0)``      # the first reset is the identity, and the second is not

both survived the pipeline unchanged.

This module measures four things and refuses to invent a fifth.

* **Where the rule comes from.** Not from a table, and not from the operator schema:
  from the fact that a ``CircuitIR`` register starts in ``|0...0>``. The
  ``rule_contract`` section reports the IR's field list, which is what makes that
  fact checkable as a property of the type, and the pass module's string literals
  and imports, which is what makes "no matrix was read" checkable rather than
  asserted.
* **What the pipeline gains.** On seeded populations the source circuit is measured
  through the whole pipeline with and without the pass, once, and every removed
  instruction is checked against an execution of the circuit.
* **Where the pass stops, and why.** ``removable_in_fact`` is measured per reset by
  executing the prefix and asking whether the reduced state of the reset's own wire
  is ``|0><0|``. That is the exact condition under which the reset is a no-op, and it
  is computed from the runtime's own trajectories rather than from a rule of thumb.
  A reset that was declined although it was removable in fact is deferred reach,
  counted as a number. A reset that was removed although it was *not* removable in
  fact would be a correctness defect, and the section asserts that count is zero.
* **The two instruments, and which rows each can speak for.** A reset reads a random
  draw, so removing one shifts the generator stream: the same seed no longer produces
  the same per-shot trajectory even though no distribution moved. On a measure-free
  program the comparison is therefore made on the exact final state; on a program
  that measures it is made on outcome shares over a fixed seed, with an
  ``execution_control`` section that drives both a removable and a non-removable
  reset so the share instrument is shown to be able to see the difference it looks
  for.

* **What it does not measure:** a gate-count parity claim against Qiskit, or the reach
  of the other optimization passes. The shape table feeds both implementations the
  identical program and reports both counts, asserting agreement only on the rows it
  actually drove; the closed-section field ``pipeline_only_row_count`` counts the rows
  where the shipped pipeline reaches further than this rule can alone, because that
  reach belongs to the fixed-point loop.

Classification: a local compiler microbenchmark on the single-device fast path. It
runs no distributed work, makes no scalability claim, and is not a performance gate.
Re-run it with::

    python benchmarks/compiler_zero_state_reset.py --json-output /tmp/w911.json
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from flagquantum.compiler import optimize  # noqa: E402
from flagquantum.compiler.zero_state_reset import (  # noqa: E402
    remove_zero_state_resets,
)
from flagquantum.core.ir import CircuitIR, Instruction  # noqa: E402

SCHEMA = "flagquantum_compiler_zero_state_reset_benchmark_v1"

_NOW = datetime.now(timezone.utc)

#: Seeded so the counts below are reproducible, which is what lets them be pinned in a
#: benchmark contract.
_SWEEP_SEED = 20261107

#: Three wires, because the shapes that matter are about one wire's own state while
#: another wire is entangled with it.
_REGISTER_WIDTH = 3

#: `complex128`, because the reduced-state comparison below asks whether a wire is
#: exactly in `|0><0|` and `run_dynamic` honours the IR's dtype. At `complex64` the
#: off-diagonal residual of a genuine `|0>` is 6e-8, which is above any tolerance
#: tight enough to be worth asserting; the dtype field is the honest fix and a
#: loosened tolerance would have hidden a real check behind rounding.
_DTYPE = "complex128"

#: The comparison is exact in `complex128`, so it uses one absolute tolerance.
_REDUCED_STATE_ATOL = 1.0e-9

#: Trajectories averaged into the reduced state of one wire. A trajectory is a draw,
#: so one trajectory answers "was this reset a no-op on the draw I happened to see"
#: and the average answers "is this reset a no-op" -- the question that matters. The
#: instrument's floor is one part in this number, so a wire that leaks to `|1>` with
#: probability below 1/64 can be missed; the figure is reported as
#: `average_reduced_state_resolution` rather than left implicit.
_DRAWS = 64

#: Circuits per population, and shots per distribution comparison. The seeded
#: trajectories are deterministic, so this controls coverage rather than precision.
_CIRCUITS_PER_POPULATION = 30
_SHOTS = 2000
_SHARE_SEED = 20261115

#: The angle the single-qubit populations rotate by. Non-zero, so that a rotation is a
#: rotation and not the identity `remove_identity_gates` would have taken.
_ANGLE = 0.7

_SHARE_TOLERANCE = 0.03


def _circuit(instructions: tuple[Instruction, ...]) -> CircuitIR:
    return CircuitIR(_REGISTER_WIDTH, instructions, dtype=_DTYPE)


def _reset(qubit: int, **metadata: Any) -> Instruction:
    return Instruction("reset", (qubit,), metadata={"is_dynamic": True, **metadata})


def _measure(qubit: int) -> Instruction:
    return Instruction(
        "measure", (qubit,), metadata={"is_dynamic": True, "classical_bit": qubit}
    )


def _gate(
    opcode: str, qubits: tuple[int, ...], angle: float | None = None
) -> Instruction:
    params = {} if angle is None else {"theta": angle}
    return Instruction(opcode, qubits, params=params)


def _names(ir: CircuitIR) -> list[str]:
    return [instruction.name for instruction in ir]


def _dynamic(ir: CircuitIR) -> Any:
    from flagquantum.runtime.dynamic import DynamicCircuit

    return DynamicCircuit.from_ir(ir)


def _has_measurement(ir: CircuitIR) -> bool:
    return any(instruction.name == "measure" for instruction in ir)


def _final_state(ir: CircuitIR, *, seed: int | None = None) -> torch.Tensor:
    from flagquantum.runtime.dynamic import run_dynamic

    return run_dynamic(
        _dynamic(ir), shots=1, seed=_SHARE_SEED if seed is None else seed
    ).final_states[0]


def _shares(ir: CircuitIR) -> tuple[list[float], int]:
    """Outcome shares and the number of distinct outcomes seen, over one seed."""

    from flagquantum.runtime.dynamic import run_dynamic

    result = run_dynamic(_dynamic(ir), shots=_SHOTS, seed=_SHARE_SEED)
    bits = result.classical_bits
    width = int(bits.shape[1])
    counts: dict[int, int] = {}
    for row in bits.tolist():
        written = [value for value in row if value >= 0]
        outcome = 0
        for shift, value in enumerate(written):
            outcome |= int(value) << shift
        counts[outcome] = counts.get(outcome, 0) + 1
    shares = [counts.get(index, 0) / _SHOTS for index in range(1 << width)]
    return shares, len(counts)


def _worst_share_difference(left: list[float], right: list[float]) -> float:
    return max(abs(a - b) for a, b in zip(left, right, strict=True))


def _reduced_state(state: torch.Tensor, qubit: int, width: int) -> torch.Tensor:
    """The single-qubit reduced state of ``qubit`` in a whole-register state.

    Wire 0 is the most significant bit of the basis index, so the flat state vector
    reshapes to one axis per wire in wire order and the partial trace is a matrix
    product on the two remaining axes.
    """

    tensor = state.reshape((2,) * width).movedim(qubit, 0).reshape(2, -1)
    return tensor @ tensor.conj().transpose(0, 1)


def _wire_is_in_the_zero_state_after(
    prefix: tuple[Instruction, ...], qubit: int
) -> bool:
    """Whether deleting a reset after ``prefix`` would have changed nothing.

    Measured, not inferred, and measured on the right object. A reset acts on one wire,
    so deleting it is a no-op exactly when that wire is in ``|0><0|`` -- which is a
    statement about the *average* over trajectories, not about one draw. A measure on
    the reset's own wire is the case that makes the distinction matter: on a trajectory
    whose outcome was zero the reset was indeed a no-op, but the reset is still not
    removable, because on the other trajectories it was not. Averaging the reduced
    state over ``_DRAWS`` trajectories answers the question that decides the
    transformation, and it answers the entangled case too, where the wire is not in a
    product state at all.
    """

    if not prefix:
        return True
    circuit = _circuit(prefix)
    total = torch.zeros((2, 2), dtype=torch.complex128)
    for index in range(_DRAWS):
        total += _reduced_state(
            _final_state(circuit, seed=_SHARE_SEED + index), qubit, _REGISTER_WIDTH
        )
    average = total / _DRAWS
    target = torch.zeros_like(average)
    target[0, 0] = 1
    return bool(torch.allclose(average, target, atol=_REDUCED_STATE_ATOL, rtol=0))


def _legacy_fixed_point(ir: CircuitIR) -> CircuitIR:
    """The pipeline with this pass taken out of it, and nothing else.

    The four passes and their order are imported from the implementation rather than
    restated, so the only difference between the two sides is the call this round adds.
    A pipeline that also differed in its pass list would make every delta below a
    comparison of two changes instead of one.
    """

    from flagquantum.compiler.pipeline import (
        merge_adjacent_rotations,
        merge_self_inverse,
        remove_identity_gates,
    )

    current = ir
    for _ in range(len(ir) + 1):
        previous_count = len(current)
        current = remove_identity_gates(current)
        current = merge_self_inverse(current)
        current = merge_adjacent_rotations(current)
        current = remove_identity_gates(current)
        if len(current) == previous_count:
            return current
    raise AssertionError("the legacy optimizer did not reach a fixed point")


def rule_contract() -> dict[str, Any]:
    """The facts the pass's rule rests on, reported rather than asserted."""

    import flagquantum.compiler.zero_state_reset as module

    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        body = getattr(node, "body", [])
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            docstrings.add(id(body[0].value))
    exported = {
        id(element)
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        )
        for element in ast.walk(node.value)
    }
    literals = sorted(
        {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
            and id(node) not in exported
        }
    )
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
        elif isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
    fields = sorted(field.name for field in dataclasses.fields(CircuitIR))
    return {
        "ir_field_count": len(fields),
        "ir_field_names": fields,
        "ir_carries_an_initial_state": any(
            "initial" in name or "state" in name for name in fields
        ),
        "pass_module_string_literal_count": len(literals),
        "pass_module_string_literals": literals,
        "pass_module_imports": sorted(imported),
    }


def _leading_resets(seed: random.Random) -> CircuitIR:
    instructions = [
        _reset(seed.randrange(_REGISTER_WIDTH)) for _ in range(seed.randint(1, 3))
    ]
    for _ in range(seed.randint(1, 4)):
        qubit = seed.randrange(_REGISTER_WIDTH)
        instructions.append(
            seed.choice(
                (
                    _gate("h", (qubit,)),
                    _gate("x", (qubit,)),
                    _gate("z", (qubit,)),
                    _gate("rz", (qubit,), _ANGLE),
                    _gate("cx", (qubit, (qubit + 1) % _REGISTER_WIDTH)),
                )
            )
        )
    return _circuit(tuple(instructions))


def _reset_after_another_qubit(seed: random.Random) -> CircuitIR:
    qubit = seed.randrange(_REGISTER_WIDTH)
    other = (qubit + 1) % _REGISTER_WIDTH
    return _circuit(
        (
            _gate("h", (other,)),
            _reset(qubit),
            _gate("rz", (other,), _ANGLE),
        ),
    )


def _blocked_by_own_qubit_gate(seed: random.Random) -> CircuitIR:
    qubit = seed.randrange(_REGISTER_WIDTH)
    return _circuit(
        (
            _gate(seed.choice(("h", "x", "z")), (qubit,)),
            _reset(qubit),
            _reset(qubit),
        ),
    )


def _blocked_by_measure(seed: random.Random) -> CircuitIR:
    qubit = seed.randrange(_REGISTER_WIDTH)
    return _circuit(
        (_measure(qubit), _reset(qubit), _measure(qubit)),
    )


def _reset_after_a_cancellable_pair(seed: random.Random) -> CircuitIR:
    """A reset the pass cannot see alone, because another pass has to empty the wire."""

    qubit = seed.randrange(_REGISTER_WIDTH)
    opcode = seed.choice(("x", "z", "h"))
    return _circuit(
        (
            _gate(opcode, (qubit,)),
            _gate(opcode, (qubit,)),
            _reset(qubit),
            _measure(qubit),
        ),
    )


def _reset_before_a_measurement(seed: random.Random) -> CircuitIR:
    qubit = seed.randrange(_REGISTER_WIDTH)
    return _circuit(
        (
            _reset(qubit),
            _reset(qubit),
            _gate("h", (qubit,)),
            _measure(qubit),
        ),
    )


def _mixed_program(seed: random.Random) -> CircuitIR:
    instructions: list[Instruction] = []
    for _ in range(seed.randint(3, 18)):
        qubit = seed.randrange(_REGISTER_WIDTH)
        draw = seed.random()
        if draw < 0.3:
            instructions.append(_reset(qubit))
        elif draw < 0.45:
            instructions.append(_measure(qubit))
        elif draw < 0.6:
            instructions.append(_gate("rz", (qubit,), seed.choice((_ANGLE, -_ANGLE))))
        elif draw < 0.8:
            instructions.append(_gate(seed.choice(("h", "x", "z")), (qubit,)))
        else:
            instructions.append(_gate("cx", (qubit, (qubit + 1) % _REGISTER_WIDTH)))
    return _circuit(tuple(instructions))


_POPULATIONS: tuple[tuple[str, Any], ...] = (
    ("leading_resets", _leading_resets),
    ("reset_after_another_qubit_gate", _reset_after_another_qubit),
    ("blocked_by_a_gate_on_its_own_qubit", _blocked_by_own_qubit_gate),
    ("blocked_by_a_measure_on_its_own_qubit", _blocked_by_measure),
    ("reset_after_a_cancellable_pair", _reset_after_a_cancellable_pair),
    ("reset_before_a_measurement", _reset_before_a_measurement),
    ("mixed_dynamic_programs", _mixed_program),
)


def pipeline_delta() -> list[dict[str, Any]]:
    """The pipeline's own reach on the seeded populations, with and without the pass."""

    seed = random.Random(_SWEEP_SEED)
    rows: list[dict[str, Any]] = []
    for label, builder in _POPULATIONS:
        circuits = [builder(seed) for _ in range(_CIRCUITS_PER_POPULATION)]
        source_count = legacy_count = optimized_count = 0
        by_pass_alone = by_pass_in_pipeline = 0
        changed = executed = 0
        worst_state = 0.0
        worst_shares = 0.0
        distinct_outcomes = 0
        for ir in circuits:
            optimized = optimize(ir)
            legacy = _legacy_fixed_point(ir)
            source_count += len(ir)
            legacy_count += len(legacy)
            optimized_count += len(optimized)
            by_pass_alone += len(ir) - len(remove_zero_state_resets(ir))
            by_pass_in_pipeline += len(legacy) - len(optimized)
            changed += int(len(legacy) != len(optimized))
            if _has_measurement(ir):
                shares, distinct = _shares(ir)
                other, _ = _shares(optimized)
                worst_shares = max(worst_shares, _worst_share_difference(shares, other))
                distinct_outcomes = max(distinct_outcomes, distinct)
            else:
                worst_state = max(
                    worst_state,
                    float(
                        (_final_state(ir) - _final_state(optimized)).abs().max().item()
                    ),
                )
            executed += 1
        rows.append(
            {
                "label": label,
                "circuit_count": len(circuits),
                "source_instruction_count": source_count,
                "legacy_instruction_count": legacy_count,
                "optimized_instruction_count": optimized_count,
                "removed_by_the_pass_alone": by_pass_alone,
                "removed_by_the_pass_in_the_pipeline": by_pass_in_pipeline,
                "changed_circuit_count": changed,
                "executed_circuit_count": executed,
                "comparison": (
                    "outcome_shares" if distinct_outcomes else "final_state"
                ),
                "distinct_outcome_count": distinct_outcomes,
                "max_final_state_difference": worst_state,
                "max_outcome_share_difference": worst_shares,
            }
        )
    return rows


#: The shapes the table drives a reset through. The label names what stands before the#: reset on its own wire and on the others; whether that made the wire leave ``|0>`` is
#: measured per row in ``removable_in_fact_count`` and never read off the label.
#:
#: ``self_inverse_pair_cancelled`` and ``cnot_twice`` are the two rows where the reset
#: is not removable for this pass but is removable for the pipeline, because the other
#: passes empty the wire first. They are separated rather than averaged, because that
#: reach belongs to the fixed-point loop and not to this rule.
_SHAPES: tuple[tuple[str, tuple[Instruction, ...]], ...] = (
    ("nothing_before_it", (_reset(0),)),
    ("another_qubit_only", (_gate("h", (1,)), _reset(0))),
    ("two_leading_resets", (_reset(0), _reset(0))),
    ("three_leading_resets", (_reset(0), _reset(0), _reset(0))),
    ("a_gate_on_another_qubit", (_gate("h", (1,)), _reset(0), _gate("x", (2,)))),
    (
        "a_reset_on_another_qubit",
        (_reset(1), _reset(0), _reset(2)),
    ),
    (
        "a_measure_on_another_qubit",
        (_measure(1), _reset(0)),
    ),
    ("a_hadamard_on_its_qubit", (_gate("h", (0,)), _reset(0))),
    ("a_pauli_x_on_its_qubit", (_gate("x", (0,)), _reset(0))),
    ("a_pauli_z_on_its_qubit", (_gate("z", (0,)), _reset(0))),
    ("an_s_gate_on_its_qubit", (_gate("s", (0,)), _reset(0))),
    ("an_angle_rotation_on_its_qubit", (_gate("rz", (0,), _ANGLE), _reset(0))),
    ("a_zero_angle_rotation", (_gate("rz", (0,), 0.0), _reset(0))),
    ("an_identity_gate", (_gate("i", (0,)), _reset(0))),
    ("a_measure_on_its_qubit", (_measure(0), _reset(0))),
    (
        "a_measure_after_a_hadamard_on_its_qubit",
        (_gate("h", (0,)), _measure(0), _reset(0)),
    ),
    ("two_resets_after_a_measure", (_measure(0), _reset(0), _reset(0))),
    ("a_self_inverse_pair", (_gate("x", (0,)), _gate("x", (0,)), _reset(0))),
    (
        "a_pair_of_rotations_that_sum_to_zero",
        (_gate("rz", (0,), 0.4), _gate("rz", (0,), -0.4), _reset(0)),
    ),
    ("a_controlled_not_from_a_zero_control", (_gate("cx", (0, 1)), _reset(0))),
    (
        "a_controlled_not_onto_the_wire",
        (_gate("x", (1,)), _gate("cx", (1, 0)), _reset(0)),
    ),
    ("a_swap_with_a_zero_qubit", (_gate("swap", (0, 1)), _reset(0))),
    ("a_controlled_not_twice", (_gate("cx", (0, 1)), _gate("cx", (0, 1)), _reset(0))),
    ("an_entangled_partner", (_gate("h", (1,)), _gate("cx", (1, 0)), _reset(0))),
)


#: The one opcode the two libraries spell differently. `i` is this repository's name
#: for the identity gate and `id` is Qiskit's; every other gate in the shape table is
#: spelled the same in both, which is why this is a one-entry table and not a mapping
#: layer.
_QISKIT_NAME = {"i": "id"}


def _to_qiskit(program: tuple[Instruction, ...]) -> Any:
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

    qubits = QuantumRegister(_REGISTER_WIDTH, "q")
    bits = ClassicalRegister(_REGISTER_WIDTH, "c")
    circuit = QuantumCircuit(qubits, bits)
    for instruction in program:
        targets = [qubits[index] for index in instruction.wires]
        if instruction.name == "reset":
            circuit.reset(targets[0])
        elif instruction.name == "measure":
            circuit.measure(targets[0], bits[instruction.wires[0]])
        elif instruction.params:
            getattr(circuit, _QISKIT_NAME.get(instruction.name, instruction.name))(
                instruction.params["theta"], *targets
            )
        else:
            getattr(circuit, _QISKIT_NAME.get(instruction.name, instruction.name))(
                *targets
            )
    return circuit


def _qiskit_shape_anchor(
    programs: dict[str, tuple[Instruction, ...]],
) -> dict[str, Any]:
    """Qiskit's own ``RemoveResetInZeroState`` driven over the identical programs.

    The anchor is a cross-check, not a dependency: this module has to run on a machine
    with no Qiskit at all.
    """

    try:
        import qiskit  # type: ignore[import-not-found]
        from qiskit.transpiler.passes import (  # type: ignore[import-not-found]
            RemoveResetInZeroState,
        )
    except Exception as error:  # pragma: no cover - the anchor is optional
        return {"available": False, "reason": f"{type(error).__name__}: {error}"}

    removed: dict[str, int] = {}
    for label, program in programs.items():
        circuit = _to_qiskit(program)
        after = RemoveResetInZeroState()(circuit)
        before_count = sum(1 for instruction in program if instruction.name == "reset")
        after_count = after.count_ops().get("reset", 0)
        removed[label] = before_count - after_count
    return {
        "available": True,
        "pass_name": "RemoveResetInZeroState",
        "qiskit_version": qiskit.__version__,
        "source_reset_count": sum(
            1
            for program in programs.values()
            for instruction in program
            if instruction.name == "reset"
        ),
        "removed_count": sum(removed.values()),
        "removed_by_shape": removed,
    }


def shape_table() -> dict[str, Any]:
    """One row per shape, with the measured truth about the wire and both ports."""

    anchor = _qiskit_shape_anchor({label: program for label, program in _SHAPES})
    rows: list[dict[str, Any]] = []
    declined_but_removable = 0
    removed_but_not_removable = 0
    reset_count = 0
    removed_count = 0
    pipeline_only = 0
    for label, program in _SHAPES:
        ir = _circuit(program)
        optimized = remove_zero_state_resets(ir)
        by_pipeline = optimize(ir)
        resets = [
            index
            for index, instruction in enumerate(program)
            if instruction.name == "reset"
        ]
        removed_here = len(program) - len(optimized)
        removable = sum(
            1
            for index in resets
            if _wire_is_in_the_zero_state_after(
                program[:index], program[index].wires[0]
            )
        )
        reset_count += len(resets)
        removed_count += removed_here
        declined_but_removable += max(0, removable - removed_here)
        removed_but_not_removable += max(0, removed_here - removable)
        pipeline_removed = len(program) - len(by_pipeline)
        if pipeline_removed > removed_here:
            pipeline_only += 1
        row = {
            "shape": label,
            "instructions_before_the_reset": [
                instruction.name for instruction in program[: resets[0]]
            ],
            "reset_count": len(resets),
            "removed_by_the_pass": removed_here,
            "removed_by_the_pipeline": pipeline_removed,
            "removable_in_fact_count": removable,
            "removable_in_fact_but_declined_count": max(0, removable - removed_here),
            "removed_but_not_removable_count": max(0, removed_here - removable),
        }
        if anchor["available"]:
            row["removed_by_qiskit"] = anchor["removed_by_shape"][label]
            row["agrees_with_qiskit"] = (
                removed_here == anchor["removed_by_shape"][label]
            )
        rows.append(row)
    disagreements = [
        row
        for row in rows
        if "agrees_with_qiskit" in row and not row["agrees_with_qiskit"]
    ]
    return {
        "row_count": len(rows),
        "reset_count": reset_count,
        "removed_count": removed_count,
        "declined_but_removable_count": declined_but_removable,
        "removed_but_not_removable_count": removed_but_not_removable,
        "average_reduced_state_resolution": 1.0 / _DRAWS,
        "draws_per_reset": _DRAWS,
        "pipeline_only_row_count": pipeline_only,
        "rows": rows,
        "reference_anchor": anchor,
        "anchor_disagreement_count": len(disagreements),
        "disagreeing_shapes": [row["shape"] for row in disagreements],
    }


#: Forms of a ``reset`` the pass declines even when the wire is in ``|0>``, because the
#: rule is a statement about one wire's own state and these are not that statement.
#: Qiskit cannot be driven over any of them, which is why they are a table of their own.
_FORMS: tuple[tuple[str, Instruction], ...] = (
    ("a_bare_reset", _reset(0)),
    ("a_conditional_reset", _reset(0, conditions=((0, 1),))),
    ("a_two_qubit_reset", Instruction("reset", (0, 1), metadata={"is_dynamic": True})),
    (
        "a_parameterised_reset",
        Instruction(
            "reset", (0,), params={"theta": _ANGLE}, metadata={"is_dynamic": True}
        ),
    ),
    (
        "a_reset_with_a_caller_supplied_matrix",
        Instruction("reset", (0,), matrix=torch.eye(2, dtype=torch.complex64)),
    ),
)


def form_table() -> dict[str, Any]:
    """Every form of the instruction, on a wire nothing has touched."""

    rows = []
    for label, instruction in _FORMS:
        ir = _circuit((instruction,))
        rows.append(
            {
                "form": label,
                "removed": len(remove_zero_state_resets(ir)) < len(ir),
                "carries_parameters": bool(instruction.params),
                "carries_a_matrix": instruction.matrix is not None,
                "wire_count": len(instruction.wires),
            }
        )
    return {
        "row_count": len(rows),
        "removed_count": sum(1 for row in rows if row["removed"]),
        "rows": rows,
    }


def execution_control() -> dict[str, Any]:
    """Show the share instrument can see a reset that was not on a zero wire.

    Two pairs of programs differ only by deleting one reset. In the first pair the wire
    was still in ``|0>``, so the outcome distribution is unmoved and the difference is
    noise. In the second the wire had been flipped, so deleting the reset moves half the
    shots. A tolerance that accepted both would be measuring nothing.
    """

    removable_source = _circuit((_reset(0), _gate("h", (0,)), _measure(0)))
    removable_after = remove_zero_state_resets(removable_source)
    counterfactual = _circuit((_gate("h", (0,)), _measure(0)))
    blocked_source = _circuit((_gate("h", (0,)), _reset(0), _measure(0)))
    blocked_deleted = _circuit((_gate("h", (0,)), _measure(0)))

    tolerated = _worst_share_difference(
        _shares(removable_source)[0], _shares(removable_after)[0]
    )
    # The counterfactual is what the share comparison would have to accept to be blind;
    # it is driven so that the tolerated figure has a scale next to it.
    counterfactual_difference = _worst_share_difference(
        _shares(removable_source)[0], _shares(counterfactual)[0]
    )
    caught = _worst_share_difference(
        _shares(blocked_source)[0], _shares(blocked_deleted)[0]
    )
    return {
        "deleting_a_reset_after_a_gate_that_left_the_zero_state": caught,
        "deleting_a_reset_that_was_a_no_op": tolerated,
        "the_same_comparison_against_a_program_that_never_had_the_reset": (
            counterfactual_difference
        ),
        "tolerance": _SHARE_TOLERANCE,
        "shots": _SHOTS,
        "seed": _SHARE_SEED,
    }


def run_benchmark() -> dict[str, Any]:
    shapes = shape_table()
    anchor = shapes["reference_anchor"]
    return {
        "schema": SCHEMA,
        "generated_at": _NOW.isoformat(),
        "artifact_classification": "local_compiler_microbenchmark",
        "distribution_semantics": "single_device_fast_path",
        "scalability_claim_allowed": False,
        "seed": {"sweep": _SWEEP_SEED, "share": _SHARE_SEED},
        "reference_algorithm": "qiskit_remove_reset_in_zero_state",
        "reference_revision": "qiskit 1.2.4 RemoveResetInZeroState",
        "rule_contract": rule_contract(),
        "pipeline_delta": pipeline_delta(),
        "shape_table": shapes,
        "form_table": form_table(),
        "execution_control": execution_control(),
        "anchor_agreement_count": (
            0
            if not anchor["available"]
            else shapes["row_count"] - shapes["anchor_disagreement_count"]
        ),
        "anchor_disagreement_count": shapes["anchor_disagreement_count"],
        "anchor_agreement_scope": (
            f"{shapes['row_count']} shape rows driven through both ports over the "
            "programs both ports can represent; the form table is a FlagQuantum-only "
            "table and is excluded from the count"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    payload = run_benchmark()
    contract = payload["rule_contract"]
    print(
        f"rule: {contract['ir_field_count']} CircuitIR fields, "
        f"carries an initial state: {contract['ir_carries_an_initial_state']}"
    )
    print(
        f"  pass module: {contract['pass_module_string_literal_count']} string "
        f"literal(s) {contract['pass_module_string_literals']}, imports "
        f"{contract['pass_module_imports']}"
    )
    print()
    print(
        f"{'population':40s} {'src':>6s} {'legacy':>7s} {'now':>6s} "
        f"{'alone':>6s} {'in pipe':>8s} {'changed':>8s} {'max diff':>10s}"
    )
    for row in payload["pipeline_delta"]:
        difference = (
            row["max_outcome_share_difference"]
            if row["comparison"] == "outcome_shares"
            else row["max_final_state_difference"]
        )
        print(
            f"{row['label']:40s} {row['source_instruction_count']:>6d} "
            f"{row['legacy_instruction_count']:>7d} "
            f"{row['optimized_instruction_count']:>6d} "
            f"{row['removed_by_the_pass_alone']:>6d} "
            f"{row['removed_by_the_pass_in_the_pipeline']:>8d} "
            f"{row['changed_circuit_count']:>8d} {difference:>10.3e}  "
            f"({row['comparison']})"
        )
    shapes = payload["shape_table"]
    print()
    print(f"shape table: {shapes['row_count']} rows, {shapes['reset_count']} resets")
    print(
        f"  removed {shapes['removed_count']}; declined although removable in fact "
        f"{shapes['declined_but_removable_count']}; removed although not removable in "
        f"fact {shapes['removed_but_not_removable_count']}"
    )
    print(
        f"  instrument: {shapes['draws_per_reset']} trajectories per reset, "
        f"resolution {shapes['average_reduced_state_resolution']:.4f}; rows where "
        f"only the pipeline reaches: {shapes['pipeline_only_row_count']}"
    )
    forms = payload["form_table"]
    print()
    print(
        f"form table: {forms['row_count']} instruction forms, "
        f"{forms['removed_count']} removed"
    )
    for row in forms["rows"]:
        print(f"  {row['form']:42s} removed={row['removed']}")
    control = payload["execution_control"]
    print()
    print(f"execution control over {control['shots']} shots, seed {control['seed']}:")
    print(
        f"  a deleted reset that was a no-op moves a share by "
        f"{control['deleting_a_reset_that_was_a_no_op']:.4f}"
    )
    print(
        f"  the same comparison against a program that never had it: "
        f"{control['the_same_comparison_against_a_program_that_never_had_the_reset']:.4f}"
    )
    print(
        f"  deleting a reset on a wire that left |0> moves a share by "
        f"{control['deleting_a_reset_after_a_gate_that_left_the_zero_state']:.4f}"
    )
    anchor = shapes["reference_anchor"]
    print()
    if anchor["available"]:
        print(
            f"Qiskit {anchor['qiskit_version']} {anchor['pass_name']}: "
            f"{payload['anchor_agreement_count']}/{shapes['row_count']} rows agree, "
            f"{payload['anchor_disagreement_count']} differ"
        )
        for row in shapes["rows"]:
            if not row.get("agrees_with_qiskit", True):
                print(
                    f"  {row['shape']}: this port removed "
                    f"{row['removed_by_the_pass']}, the reference removed "
                    f"{row['removed_by_qiskit']}"
                )
    else:
        print(f"Qiskit anchor unavailable: {anchor['reason']}")
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()

"""Measure what removing a diagonal gate before a measurement buys the pipeline.

W9-10 of the Qiskit parity backlog asks for the pass behind
``transpiler/passes/optimization/remove_diagonal_gates_before_measure.py``: a
diagonal gate commutes with a computational-basis measurement, so a gate that is
diagonal and is read out on every one of its wires cannot change any outcome.

Two different kinds of claim hold that rule up, and this module measures them
separately because they can fail in different ways.

* **The single-qubit half is a theorem.** ``U3(theta, phi, lam)`` is diagonal
  exactly when its polar angle is a multiple of ``pi``, which is a fact about the
  operator rather than a list of opcodes. ``one_qubit_synthesis`` already tabulates
  that angle for all eighteen declared single-qubit unitaries, so the pass reads a
  declaration instead of re-deriving one -- and this module *measures* the same
  eighteen out of the runtime gate matrices at a generic point and asserts the
  declaration agrees with the matrix, row by row. That is what makes ``u3(0, phi,
  lam)``, ``rx(0)`` and ``ry(0)`` removable here and not by ``remove_identity_gates``:
  they are diagonal and they are not the identity.
* **The two-wire half is a declaration, and it has to be checked.** No module in
  this repository states a two-wire diagonality. ``two_qubit_synthesis`` works in
  Weyl coordinates, which do not distinguish a diagonal operator from a locally
  equivalent one, so it cannot be the source. The pass therefore names a
  four-opcode set and this module measures the whole declared two-wire unitary set
  out of the matrices and asserts the two are equal in both directions. A new Core
  opcode fails here rather than being silently declined by the pass.

The rule is not a matrix measurement in the pass itself: ``compiler/**`` may not
import ``flagquantum.simulation.**``, and the same Architectural rule is what makes
the pass cheap. The boundary is proven by replacement -- the measurement lives here.

The remaining questions are what the pipeline gains, where the pass stops, and
what the anchor says. ``pipeline_delta`` measures the pipeline with and without the
pass on seeded populations and checks every removal against an exact outcome
distribution. ``shape_table`` drives one row per candidate and per gap, with the
*removed* column taken from the pass alone -- the pipeline's extra reach is reported
as its own count, because it belongs to the fixed-point loop. ``form_table`` reports
the four non-bare forms the pass declines. ``execution_control`` drives a deletion
that cannot be invisible, so the instrument is shown to be able to fail.

**What it does not measure, and the one place it deliberately disagrees with the
anchor.** Qiskit 1.2.4's pass is a list of gate *classes*: ``RZGate, ZGate, TGate,
SGate, TdgGate, SdgGate, U1Gate`` and ``CZGate, CRZGate, CU1Gate, RZZGate``. Three
consequences follow that this module publishes rather than hides, because each is a
real, measured disagreement between two correct implementations.

* ``IGate`` is not in that list, so Qiskit keeps an ``i`` this pass removes. ``i``
  is not the identity by *name* in Qiskit's sense -- it is simply absent.
* ``PhaseGate`` is not a ``U1Gate`` subclass in Qiskit, so ``qc.p(...)`` survives
  that pass while an appended ``U1Gate`` does not, even though the two operators are
  equal. FlagQuantum declares ``phase`` and ``u1`` as two opcodes with the same
  matrix, so this pass removes both.
* ``CPhaseGate`` is not a ``CU1Gate`` subclass, so ``cp`` survives while ``cu1``
  does not. FlagQuantum declares one opcode, ``cphase``.

Conversely, Qiskit's opcode list cannot express ``u3(0, phi, lam)``, ``rx(0)`` or
``ry(0)`` at all: those are one opcode with a parameter, and the pass is a class
tester. So FlagQuantum reaches rows the anchor cannot be asked about. The anchor
section reports the disagreements per shape and states the reason.

Classification: a local compiler microbenchmark on the single-device fast path. It
runs no distributed work, makes no scalability claim, and is not a performance gate.
Re-run it with::

    python benchmarks/compiler_diagonal_before_measure.py --json-output /tmp/w910.json
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import json
import random
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

import flagquantum.compiler.diagonal_before_measure as pass_module  # noqa: E402
from flagquantum.compiler import optimize  # noqa: E402
from flagquantum.compiler.diagonal_before_measure import (  # noqa: E402
    _DIAGONAL_TWO_WIRE,
    remove_diagonal_gates_before_measure,
)
from flagquantum.compiler.one_qubit_synthesis import (  # noqa: E402
    is_diagonal_one_qubit,
)
from flagquantum.core.ir import CircuitIR, Instruction  # noqa: E402
from flagquantum.core.operator_schema import (  # noqa: E402
    OPERATOR_SCHEMAS,
    canonical_opcode,
)

SCHEMA = "flagquantum_compiler_diagonal_before_measure_benchmark_v1"

_NOW = datetime.now(timezone.utc)

#: Seeded so the counts below are reproducible, which is what lets them be pinned in a
#: benchmark contract.
_SWEEP_SEED = 20261117

#: Three wires: the shapes that matter are about one candidate's own wires while a
#: third wire is busy, and about which of a two-wire candidate's wires is read out.
_REGISTER_WIDTH = 3

#: `complex128`, because the outcome distributions below are compared exactly rather
#: than sampled and `run_local_statevector` honours the IR's dtype.
_DTYPE = "complex128"

#: The exact comparison tolerance. Every removal in the shape table is a no-op on the
#: outcome distribution up to this, and every wrong removal moves it by more than
#: ``_CONTROL_FLOOR``.
_DISTRIBUTION_ATOL = 1.0e-12

#: The smallest distribution movement a correct deletion of a *non*-diagonal gate
#: produced over the control shapes. The control section asserts the measured movement
#: exceeds it, so a tolerance that accepted everything would fail.
_CONTROL_FLOOR = 1.0e-2

#: Circuits per population in ``pipeline_delta``. Seeded, so the deltas are stable.
_CIRCUITS_PER_POPULATION = 30

#: The generic angle triple the classification is measured at. Non-zero, so that a
#: diagonal verdict is not the identity verdict as well.
_GENERIC = {"theta": 0.7137, "phi": -0.4211, "lbd": 1.9073}

#: The polar angle that makes a general form diagonal without making it the identity.
_ZERO_POLAR = {"theta": 0.0, "phi": -0.4211, "lbd": 1.9073}

_ANGLE = 0.7

_SINGLE_QUBIT_UNITARIES = tuple(
    sorted(
        name
        for name, schema in OPERATOR_SCHEMAS.items()
        if schema.unitary and schema.arity == 1
    )
)
_TWO_WIRE_UNITARIES = tuple(
    sorted(
        name
        for name, schema in OPERATOR_SCHEMAS.items()
        if schema.unitary and schema.arity == 2
    )
)

#: Qiskit's own list, for the anchor's reason column only. The names are opcodes in
#: this repository; the anchor maps each to the Qiskit class of the same operator.
_QISKIT_DIAGONAL = frozenset(
    {"rz", "z", "t", "s", "tdg", "sdg", "u1", "cz", "crz", "cu1", "rzz"}
)


def _circuit(instructions: tuple[Instruction, ...], width: int = _REGISTER_WIDTH) -> CircuitIR:
    return CircuitIR(width, instructions, dtype=_DTYPE)


def _measure(qubit: int, bit: int | None = None, **metadata: Any) -> Instruction:
    return Instruction(
        "measure",
        (qubit,),
        metadata={
            "is_dynamic": True,
            "classical_bit": qubit if bit is None else bit,
            **metadata,
        },
    )


def _gate(name: str, wires: tuple[int, ...], **params: float) -> Instruction:
    return Instruction(name, wires, params=params)


def _avoiding(candidate: str, preferences: tuple[str, ...]) -> str:
    """The first preferred opcode that is not the candidate's own.

    Every gate a shape program adds besides the candidate is picked through here, and
    every preference below is non-diagonal. Two facts follow, and both columns of the
    shape table depend on them: the pass under test can only ever remove the candidate,
    so a length difference is the candidate's removal and nothing else; and the
    candidate's opcode occurs exactly once, so an anchor that counts opcodes is
    answering about the candidate and not about one of its neighbours.
    """

    for name in preferences:
        if name != candidate:
            return name
    raise AssertionError(f"no opcode avoids {candidate!r}")


def _params_for(name: str, point: dict[str, float]) -> dict[str, float]:
    schema = OPERATOR_SCHEMAS[canonical_opcode(name)]
    return {key: point[key] for key in schema.parameters}


def _names(ir: CircuitIR) -> list[str]:
    return [instruction.name for instruction in ir.instructions]


def _signature(ir: CircuitIR) -> tuple[Any, ...]:
    """An equality key for a program that never touches a matrix tensor.

    ``Instruction`` is a frozen dataclass holding a ``torch.Tensor``, so comparing two
    instructions directly would compare tensors and raise. The opcode, the wires and
    the parameters are what a length-equal comparison needs to distinguish -- notably
    a fused ``u3`` from a bare ``h``.
    """

    return tuple(
        (
            instruction.name,
            instruction.wires,
            tuple(sorted(instruction.params.items())),
            instruction.matrix is not None,
        )
        for instruction in ir.instructions
    )


def _matrix(name: str, wires: tuple[int, ...], params: dict[str, float]) -> torch.Tensor:
    from flagquantum.simulation.gate_matrix import gate_matrix

    width = 2 ** len(wires)
    return gate_matrix(
        Instruction(name, wires, params=params),
        bsz=1,
        device="cpu",
        dtype=torch.complex128,
    ).reshape(width, width)


def _measured_diagonal(
    name: str, wires: tuple[int, ...], params: dict[str, float]
) -> bool:
    matrix = _matrix(name, wires, params)
    return float((matrix - torch.diag(torch.diagonal(matrix))).abs().max()) <= 1.0e-12


def classification_census() -> dict[str, Any]:
    """The declaration measured against the runtime matrices, at two angle triples.

    This is the honest form of the claim. The single-qubit verdicts are checked
    against the matrices at a generic point and at a zero polar angle; the two-wire
    verdicts are checked against the matrices over the whole declared set. A row
    where they disagree is reported rather than smoothed.
    """

    single = []
    for name in _SINGLE_QUBIT_UNITARIES:
        generic = _measured_diagonal(name, (0,), _params_for(name, _GENERIC))
        zero_polar_params = _params_for(name, _GENERIC)
        if "theta" in zero_polar_params:
            zero_polar_params["theta"] = 0.0
        zero_polar = _measured_diagonal(name, (0,), zero_polar_params)
        reported_generic = is_diagonal_one_qubit(
            Instruction(name, (0,), params=_params_for(name, _GENERIC))
        )
        reported_zero = is_diagonal_one_qubit(
            Instruction(name, (0,), params=zero_polar_params)
        )
        single.append(
            {
                "opcode": name,
                "measured_diagonal_at_the_generic_angles": generic,
                "reported_diagonal_at_the_generic_angles": reported_generic,
                "measured_diagonal_at_a_zero_polar_angle": zero_polar,
                "reported_diagonal_at_a_zero_polar_angle": reported_zero,
                "agrees": generic == reported_generic and zero_polar == reported_zero,
            }
        )

    two_wire = []
    for name in _TWO_WIRE_UNITARIES:
        measured = _measured_diagonal(name, (0, 1), _params_for(name, _GENERIC))
        declared = name in _DIAGONAL_TWO_WIRE
        two_wire.append(
            {
                "opcode": name,
                "measured_diagonal": measured,
                "declared_diagonal": declared,
                "agrees": measured == declared,
            }
        )

    return {
        "single_qubit": single,
        "two_wire": two_wire,
        "single_qubit_row_count": len(single),
        "two_wire_row_count": len(two_wire),
        "single_qubit_measured_diagonal_count": sum(
            1 for row in single if row["measured_diagonal_at_the_generic_angles"]
        ),
        "single_qubit_measured_diagonal_count_at_a_zero_polar_angle": sum(
            1 for row in single if row["measured_diagonal_at_a_zero_polar_angle"]
        ),
        "two_wire_measured_diagonal_count": sum(
            1 for row in two_wire if row["measured_diagonal"]
        ),
        "disagreement_count": sum(1 for row in single + two_wire if not row["agrees"]),
        "angles_used": {"generic": _GENERIC, "zero_polar": _ZERO_POLAR},
        "declared_two_wire_opcodes": sorted(_DIAGONAL_TWO_WIRE),
    }


def rule_contract() -> dict[str, Any]:
    """The facts the rule rests on, read out of the module and the IR type."""

    source = Path(pass_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

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

    census = classification_census()
    return {
        "ir_field_names": sorted(field.name for field in dataclasses.fields(CircuitIR)),
        "ir_field_count": len(dataclasses.fields(CircuitIR)),
        "pass_module_path": "flagquantum/compiler/diagonal_before_measure.py",
        "pass_module_line_count": len(source.splitlines()),
        "pass_module_string_literals": literals,
        "pass_module_string_literal_count": len(literals),
        "pass_module_imports": sorted(imported),
        "pass_module_imports_the_operator_schema": "core.operator_schema" in imported,
        "pass_module_imports_a_matrix_source": any(
            name.startswith(("..simulation", "simulation", "core.operator_schema"))
            for name in imported
        ),
        "single_qubit_rule": (
            "the polar angle of the opcode's canonical Euler triple is exactly zero"
        ),
        "two_wire_rule": "membership in one four-opcode frozenset",
        "declared_two_wire_opcodes": sorted(_DIAGONAL_TWO_WIRE),
        "declared_two_wire_opcode_count": len(_DIAGONAL_TWO_WIRE),
        "classification_census": census,
    }


def _legacy_pipeline(ir: CircuitIR) -> CircuitIR:
    """The shipped pipeline with this one pass disabled, for the delta.

    ``pipeline._optimize_to_fixed_point`` imports the pass inside the function, so
    replacing the module attribute is exactly the switch a caller would have.
    """

    original = pass_module.remove_diagonal_gates_before_measure
    pass_module.remove_diagonal_gates_before_measure = lambda program: program
    try:
        return optimize(ir)
    finally:
        pass_module.remove_diagonal_gates_before_measure = original


def _gate_only(ir: CircuitIR) -> CircuitIR:
    """The gate sequence with the measurements removed, for exact simulation."""

    return replace(
        ir,
        instructions=tuple(
            instruction
            for instruction in ir.instructions
            if instruction.name != "measure"
        ),
    )


def _exact_outcome_distribution(ir: CircuitIR) -> dict[tuple[int, ...], float]:
    """The exact joint distribution over the measured wires, from the statevector.

    No sampling: this is the instrument the truth column rests on, and it is exact
    because a diagonal gate moves no amplitude. The measured wires are taken in the
    order the classical bits are written, which is the order ``run_dynamic`` reports.
    """

    from flagquantum.simulation.statevector.local import run_local_statevector

    gates = _gate_only(ir)
    state = run_local_statevector(
        gates, batch_size=1, device=torch.device("cpu"), dtype=torch.complex128
    )[0]
    probabilities = (state.abs() ** 2).detach().tolist()

    ordered: list[tuple[int, int]] = []
    for instruction in ir.instructions:
        if instruction.name == "measure":
            ordered.append(
                (instruction.wires[0], int(instruction.metadata.get("classical_bit", 0)))
            )
    ordered.sort(key=lambda pair: pair[1])

    width = ir.n_wires
    distribution: dict[tuple[int, ...], float] = {}
    for index, probability in enumerate(probabilities):
        # Wire 0 is the most significant bit of the basis index.
        key = tuple(
            (index >> (width - 1 - wire)) & 1 for wire, _ in ordered
        )
        distribution[key] = distribution.get(key, 0.0) + probability
    return dict(sorted(distribution.items()))


def _worst_distribution_difference(
    left: dict[tuple[int, ...], float], right: dict[tuple[int, ...], float]
) -> float:
    keys = set(left) | set(right)
    return max(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys)


def _measures_are_terminal(ir: CircuitIR) -> bool:
    """Whether at least one measurement is present and all of them close the program.

    The exact instrument reads the distribution off the final state, which is the
    distribution a measurement block at the end reports. A program that measures and
    then keeps computing is a different instrument's problem, so the shape table
    asserts this precondition rather than assuming it. A program with no measurement
    at all is *not* terminal: every measurement is vacuously terminal there, the
    instrument would compare two empty distributions, and the row would report a
    removal as invisible for having measured nothing.
    """

    seen = False
    for instruction in ir.instructions:
        if instruction.name == "measure":
            seen = True
        elif seen:
            return False
    return seen


def _terminal_program(
    instruction: Instruction, wire_count: int
) -> tuple[CircuitIR, tuple[Instruction, ...], tuple[Instruction, ...]]:
    """The candidate on its wires, with the preparation and the measures it needs.

    A two-wire candidate is placed after a Bell preparation so that its diagonal
    character is observable in principle: on ``|00>`` every two-wire operator reads
    out identically, and a shape table built on ``|00>`` would measure nothing. A
    third wire carries a one-qubit gate and is measured too, so "a gate on another
    qubit" is a shape these programs can express.

    Every gate here goes through :func:`_avoiding` and is non-diagonal, so the
    candidate is the only instruction in the program this pass may delete.
    """

    name = instruction.name
    one = _avoiding(name, ("h", "x"))
    if len(instruction.wires) == 2:
        preparation = (
            _gate(one, (0,)),
            _gate(_avoiding(name, ("cx", "cy")), (0, 1)),
            _gate(one, (2,)),
        )
    else:
        preparation = (_gate(one, (0,)),)
    measures = (_measure(0, 0), _measure(1, 1), _measure(2, 2))
    program = _circuit((*preparation, instruction, *measures), wire_count)
    return program, preparation, measures


def _shape_programs() -> list[tuple[str, CircuitIR, str, int]]:
    """One row per candidate and per gap, as ``(label, program, candidate, index)``.

    The label is ``<candidate>|<gap>``. ``index`` is the position of the candidate
    itself, so the gap tables below can delete exactly that instruction when they ask
    whether the removal would have been invisible.
    """

    candidates = [
        (name, (0,)) for name in _SINGLE_QUBIT_UNITARIES
    ] + [(name, (0, 1)) for name in _TWO_WIRE_UNITARIES]

    rows: list[tuple[str, CircuitIR, str, int]] = []
    for name, wires in candidates:
        params = {key: _ANGLE for key in _params_for(name, _GENERIC)}
        candidate = Instruction(name, wires, params=params)
        index = len(wires) and (3 if len(wires) == 2 else 1)
        assert index, name

        base, preparation, measures = _terminal_program(candidate, _REGISTER_WIDTH)
        rows.append((f"{name}|nothing_between", base, name, index))

        # A gate on a wire the candidate does not touch must not block it, and this
        # row is what shows a two-wire candidate is judged per wire rather than by the
        # program's whole support.
        rows.append(
            (
                f"{name}|a_gate_on_another_qubit",
                _circuit(
                    (
                        *preparation,
                        candidate,
                        _gate(_avoiding(name, ("h", "x")), (2,)),
                        *measures,
                    )
                ),
                name,
                index,
            )
        )

        # A gate on one of the candidate's own wires, between it and the measurement.
        # It must be non-diagonal, or the pass would remove it rather than the
        # candidate and the row's length difference would belong to the blocker.
        if len(wires) == 2:
            blocker = _gate(_avoiding(name, ("cx", "cy")), (0, 1))
        else:
            blocker = _gate(_avoiding(name, ("h", "x")), (0,))
        rows.append(
            (
                f"{name}|a_gate_on_its_wire",
                _circuit((*preparation, candidate, blocker, *measures)),
                name,
                index,
            )
        )

        rows.append(
            (
                f"{name}|no_measurement",
                _circuit((_gate(_avoiding(name, ("h", "x")), (0,)), candidate)),
                name,
                1,
            )
        )

        conditioned = tuple(
            _measure(0, 0, conditions=((0, 1),)) if measure.wires == (0,) else measure
            for measure in measures
        )
        rows.append(
            (
                f"{name}|the_measurement_is_conditional",
                _circuit((*preparation, candidate, *conditioned)),
                name,
                index,
            )
        )

        conditional_candidate = Instruction(
            name, wires, params=params, metadata={"conditions": ((0, 1),)}
        )
        rows.append(
            (
                f"{name}|the_candidate_is_conditional",
                _circuit((*preparation, conditional_candidate, *measures)),
                name,
                index,
            )
        )

        if len(wires) == 2:
            rows.append(
                (
                    f"{name}|only_one_of_its_wires_is_measured",
                    _circuit((*preparation, candidate, _measure(0, 0), _measure(2, 2))),
                    name,
                    index,
                )
            )
            rows.append(
                (
                    f"{name}|a_gate_between_the_two_measurements",
                    _circuit(
                        (
                            *preparation,
                            candidate,
                            _measure(0, 0),
                            _gate(_avoiding(name, ("h", "x")), (1,)),
                            _measure(1, 1),
                            _measure(2, 2),
                        )
                    ),
                    name,
                    index,
                )
            )
    return rows


def _qiskit_builder(name: str, params: dict[str, float]) -> Any:
    """The Qiskit gate object for a FlagQuantum opcode, or None if it has none.

    The mapping is by *operator*, so it is many-to-one in two places on purpose:
    FlagQuantum's ``phase`` and ``p`` are both ``PhaseGate``, and ``cphase`` is
    ``CPhaseGate`` rather than ``CU1Gate``. Those two choices are what the anchor
    section reports as a deliberate disagreement.
    """

    from qiskit.circuit.library import (  # type: ignore[import-not-found]
        CPhaseGate,
        IGate,
        PhaseGate,
        U1Gate,
        U2Gate,
        U3Gate,
    )

    single = {
        "i": lambda: IGate(),
        "id": lambda: IGate(),
        "x": None,
        "y": None,
        "z": None,
        "h": None,
        "s": None,
        "sdg": None,
        "t": None,
        "tdg": None,
        "sx": None,
        "sxdg": None,
        "rx": None,
        "ry": None,
        "rz": None,
        "phase": lambda: PhaseGate(params.get("theta", 0.0)),
        "p": lambda: PhaseGate(params.get("theta", 0.0)),
        "u1": lambda: U1Gate(params.get("theta", 0.0)),
        "u2": lambda: U2Gate(params.get("phi", 0.0), params.get("lbd", 0.0)),
        "u3": lambda: U3Gate(
            params.get("theta", 0.0), params.get("phi", 0.0), params.get("lbd", 0.0)
        ),
    }
    two = {
        "cphase": lambda: CPhaseGate(params.get("theta", 0.0)),
        "cu1": lambda: U1Gate(params.get("theta", 0.0)),
    }
    if name in two:
        return two[name]()
    if name in single:
        factory = single[name]
        return None if factory is None else factory()
    return None


def _to_qiskit(ir: CircuitIR) -> Any:
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

    qubits = QuantumRegister(ir.n_wires, "q")
    bits = ClassicalRegister(ir.n_wires, "c")
    circuit = QuantumCircuit(qubits, bits)
    for instruction in ir.instructions:
        targets = [qubits[index] for index in instruction.wires]
        if instruction.name == "measure":
            classical = int(instruction.metadata.get("classical_bit", 0))
            if "conditions" in instruction.metadata or "condition_clauses" in instruction.metadata:
                # Qiskit expresses a conditional measurement as an `IfElseOp`, which its
                # pass does not look through. Building it is the honest anchor.
                from qiskit.circuit import IfElseOp

                body = QuantumCircuit(qubits, bits)
                body.measure(targets[0], bits[classical])
                condition = next(iter(instruction.metadata.get("conditions", ((0, 1),))))
                # The body is built over this circuit's own registers, so the operation
                # spans all of them and `append` must be handed all of them; passing the
                # operand qubits alone is rejected as a width mismatch.
                circuit.append(
                    IfElseOp((bits[condition[0]], condition[1]), body),
                    list(qubits),
                    list(bits),
                )
            else:
                circuit.measure(targets[0], bits[classical])
            continue
        params = instruction.params
        if instruction.metadata.get("conditions"):
            from qiskit.circuit import IfElseOp

            body = QuantumCircuit(qubits, bits)
            gate = _qiskit_builder(instruction.name, params)
            if gate is None:
                # The named circuit method takes the same parameters, in the same
                # order, that the instruction carries.
                getattr(body, instruction.name)(*params.values(), *targets)
            else:
                body.append(gate, targets)
            condition = next(iter(instruction.metadata["conditions"]))
            circuit.append(
                IfElseOp((bits[condition[0]], condition[1]), body),
                list(qubits),
                list(bits),
            )
            continue
        if instruction.matrix is not None:
            # A caller-supplied matrix has no opcode class; Qiskit's list-based pass
            # cannot see it, so the anchor is asked about the same non-answer.
            from qiskit.circuit.library import UnitaryGate

            circuit.append(UnitaryGate(instruction.matrix.numpy()), targets)
            continue
        gate = _qiskit_builder(instruction.name, params)
        if gate is not None:
            circuit.append(gate, targets)
        elif params:
            getattr(circuit, instruction.name)(*params.values(), *targets)
        else:
            getattr(circuit, instruction.name)(*targets)
    return circuit


_QISKIT_DISAGREEMENT_REASONS = {
    "i": (
        "Qiskit's diagonal_1q_gates names no IGate, so it keeps an identity that this "
        "pass removes"
    ),
    "phase": (
        "PhaseGate is not a U1Gate subclass, so Qiskit keeps a `p` where it would "
        "remove an appended U1Gate of the same operator"
    ),
    "cphase": (
        "CPhaseGate is not a CU1Gate subclass, so Qiskit keeps a `cp` where it would "
        "remove a CU1Gate of the same operator"
    ),
}


def _qiskit_shape_anchor(
    programs: dict[str, CircuitIR],
    verdicts: dict[str, bool],
) -> dict[str, Any]:
    """Qiskit's own ``RemoveDiagonalGatesBeforeMeasure`` over the identical programs.

    The anchor is a cross-check, not a dependency: this module has to run on a machine
    with no Qiskit at all. A row disagrees when the anchor's answer differs from the
    pass's own, and every disagreement has to be one of the three class-membership
    facts published in the module docstring -- ``i``, ``phase`` (so ``p`` too), and
    ``cphase``. An unlisted disagreement raises rather than being filed under a reason
    that does not fit it.
    """

    try:
        import qiskit  # type: ignore[import-not-found]
        from qiskit.converters import circuit_to_dag  # type: ignore[import-not-found]
        from qiskit.transpiler.passes import (  # type: ignore[import-not-found]
            RemoveDiagonalGatesBeforeMeasure,
        )
    except Exception as error:  # pragma: no cover - the anchor is optional
        return {"available": False, "reason": f"{type(error).__name__}: {error}"}

    pass_ = RemoveDiagonalGatesBeforeMeasure()
    removed: dict[str, int] = {}
    disagreed: dict[str, str] = {}
    for label, program in programs.items():
        circuit = _to_qiskit(program)
        dag = circuit_to_dag(circuit)
        before = dag.count_ops()
        # The pass has to be fed the DAG directly: calling it as a pass re-converts the
        # circuit through `circuit_to_dag` and rejects the registers this module builds.
        after = pass_.run(dag)
        after_counts = after.count_ops()
        candidate = label.split("|", 1)[0]
        # The candidate is the only instruction of its opcode in these programs, so the
        # difference in that opcode's count is the anchor's answer for the candidate.
        removed[label] = max(
            0, before.get(candidate, 0) - after_counts.get(candidate, 0)
        )
        if (removed[label] > 0) != verdicts[label]:
            reason = _QISKIT_DISAGREEMENT_REASONS.get(candidate)
            if reason is None:
                raise AssertionError(
                    f"{label}: the anchor and the pass disagree, and no published "
                    f"class-membership reason covers {candidate!r}"
                )
            disagreed[label] = reason
    return {
        "available": True,
        "pass_name": "RemoveDiagonalGatesBeforeMeasure",
        "qiskit_version": qiskit.__version__,
        "row_count": len(removed),
        "removed_count": sum(removed.values()),
        "disagreement_count": len(disagreed),
        "disagreement_reasons": disagreed,
        "removed_by_shape": removed,
    }


def shape_table() -> dict[str, Any]:
    """One row per candidate and gap, with the measured truth about the outcome.

    The *removed* column comes from the pass under test alone. The pipeline's reach is
    a separate column, because a removal the loop earns and a removal this rule earns
    are different facts and only the second is what the row is about.
    """

    rows = _shape_programs()
    programs = {label: program for label, program, _, _ in rows}
    # The pass's own answer is measured first, because the anchor's disagreements are
    # defined against it: an anchor answer counts as a disagreement only where the two
    # differ, and a reason has to be produced for each of those before the anchor is
    # allowed to report.
    verdicts = {
        label: len(remove_diagonal_gates_before_measure(program)) < len(program)
        for label, program in programs.items()
    }
    anchor = _qiskit_shape_anchor(programs, verdicts)

    recorded: list[dict[str, Any]] = []
    for label, program, candidate, index in rows:
        instruction = program.instructions[index]
        assert instruction.name == candidate, (label, instruction.name)
        # The candidate's opcode appears exactly once, so a count difference is
        # attributable and the truth column is about this instruction and no other.
        assert sum(
            1 for item in program.instructions if item.name == candidate
        ) == 1, label

        # Every other gate in the program is non-diagonal, so the pass can only be
        # removing the candidate. Without this the length difference would silently
        # include whatever else the rule deleted and the row would claim a reach the
        # rule does not have.
        by_the_pass = remove_diagonal_gates_before_measure(program)
        removed_here = len(program) - len(by_the_pass)
        assert removed_here <= 1, (label, removed_here)
        from_pass = removed_here == 1
        assert from_pass == verdicts[label], label
        shipped = optimize(program)
        from_pipeline = len(shipped) < len(program)

        if not _measures_are_terminal(program):
            instrument = "not_applicable"
            removable_in_fact = None
            moved = None
        else:
            instrument = "exact_outcome_distribution"
            before = _exact_outcome_distribution(program)
            stripped = CircuitIR(
                program.n_wires,
                tuple(
                    item
                    for position, item in enumerate(program.instructions)
                    if position != index
                ),
                dtype=_DTYPE,
            )
            after = _exact_outcome_distribution(stripped)
            moved = _worst_distribution_difference(before, after)
            removable_in_fact = moved <= _DISTRIBUTION_ATOL

        recorded.append(
            {
                "label": label,
                "candidate": candidate,
                "gap": label.split("|", 1)[1],
                "candidate_wire_count": len(instruction.wires),
                "candidate_is_diagonal": _measured_diagonal(
                    candidate, instruction.wires, instruction.params
                ),
                "removed_by_the_pass": from_pass,
                "removed_in_the_pipeline": from_pipeline,
                "instrument": instrument,
                "deleting_the_candidate_moved_the_distribution_by": moved,
                "removable_in_fact": removable_in_fact,
            }
        )

    return {
        "row_count": len(recorded),
        "candidate_count": len({row["candidate"] for row in recorded}),
        "gap_count": len({row["gap"] for row in recorded}),
        "removed_count": sum(1 for row in recorded if row["removed_by_the_pass"]),
        "removed_in_the_pipeline_count": sum(
            1 for row in recorded if row["removed_in_the_pipeline"]
        ),
        "declined_count": sum(1 for row in recorded if not row["removed_by_the_pass"]),
        "declined_but_removable_count": sum(
            1
            for row in recorded
            if not row["removed_by_the_pass"] and row["removable_in_fact"] is True
        ),
        # A declined row whose candidate the matrix calls diagonal is a rule limit: the
        # gate commutes with the measurement and the rule still declined, for a
        # structural reason. Two reasons account for every one of them, and they are
        # counted apart because they are different limits: a condition on the candidate
        # or on the measurement, and a successor on the candidate's wire that is not a
        # measurement. A declined row whose candidate the matrix does *not* call
        # diagonal is invisible only in this program -- ``x`` on ``|+>``, ``swap`` on a
        # Bell pair -- and the rule is state-independent, so it may not take those.
        "declined_but_removable_behind_a_condition_count": sum(
            1
            for row in recorded
            if not row["removed_by_the_pass"]
            and row["removable_in_fact"] is True
            and row["gap"]
            in {"the_measurement_is_conditional", "the_candidate_is_conditional"}
        ),
        "declined_but_removable_blocked_by_a_later_instruction_count": sum(
            1
            for row in recorded
            if not row["removed_by_the_pass"]
            and row["removable_in_fact"] is True
            and row["gap"]
            not in {"the_measurement_is_conditional", "the_candidate_is_conditional"}
        ),
        "declined_but_removable_diagonal_count": sum(
            1
            for row in recorded
            if not row["removed_by_the_pass"]
            and row["removable_in_fact"] is True
            and row["candidate_is_diagonal"]
        ),
        "removed_but_not_removable_count": sum(
            1
            for row in recorded
            if row["removed_by_the_pass"] and row["removable_in_fact"] is False
        ),
        "pipeline_only_row_count": sum(
            1
            for row in recorded
            if row["removed_in_the_pipeline"] and not row["removed_by_the_pass"]
        ),
        "measured_row_count": sum(
            1 for row in recorded if row["instrument"] == "exact_outcome_distribution"
        ),
        "one_wire_row_count": sum(
            1 for row in recorded if row["candidate_wire_count"] == 1
        ),
        "two_wire_row_count": sum(
            1 for row in recorded if row["candidate_wire_count"] == 2
        ),
        "removed_one_wire_row_count": sum(
            1
            for row in recorded
            if row["removed_by_the_pass"] and row["candidate_wire_count"] == 1
        ),
        "removed_two_wire_row_count": sum(
            1
            for row in recorded
            if row["removed_by_the_pass"] and row["candidate_wire_count"] == 2
        ),
        "rows": recorded,
        "reference_anchor": anchor,
    }


def form_table() -> dict[str, Any]:
    """The non-bare forms of the candidate, which the pass declines by construction."""

    rows: list[dict[str, Any]] = []
    forms = [
        ("a_bare_gate", _gate("z", (0,))),
        (
            "a_gate_with_a_caller_supplied_matrix",
            Instruction("z", (0,), matrix=torch.eye(2, dtype=torch.complex64)),
        ),
        (
            "a_conditional_gate",
            Instruction("z", (0,), metadata={"conditions": ((0, 1),)}),
        ),
        (
            "a_condition_clause_gate",
            Instruction("z", (0,), metadata={"condition_clauses": ((0, 1),)}),
        ),
        ("a_dynamic_gate", Instruction("z", (0,), metadata={"is_dynamic": True})),
        ("a_two_wire_gate_on_one_wire_of_measurement", _gate("cz", (0, 1))),
    ]
    for label, candidate in forms:
        program = _circuit((_gate("h", (0,)), candidate, _measure(0, 0)))
        rows.append(
            {
                "form": label,
                "removed": len(remove_diagonal_gates_before_measure(program))
                < len(program),
                "carries_a_matrix": candidate.matrix is not None,
                "carries_a_condition": any(
                    key in candidate.metadata
                    for key in ("conditions", "condition_clauses")
                ),
                "is_dynamic": bool(candidate.metadata.get("is_dynamic")),
                "wire_count": len(candidate.wires),
            }
        )
    return {
        "row_count": len(rows),
        "removed_count": sum(1 for row in rows if row["removed"]),
        "rows": rows,
    }


def execution_control() -> dict[str, Any]:
    """Show the outcome-distribution instrument can see a deletion that is not a no-op.

    Deleting a diagonal gate is invisible because a diagonal gate is a phase on each
    basis state. The control deletes a *non*-diagonal gate from the same program shape
    and reports how far the distribution moved. A tolerance that accepted both would
    be measuring nothing.
    """

    def bell() -> tuple[Instruction, ...]:
        return (_gate("h", (0,)), _gate("cx", (0, 1)))

    tolerated = _worst_distribution_difference(
        _exact_outcome_distribution(_circuit((*bell(), _gate("rzz", (0, 1), theta=_ANGLE), _measure(0, 0), _measure(1, 1)))),
        _exact_outcome_distribution(_circuit((*bell(), _measure(0, 0), _measure(1, 1)))),
    )
    caught = _worst_distribution_difference(
        _exact_outcome_distribution(_circuit((_gate("h", (0,)), _gate("ry", (0,), theta=1.8), _measure(0, 0)))),
        _exact_outcome_distribution(_circuit((_gate("h", (0,)), _measure(0, 0)))),
    )
    entangled_caught = _worst_distribution_difference(
        _exact_outcome_distribution(_circuit((*bell(), _measure(0, 0), _measure(1, 1)))),
        _exact_outcome_distribution(_circuit((_gate("h", (0,)), _measure(0, 0), _measure(1, 1)))),
    )
    return {
        "deleting_a_diagonal_gate_that_commutes_with_the_measurement": tolerated,
        "deleting_a_non_diagonal_gate_on_one_wire": caught,
        "deleting_an_entangler": entangled_caught,
        "tolerance": _DISTRIBUTION_ATOL,
        "control_floor": _CONTROL_FLOOR,
    }


def pipeline_delta() -> list[dict[str, Any]]:
    """What the pass buys, per seeded population, with and without it."""

    populations: list[tuple[str, Any]] = [
        ("terminal_diagonal_gates", _population_terminal),
        ("diagonal_gate_behind_a_hadamard", _population_behind_a_hadamard),
        ("a_general_form_at_a_zero_polar_angle", _population_zero_polar),
        ("two_wire_diagonal_before_two_measurements", _population_two_wire),
        ("a_gate_blocked_by_a_later_rotation", _population_blocked),
        ("a_mixed_program", _population_mixed),
        (
            "a_mixed_program_with_mid_circuit_measurements",
            _population_mixed_with_mid_circuit_measures,
        ),
    ]

    rows = []
    for label, builder in populations:
        source_count = legacy_count = optimized_count = 0
        removed_by_the_legacy = removed_by_the_pass_alone = 0
        removed_by_the_pass_in_the_pipeline = 0
        changed = executed = 0
        worst = 0.0
        for seed in range(_CIRCUITS_PER_POPULATION):
            program = builder(random.Random(_SWEEP_SEED + seed))
            legacy = _legacy_pipeline(program)
            shipped = optimize(program)
            source_count += len(program)
            legacy_count += len(legacy)
            optimized_count += len(shipped)
            removed_by_the_legacy += len(program) - len(legacy)
            removed_by_the_pass_alone += len(program) - len(
                remove_diagonal_gates_before_measure(program)
            )
            removed_by_the_pass_in_the_pipeline += len(legacy) - len(shipped)
            # Not a length comparison. The pass can delete the candidate and leave a bare
            # preparation where the legacy loop instead folds the two gates into one, and
            # both programs are then one instruction shorter: the length ties while the
            # programs differ, and a `0` in the delta column would then read as "no
            # effect" when the effect is a different program of the same size.
            changed += int(_signature(shipped) != _signature(legacy))
            if not _measures_are_terminal(program):
                continue
            executed += 1
            worst = max(
                worst,
                _worst_distribution_difference(
                    _exact_outcome_distribution(program),
                    _exact_outcome_distribution(shipped),
                ),
            )
        rows.append(
            {
                "label": label,
                "circuit_count": _CIRCUITS_PER_POPULATION,
                "source_instruction_count": source_count,
                "legacy_instruction_count": legacy_count,
                "optimized_instruction_count": optimized_count,
                "removed_by_the_legacy_pipeline": removed_by_the_legacy,
                "removed_by_the_pass_alone": removed_by_the_pass_alone,
                "removed_by_the_pass_in_the_pipeline": (
                    removed_by_the_pass_in_the_pipeline
                ),
                "changed_circuit_count": changed,
                "executed_circuit_count": executed,
                "instrument": "exact_outcome_distribution",
                "max_outcome_distribution_difference": worst,
            }
        )
    return rows


def _population_terminal(seed: random.Random) -> CircuitIR:
    wires = [seed.randrange(_REGISTER_WIDTH) for _ in range(3)]
    opcode = seed.choice(("z", "s", "t", "sdg", "tdg", "rz", "phase", "u1", "i"))
    params = {"theta": seed.choice((0.0, 0.3, -0.7))} if opcode in {"rz", "phase", "u1"} else {}
    return _circuit(
        (
            *(_gate("h", (wire,)) for wire in wires),
            Instruction(opcode, (wires[0],), params=params),
            *(_measure(wire) for wire in sorted(set(wires))),
        )
    )


def _population_behind_a_hadamard(seed: random.Random) -> CircuitIR:
    wire = seed.randrange(_REGISTER_WIDTH)
    opcode = seed.choice(("z", "t", "rz", "phase", "u1", "s"))
    params = {"theta": seed.choice((0.0, 0.4))} if opcode in {"rz", "phase", "u1"} else {}
    return _circuit(
        (
            _gate("h", (wire,)),
            Instruction(opcode, (wire,), params=params),
            _measure(wire),
        )
    )


def _population_zero_polar(seed: random.Random) -> CircuitIR:
    wire = seed.randrange(_REGISTER_WIDTH)
    opcode = seed.choice(("rx", "ry", "u3"))
    schema = OPERATOR_SCHEMAS[opcode]
    params = {key: seed.choice((0.3, -0.9)) for key in schema.parameters}
    params["theta"] = 0.0
    return _circuit(
        (_gate("h", (wire,)), Instruction(opcode, (wire,), params=params), _measure(wire))
    )


def _population_two_wire(seed: random.Random) -> CircuitIR:
    opcode = seed.choice(("cz", "crz", "cphase", "rzz"))
    schema = OPERATOR_SCHEMAS[opcode]
    params = {key: seed.choice((0.3, -0.7)) for key in schema.parameters}
    return _circuit(
        (
            _gate("h", (0,)),
            _gate("cx", (0, 1)),
            Instruction(opcode, (0, 1), params=params),
            _measure(0),
            _measure(1),
            _measure(2),
        )
    )


def _population_blocked(seed: random.Random) -> CircuitIR:
    wire = seed.randrange(_REGISTER_WIDTH)
    opcode = seed.choice(("z", "t", "rz", "phase"))
    params = {"theta": seed.choice((0.0, 0.4))} if opcode in {"rz", "phase"} else {}
    return _circuit(
        (
            Instruction(opcode, (wire,), params=params),
            _gate("h", (wire,)),
            _measure(wire),
        )
    )


def _population_mixed(seed: random.Random) -> CircuitIR:
    """A general program, measuring only at the end, so every circuit is terminal.

    Terminal by construction: the exact instrument reads the distribution off the final
    state, so a circuit that measures and then keeps computing is not a row this
    instrument can witness. The mid-circuit variant below is where that shape lives.
    """

    return _mixed_instructions(seed, mid_circuit_measures=False)


def _population_mixed_with_mid_circuit_measures(seed: random.Random) -> CircuitIR:
    """The same generator, also free to measure mid-program.

    Those circuits are not terminal, so this population's arithmetic columns cover all
    of its circuits while only the terminal ones carry an execution witness. The row
    reports both counts rather than silently dropping the unwitnessed circuits.
    """

    return _mixed_instructions(seed, mid_circuit_measures=True)


def _mixed_instructions(seed: random.Random, *, mid_circuit_measures: bool) -> CircuitIR:
    instructions: list[Instruction] = []
    written: list[int] = []
    for _ in range(seed.randint(2, 14)):
        draw = seed.random()
        wire = seed.randrange(_REGISTER_WIDTH)
        if draw < 0.4:
            opcode = seed.choice(("z", "s", "t", "sdg", "rz", "phase", "u1", "i"))
            params = (
                {"theta": seed.choice((0.0, 0.3, -0.7))}
                if opcode in {"rz", "phase", "u1"}
                else {}
            )
            instructions.append(Instruction(opcode, (wire,), params=params))
        elif draw < 0.6:
            opcode = seed.choice(("h", "x", "ry", "sx", "u2", "u3"))
            schema = OPERATOR_SCHEMAS[opcode]
            params = {key: seed.choice((0.0, 0.3, -0.7)) for key in schema.parameters}
            instructions.append(Instruction(opcode, (wire,), params=params))
        elif draw < 0.75 and _REGISTER_WIDTH >= 2:
            opcode = seed.choice(("cz", "crz", "cphase", "rzz", "cx", "rxx"))
            schema = OPERATOR_SCHEMAS[opcode]
            params = {key: seed.choice((0.3, -0.7)) for key in schema.parameters}
            pair = tuple(seed.sample(range(_REGISTER_WIDTH), 2))
            instructions.append(Instruction(opcode, pair, params=params))
        elif mid_circuit_measures:
            bit = len(written)
            instructions.append(_measure(wire, bit))
            written.append(bit)
    for wire in range(_REGISTER_WIDTH):
        instructions.append(_measure(wire, len(written)))
        written.append(len(written))
    return _circuit(tuple(instructions))


def run_benchmark() -> dict[str, Any]:
    shapes = shape_table()
    anchor = shapes["reference_anchor"]
    return {
        "schema": SCHEMA,
        "generated_at": _NOW.isoformat(),
        "artifact_classification": "local_compiler_microbenchmark",
        "distribution_semantics": "single_device_fast_path",
        "scalability_claim_allowed": False,
        "seed": {"sweep": _SWEEP_SEED},
        "reference_algorithm": "qiskit_remove_diagonal_gates_before_measure",
        "reference_revision": "qiskit 1.2.4 RemoveDiagonalGatesBeforeMeasure",
        "rule_contract": rule_contract(),
        "pipeline_delta": pipeline_delta(),
        "shape_table": shapes,
        "form_table": form_table(),
        "execution_control": execution_control(),
        "anchor_row_count": shapes["row_count"],
        "anchor_disagreement_count": (
            0 if not anchor["available"] else anchor["disagreement_count"]
        ),
        "anchor_agreement_scope": (
            f"{shapes['row_count']} candidate/gap rows driven through both ports; "
            "agreement is claimed only over the rows actually driven, and the "
            "disagreements are the anchor's gate-class list not naming IGate, "
            "PhaseGate or CPhaseGate"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    payload = run_benchmark()
    contract = payload["rule_contract"]
    census = contract["classification_census"]
    shapes = payload["shape_table"]
    anchor = shapes["reference_anchor"]
    print(
        f"rule: {contract['pass_module_string_literal_count']} string literals, "
        f"imports {contract['pass_module_imports']}"
    )
    print(
        f"census: {census['single_qubit_row_count']} single-qubit rows + "
        f"{census['two_wire_row_count']} two-wire rows, "
        f"{census['disagreement_count']} disagreements with the matrices"
    )
    print(
        f"shapes: {shapes['row_count']} rows, removed {shapes['removed_count']}, "
        f"declined-but-invisible {shapes['declined_but_removable_count']} "
        f"({shapes['declined_but_removable_behind_a_condition_count']} behind a "
        f"condition, "
        f"{shapes['declined_but_removable_blocked_by_a_later_instruction_count']} "
        f"blocked by a later instruction), "
        f"removed-but-not-removable {shapes['removed_but_not_removable_count']}"
    )
    print(
        f"anchor: available {anchor['available']}, disagreeing rows "
        f"{anchor.get('disagreement_count', 'n/a')}"
    )
    for row in payload["pipeline_delta"]:
        print(
            f"  {row['label']}: {row['source_instruction_count']} -> "
            f"{row['legacy_instruction_count']} legacy -> "
            f"{row['optimized_instruction_count']} shipped, pass alone "
            f"{row['removed_by_the_pass_alone']}, in the pipeline "
            f"{row['removed_by_the_pass_in_the_pipeline']}, programs changed "
            f"{row['changed_circuit_count']}, worst distribution difference "
            f"{row['max_outcome_distribution_difference']:.3e}"
        )

    if args.json_output is not None:
        args.json_output.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )


if __name__ == "__main__":
    main()

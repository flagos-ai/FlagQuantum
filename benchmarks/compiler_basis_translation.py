"""Measure the equivalence table that carries named gates into a target basis.

W8-04 of the Qiskit parity backlog asks for the basis translator Qiskit keeps in
``transpiler/passes/basis``. Before it, a *named* gate had no route into a basis
that did not carry it: the legalizer held four hand-written rewrite rules and
refused everything else, because turning a named gate into an entangler basis
through KAK would have needed that gate's matrix, and a named gate's matrix
belongs to ``flagquantum.simulation``, which the Compiler layer must not import.

``basis_translation`` closes that with a table of exact identities plus a
deterministic search that composes them. This module records what the table buys,
rule by rule and basis by basis:

* Per rule, the identity's declared leaves against the leaves it actually emits
  on a ``rz``/``sx``/``cx`` target, and whether the emitted program is *exactly*
  the source or only equal up to one global phase. Seventeen of the eighteen
  entries are exact; ``cphase`` is not, because turning a controlled rotation
  into a controlled phase leaves a phase behind and CircuitIR cannot record it.
  The table stores the shortest statement of each identity rather than its
  closure, so most rules emit more than they declare: ``cry`` declares three
  leaves and emits seven, and ``cswap`` declares a ``ccx`` that expands to
  fifteen.
* Per basis, how many of the eighteen table opcodes reach it by name, split by
  arity so the one-wire, two-wire and three-wire reaches can be read apart. The
  number is not the two-qubit reach -- a `cx` basis reaches five two-qubit opcodes
  on the table alone and the rest through the Euler synthesis, which
  ``compiler_two_qubit_synthesis`` measures.
* The Gray-code form of ``ccx`` that W8-05 measured and did not take. Qiskit
  splits the multi-controlled family three ways, but ``MCXGrayCode`` returns
  ``CCXGate`` itself at two controls and builds a Gray-code chain only from five;
  at the one arity FlagQuantum IR declares, all three are the same circuit. The
  Gray-code statement is shorter (seven leaves against fifteen) and equally
  exact, but it costs eight two-qubit gates on the interaction basis against six,
  and its further expansion runs through the one table entry that is not exact.
  Both forms are measured here rather than asserted.
* The one cross-framework number: Qiskit's own ``BasisTranslator`` over the
  standard equivalence library, on the same named gates into the same target
  bases. Four of the five bases agree opcode for opcode. The fifth, an
  ``rzz``-only rotating-frame basis, is a recorded gap and not a claim of
  parity: Qiskit reaches every name there, this table reaches eight. The
  table routes every entangling rule down to a *sink* of ``cx`` or ``cz`` -- the
  two names are each other's rule, so a basis that publishes neither cannot
  build either -- and that basis publishes only a parametrized ``rzz``. Qiskit's
  library bottoms out in ``rzz`` instead, and the anchor measures the price:
  every row records whether Qiskit's output carries an angle the source never
  wrote, and Qiskit introduces ``pi/2``, ``pi``, ``3*pi/2``, ``3*pi`` and halved
  copies of the caller's own angle on every basis whose sink is not already
  native. A rule here may introduce no angle at all, so the table stops at the
  names it can express from the caller's own parameters.

Two-qubit entangler counts are compared against Qiskit directly, because that is
the quantity both ports agree on: ``ccx`` costs six on either side and ``cswap``
eight. Raw instruction counts are not comparable and are not compared -- this
port spells a ``t`` as an ``rz``/``sx`` pair where Qiskit's ZSX decomposer emits
the ``sx`` first, so the same entangler count lands on a different instruction
count.

Classification: a local compiler microbenchmark on the single-device fast path.
It runs no distributed work, makes no scalability claim, and is not a performance
gate. Re-run it with::

    python benchmarks/compiler_basis_translation.py --json-output /tmp/w804.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from benchmarks.compiler_two_qubit_synthesis import (
    _NOW,
    DEFAULT_BASES,
    Basis,
    baseline_reach,
    snapshot,
)
from flagquantum.compiler.basis_translation import EQUIVALENCE_RULES, _gate
from flagquantum.compiler.native_gate_legalization import (
    NativeGateLegalizationError,
    legalize_native_gates,
)
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.operator_schema import OPERATOR_SCHEMAS, get_operator_schema
from flagquantum.core.target_capabilities import TargetCapabilitySnapshot
from flagquantum.simulation.statevector.local import run_local_statevector

SCHEMA = "flagquantum_compiler_basis_translation_benchmark_v1"
COMPLEX = torch.complex128

#: Every opcode the equivalence table has an entry for, in a stable order.
TABLE_OPCODES: tuple[str, ...] = tuple(sorted(EQUIVALENCE_RULES))

_ANGLES: dict[str, float] = {"theta": 0.7137, "phi": -0.4211, "lbd": 1.9073}

#: The cost basis: a z-rotation, a pi/2 pulse, and one entangler. This is what a
#: real `cx`-class target publishes, so the leaf count recorded below is what a
#: consumer actually pays, Euler synthesis and the table's own composition
#: included.
_INTERACTION = ("rz", "sx", "cx")

#: The fidelity basis declares every leaf any entry can name, so a rule is
#: measured on its own closed form instead of on whatever a narrower basis
#: synthesizes it into. Fidelity is *not* measured on `_INTERACTION`, because
#: there an `h` leaf becomes an Euler decomposition that carries its own global
#: phase -- real, but a property of the basis rather than of the rule, and the
#: per-rule table must not absorb it.
_LEAF_UNION = (
    "ccx",
    "crx",
    "crz",
    "cx",
    "cz",
    "h",
    "rxx",
    "rz",
    "rzz",
    "s",
    "sdg",
    "t",
    "tdg",
    "z",
)

#: The two arity-3 unitary opcodes FlagQuantum IR declares, and therefore the
#: whole of its multi-controlled surface. Qiskit's three MCX constructions cannot
#: be told apart on it.
_MULTI_CONTROLLED = ("ccx", "cswap")

#: The declared opcodes that act on more than one wire. Counting these is how the
#: cost of a form is compared across ports: the raw instruction count is not
#: comparable, because this port spells a `t` as an `rz`/`sx` pair where Qiskit's
#: ZSX decomposer emits the `sx` first.
_ENTANGLER_OPCODES = frozenset(
    name
    for name, schema in OPERATOR_SCHEMAS.items()
    if schema.unitary and schema.arity >= 2
)

#: `cphase` is the one entry that is not exact: its translation is equal to its
#: source only up to a global phase. The exactness check is asserted in both
#: directions so an entry cannot change side silently.
_PHASE_CARRYING_RULES = frozenset({"cphase"})


def _snapshot(*opcodes: str) -> TargetCapabilitySnapshot:
    """A verified, in-window snapshot declaring ``opcodes`` as native."""

    return snapshot(
        Basis(
            "benchmark-basis",
            tuple(
                {"name": name, "parameters": tuple(OPERATOR_SCHEMAS[name].parameters)}
                for name in opcodes
            ),
            None,
            None,
            "sx",
        )
    )


def source_instruction(opcode: str) -> Instruction:
    """One instance of ``opcode``, carrying an angle the caller owns."""

    schema = OPERATOR_SCHEMAS[opcode]
    params = {name: _ANGLES[name] for name in schema.parameters}
    return Instruction(opcode, tuple(range(schema.arity)), params=params)


def _preparation(n_wires: int) -> tuple[Instruction, ...]:
    """One `h` per wire, so a diagonal gate is distinguishable from identity."""

    return tuple(Instruction("h", (wire,)) for wire in range(n_wires))


def _probe(source: Instruction) -> CircuitIR:
    """The gate under `h` on every wire, legalized and run as a whole."""

    return CircuitIR(
        len(source.wires),
        (*_preparation(len(source.wires)), source),
        dtype="complex128",
    )


def _state(program: CircuitIR):
    return run_local_statevector(
        program, batch_size=1, device=torch.device("cpu"), dtype=COMPLEX
    ).reshape(-1)


def legalize(program: CircuitIR, basis: TargetCapabilitySnapshot):
    """Legalize ``program``, returning the error text instead of raising."""

    try:
        return legalize_native_gates(program, snapshot=basis, evaluated_at=_NOW)
    except NativeGateLegalizationError as error:
        return str(error)


def _preparation_leaf_count(n_wires: int) -> int:
    """How many leaves the preparation alone contributes to the interaction basis."""

    result = legalize(
        CircuitIR(n_wires, _preparation(n_wires), dtype="complex128"),
        _snapshot(*_INTERACTION),
    )
    assert not isinstance(result, str), result
    return len(result.program.instructions)


def rule_row(opcode: str) -> dict[str, Any]:
    """One table entry: declared form, emitted cost, and fidelity of the rule."""

    source = source_instruction(opcode)
    declared = EQUIVALENCE_RULES[opcode][0].build(source)
    n_wires = len(source.wires)

    # Cost: what the rule costs once a real target has to carry its leaves.
    cost = legalize(_probe(source), _snapshot(*_INTERACTION))
    assert not isinstance(cost, str), (opcode, cost)
    leaves = [item.name for item in cost.program.instructions]
    emitted = leaves[_preparation_leaf_count(n_wires) :]
    assert emitted, opcode

    # Fidelity: the rule's own leaves are native, so exactly one rewrite step
    # runs and the comparison is of the identity rather than of the basis.
    translation = legalize(_probe(source), _snapshot(*_LEAF_UNION))
    assert not isinstance(translation, str), (opcode, translation)
    reference = _state(_probe(source))
    actual = _state(translation.program)
    overlap = torch.vdot(reference, actual)
    magnitude = float(torch.abs(overlap))
    difference = float(torch.max(torch.abs(actual - reference)))
    exact = abs(magnitude - 1.0) < 1e-12 and difference < 1e-12
    if opcode in _PHASE_CARRYING_RULES:
        # Non-vacuity: the entry is listed as inexact exactly because it is; a
        # change that made it exact has to fail here rather than pass silently.
        assert not exact, opcode
    else:
        assert exact, (opcode, magnitude, difference)

    return {
        "opcode": opcode,
        "arity": n_wires,
        "declared_leaf_opcodes": [item.name for item in declared],
        "declared_leaf_count": len(declared),
        "emitted_leaf_opcodes": emitted,
        "emitted_leaf_count": len(emitted),
        "cost_basis": list(_INTERACTION),
        "fidelity_basis": list(_LEAF_UNION),
        "state_overlap_magnitude": magnitude,
        "global_phase_radians": float(torch.angle(overlap)),
        "max_raw_state_difference": difference,
        "exact": exact,
    }


def gray_code_form(source: Instruction) -> tuple[Instruction, ...]:
    """The Gray-code statement of ``ccx``, as the rule that was not taken.

    ``MCX = H . MCU1(pi) . H``, and Qiskit's ``_gray_code_chain`` builds
    ``MCU1(pi)`` at two controls as three ``CU1(pi/2)`` around two ``cx``. Written
    against the names this IR declares, that is two ``h``, three ``cphase`` and
    two ``cx``.

    It is a real candidate and not a straw man: it declares seven leaves where the
    taken rule declares fifteen, and it is exact for the same reason the taken one
    is. It loses on what a consumer pays -- the three ``cphase`` each expand to
    two ``cx`` and three ``rz`` -- and it inherits ``cphase``'s missing global
    phase one level further out. This function exists so the anchor measures both
    and the numbers, not the preference, decide.
    """

    control0, control1, target = source.wires
    metadata = source.metadata
    quarter_turn = 0.5 * math.pi
    return (
        _gate("h", (target,), metadata),
        _gate("cphase", (control1, target), metadata, quarter_turn),
        _gate("cx", (control1, control0), metadata),
        _gate("cphase", (control0, target), metadata, -quarter_turn),
        _gate("cx", (control1, control0), metadata),
        _gate("cphase", (control0, target), metadata, quarter_turn),
        _gate("h", (target,), metadata),
    )


def _form_evidence(
    opcode: str, build: Callable[[Instruction], tuple[Instruction, ...]], *, label: str
) -> dict[str, Any]:
    """Declared size, interaction-basis cost and exactness of one candidate form.

    Both quantities are measured on the form's *own* declared leaves rather than on
    the source instruction, because the two candidates are different statements of
    the same gate and running the source through the table would measure whichever
    rule the table prefers rather than the form under test.

    Fidelity is measured on a basis that declares every leaf any entry can name, so
    at most one rewrite step runs per leaf and the comparison is of the identity
    rather than of a basis that cannot hold it.
    """

    source = source_instruction(opcode)
    declared = build(source)
    leaves = [item.name for item in declared]
    fidelity_basis = tuple(sorted({*leaves, *_LEAF_UNION}))
    n_wires = len(source.wires)
    # Fidelity: the form's leaves under the same preparation the source is probed
    # with, so the two states describe the same circuit.
    prepared = CircuitIR(
        n_wires, (*_preparation(n_wires), *declared), dtype="complex128"
    )
    translation = legalize(prepared, _snapshot(*fidelity_basis))
    assert not isinstance(translation, str), (label, translation)
    reference = _state(_probe(source))
    actual = _state(translation.program)
    difference = float(torch.max(torch.abs(actual - reference)))
    # Cost: the form's leaves alone, so the preparation is not charged to it.
    cost = legalize(
        CircuitIR(n_wires, declared, dtype="complex128"), _snapshot(*_INTERACTION)
    )
    assert not isinstance(cost, str), (label, cost)
    emitted = [item.name for item in cost.program.instructions]
    return {
        "form": label,
        "declared_leaf_count": len(declared),
        "declared_leaf_opcodes": leaves,
        "interaction_leaf_count": len(emitted),
        "interaction_entangler_count": sum(
            1 for name in emitted if name in _ENTANGLER_OPCODES
        ),
        "fidelity_basis": list(fidelity_basis),
        "max_raw_state_difference": difference,
        "exact": difference < 1e-12,
    }


def gray_code_evidence() -> dict[str, Any]:
    """``ccx`` as the table states it against the Gray-code statement it did not."""

    taken = _form_evidence(
        "ccx", EQUIVALENCE_RULES["ccx"][0].build, label="identity-table"
    )
    gray = _form_evidence("ccx", gray_code_form, label="gray-code")
    # Non-vacuity: the two forms must actually differ, or the comparison above
    # would be reporting the same circuit twice.
    assert taken["declared_leaf_count"] != gray["declared_leaf_count"], taken
    assert taken["interaction_entangler_count"] < gray["interaction_entangler_count"], (
        taken["interaction_entangler_count"],
        gray["interaction_entangler_count"],
    )
    return {
        "opcode": "ccx",
        "qiskit_counterpart": "MCXGrayCode",
        "qiskit_note": (
            "MCXGrayCode returns CCXGate at two controls and builds a Gray-code "
            "chain from five; C3XGate and C4XGate cover three and four, and "
            "FlagQuantum IR declares no arity above three"
        ),
        "taken": taken["form"],
        "reason_not_taken": (
            "the Gray-code statement is shorter but costs two more two-qubit "
            "gates on the interaction basis, and its further expansion depends on "
            "cphase, the one table entry that is not exact"
        ),
        "forms": [taken, gray],
    }


def name_evidence(basis: Basis) -> dict[str, Any]:
    """How many table opcodes reach ``basis`` by name, and at what cost."""

    reached: dict[str, int] = {}
    refused: dict[str, str] = {}
    for opcode in TABLE_OPCODES:
        result = legalize(
            CircuitIR(
                OPERATOR_SCHEMAS[opcode].arity,
                (source_instruction(opcode),),
                dtype="complex128",
            ),
            snapshot(basis),
        )
        if isinstance(result, str):
            refused[opcode] = result
            continue
        reached[opcode] = len(result.program.instructions)
    by_arity: dict[str, dict[str, int]] = {}
    for arity in (1, 2, 3):
        names = [
            opcode
            for opcode in TABLE_OPCODES
            if OPERATOR_SCHEMAS[opcode].arity == arity
        ]
        by_arity[str(arity)] = {
            "opcode_count": len(names),
            "reached_count": sum(1 for name in names if name in reached),
        }
    return {
        "label": basis.label,
        "reached_count": len(reached),
        "reached_leaf_counts": reached,
        "refused_count": len(refused),
        "refused": refused,
        "reached_by_arity": by_arity,
        # The same quantity `compiler_two_qubit_synthesis` reports, over the
        # eleven declared two-qubit opcodes only, so the two measurements can
        # be read side by side. It is deliberately not the count above, which
        # covers all eighteen table entries including the one- and three-wire
        # ones.
        "two_qubit_identity_table_reach_count": baseline_reach(basis)[
            "legalized_opcode_count"
        ],
    }


def _multi_controlled_rules() -> list[dict[str, Any]]:
    """The two arity-3 entries stated as the table states them."""

    rows: list[dict[str, Any]] = []
    for opcode in _MULTI_CONTROLLED:
        source = source_instruction(opcode)
        declared = EQUIVALENCE_RULES[opcode][0].build(source)
        rows.append(
            {
                "opcode": opcode,
                "aliases": list(get_operator_schema(opcode).aliases),
                "declared_leaf_count": len(declared),
                "declared_leaf_opcodes": [item.name for item in declared],
                "wires": list(source.wires),
            }
        )
    return rows


def multi_controlled_evidence(basis: Basis) -> dict[str, Any]:
    """How the two arity-3 opcodes reach ``basis``, before the table and after.

    Before W8-05 both were refused on every basis, so the before-count is not
    recomputed here: it is asserted to be zero by the contract test rather than
    re-derived from a table that no longer exists. What is measured is the
    entangler count each opcode costs once it lands, which is the one number both
    ports can be compared on.
    """

    reached: dict[str, int] = {}
    entanglers: dict[str, int] = {}
    refused: dict[str, str] = {}
    for opcode in _MULTI_CONTROLLED:
        result = legalize(
            CircuitIR(3, (source_instruction(opcode),), dtype="complex128"),
            snapshot(basis),
        )
        if isinstance(result, str):
            refused[opcode] = result
            continue
        emitted = [item.name for item in result.program.instructions]
        reached[opcode] = len(emitted)
        entanglers[opcode] = sum(1 for name in emitted if name in _ENTANGLER_OPCODES)
    return {
        "label": basis.label,
        "reached_count": len(reached),
        "reached_leaf_counts": reached,
        "entangler_counts": entanglers,
        "refused": refused,
    }


def _qiskit_gates() -> dict[str, Any]:
    """The table's opcodes as Qiskit gate objects, at the benchmark's angles."""

    from qiskit.circuit.library import (  # type: ignore[import-not-found]
        CCXGate,
        CPhaseGate,
        CRXGate,
        CRYGate,
        CRZGate,
        CSwapGate,
        CXGate,
        CYGate,
        CZGate,
        RXGate,
        RXXGate,
        RYGate,
        RYYGate,
        RZZGate,
        SdgGate,
        SwapGate,
        XGate,
        ZGate,
    )

    theta = _ANGLES["theta"]
    builders = {
        "cx": CXGate,
        "cz": CZGate,
        "cy": CYGate,
        "swap": SwapGate,
        "rzz": lambda: RZZGate(theta),
        "crz": lambda: CRZGate(theta),
        "cphase": lambda: CPhaseGate(theta),
        "rxx": lambda: RXXGate(theta),
        "ryy": lambda: RYYGate(theta),
        "crx": lambda: CRXGate(theta),
        "cry": lambda: CRYGate(theta),
        "x": XGate,
        "z": ZGate,
        "rx": lambda: RXGate(theta),
        "ry": lambda: RYGate(theta),
        "sdg": SdgGate,
        "ccx": CCXGate,
        "cswap": CSwapGate,
    }
    assert set(builders) == set(TABLE_OPCODES), sorted(
        set(builders) ^ set(TABLE_OPCODES)
    )
    return {name: builder() for name, builder in builders.items()}


def _qiskit_anchor(
    measured: list[dict[str, Any]], multi_controlled_measured: list[dict[str, Any]]
) -> dict[str, Any]:
    """The same named reach from Qiskit's ``BasisTranslator``, when importable.

    The anchor is a cross-check, not a dependency: this module is a local
    benchmark and CI does not install Qiskit for it. It is reported as absent
    rather than failed when Qiskit cannot be imported.

    Each row also records whether Qiskit's output carries an angle the source
    instruction never wrote. That is the measurement behind the one recorded
    disagreement: Qiskit's equivalence library may introduce a constant or a
    halved angle, and the table here may not.
    """

    try:
        from qiskit.circuit import QuantumCircuit  # type: ignore[import-not-found]
        from qiskit.circuit.equivalence_library import (  # type: ignore[import-not-found]
            StandardEquivalenceLibrary,
        )
        from qiskit.transpiler.passes import (  # type: ignore[import-not-found]
            BasisTranslator,
        )
    except Exception as error:  # pragma: no cover - environment dependent
        return {"available": False, "reason": str(error)}

    gates = _qiskit_gates()
    ours = {row["label"]: row for row in measured}
    port_multi_controlled = {row["label"]: row for row in multi_controlled_measured}
    rows: list[dict[str, Any]] = []
    for basis in DEFAULT_BASES:
        target = [
            item["name"] if isinstance(item, dict) else str(item)
            for item in basis.gates
        ]
        translator = BasisTranslator(StandardEquivalenceLibrary, target)
        reached: list[str] = []
        refused: dict[str, str] = {}
        introduced: dict[str, list[float]] = {}
        multi_controlled: dict[str, int] = {}
        for opcode in TABLE_OPCODES:
            gate = gates[opcode]
            circuit = QuantumCircuit(OPERATOR_SCHEMAS[opcode].arity)
            circuit.append(gate, list(range(circuit.num_qubits)))
            try:
                translated = translator(circuit)
            except Exception as error:  # pragma: no cover - Qiskit dependent
                refused[opcode] = type(error).__name__
                continue
            reached.append(opcode)
            own = [float(value) for value in gate.params]
            written = [
                float(value)
                for item in translated.data
                for value in item.operation.params
            ]
            extra = sorted(
                {
                    round(value, 12)
                    for value in written
                    if not any(abs(value - value_own) < 1e-9 for value_own in own)
                }
            )
            if extra:
                introduced[opcode] = extra
            if opcode in _MULTI_CONTROLLED:
                multi_controlled[opcode] = sum(
                    1 for item in translated.data if item.operation.num_qubits >= 2
                )
        rows.append(
            {
                "label": basis.label,
                "target_basis": target,
                "reached_count": len(reached),
                "reached": reached,
                "refused": refused,
                "introduced_angles": introduced,
                "introduced_angle_opcode_count": len(introduced),
                "multi_controlled_entangler_counts": multi_controlled,
                "port_reached_count": ours[basis.label]["reached_count"],
                "count_agrees": len(reached) == ours[basis.label]["reached_count"],
                "opcode_agrees": sorted(reached)
                == sorted(ours[basis.label]["reached_leaf_counts"]),
                # Non-empty as well as equal: a basis where neither port reaches
                # either opcode agrees vacuously, and that is not a comparison.
                "multi_controlled_entanglers_agree": bool(multi_controlled)
                and multi_controlled
                == port_multi_controlled[basis.label]["entangler_counts"],
            }
        )
    # Non-vacuity, and the recorded mechanism: the bases that disagree are
    # exactly the bases publishing neither `cx` nor `cz`, so the gap is the
    # table's two-name sink and nothing else.
    assert sorted(row["label"] for row in rows if not row["count_agrees"]) == sorted(
        row["label"] for row in rows if not {"cx", "cz"} & set(row["target_basis"])
    ), [row["label"] for row in rows if not row["count_agrees"]]
    # The multi-controlled entangler comparison is only the same quantity where
    # both ports reach both opcodes. Asserting that set rather than trusting the
    # equality keeps an all-refused run from reporting agreement with itself.
    both_reached = sorted(
        row["label"]
        for row in rows
        if len(row["multi_controlled_entangler_counts"]) == len(_MULTI_CONTROLLED)
        and len(port_multi_controlled[row["label"]]["entangler_counts"])
        == len(_MULTI_CONTROLLED)
    )
    assert both_reached, [row["label"] for row in rows]
    assert (
        sorted(row["label"] for row in rows if row["multi_controlled_entanglers_agree"])
        == both_reached
    ), both_reached
    return {
        "available": True,
        "translator": "BasisTranslator(StandardEquivalenceLibrary, target)",
        "reference_revision": "qiskit 1.2.4",
        "compared_opcode_count": len(TABLE_OPCODES),
        "basis_count": len(rows),
        "count_agreement_count": sum(1 for row in rows if row["count_agrees"]),
        "opcode_agreement_count": sum(1 for row in rows if row["opcode_agrees"]),
        "disagreements": [row["label"] for row in rows if not row["count_agrees"]],
        "multi_controlled": sorted(_MULTI_CONTROLLED),
        "multi_controlled_compared_bases": both_reached,
        "multi_controlled_agreement_count": sum(
            1 for row in rows if row["multi_controlled_entanglers_agree"]
        ),
        "multi_controlled_entangler_counts": {
            row["label"]: row["multi_controlled_entangler_counts"] for row in rows
        },
        "port_multi_controlled_entangler_counts": {
            row["label"]: port_multi_controlled[row["label"]]["entangler_counts"]
            for row in rows
        },
        "rows": rows,
    }


def run_benchmark() -> dict[str, Any]:
    """Measure the table's rules, its named reach, and the Qiskit anchor."""

    rules = [rule_row(opcode) for opcode in TABLE_OPCODES]
    reach = [name_evidence(basis) for basis in DEFAULT_BASES]
    multi_controlled = [multi_controlled_evidence(basis) for basis in DEFAULT_BASES]
    return {
        "schema": SCHEMA,
        "artifact_classification": "local_compiler_microbenchmark",
        "distribution_semantics": "single_device_fast_path",
        "scalability_claim_allowed": False,
        "reference_algorithm": "qiskit_basis_translator",
        "reference_revision": "qiskit 1.2.4 BasisTranslator(StandardEquivalenceLibrary)",
        "table_contract": (
            "each entry is an exact identity that forwards the source "
            "instruction's own parameters and metadata; no entry introduces a "
            "constant angle"
        ),
        "phase_contract": "equal up to one global phase, which CircuitIR cannot record",
        "table_opcode_count": len(TABLE_OPCODES),
        "table_opcodes": list(TABLE_OPCODES),
        "exact_rule_count": sum(1 for row in rules if row["exact"]),
        "phase_carrying_rules": sorted(_PHASE_CARRYING_RULES),
        "named_two_qubit_opcodes": [
            name
            for name, schema in sorted(OPERATOR_SCHEMAS.items())
            if schema.unitary and schema.arity == 2
        ],
        "named_multi_controlled_opcodes": list(_MULTI_CONTROLLED),
        "multi_controlled_before_reach_count": 0,
        "multi_controlled_rules": _multi_controlled_rules(),
        "multi_controlled_form_comparison": gray_code_evidence(),
        "multi_controlled_name_reach": multi_controlled,
        "rules": rules,
        "name_reach": reach,
        "reference_anchor": _qiskit_anchor(reach, multi_controlled),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    payload = run_benchmark()
    print(
        f"{payload['table_opcode_count']} equivalence rules, "
        f"{payload['exact_rule_count']} exact"
    )
    print(
        f"{'opcode':10s} {'declared':>9s} {'emitted':>8s} {'overlap':>18s} "
        f"{'raw diff':>10s}  exact"
    )
    for row in payload["rules"]:
        print(
            f"{row['opcode']:10s} {row['declared_leaf_count']:>9d} "
            f"{row['emitted_leaf_count']:>8d} "
            f"{row['state_overlap_magnitude']:>18.15f} "
            f"{row['max_raw_state_difference']:>10.3e}  {row['exact']}"
        )
    print()
    print(f"{'basis':28s} {'reached':>9s} {'refused':>8s}")
    for row in payload["name_reach"]:
        by_arity = row["reached_by_arity"]
        print(
            f"{row['label']:28s} "
            f"{row['reached_count']:>3d}/{payload['table_opcode_count']:<5d} "
            f"{row['refused_count']:>8d}  "
            f"1q {by_arity['1']['reached_count']}/{by_arity['1']['opcode_count']} "
            f"2q {by_arity['2']['reached_count']}/{by_arity['2']['opcode_count']} "
            f"3q {by_arity['3']['reached_count']}/{by_arity['3']['opcode_count']}"
        )
    comparison = payload["multi_controlled_form_comparison"]
    print()
    print(f"ccx form comparison ({comparison['taken']} taken):")
    for row in comparison["forms"]:
        print(
            f"  {row['form']:16s} declared {row['declared_leaf_count']:>3d} "
            f"emitted {row['interaction_leaf_count']:>3d} "
            f"2q {row['interaction_entangler_count']:>3d} "
            f"exact {row['exact']}"
        )
    anchor = payload["reference_anchor"]
    if anchor["available"]:
        print()
        print(
            f"Qiskit anchor: {anchor['opcode_agreement_count']}/"
            f"{anchor['basis_count']} bases agree opcode for opcode; "
            f"{anchor['count_agreement_count']}/{anchor['basis_count']} agree on "
            "the count"
        )
        for row in anchor["rows"]:
            mark = " " if row["opcode_agrees"] else "*"
            print(
                f" {mark}{row['label']:28s} qiskit {row['reached_count']:>2d}/"
                f"{anchor['compared_opcode_count']:<3d} port "
                f"{row['port_reached_count']:>2d}"
            )
        print(
            f" multi-controlled entanglers agree on "
            f"{anchor['multi_controlled_agreement_count']}/"
            f"{len(anchor['multi_controlled_compared_bases'])} compared bases: "
            f"{anchor['multi_controlled_entangler_counts']}"
        )
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()

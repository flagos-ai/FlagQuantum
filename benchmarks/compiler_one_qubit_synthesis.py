"""Measure how far one-qubit Euler synthesis extends native-gate legalization.

W8-01 of the Qiskit parity backlog asks for the single-qubit synthesis Qiskit
keeps in ``synthesis/one_qubit``. FlagQuantum had four hand-written rewrite rules
and no synthesis of one-qubit unitaries at all, so a target whose evidence
publishes a z-rotation and ``sx`` could not legalize ``h``, ``s``, ``t``, ``ry``,
``rx``, ``u2``, or ``u3``, which is the whole of an IBM-style basis's one-qubit
group beyond the two gates the basis already names.

``one_qubit_synthesis`` closes that with one identity,

    ``U3(theta, phi, lam) == RZ(lam) SX RZ(theta - pi) SX RZ(phi - pi)``

up to a global phase, plus tabulated angle triples for the parameter-free
opcodes, which keeps them exact instead of solved. The hand-written rules stay
ahead of it, so a basis that can carry a gate exactly keeps its exact form.

This module measures the reach that buys, on the bases this repository already
records, and pins the one property a consumer has to know about:

* ``rz`` + ``sx`` bases legalize **18 of 18** declared single-qubit unitary
  opcodes, from 3 of 18 before this change. ``rx``/``ry``-only and ``h``/``s``/``t``
  bases are unchanged, because neither publishes a z-rotation to synthesize
  against.
* The synthesized program is equal to its source up to **one global phase**. On a
  two-wire program mixing `h`, `cx`, `rx`, `u3`, `sdg`, `y`, and `t` the state
  overlap magnitude is 1.000000000000000 with a phase of -0.927 rad, and the raw
  statevector difference is 0.91. A z-rotation and ``sx`` generate a determinant
  group whose phases are multiples of ``i``, so no basis of those two can carry
  ``h`` exactly, and FlagQuantum IR has no field for the difference. Expectation
  values are unaffected; a consumer comparing raw statevectors is not.

Classification: a local compiler microbenchmark on the single-device fast path.
It runs no distributed work, makes no scalability claim, and is not a performance
gate. Re-run it with::

    python benchmarks/compiler_one_qubit_synthesis.py --json-output /tmp/w801.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

import torch

from flagquantum.compiler.basis_translation import translate
from flagquantum.compiler.native_gate_legalization import (
    NativeGateLegalizationError,
    _native_descriptors,
    _supports,
    legalize_native_gates,
)
from flagquantum.compiler.one_qubit_synthesis import synthesize_one_qubit
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.operator_schema import OPERATOR_SCHEMAS
from flagquantum.core.target_capabilities import (
    CapabilityFact,
    CapabilityScope,
    EvidenceLevel,
    EvidenceReference,
    FactExposure,
    FactSource,
    SupportStatus,
    TargetCapabilitySnapshot,
    TargetIdentity,
)
from flagquantum.simulation.statevector.local import run_local_statevector

SCHEMA = "flagquantum_compiler_one_qubit_synthesis_benchmark_v1"

_NOW = datetime(2026, 9, 10, 8, 0, tzinfo=timezone.utc)
_ANGLES: dict[str, float] = {"theta": 0.7137, "phi": -0.4211, "lbd": 1.9073}


@dataclass(frozen=True)
class Basis:
    """One target basis, named by the gates its evidence publishes."""

    label: str
    gates: tuple[dict[str, Any], ...]
    z_rotation: str | None


def _gate(name: str) -> dict[str, Any]:
    return {"name": name, "parameters": tuple(OPERATOR_SCHEMAS[name].parameters)}


# The two IBM-style shapes a QPU basis publishes, and the three bases this
# repository already records elsewhere. ``deployment-cloud-simulator`` is the
# ten-gate set the cloud simulator advertises.
DEFAULT_BASES: tuple[Basis, ...] = (
    Basis("ibm-rz-sx-cx", (_gate("rz"), _gate("sx"), _gate("x"), _gate("cx")), "rz"),
    # Heron-class hardware publishes ECR rather than CX, but ECR is not a
    # declared FlagQuantum opcode, so the closest real basis is `cz`.
    Basis("ibm-heron-cz", (_gate("rz"), _gate("sx"), _gate("x"), _gate("cz")), "rz"),
    Basis("rotational", (_gate("rz"), _gate("rx"), _gate("cz")), "rz"),
    Basis("clifford-t", (_gate("h"), _gate("s"), _gate("t"), _gate("cx")), None),
    Basis(
        "deployment-cloud-simulator",
        tuple(
            _gate(name)
            for name in ("x", "y", "z", "h", "rx", "ry", "rz", "cx", "cz", "swap")
        ),
        "rz",
    ),
)

# One instruction of every declared arity-1 unitary opcode, fully parameter-bound
# so nothing here depends on a caller leaving a rotation unbound.
SINGLE_QUBIT_OPCODES: tuple[str, ...] = tuple(
    sorted(
        name
        for name, schema in OPERATOR_SCHEMAS.items()
        if schema.unitary and schema.arity == 1
    )
)


def mixed_program(entangler: str) -> CircuitIR:
    """An entangled program mixing synthesized and directly supported gates.

    It carries a tabulated gate (``h``, ``sdg``, ``y``, ``t``), two parameterized
    rotations, and two two-wire operations, so the phase measurement is taken on
    an entangled state rather than on a product state, where a relative phase
    could hide.
    """

    return CircuitIR(
        2,
        (
            Instruction("h", (0,)),
            Instruction(entangler, (0, 1)),
            Instruction("rx", (0,), params={"theta": 0.7137}),
            Instruction("u3", (1,), params={"theta": -1.2, "phi": 0.4, "lbd": 2.1}),
            Instruction("sdg", (0,)),
            Instruction("y", (1,)),
            Instruction(entangler, (1, 0)),
            Instruction("t", (1,)),
        ),
        dtype="complex128",
    )


# The two-wire gate each basis publishes, used to build its mixed program.
_ENTANGLER: dict[str, str] = {
    "ibm-rz-sx-cx": "cx",
    "ibm-heron-cz": "cz",
    "rotational": "cz",
    "clifford-t": "cx",
    "deployment-cloud-simulator": "cx",
}

COMPLEX = torch.complex128


def single_qubit_instruction(opcode: str) -> Instruction:
    """One arity-1 instruction of ``opcode`` with every declared angle bound."""

    schema = OPERATOR_SCHEMAS[opcode]
    return Instruction(
        opcode, (0,), params={name: _ANGLES[name] for name in schema.parameters}
    )


def snapshot(basis: Basis) -> TargetCapabilitySnapshot:
    """A verified, in-window capability snapshot declaring ``basis``'s gates."""

    scope = CapabilityScope(device_ids=("w801:0",))
    return TargetCapabilitySnapshot(
        target_identity=TargetIdentity(
            target_id=f"w801-{basis.label}",
            target_class="test",
            provider="flagquantum.test",
            provider_version="1",
            target_revision="1",
            environment_id="w801-environment",
        ),
        scope=scope,
        captured_at=_NOW.isoformat(),
        valid_until=_NOW.replace(hour=9).isoformat(),
        facts=(
            CapabilityFact(
                name="gates.native",
                value=basis.gates,
                support_status=SupportStatus.VERIFIED,
                fact_exposure=FactExposure.DECLARED,
                source=FactSource(kind="w801_probe", ref="w801-evidence"),
            ),
        ),
        evidence_refs=(
            EvidenceReference(
                evidence_id="w801-evidence",
                sha256="b" * 64,
                level=EvidenceLevel.BASIC,
                scope=scope,
            ),
        ),
    )


def legalize(program: CircuitIR, basis: Basis):
    """Legalize ``program``, returning the error text instead of raising."""

    try:
        return legalize_native_gates(
            program, snapshot=snapshot(basis), evaluated_at=_NOW
        )
    except NativeGateLegalizationError as error:
        return str(error)


def baseline_reach(basis: Basis) -> dict[str, Any]:
    """How far the equivalence table alone reaches, with no synthesis.

    A basis supports an opcode directly, or an exact identity rewrites it into
    gates the basis supports. Four one-wire identities exist -- `x`, `rx`, `ry`
    and `sdg` -- and this is the number the reach below is measured against,
    computed from the same descriptors rather than quoted. It is the same
    quantity `compiler_two_qubit_synthesis` reports, so the two benchmarks can
    be read side by side.
    """

    descriptors = _native_descriptors(snapshot(basis), evaluated_at=_NOW)
    legalized: list[str] = []
    for opcode in SINGLE_QUBIT_OPCODES:
        instruction = single_qubit_instruction(opcode)
        if _supports(descriptors, instruction):
            legalized.append(opcode)
            continue
        replacement = translate(instruction, can_run=partial(_supports, descriptors))
        if replacement is not None and all(
            _supports(descriptors, leaf) for leaf in replacement
        ):
            legalized.append(opcode)
    return {
        "label": basis.label,
        "legalized_opcode_count": len(legalized),
        "legalized_opcodes": legalized,
    }


def reach(basis: Basis) -> dict[str, Any]:
    """How much of the declared single-qubit group ``basis`` can legalize."""

    legalized: list[str] = []
    unresolved: dict[str, str] = {}
    leaves: dict[str, int] = {}
    for opcode in SINGLE_QUBIT_OPCODES:
        result = legalize(
            CircuitIR(1, (single_qubit_instruction(opcode),), dtype="complex128"),
            basis,
        )
        if isinstance(result, str):
            unresolved[opcode] = result
            continue
        legalized.append(opcode)
        # A basis-native opcode needs no record, so its replacement is empty.
        leaves[opcode] = (
            len(result.decompositions[0].replacement_opcodes)
            if result.decompositions
            else 0
        )
    native = {
        item.name if isinstance(item, str) else str(item["name"])
        for item in basis.gates
    }
    before = baseline_reach(basis)
    return {
        "label": basis.label,
        "declared_opcode_count": len(SINGLE_QUBIT_OPCODES),
        "legalized_opcode_count": len(legalized),
        "unresolved_opcode_count": len(unresolved),
        "legalized_opcodes": legalized,
        "unresolved_errors": unresolved,
        "replacement_lengths": leaves,
        "max_replacement_length": max(leaves.values()) if leaves else 0,
        "native_opcodes": sorted(native),
        "identity_table_reach_count": before["legalized_opcode_count"],
        "identity_table_reach_opcodes": before["legalized_opcodes"],
    }


def phase_evidence(basis: Basis) -> dict[str, Any]:
    """Measure what legalizing an entangled mixed program does to the state."""

    source = mixed_program(_ENTANGLER[basis.label])
    result = legalize(source, basis)
    if isinstance(result, str):
        return {"label": basis.label, "legalized": False, "error": result}
    original = run_local_statevector(
        source, batch_size=1, device=torch.device("cpu"), dtype=COMPLEX
    ).reshape(-1)
    legalized = run_local_statevector(
        result.program, batch_size=1, device=torch.device("cpu"), dtype=COMPLEX
    ).reshape(-1)
    overlap = torch.vdot(original, legalized)
    return {
        "label": basis.label,
        "legalized": True,
        "entangler": _ENTANGLER[basis.label],
        "source_gate_count": len(source.instructions),
        "legalized_gate_count": len(result.program.instructions),
        "decomposition_count": len(result.decompositions),
        "native_only": sorted(
            {item.name for item in result.program.instructions}
            - {str(item["name"]) for item in basis.gates}
        ),
        "state_overlap_magnitude": float(torch.abs(overlap)),
        "global_phase_radians": float(torch.angle(overlap)),
        "max_raw_state_difference": float(torch.max(torch.abs(legalized - original))),
    }


def synthesis_table() -> list[dict[str, Any]]:
    """The leaf sequence of every opcode, before the legalizer sees it."""

    rows: list[dict[str, Any]] = []
    for opcode in SINGLE_QUBIT_OPCODES:
        if opcode == "rz":
            continue
        leaves = synthesize_one_qubit(single_qubit_instruction(opcode), z_rotation="rz")
        assert leaves is not None, opcode
        rows.append(
            {
                "opcode": opcode,
                "leaf_count": len(leaves),
                "leaf_opcodes": [leaf.name for leaf in leaves],
                "leaf_angles": [
                    None if not leaf.params else float(leaf.params["theta"])
                    for leaf in leaves
                ],
            }
        )
    return rows


def run_benchmark(*, bases: tuple[Basis, ...] = DEFAULT_BASES) -> dict[str, Any]:
    """Measure reach and phase on every basis of ``bases``."""

    rows = [reach(basis) for basis in bases]
    phases = [phase_evidence(basis) for basis in bases]
    return {
        "schema": SCHEMA,
        "artifact_classification": "local_compiler_microbenchmark",
        "distribution_semantics": "single_device_fast_path",
        "scalability_claim_allowed": False,
        "reference_algorithm": "qiskit_one_qubit_euler_decomposer",
        "reference_revision": "qiskit 1.2.4 OneQubitEulerDecomposer('ZSX'/'ZYZ')",
        "synthesis_identity": "U3(theta, phi, lam) == RZ(lam) SX RZ(theta - pi) SX RZ(phi - pi)",
        "phase_contract": "equal up to one global phase, which CircuitIR cannot record",
        "declared_single_qubit_unitary_opcodes": list(SINGLE_QUBIT_OPCODES),
        "basis_count": len(rows),
        "reach": rows,
        "phase": phases,
        "hand_written_rewrite_opcodes": ("x", "rx", "ry", "swap"),
        "synthesis": synthesis_table(),
        "reference_anchor": _qiskit_anchor(),
    }


def _qiskit_anchor() -> dict[str, Any]:
    """The same table from Qiskit's ZSX decomposer, when Qiskit is importable.

    The anchor is a cross-check, not a dependency: this module is a local
    benchmark and must run on a machine with no Qiskit at all. Where the two
    disagree the sequences are phase- or count-equivalent forms rather than
    errors, which is why the anchor reports the sequences and lets a reader
    compare rather than asserting equality.
    """

    try:
        import qiskit  # type: ignore[import-not-found]
        from qiskit.circuit.library import (  # type: ignore[import-not-found]
            HGate,
            IGate,
            PhaseGate,
            RXGate,
            RYGate,
            RZGate,
            SdgGate,
            SGate,
            SXdgGate,
            SXGate,
            TdgGate,
            TGate,
            U2Gate,
            U3Gate,
            XGate,
            YGate,
            ZGate,
        )
    except ImportError as error:  # pragma: no cover - depends on the environment
        return {"available": False, "reason": str(error)}
    try:
        # Qiskit 1.x keeps the decomposer under `synthesis`; 2.x re-exports it
        # from `quantum_info`.
        from qiskit.synthesis.one_qubit import (  # type: ignore[import-not-found]
            OneQubitEulerDecomposer,
        )
    except ImportError:
        try:
            from qiskit.quantum_info import (  # type: ignore[import-not-found]
                OneQubitEulerDecomposer,
            )
        except ImportError as error:  # pragma: no cover - environment dependent
            return {"available": False, "reason": str(error)}

    gates = {
        "i": IGate(),
        "x": XGate(),
        "y": YGate(),
        "z": ZGate(),
        "h": HGate(),
        "s": SGate(),
        "sdg": SdgGate(),
        "t": TGate(),
        "tdg": TdgGate(),
        "sx": SXGate(),
        "sxdg": SXdgGate(),
        "rx": RXGate(0.7137),
        "ry": RYGate(0.7137),
        "rz": RZGate(0.7137),
        "phase": PhaseGate(0.7137),
        "u1": PhaseGate(0.7137),
        "u2": U2Gate(-0.4211, 1.9073),
        "u3": U3Gate(0.7137, -0.4211, 1.9073),
    }
    decomposer = OneQubitEulerDecomposer("ZSX")
    rows: list[dict[str, Any]] = []
    for opcode in SINGLE_QUBIT_OPCODES:
        if opcode == "rz":
            continue
        reference = [item.operation.name for item in decomposer(gates[opcode]).data]
        leaves = synthesize_one_qubit(single_qubit_instruction(opcode), z_rotation="rz")
        assert leaves is not None, opcode
        port = [leaf.name for leaf in leaves]
        rows.append(
            {
                "opcode": opcode,
                "qiskit_gate_names": reference,
                "port_gate_names": port,
                "identical": reference == port,
                "same_length": len(reference) == len(port),
            }
        )
    return {
        "available": True,
        # The library reading, not the pinned `reference_revision` above, so that
        # every anchor in this directory answers "which lane?" the same way. The
        # counts below are the ones this module measured as stable across the
        # certified lanes, which is a reason to read them as stable rather than a
        # reason to leave the lane unnamed.
        "qiskit_version": qiskit.__version__,
        "decomposer": "OneQubitEulerDecomposer('ZSX')",
        "identical_gate_count": sum(1 for row in rows if row["identical"]),
        "same_length_count": sum(1 for row in rows if row["same_length"]),
        "compared_gate_count": len(rows),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    payload = run_benchmark()
    print(
        f"{payload['declared_single_qubit_unitary_opcodes'].__len__()} declared "
        f"single-qubit unitary opcodes on {payload['basis_count']} bases"
    )
    print(
        f"{'basis':28s} {'before':>7s} {'after':>7s} {'max leaves':>11s} "
        f"{'unresolved':>11s}"
    )
    for row in payload["reach"]:
        print(
            f"{row['label']:28s} "
            f"{row['identity_table_reach_count']:>4d}/18 "
            f"{row['legalized_opcode_count']:>4d}/18 "
            f"{row['max_replacement_length']:>11d} "
            f"{row['unresolved_opcode_count']:>11d}"
        )
    print()
    print(
        f"{'basis':28s} {'gates':>11s} {'overlap':>18s} {'phase':>9s} {'raw diff':>9s}"
    )
    for row in payload["phase"]:
        if not row["legalized"]:
            print(f"{row['label']:28s} not legalized: {row['error']}")
            continue
        print(
            f"{row['label']:28s} {row['source_gate_count']:>4d}->"
            f"{row['legalized_gate_count']:<6d} "
            f"{row['state_overlap_magnitude']:>18.15f} "
            f"{row['global_phase_radians']:>+9.6f} "
            f"{row['max_raw_state_difference']:>9.6f}"
        )
    anchor = payload["reference_anchor"]
    if anchor["available"]:
        print()
        print(
            f"Qiskit {anchor['decomposer']} anchor: "
            f"{anchor['identical_gate_count']}/{anchor['compared_gate_count']} "
            "opcodes produce the same gate sequence"
        )
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()

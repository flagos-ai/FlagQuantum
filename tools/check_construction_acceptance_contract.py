#!/usr/bin/env python3
"""Validate the construction-acceptance contract against the implementation.

`contracts/construction-acceptance-contract.toml` records what must be true of a program
`Circuit.compose` built: its IR must equal the hand-built program's IR, and every
execution mode that accepts the request must agree. This gate reads that record back
against the code.

Three groups of checks live here, and they fail for different reasons:

- the census, which fails the moment the construction layer grows a member recorded as
  absent or loses one recorded as present. This is the fail-closed mechanism: acceptance
  that silently stops covering a new member is the failure this contract exists to stop.
- the IR identity claim, which is deterministic and cheap, so it is checked here rather
  than left to a test whose failure output is harder to read.
- the corrections, which re-measure every identifier the plan row named that does not
  exist. A correction that stops being true is a stale document, not a passing gate.

The numerical claims — cross-mode agreement, distributed agreement, tolerances — belong
to the conformance test the contract names, because they need pytest's tolerance
reporting to be readable when they fail.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 and older
    import tomli as tomllib

import flagquantum as fq
from flagquantum.core.ir import IR_VERSION, CircuitIR
from flagquantum.core.operator_schema import _SCHEMAS
from flagquantum.runtime.options import _MODES

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "contracts" / "construction-acceptance-contract.toml"

#: Members the census reads off `Circuit`. `__pow__` is the operator spelling of
#: `power`; measuring both keeps the census honest about "absent as a method and absent
#: as an operator", which is one fact and must not be counted as two.
CENSUS_ATTRIBUTES = ("compose", "adjoint", "control", "power", "__pow__")


def _load_toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _test_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    }


def _block_program(width: int) -> fq.Circuit:
    """The block that each contracted placement places."""

    block = fq.Circuit(width)
    for position in range(width):
        block.h(position)
        if position + 1 < width:
            block.cnot(position, position + 1)
    return block


def _hand_built(n_qubits: int, targets: tuple[int, ...]) -> fq.Circuit:
    """The same block emitted directly on the placed qubits, written by hand."""

    circuit = fq.Circuit(n_qubits)
    for offset, target in enumerate(targets):
        circuit.h(target)
        if offset + 1 < len(targets):
            circuit.cnot(target, targets[offset + 1])
    return circuit


def _instruction_signature(circuit: fq.Circuit) -> tuple[Any, ...]:
    return tuple(
        (
            instruction.name,
            tuple(instruction.wires),
            tuple(sorted((k, repr(v)) for k, v in dict(instruction.params).items())),
        )
        for instruction in circuit.to_ir().instructions
    )


def _census_errors(contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    subject = contract.get("subject", {})
    census = contract.get("census", {})
    instance = fq.Circuit(2)
    measured = {name: hasattr(instance, name) for name in CENSUS_ATTRIBUTES}

    for entry in subject.get("covered", ()):
        attribute = entry.rsplit(".", 1)[-1]
        if not measured.get(attribute, False):
            errors.append(f"contracted construction member {entry!r} is not present")
    if census.get("covered_members_are_present") is not True:
        errors.append("census.covered_members_are_present must be true")

    for entry in subject.get("not_covered", ()):
        attribute = entry.rsplit(".", 1)[-1]
        if measured.get(attribute, False):
            errors.append(
                f"construction member {entry!r} is contracted as absent but is present; "
                "the acceptance must be extended to cover it"
            )
    for attribute in census.get("also_measured_absent", ()):
        bare = attribute.rsplit(".", 1)[-1]
        if measured.get(bare, False):
            errors.append(f"{attribute!r} is contracted as absent but is present")
    for attribute in census.get("also_measured_absent", ()):
        bare = attribute.rsplit(".", 1)[-1]
        if bare not in measured:
            errors.append(
                f"census measures {attribute!r} but this gate does not read it"
            )

    if int(contract.get("measured_root_export_count", -1)) != len(fq.__all__):
        errors.append(
            "measured_root_export_count is stale: "
            f"contract says {contract.get('measured_root_export_count')}, "
            f"fq.__all__ has {len(fq.__all__)}"
        )
    if contract.get("measured_ir_version") != IR_VERSION:
        errors.append(
            f"measured_ir_version is stale: contract says "
            f"{contract.get('measured_ir_version')!r}, IR_VERSION is {IR_VERSION!r}"
        )
    return errors


def _ir_identity_errors(contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    identity = contract.get("ir_identity", {})
    if identity.get("tolerance") != "exact":
        errors.append("ir_identity.tolerance must be exact")
    declare = tuple(identity.get("measures", ()))

    placements: list[tuple[int, tuple[int, ...]]] = [
        (int(row["n_qubits"]), tuple(int(q) for q in row["qubits"]))
        for row in contract.get("placement", ())
    ]
    mapped: list[tuple[int, tuple[int, ...]]] = []
    for row in contract.get("placement_map", ()):
        mapping = {int(k): int(v) for k, v in row["mapping"].items()}
        mapped.append(
            (int(row["n_qubits"]), tuple(mapping[k] for k in sorted(mapping)))
        )

    if not placements and not mapped:
        errors.append("the contract lists no placement to measure")

    for n_qubits, targets in placements:
        if len(set(targets)) != len(targets):
            errors.append(f"placement {targets} repeats a qubit")
            continue
        composed = fq.Circuit(n_qubits)
        composed.compose(_block_program(len(targets)), qubits=targets)
        _compare(errors, f"qubits={targets}", n_qubits, composed, targets, declare)

    for n_qubits, targets in mapped:
        if len(set(targets)) != len(targets):
            errors.append(f"placement map {targets} repeats a qubit")
            continue
        mapping = {offset: target for offset, target in enumerate(targets)}
        composed = fq.Circuit(n_qubits)
        composed.compose(_block_program(len(targets)), qubit_map=mapping)
        _compare(errors, f"qubit_map={mapping}", n_qubits, composed, targets, declare)
    return errors


def _compare(
    errors: list[str],
    label: str,
    n_qubits: int,
    composed: fq.Circuit,
    targets: tuple[int, ...],
    declare: tuple[str, ...],
) -> None:
    hand = _hand_built(n_qubits, targets)
    hand_ir, composed_ir = hand.to_ir(), composed.to_ir()
    checks = {
        "to_dict": hand_ir.to_dict() == composed_ir.to_dict(),
        "content_hash": hand_ir.content_hash == composed_ir.content_hash,
        "n_wires": hand_ir.n_wires == composed_ir.n_wires == n_qubits,
        "instruction_signature": _instruction_signature(hand)
        == _instruction_signature(composed),
    }
    for name in declare:
        if name not in checks:
            errors.append(f"ir_identity measures {name!r}, which this gate cannot read")
            continue
        if not checks[name]:
            errors.append(
                f"placement {label} on {n_qubits} qubits: composed {name} differs "
                "from the hand-built program"
            )
    expected = 2 * len(targets) - 1
    if len(composed_ir.instructions) != expected:
        errors.append(
            f"placement {label}: emitted {len(composed_ir.instructions)} instructions, "
            f"the contracted count is 2 * width - 1 = {expected}"
        )
    if not isinstance(composed_ir, CircuitIR):
        errors.append(f"placement {label}: to_ir() did not return a CircuitIR")


def _mode_partition_errors(contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    modes = contract.get("modes", {})
    serving = list(modes.get("serving", ()))
    refused = list(modes.get("refused", ()))
    measured = set(_MODES)
    if serving and refused and set(serving) & set(refused):
        errors.append(
            f"modes.serving and modes.refused overlap: {sorted(set(serving) & set(refused))}"
        )
    combined = set(serving) | set(refused)
    if combined != measured:
        errors.append(
            "modes.serving plus modes.refused does not equal the measured mode set: "
            f"only in contract {sorted(combined - measured)}, "
            f"only in the implementation {sorted(measured - combined)}"
        )
    if modes.get("serving_and_refused_must_partition_the_mode_set") is not True:
        errors.append(
            "modes.serving_and_refused_must_partition_the_mode_set must be true"
        )
    return errors


def _correction_errors(contract: dict[str, Any]) -> list[str]:
    """Re-measure every identifier the plan row named that does not exist.

    Each row is a claim about the implementation, not a note. A row whose `still_absent`
    is true must still be absent; a row that stopped being absent means this contract has
    gone stale and the acceptance text needs revisiting.
    """

    errors: list[str] = []
    instance = fq.Circuit(2).to_ir()
    run_signature = inspect.signature(fq.run)
    plan_signature = inspect.signature(fq.plan)
    remeasured = {
        "CircuitIR.n_qubits": hasattr(instance, "n_qubits"),
        "CircuitIR.semantic_fingerprint": hasattr(instance, "semantic_fingerprint"),
        "ExecutionOptions(mode='distributed_statevector')": "distributed_statevector"
        in _MODES,
        "cross-mode agreement": False,
    }
    for row in contract.get("plan_corrections", ()):
        field = row.get("field")
        if field not in remeasured:
            errors.append(
                f"plan correction {field!r} has no re-measurement in this gate"
            )
            continue
        if not row.get("still_absent"):
            continue
        if remeasured[field]:
            errors.append(
                f"plan correction {field!r} claims the name is absent, but it is present; "
                "the acceptance text is stale"
            )
        if not str(row.get("measured_spelling", "")).strip():
            errors.append(
                f"plan correction {field!r} does not record the spelling to use"
            )
        if not str(row.get("consequence", "")).strip():
            errors.append(f"plan correction {field!r} does not record the consequence")

    distributed = contract.get("distributed_member", {})
    if distributed.get("reachable_through_execution_options") is not False:
        errors.append(
            "distributed_member.reachable_through_execution_options must be false"
        )
    if distributed.get("root_run_accepts_world_size") is not False:
        errors.append("distributed_member.root_run_accepts_world_size must be false")
    if "world_size" in run_signature.parameters:
        errors.append("fq.run now accepts world_size; distributed_member is stale")
    if "world_size" in plan_signature.parameters:
        errors.append("fq.plan now accepts world_size; distributed_member is stale")
    return errors


def _verification_errors(contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    verification = contract.get("verification", {})
    for key in ("contract_test", "gate"):
        value = verification.get(key)
        if not isinstance(value, str) or not value.endswith(".py"):
            errors.append(f"verification.{key} must name a Python file")
        elif not (ROOT / value).is_file():
            errors.append(f"verification.{key} path {value!r} does not exist")

    raw = verification.get("contract_test")
    if not isinstance(raw, str) or not (ROOT / raw).is_file():
        return errors
    names = _test_names(ROOT / raw)
    requirements = verification.get("requirements", {})
    covered = list(contract.get("subject", {}).get("covered", ()))
    missing = [member for member in covered if member not in requirements]
    if missing:
        errors.append(f"contracted members without a requirement row: {missing}")
    for member, test in requirements.items():
        if member not in covered:
            errors.append(f"requirement names absent member {member!r}")
        if test not in names:
            errors.append(f"requirement test {test!r} for {member!r} does not exist")
    return errors


def contract_errors(contract: dict[str, Any]) -> list[str]:
    errors = _census_errors(contract)
    errors += _ir_identity_errors(contract)
    errors += _mode_partition_errors(contract)
    errors += _correction_errors(contract)
    errors += _verification_errors(contract)
    return errors


def main() -> int:
    errors = contract_errors(_load_toml(CONTRACT))
    if errors:
        print("\n".join(errors))
        return 1
    print(
        "Construction acceptance contract passed: "
        f"{len(fq.__all__)} exports, opcode census {len(_SCHEMAS)}, "
        f"IR_VERSION {IR_VERSION}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

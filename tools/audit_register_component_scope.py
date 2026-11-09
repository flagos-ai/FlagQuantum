"""Inspect the existing Program IR boundary before adding register components.

Run from the checkout with ``python -m tools.audit_register_component_scope``.
This is a read-only capability audit, not a second IR or a replacement verifier.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import fields
from pathlib import Path

from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import (
    BlockRecord,
    FunctionRecord,
    Operation,
    ProgramRecord,
    ValueRef,
    circuit_to_program,
    opcode_signature,
    program_to_circuit,
    verify_program_record,
)
from flagquantum.core.ir.program.registers import Register, RegisterType
from flagquantum.dynamic import DynamicCircuit


def _operation_probe(opcode: str, kind: str) -> list[str]:
    before, after = ValueRef("entry", 0, kind), ValueRef("entry", 1, kind)
    block = BlockRecord(
        "entry",
        arguments=(before,),
        operations=(Operation(opcode, (before,), (after,)),),
    )
    record = ProgramRecord((FunctionRecord("main", (block,)),))
    return sorted({diagnostic.code for diagnostic in verify_program_record(record)})


def _finite_call_probe() -> list[str]:
    value = ValueRef("entry", 0, "qubit")
    output = ValueRef("entry", 1, "qubit")
    leaf = FunctionRecord(
        "leaf", (BlockRecord("entry", (value,), return_values=(value,)),)
    )
    main = FunctionRecord(
        "main",
        (
            BlockRecord(
                "entry",
                (value,),
                operations=(Operation("call", (value,), (output,), callee="leaf"),),
                return_values=(output,),
            ),
        ),
    )
    return sorted(
        {item.code for item in verify_program_record(ProgramRecord((main, leaf)))}
    )


def _register_rebinding_probe() -> list[str]:
    dtype = RegisterType("uint", 1)
    word = ValueRef("entry", 0, "register")
    bit = ValueRef("entry", 1, "qubit")
    returned = ValueRef("entry", 2, "register")
    function = FunctionRecord(
        "main",
        (
            BlockRecord(
                "entry",
                (word,),
                operations=(
                    Operation("split_register", (word,), (bit,)),
                    Operation(
                        "join_register", (bit,), (returned,), register_type=dtype
                    ),
                ),
                return_values=(returned,),
            ),
        ),
        registers=(Register("word", dtype),),
    )
    return sorted({d.code for d in verify_program_record(ProgramRecord((function,)))})


def _round_trip(circuit: CircuitIR) -> dict[str, object]:
    converted = circuit_to_program(circuit)
    if converted.record is None:
        raise RuntimeError(f"Existing circuit lowering failed: {converted.diagnostics}")
    restored = program_to_circuit(converted.record)
    return {
        "lowering_state": converted.state,
        "restoration_state": restored.state,
        "diagnostics": [
            diagnostic.code
            for diagnostic in (*converted.diagnostics, *restored.diagnostics)
        ],
        "verified": not verify_program_record(converted.record),
        "canonical_payload_equal": (
            restored.canonical_payload
            == converted.canonical_payload
            == circuit.to_dict()
        ),
    }


def audit_program_scope(root: Path) -> dict[str, object]:
    """Return observed support and content fingerprints without mutating contracts."""
    contract_path = "contracts/multi-level-ir-internal-v1-candidate.json"
    contract = json.loads((root / contract_path).read_text())
    dynamic = DynamicCircuit(3)
    dynamic.h(0)
    dynamic.measure(0, classical_bit=2)
    dynamic.reset(0)
    dynamic.measure(1, classical_bit=0)
    dynamic.conditional("x", 2, any_of=({0: 0}, {2: 1}))
    protected_paths = (
        contract_path,
        "docs/public_api_v1.json",
        "flagquantum/core/ir/program/model.py",
        "flagquantum/core/ir/program/verifier.py",
        "flagquantum/core/ir/program/call_verifier.py",
        "flagquantum/core/ir/program/register_verifier.py",
        "flagquantum/core/ir/program/lowering.py",
        "flagquantum/core/ir/program/physical_codec.py",
        "flagquantum/core/operator_schema.py",
    )
    signature = opcode_signature("x")
    if signature is None:
        raise RuntimeError("The existing X opcode has no Program IR signature")
    return {
        "basis": "observed_internal_program_scope",
        "contract_version": contract["version"],
        "contract_status": contract["status"],
        "sha256": {
            path: hashlib.sha256((root / path).read_bytes()).hexdigest()
            for path in protected_paths
        },
        "record_fields": {
            record.__name__: [field.name for field in fields(record)]
            for record in (
                ValueRef,
                Operation,
                BlockRecord,
                FunctionRecord,
                ProgramRecord,
            )
        },
        "standalone_metadata_fields": {
            record.__name__: [field.name for field in fields(record)]
            for record in (RegisterType, Register)
        },
        "gate_signature": {
            "opcode": signature.opcode,
            "operand_kinds": list(signature.operand_kinds),
            "result_kinds": list(signature.result_kinds),
        },
        "probes": {
            "gate_on_qubit": _operation_probe("x", "qubit"),
            "gate_on_index": _operation_probe("x", "index"),
            "component_call": _operation_probe("register_call", "qubit"),
            "finite_gate_call": _finite_call_probe(),
            "typed_register_rebinding": _register_rebinding_probe(),
            "quantum_register_kind": _operation_probe("x", "quantum_register"),
        },
        "round_trips": {
            "static": _round_trip(
                CircuitIR(2, (Instruction("h", (0,)), Instruction("cx", (0, 1))))
            ),
            "dynamic": _round_trip(dynamic.to_ir()),
        },
        "claims": {
            "register_component_execution": False,
            "finite_gate_call_structure": not _finite_call_probe(),
            "typed_register_structure": not _register_rebinding_probe(),
            "standalone_register_metadata": True,
            "complete_source_register_metadata_ported": False,
            "resource_dag_is_execution_graph": False,
            "protected_contract_changed": contract["version"] != "0.5",
        },
    }


def main() -> None:
    """Print evidence to stdout; the caller chooses whether to save it."""
    print(
        json.dumps(audit_program_scope(Path(__file__).resolve().parents[1]), indent=2)
    )


if __name__ == "__main__":
    main()

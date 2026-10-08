"""The program level: the value graph, its type signatures, and its lowering.

`contracts/multi-level-ir-internal-v1-candidate.json` declares what the internal
levels must satisfy and `flagquantum/core/ir/levels.py` is the boundary a caller
crosses; this package is the program level behind it. `model.py` owns the value,
block, and operation model, `verifier.py` enforces it, and `lowering.py` derives
that model from a public `CircuitIR` and back.

Each level owns its own verification module rather than sharing one at the level
boundary: engineering decision principle 11 admits a new layer only for a distinct
current responsibility or a second concrete use, and a shared verifier would have
one level to verify.

This package is internal. It is not re-exported from `flagquantum.core.ir`, is
absent from `docs/public_api_v1.json`, and adds no root export.
"""

from .lowering import circuit_to_program, classify_public_circuit, program_to_circuit
from .model import (
    CONSTANT_OPCODE,
    INVALID_INPUT,
    LINEAR_VALUE_KINDS,
    SUPPORT_STATES,
    SUPPORTED_EXACT,
    TERMINATORS,
    UNSUPPORTED_WITH_DIAGNOSTICS,
    VALUE_KINDS,
    BlockRecord,
    Edge,
    FunctionRecord,
    LevelConversion,
    OpcodeSignature,
    Operation,
    ProgramRecord,
    ValueRef,
    opcode_signature,
)
from .verifier import verify_program_record

__all__ = (
    "CONSTANT_OPCODE",
    "INVALID_INPUT",
    "LINEAR_VALUE_KINDS",
    "SUPPORTED_EXACT",
    "SUPPORT_STATES",
    "TERMINATORS",
    "UNSUPPORTED_WITH_DIAGNOSTICS",
    "VALUE_KINDS",
    "BlockRecord",
    "Edge",
    "FunctionRecord",
    "LevelConversion",
    "OpcodeSignature",
    "Operation",
    "ProgramRecord",
    "ValueRef",
    "circuit_to_program",
    "classify_public_circuit",
    "opcode_signature",
    "program_to_circuit",
    "verify_program_record",
)

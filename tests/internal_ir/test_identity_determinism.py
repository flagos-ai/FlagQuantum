from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from flagquantum._compiler.importers.circuit_ir import import_circuit_ir
from flagquantum.core.ir import CircuitIR, Instruction

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]


def _probe(hash_seed: int) -> dict[str, object]:
    script = """
import json
from flagquantum._compiler.importers.circuit_ir import import_circuit_ir
from flagquantum._compiler.passes.canonicalize import CanonicalizeAttributesPass
from flagquantum._compiler.passes.manager import PassManager
from flagquantum.core.ir import CircuitIR, Instruction
source = CircuitIR(2, (
    Instruction('h', (0,)),
    Instruction('cx', (0, 1)),
    Instruction('rz', (1,), {'theta': 0.25}),
), dtype='complex128')
module = import_circuit_ir(source).imported.module
print(json.dumps({
    'program_identity': module.program_identity,
    'canonical_debug_encoding': json.dumps(
        module.canonical(), sort_keys=True, separators=(',', ':')
    ),
    'pipeline_digest': PassManager(
        (CanonicalizeAttributesPass(),)
    ).pipeline_digest,
}, sort_keys=True))
"""
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = str(hash_seed)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return json.loads(completed.stdout)


def test_identity_debug_encoding_and_pipeline_digest_ignore_hash_seed() -> None:
    first = _probe(1)
    second = _probe(8675309)

    assert first == second
    assert len(first["program_identity"]) == 64
    assert len(first["pipeline_digest"]) == 64


def test_immutable_program_identities_are_cached_without_changing_digest() -> None:
    imported = import_circuit_ir(
        CircuitIR(1, (Instruction("rz", (0,), {"theta": 0.25}),))
    ).imported
    assert imported is not None
    module = imported.module

    assert "program_identity" not in module.__dict__
    first_module_identity = module.program_identity
    assert module.__dict__["program_identity"] == first_module_identity
    assert module.program_identity is first_module_identity

    assert "internal_program_identity" not in imported.__dict__
    first_import_identity = imported.internal_program_identity
    assert imported.__dict__["internal_program_identity"] == first_import_identity
    assert imported.internal_program_identity is first_import_identity

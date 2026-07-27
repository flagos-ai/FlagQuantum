#!/usr/bin/env python
"""One-shot mechanical extraction of Circuit statevector helpers.

Kept as an auditable refactoring recipe. It refuses to run after the extraction
or when source anchors differ, so it cannot silently rewrite future code.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "flagquantum" / "circuit.py"
TARGET = ROOT / "flagquantum" / "circuit_statevector.py"

text = SOURCE.read_text(encoding="utf-8")
start = text.index("_FIXED_GATE_CACHE:")
end = text.index("\n\nclass Circuit:")
if TARGET.exists():
    raise SystemExit(f"{TARGET} already exists")

helpers = text[start:end]
target = '''"""Private statevector compilation and tensor helpers for :mod:`circuit`."""

from __future__ import annotations

import os
from dataclasses import dataclass
from numbers import Number
from typing import Any, Mapping, Sequence

import torch

from .core.ir import Instruction
from .core.operator_schema import canonical_opcode
from .core.parameters import value_to_tensor
from .ops.matrices import GATE_MAT_DICT

''' + helpers + "\n"

exports = """from .circuit_statevector import (
    _DIAGONAL_STATEVECTOR_GATES,
    _StatevectorCXSequenceStep,
    _StatevectorFusedGateStep,
    _StatevectorGateStep,
    _StatevectorRXRZLoopStep,
    _apply_cx_permutation,
    _apply_diagonal_matrix,
    _apply_fixed_permutation,
    _apply_matrix,
    _apply_rx_rz_loop,
    _apply_single_qubit_fixed,
    _batched_rx_ry_rz_matrices,
    _bits_from_indices,
    _canonical_name,
    _compile_statevector_program,
    _fused_gate_matrix,
    _gate_matrix,
    _gate_parameter_tensor,
    _normalize_wires,
    _parameter_tensor,
    _statevector_layout,
)
"""
SOURCE.write_text(text[:start] + exports + text[end:], encoding="utf-8")
TARGET.write_text(target, encoding="utf-8")

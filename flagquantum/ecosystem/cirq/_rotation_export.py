#  Copyright 2025 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

# Adapted from Qualtran v0.7.0 rotation_synthesis/matrix/_clifford_t_repr.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/rotation_synthesis/matrix/_clifford_t_repr.py

"""Optional Cirq export for a synthesized single-qubit rotation."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from ...synth import _su2_ct
from ...synth._clifford_t_repr import to_sequence
from .conversion import _cirq

if TYPE_CHECKING:
    import cirq


def to_cirq(
    matrix: _su2_ct.SU2CliffordT, fmt: str, q: cirq.Qid | None = None
) -> tuple[cirq.Operation, ...]:
    """Export a synthesized matrix as Cirq operations on the given qubit."""
    cirq = _cirq()
    q = q if q is not None else cirq.q(0)
    gates = {
        "I": cirq.I,
        "H": cirq.H,
        "S": cirq.S,
        "X": cirq.X,
        "Y": cirq.Y,
        "Z": cirq.Z,
        "Tx": cirq.rx(math.pi / 4),
        "Ty": cirq.ry(math.pi / 4),
        "Tz": cirq.rz(math.pi / 4),
        "Tx*": cirq.rx(-math.pi / 4),
        "Ty*": cirq.ry(-math.pi / 4),
        "Tz*": cirq.rz(-math.pi / 4),
    }
    return tuple(gates[label](q) for label in to_sequence(matrix, fmt))

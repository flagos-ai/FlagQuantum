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

"""Export synthesized single-qubit words as Quirk cell tokens."""

from ..synth import _su2_ct
from ..synth._clifford_t_repr import to_sequence


def _to_quirk_name(name: str, allow_global_phase: bool = False) -> str:
    if allow_global_phase:
        if name == "I":
            return "1"
        if name in ("X", "Y", "Z", "H"):
            return '"' + name + '"'
        if name == "S":
            return '"Z^½"'
        if name == "S*":
            return '"Z^-½"'
        if name.startswith("T"):
            if name.endswith("*"):
                return '"' + name[1].upper() + "^-¼" + '"'
            return '"' + name[1].upper() + "^¼" + '"'
    else:
        if name == "I":
            return "1"
        if name == "H":
            return '"H"'
        if name in ("X", "Y", "Z"):
            return '{"id":"R%sft","arg":"-pi"}' % (name.lower())
        if name == "S":
            return '{"id":"Rzft","arg":"pi/2"}'
        if name == "S*":
            return '{"id":"Rzft","arg":"-pi/2"}'
        if name.startswith("T"):
            angle = ["pi/4", "-pi/4"][name.endswith("*")]
            return '{"id":"R%sft","arg":"%s"}' % (name[1].lower(), angle)
    raise ValueError(f"{name=} is not supported")


def to_quirk(
    matrix: _su2_ct.SU2CliffordT, fmt: str, allow_global_phase: bool = False
) -> tuple[str, ...]:
    """Return the source-format Quirk tokens in circuit order.

    Args:
        matrix: The matrix to represent.
        fmt: The gates to use (see the documentation of to_sequence).
        allow_global_phase: whether the result can have a global phase or not.
            If false, each gate (except H) gets replaced by SU2 version:
                - S -> Rz(pi/2)
                - T -> Rz(pi/4)
                - X -> Rx(-pi)
                - Y -> Ry(-pi)
                - Z -> Rz(-pi)
            A prefix of SXSX is added for each H, modulo four.
    Returns:
        A tuple quirk symbols.
    """
    sequence = to_sequence(matrix, fmt)
    phase_correction = tuple[str, ...]()
    if not allow_global_phase:
        phase = sum(g == "H" for g in sequence) % 4
        phase_correction = ('"Z^½"', '"X"', '"Z^½"', '"X"') * phase
    return phase_correction + tuple(
        _to_quirk_name(name, allow_global_phase) for name in sequence
    )

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

"""Synthesize exact scaled matrices into circuit-order Clifford+T words."""

from typing import cast

from . import _su2_ct, _zsqrt2
from ._rotation_generation import generate_cliffords

_TWO = _zsqrt2.ZSqrt2(2)


def clifford(matrix: _su2_ct.SU2CliffordT) -> tuple[str, ...]:
    assert matrix.det() == 2
    cliffords = generate_cliffords()
    if matrix in cliffords:
        return cliffords[matrix]
    return ("Z", "X", "Z", "X") + cliffords[-matrix]


def _xyz_sequence(matrix: _su2_ct.SU2CliffordT) -> tuple[str, ...]:
    seq = []
    while matrix.det() > _TWO:
        t = _su2_ct._key_map()[matrix._key]
        nxt = (_su2_ct.GATE_MAP[t].adjoint() @ matrix).scale_down()
        assert nxt is not None
        assert nxt.det() < matrix.det()
        seq.append(t)
        matrix = nxt
    return clifford(matrix) + tuple(seq[::-1])


_T_list = [
    ("Tz", _su2_ct.Tz),
    ("Tx", _su2_ct.Tx),
    ("Tz*", _su2_ct.Tz.adjoint()),
    ("Tx*", _su2_ct.Tx.adjoint()),
]


def _xz_sequence(
    matrix: _su2_ct.SU2CliffordT, use_hs: bool = True, prv: str = "dummy"
) -> tuple[str, ...] | None:
    if matrix.det() == 2:
        return clifford(matrix)
    cliffords = [_su2_ct.ISqrt2]
    if use_hs:
        cliffords.append(_su2_ct.HSqrt2)
        cliffords.append(_su2_ct.HSqrt2 @ _su2_ct.SSqrt2)
    pref = prv.removesuffix("*")
    for c in cliffords:
        for name, t in _T_list:
            if name.startswith(pref):
                continue
            rotated = c.adjoint() @ matrix
            new = (t.adjoint() @ rotated).scale_down()
            if new is None or not new.is_valid():
                continue
            seq = _xz_sequence(new, False, name)
            if seq is None:
                continue
            gates = cast(tuple[str, ...], c.gates)
            return seq + (name,) + gates
    return None


def to_sequence(matrix: _su2_ct.SU2CliffordT, fmt: str = "xyz") -> tuple[str, ...]:
    r"""Returns a sequence of Clifford+T that produces the given matrix.

    Allowable format strings are
     - 'xyz' uses Tx, Ty, Tz gates.
     - 'xz' uses $Tx, Tz, Tx^\dagger, Tz^\dagger$

    Args:
        matrix: The matrix to represent.
        fmt: A string from the set {'xz', 'xyz'} representing the allowed T gates described above.

    Returns:
        A tuple of strings representing the gates.

    Raises:
        ValueError: If `fmt` is not supported.
    """
    if fmt == "xyz":
        return _xyz_sequence(matrix)
    if fmt == "xz":
        return cast(tuple[str, ...], _xz_sequence(matrix))
    raise ValueError(f"{type=} is not supported")

# Adapted in part from OpenFermion, Apache-2.0. See third_party/openfermion.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""General JW, prefix-parity and Fenwick Bravyi-Kitaev encodings."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from ...core.algebra.pauli import PauliOperator, count, pauli
from ...observables.fermion import FermionOperator

Encoding = Literal["jordan_wigner", "parity", "bravyi_kitaev"]
ENCODINGS = ("jordan_wigner", "parity", "bravyi_kitaev")


def encode_occupation(
    occupation: Sequence[int], *, encoding: Encoding
) -> tuple[int, ...]:
    """Return encoded bits, qubit zero first, from an occupation vector."""
    if encoding not in ENCODINGS:
        raise ValueError(f"unsupported encoding {encoding!r}")
    bits = tuple(count(bit, "occupation bit") for bit in occupation)
    if not bits or any(bit > 1 for bit in bits):
        raise ValueError("occupation must contain at least one binary digit")
    if encoding == "jordan_wigner":
        return bits
    if encoding == "parity":
        return tuple(sum(bits[: i + 1]) % 2 for i in range(len(bits)))
    return tuple(
        sum(bits[i + 1 - ((i + 1) & -(i + 1)) : i + 1]) % 2 for i in range(len(bits))
    )


@dataclass(frozen=True, slots=True)
class EncodedOperator:
    """An operator and the complete binary basis convention it uses."""

    operator: PauliOperator
    encoding: Encoding
    n_modes: int

    def __post_init__(self) -> None:
        if not isinstance(self.operator, PauliOperator):
            raise TypeError("operator must be a native PauliOperator")
        object.__setattr__(self, "n_modes", count(self.n_modes, "n_modes", minimum=1))
        if self.encoding not in ENCODINGS:
            raise ValueError(f"unsupported encoding {self.encoding!r}")
        if any(q >= self.n_modes for t in self.operator.terms for q, _ in t.factors):
            raise ValueError("operator acts outside the declared mode space")

    @property
    def n_qubits(self) -> int:
        return self.n_modes

    def reference_bits(self, occupation: Sequence[int]) -> tuple[int, ...]:
        """Encode a reference using this result's exact width and convention."""
        if len(occupation) != self.n_modes:
            raise ValueError("reference occupation does not match the mode space")
        return encode_occupation(occupation, encoding=self.encoding)


def _bk_ladder(mode: int, creating: bool, width: int) -> PauliOperator:
    update = {mode}
    index = mode + 1
    index += index & -index
    while index <= width:
        update.add(index - 1)
        index += index & -index
    occupation = {mode}
    index = mode
    parent = (mode + 1) & mode
    while index != parent:
        occupation.add(index - 1)
        index &= index - 1
    parity: set[int] = set()
    index = mode
    while index:
        parity.add(index - 1)
        index &= index - 1
    real = tuple((i, "x") for i in update) + tuple((i, "z") for i in parity)
    imaginary = (
        ((mode, "y"),)
        + tuple((i, "x") for i in update - {mode})
        + tuple((i, "z") for i in (parity ^ occupation) - {mode})
    )
    return pauli(0.5, real) + pauli(-0.5j if creating else 0.5j, imaginary)


def _ladder(mode: int, creating: bool, width: int, encoding: Encoding) -> PauliOperator:
    if encoding == "bravyi_kitaev":
        return _bk_ladder(mode, creating, width)
    imaginary = -0.5j if creating else 0.5j
    if encoding == "jordan_wigner":
        prefix = tuple((q, "z") for q in range(mode))
        return pauli(0.5, prefix + ((mode, "x"),)) + pauli(
            imaginary, prefix + ((mode, "y"),)
        )
    suffix = tuple((q, "x") for q in range(mode + 1, width))
    prefix = () if mode == 0 else ((mode - 1, "z"),)
    return pauli(0.5, prefix + ((mode, "x"),) + suffix) + pauli(
        imaginary, ((mode, "y"),) + suffix
    )


def encode_fermion(
    operator: FermionOperator, *, n_modes: int, encoding: Encoding = "jordan_wigner"
) -> EncodedOperator:
    """Map a general complex fermionic operator without a Hermiticity assumption.

    Args:
        operator: Existing canonical FlagQuantum fermionic value.
        n_modes: Explicit width, including idle modes.
        encoding: JW, prefix parity, or Fenwick BK basis.

    Returns:
        The Pauli operator with its encoding convention and width.

    Raises:
        ValueError: A mode is outside the declared space or encoding is unknown.
    """
    if not isinstance(operator, FermionOperator):
        raise TypeError("operator must be a native FermionOperator")
    width = count(n_modes, "n_modes", minimum=1)
    if encoding not in ENCODINGS:
        raise ValueError(f"unsupported encoding {encoding!r}")
    result = PauliOperator()
    for term in operator.terms:
        image = pauli(term.coefficient)
        for modes, creating in ((term.creations, True), (term.annihilations, False)):
            for mode in modes:
                if mode >= width:
                    raise ValueError("fermionic mode outside the declared space")
                image = image * _ladder(mode, creating, width, encoding)
        result = result + image
    return EncodedOperator(result, encoding, width)

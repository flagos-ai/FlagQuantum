"""SELECT for the sparse second-quantized chemistry Hamiltonian."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ....circuit import Circuit
from ....errors import ValidationError
from .selected_majorana_fermion import SelectedMajoranaFermion


@dataclass(frozen=True)
class SelectSparse:
    """Bind sparse chemistry SELECT with no external control register.

    The four spatial orbital registers are big-endian. The spin is the least
    significant bit of each spin-orbital address. The one-body flag selects
    the additional p/q pair and contributes its S phase.
    """

    p: tuple[int, ...]
    q: tuple[int, ...]
    r: tuple[int, ...]
    s: tuple[int, ...]
    alpha: int
    beta: int
    flag_1b: int
    sys: tuple[int, ...]

    @classmethod
    def make_on(
        cls,
        *,
        p: Sequence[int],
        q: Sequence[int],
        r: Sequence[int],
        s: Sequence[int],
        alpha: int,
        beta: int,
        flag_1b: int,
        sys: Sequence[int],
    ) -> SelectSparse:
        """Bind SELECT to FlagQuantum qubit indices."""
        return cls(
            tuple(p), tuple(q), tuple(r), tuple(s), alpha, beta, flag_1b, tuple(sys)
        )

    def append_to(self, circuit: Circuit) -> Circuit:
        """Append the phase and four Majorana instructions."""
        if not self.sys or len(self.sys) % 2:
            raise ValidationError(
                "Sparse SELECT needs a positive even number of spin orbitals."
            )
        width = (len(self.sys) // 2 - 1).bit_length()
        if any(len(register) != width for register in (self.p, self.q, self.r, self.s)):
            raise ValidationError(
                "Sparse SELECT orbital registers have the wrong width."
            )
        qubits = (
            *self.p,
            *self.q,
            *self.r,
            *self.s,
            self.alpha,
            self.beta,
            self.flag_1b,
            *self.sys,
        )
        if any(
            type(qubit) is not int or not 0 <= qubit < circuit.n_qubits
            for qubit in qubits
        ):
            raise ValidationError("Sparse SELECT qubits must be valid circuit indices.")
        if len(set(qubits)) != len(qubits):
            raise ValidationError("Sparse SELECT registers must use disjoint qubits.")

        circuit.gate("s", self.flag_1b)
        for orbital, spin, control, pauli in (
            (self.p, self.alpha, (self.flag_1b,), "Y"),
            (self.q, self.alpha, (self.flag_1b,), "X"),
            (self.r, self.beta, (), "Y"),
            (self.s, self.beta, (), "X"),
        ):
            circuit.append(
                SelectedMajoranaFermion.make_on(
                    selection=(*orbital, spin),
                    target=self.sys,
                    control=control,
                    target_gate=pauli,
                )
            )
        return circuit

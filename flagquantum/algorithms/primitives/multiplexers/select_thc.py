"""Executable SELECT for tensor hypercontraction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ....circuit import Circuit
from ....errors import ValidationError
from .thc_rotations import THCRotations

_Network = Sequence[tuple[int, int, float]]


@dataclass(frozen=True)
class SelectTHC:
    """Select one- and two-electron reflections from explicit Givens tables.

    The address registers are big-endian. The one-electron flag selects
    the one-body table for mu and suppresses the nu reflection. Spin bit
    zero selects sys_a. Success bit zero disables both reflections.
    PREPARE supplies the ordered address and spin pairs and the coefficient
    sign. Two-electron branches exchange the pairs and flip plus_mn so a
    second SELECT applies the reflections in reverse order.
    """

    mu: tuple[int, ...]
    nu: tuple[int, ...]
    succ: int
    nu_eq_mp1: int
    plus_mn: int
    plus_a: int
    plus_b: int
    sys_a: tuple[int, ...]
    sys_b: tuple[int, ...]
    networks: tuple[tuple[tuple[int, int, float], ...], ...]
    one_body_networks: tuple[tuple[tuple[int, int, float], ...], ...]

    @classmethod
    def make_on(
        cls,
        *,
        mu: Sequence[int],
        nu: Sequence[int],
        succ: int,
        nu_eq_mp1: int,
        plus_mn: int,
        plus_a: int,
        plus_b: int,
        sys_a: Sequence[int],
        sys_b: Sequence[int],
        networks: Sequence[_Network],
        one_body_networks: Sequence[_Network],
    ) -> SelectTHC:
        """Bind both reflections to the prepared selection and system qubits."""
        return cls(
            tuple(mu),
            tuple(nu),
            succ,
            nu_eq_mp1,
            plus_mn,
            plus_a,
            plus_b,
            tuple(sys_a),
            tuple(sys_b),
            tuple(map(tuple, networks)),
            tuple(map(tuple, one_body_networks)),
        )

    def append_to(self, circuit: Circuit) -> Circuit:
        """Emit the two rotate–reflect–unrotate blocks as ordinary gates."""
        width = len(self.mu)
        orbitals = len(self.sys_a)
        if (
            not self.networks
            or width != len(self.networks).bit_length()
            or len(self.nu) != width
            or orbitals == 0
            or len(self.sys_b) != orbitals
            or len(self.one_body_networks) != orbitals
            or orbitals > 1 << width
        ):
            raise ValidationError(
                "THC address, orbital, and rotation table widths disagree."
            )
        wires = (
            *self.mu,
            *self.nu,
            self.succ,
            self.nu_eq_mp1,
            self.plus_mn,
            self.plus_a,
            self.plus_b,
            *self.sys_a,
            *self.sys_b,
        )
        fragment = Circuit(len(wires), dtype=circuit.dtype, bsz=circuit.bsz)
        mu, nu = tuple(range(width)), tuple(range(width, 2 * width))
        succ, one_body, order, spin_mu, spin_nu = range(2 * width, 2 * width + 5)
        sys_a = tuple(range(2 * width + 5, 2 * width + 5 + orbitals))
        sys_b = tuple(range(2 * width + 5 + orbitals, len(wires)))
        padding = ((),) * ((1 << width) - len(self.networks))

        for address, spin, networks, first in (
            (
                (one_body, *mu),
                spin_mu,
                (*self.networks, *padding, *self.one_body_networks),
                True,
            ),
            (nu, spin_nu, self.networks, False),
        ):
            rotations = THCRotations.make_on(
                selection=address, target=sys_a, networks=networks
            )
            for a, b in zip(sys_a, sys_b, strict=True):
                fragment.cswap(spin, a, b)
            fragment.append(rotations)
            if first:
                fragment.cz(succ, sys_a[0])
            else:
                fragment.x(one_body).h(sys_a[0])
                fragment.ccx(succ, one_body, sys_a[0])
                fragment.h(sys_a[0]).x(one_body)
            fragment.append(rotations.adjoint())
            for a, b in zip(sys_a, sys_b, strict=True):
                fragment.cswap(spin, a, b)
        fragment.x(one_body).cx(one_body, order)
        fragment.cswap(one_body, spin_mu, spin_nu)
        for left, right in zip(mu, nu, strict=True):
            fragment.cswap(one_body, left, right)
        fragment.x(one_body)
        return circuit.compose(fragment, qubits=wires)

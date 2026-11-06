"""Executable SELECT for tensor hypercontraction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ....circuit import Circuit
from ....compiler.registers.phase_gradient import phase_gradient_state_ir
from ....core.classical_control import validate_classical_control
from ....errors import ValidationError
from ....runtime.dynamic.circuit import DynamicCircuit
from .thc_rotations import THCRotations

_ANGLE_BITS = 8

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
    workspace: tuple[int, ...] = ()
    measurement_erasure: bool = False

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
        workspace: Sequence[int] = (),
        measurement_erasure: bool = False,
    ) -> SelectTHC:
        """Bind both reflections to the prepared selection and system qubits.

        Pass 18 + len(mu) clean workspace qubits to use the 8-bit QROM and
        phase-gradient path. All workspace returns to zero. Without workspace,
        the operation uses exact rotations. Pass eight additional clean qubits
        to use the two-row QROAM load with a junk word. With workspace and a DynamicCircuit,
        measurement_erasure=True uses paired-block phase corrections to erase
        the loaded angles. The default uses reversible XOR unlookup. With
        measurement erasure, six more clean qubits select Qualtran's 8-bit
        carry chain, for either lookup layout.
        """
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
            tuple(workspace),
            measurement_erasure,
        )

    def append_to(self, circuit: Circuit) -> Circuit:
        """Emit the two rotate–reflect–unrotate blocks as ordinary gates."""
        return self._append_to(circuit)

    def _append_to(self, circuit: Circuit, *, measurement_start: int = 0) -> Circuit:
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
        workspace_size = 2 * _ANGLE_BITS + width + 2
        extra_carries = _ANGLE_BITS - 2
        measured_add = self.measurement_erasure and len(self.workspace) in (
            workspace_size + extra_carries,
            workspace_size + _ANGLE_BITS + extra_carries,
        )
        lookup_size = len(self.workspace) - (extra_carries if measured_add else 0)
        blocked = lookup_size == workspace_size + _ANGLE_BITS
        if self.workspace and lookup_size != workspace_size and not blocked:
            raise ValidationError(
                f"THC lookup requires {workspace_size} or {workspace_size + _ANGLE_BITS} clean work qubits."
            )
        measurement_bits: tuple[int, ...] = ()
        if self.measurement_erasure:
            if not self.workspace or not isinstance(circuit, DynamicCircuit):
                raise ValidationError(
                    "THC measurement erasure requires workspace and a DynamicCircuit."
                )
            next_bit = validate_classical_control(
                circuit.to_ir().instructions, n_qubits=circuit.n_qubits
            )
            next_bit = max(next_bit, measurement_start)
            measurement_bits = tuple(
                range(next_bit, next_bit + _ANGLE_BITS * (2 if blocked else 1))
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
            *self.workspace,
        )
        fragment = Circuit(len(wires), dtype=circuit.dtype, bsz=circuit.bsz)
        mu, nu = tuple(range(width)), tuple(range(width, 2 * width))
        succ, one_body, order, spin_mu, spin_nu = range(2 * width, 2 * width + 5)
        sys_a = tuple(range(2 * width + 5, 2 * width + 5 + orbitals))
        logical_width = 2 * width + 5 + 2 * orbitals
        sys_b = tuple(range(2 * width + 5 + orbitals, logical_width))
        if self.workspace:
            angle_data = tuple(range(logical_width, logical_width + _ANGLE_BITS))
            gradient = tuple(
                range(logical_width + _ANGLE_BITS, logical_width + 2 * _ANGLE_BITS)
            )
            carry = logical_width + 2 * _ANGLE_BITS
            scratch = tuple(range(carry + 1, logical_width + workspace_size))
            junk = tuple(
                range(logical_width + workspace_size, logical_width + lookup_size)
            )
            carries = (carry, *range(logical_width + lookup_size, len(wires)))
            for gate in phase_gradient_state_ir(gradient):
                fragment.gate(gate.name, gate.wires, params=gate.params)
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
            if self.workspace:
                rotations._append_qrom(
                    fragment,
                    angle_data,
                    gradient,
                    carries,
                    scratch,
                    measurement_bits,
                    junk,
                )
            else:
                fragment.append(rotations)
            if first:
                fragment.cz(succ, sys_a[0])
            else:
                fragment.x(one_body).h(sys_a[0])
                fragment.ccx(succ, one_body, sys_a[0])
                fragment.h(sys_a[0]).x(one_body)
            if self.workspace:
                rotations.adjoint()._append_qrom(
                    fragment,
                    angle_data,
                    gradient,
                    carries,
                    scratch,
                    measurement_bits,
                    junk,
                )
            else:
                fragment.append(rotations.adjoint())
            for a, b in zip(sys_a, sys_b, strict=True):
                fragment.cswap(spin, a, b)
        fragment.x(one_body).cx(one_body, order)
        fragment.cswap(one_body, spin_mu, spin_nu)
        for left, right in zip(mu, nu, strict=True):
            fragment.cswap(one_body, left, right)
        fragment.x(one_body)
        if self.workspace:
            for gate in phase_gradient_state_ir(gradient, inverse=True):
                fragment.gate(gate.name, gate.wires, params=gate.params)
        return circuit.compose(fragment, qubits=wires)

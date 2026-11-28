# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 MultiplexedCSwap3D and its controlled variant.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/pbc/first_quantization/select_and_prepare.py

# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/pbc/first_quantization/projectile/select_and_prepare.py

"""Components for first-quantized plane-wave chemistry."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from ..circuit import Circuit
from ..compiler._component_trace.resource_model import ComponentCost, CostCall, GateCost
from ..compiler.registers.bits.bitwise import _validate_wires
from ..compiler.registers.lookup.multiplexed_swap import (
    multiplexed_cswap_ir,
    source_multiplexed_cswap_cost,
)
from ..core.ir import CircuitIR, Instruction
from ..core.ir.program.register_types import (
    bounded_unsigned_register_type,
    opaque_register_type,
    quantum_bit_type,
)
from ..core.ir.program.registers import Register
from ..core.ir.program.signatures import Signature
from ..core.value_math import SymbolicInt
from ..errors import ValidationError


def _word(name: str, width: SymbolicInt, shape: tuple[int, ...] = ()) -> Register:
    return Register(name, opaque_register_type(width), shape=shape)


def _bit(name: str) -> Register:
    return Register(name, quantum_bit_type())


def _bounded(
    name: str, width: SymbolicInt, length: SymbolicInt | None = None
) -> Register:
    return Register(
        name,
        bounded_unsigned_register_type(width, 2**width if length is None else length),
    )


def first_quantization_signature(
    p: int, eta: int, atoms: int, m: int, *, prepare: bool
) -> Signature:
    """Describe the matching source PREPARE or SELECT port order and domains."""
    if any(type(value) is not int or value < 1 for value in (p, eta, atoms, m)):
        raise ValueError("First-quantized dimensions must be positive integers")
    return _first_quantization_declaration(p, eta, atoms, m, prepare=prepare)


def _first_quantization_declaration(
    p: SymbolicInt, eta: int, atoms: int, m: int, *, prepare: bool
) -> Signature:
    """Retain source logical widths without changing concrete API admission."""
    n_eta, n_atoms = (eta - 1).bit_length(), (atoms - 1).bit_length()
    indices = (_bounded("i", n_eta, eta), _bounded("j", n_eta, eta))
    momenta = tuple(_bounded(name, p) for name in ("r", "s", "mu"))
    transfers = tuple(_bounded(name, p + 1) for name in ("nu_x", "nu_y", "nu_z"))
    common = (*momenta, *transfers, _bounded("m", (m - 1).bit_length()))
    if prepare:
        return Signature(
            (
                _bounded("tuv", 1),
                _bounded("uv", 1),
                *indices,
                _bounded("w", 2, 3),
                *common,
                _bounded("l", n_atoms, n_atoms),
                _bit("succ_nu"),
                _bit("plus_t"),
            )
        )
    return Signature(
        (
            *(_bit(name) for name in ("tuv", "uv", "i_ne_j", "plus_t")),
            *indices,
            _bounded("w", 3),
            *common,
            _bounded("l", n_atoms),
            _word("sys", p, (eta, 3)),
        )
    )


@dataclass(frozen=True)
class MultiplexedCSwap3D:
    """Exchange an addressed particle's three momentum coordinates with junk.

    Coordinates are ordered x, y, z. Each coordinate and the selection word
    are big-endian. Junk can carry arbitrary quantum data. The explicit
    traversal workspace starts and ends in zero.
    """

    selection: tuple[int, ...]
    targets: tuple[tuple[tuple[int, ...], ...], ...]
    junk: tuple[tuple[int, ...], ...]
    workspace: tuple[int, ...] = ()
    classical_bits: tuple[int, ...] = ()
    adjoint: bool = False

    @classmethod
    def make_on(
        cls,
        *,
        selection: Sequence[int],
        targets: Sequence[Sequence[Sequence[int]]],
        junk: Sequence[Sequence[int]],
        workspace: Sequence[int] = (),
        classical_bits: Sequence[int] = (),
        adjoint: bool = False,
    ) -> "MultiplexedCSwap3D":
        """Bind particle, junk and work registers to qubit indices."""
        return cls(
            tuple(selection),
            tuple(tuple(tuple(word) for word in particle) for particle in targets),
            tuple(tuple(word) for word in junk),
            tuple(workspace),
            tuple(classical_bits),
            adjoint,
        )

    def emit(self) -> tuple[Instruction, ...]:
        """Bind the three coordinates as one contiguous logical word per row."""
        if len(self.targets) < 2:
            raise ValidationError("Momentum exchange requires at least two particles.")
        if len(self.selection) != (len(self.targets) - 1).bit_length():
            raise ValidationError(
                "Momentum exchange selection width disagrees with the particle count."
            )
        if len(self.junk) != 3 or not self.junk[0]:
            raise ValidationError(
                "Momentum exchange requires three nonempty coordinates."
            )
        width = len(self.junk[0])
        if any(
            len(particle) != 3 or any(len(word) != width for word in particle)
            for particle in (*self.targets, self.junk)
        ):
            raise ValidationError("Momentum coordinates must have equal widths.")
        return multiplexed_cswap_ir(
            (),
            (self.selection,),
            (len(self.targets),),
            tuple(tuple(wire for word in row for wire in word) for row in self.targets),
            tuple(wire for word in self.junk for wire in word),
            self.workspace,
            self.classical_bits,
            adjoint=self.adjoint,
        )

    def append_to(self, circuit: Circuit) -> Circuit:
        """Append the bound exchange and its measurement feedback."""
        ir = CircuitIR(
            circuit.n_qubits, self.emit(), metadata={"batch_size": circuit.bsz}
        )
        return circuit.compose(ir.validate())


def momentum_exchange_signature(eta: int, num_bits_p: int) -> Signature:
    """Declare the bounded selector and shaped momentum banks."""
    if type(eta) is not int or eta < 2 or type(num_bits_p) is not int or num_bits_p < 1:
        raise ValueError(
            "Momentum exchange requires at least two particles and a positive coordinate width"
        )
    return _momentum_exchange_declaration(eta, num_bits_p)


def _momentum_exchange_declaration(eta: int, num_bits_p: SymbolicInt) -> Signature:
    """Describe source exchange ports independently of its Split body."""
    selection_width = (eta - 1).bit_length()
    return Signature(
        (
            _bounded("sel", selection_width, eta),
            _word("targets", num_bits_p, (eta, 3)),
            _word("junk", num_bits_p, (3,)),
        )
    )


def source_multiplexed_cswap_3d_cost(eta: int, num_bits_p: int) -> ComponentCost:
    """Estimate the source reshaping wrapper and its shared exchange leaf."""
    signature = momentum_exchange_signature(eta, num_bits_p)
    selection_width = int(signature.get_left("sel").bitsize)
    child = source_multiplexed_cswap_cost((eta,), (selection_width,), 0, 3 * num_bits_p)
    ports = int(signature.n_qubits())
    return ComponentCost("MultiplexedCSwap3D", ports, calls=(CostCall(child),))


@dataclass(frozen=True)
class ControlledMultiplexedCSwap3D:
    """Exchange the prefix of each of three projectile words with an electron.

    Words are big-endian in x/y/z order. The locked source uses cvs for
    diagram labels and the child exchange activates on all-one controls.
    An explicit control_values pattern sets the native activation.
    The untouched suffixes may carry arbitrary states. The workspace is clean.
    """

    controls: tuple[int, ...]
    cvs: tuple[int, ...]
    selection: tuple[int, ...]
    targets: tuple[tuple[tuple[int, ...], ...], ...]
    junk: tuple[tuple[int, ...], ...]
    workspace: tuple[int, ...] = ()
    classical_bits: tuple[int, ...] = ()
    adjoint: bool = False
    control_values: tuple[int, ...] | None = None

    @classmethod
    def make_on(
        cls,
        *,
        controls: Sequence[int],
        cvs: int | Sequence[int],
        selection: Sequence[int],
        targets: Sequence[Sequence[Sequence[int]]],
        junk: Sequence[Sequence[int]],
        workspace: Sequence[int] = (),
        classical_bits: Sequence[int] = (),
        adjoint: bool = False,
        control_values: Sequence[int] | None = None,
    ) -> "ControlledMultiplexedCSwap3D":
        """Bind wires, with optional explicit activation in control order."""
        return cls(
            tuple(controls),
            (cvs,) if isinstance(cvs, int) else tuple(cvs),
            tuple(selection),
            tuple(tuple(tuple(word) for word in row) for row in targets),
            tuple(tuple(word) for word in junk),
            tuple(workspace),
            tuple(classical_bits),
            adjoint,
            None if control_values is None else tuple(control_values),
        )

    def _activation(self) -> tuple[int, ...]:
        if (
            not self.controls
            or len(self.controls) != len(self.cvs)
            or any(type(value) is not int or value not in (0, 1) for value in self.cvs)
        ):
            raise ValidationError(
                "Supply at least one control and one binary label per control."
            )
        values = self.control_values
        if values is None:
            return (1,) * len(self.controls)
        if len(values) != len(self.controls) or any(
            type(value) is not int or value not in (0, 1) for value in values
        ):
            raise ValidationError("Supply one binary activation value per control.")
        return values

    def _coordinate_widths(self) -> tuple[int, int]:
        if (
            len(self.targets) < 2
            or len(self.selection) != (len(self.targets) - 1).bit_length()
        ):
            raise ValidationError(
                "Supply at least two particles and their bounded selection word."
            )
        if len(self.targets[0]) != 3 or not self.targets[0][0] or len(self.junk) != 3:
            raise ValidationError(
                "Momentum exchange requires three nonempty coordinates."
            )
        width = len(self.targets[0][0])
        junk_width = len(self.junk[0])
        if junk_width < width or any(len(word) != junk_width for word in self.junk):
            raise ValidationError(
                "Projectile coordinates must have equal widths at least as wide as electron coordinates."
            )
        if any(
            len(row) != 3 or any(len(word) != width for word in row)
            for row in self.targets
        ):
            raise ValidationError("Electron coordinates must have equal widths.")
        return width, junk_width

    def emit(self) -> tuple[Instruction, ...]:
        """Exchange coordinate prefixes under the bound activation."""
        values = self._activation()
        width, _ = self._coordinate_widths()
        targets = tuple(
            tuple(wire for word in row for wire in word) for row in self.targets
        )
        _validate_wires(
            (
                *self.controls,
                *self.selection,
                *(wire for row in targets for wire in row),
                *(wire for word in self.junk for wire in word),
                *self.workspace,
            )
        )
        flips = tuple(
            Instruction("x", (wire,))
            for wire, value in zip(self.controls, values, strict=True)
            if value == 0
        )
        exchange = multiplexed_cswap_ir(
            self.controls,
            (self.selection,),
            (len(targets),),
            targets,
            tuple(wire for word in self.junk for wire in word[:width]),
            self.workspace,
            self.classical_bits,
            adjoint=self.adjoint,
        )
        return (*flips, *exchange, *flips)

    def resource_model(self) -> ComponentCost:
        """Price the source exchange plus the explicit control flips."""
        values = self._activation()
        width, junk_width = self._coordinate_widths()
        exchange = source_controlled_multiplexed_cswap_3d_cost(
            len(self.targets), width, junk_width, len(values)
        )
        return replace(
            exchange,
            gates=exchange.gates.plus(GateCost(clifford=2 * values.count(0)), 1),
        )

    def append_to(self, circuit: Circuit) -> Circuit:
        """Append the bound exchange and its measurement feedback."""
        gates = self.emit()
        if any(wire >= circuit.n_qubits for word in self.junk for wire in word):
            raise ValidationError("Projectile coordinate exceeds circuit width.")
        ir = CircuitIR(circuit.n_qubits, gates, metadata={"batch_size": circuit.bsz})
        return circuit.compose(ir.validate())


def source_controlled_multiplexed_cswap_3d_cost(
    eta: int, num_bits_p: int, num_bits_n: int, control_count: int
) -> ComponentCost:
    """Estimate the source exchange while retaining live projectile suffixes."""
    if (
        any(type(n) is not int for n in (eta, num_bits_p, num_bits_n, control_count))
        or eta < 2
        or not 1 <= num_bits_p <= num_bits_n
        or control_count < 1
    ):
        raise ValueError(
            "Controlled momentum exchange requires eta >= 2, 1 <= p <= n, and controls."
        )
    selection_width = (eta - 1).bit_length()
    child = source_multiplexed_cswap_cost(
        (eta,), (selection_width,), control_count, 3 * num_bits_p
    )
    ports = control_count + selection_width + 3 * (eta * num_bits_p + num_bits_n)
    return ComponentCost(
        "ControlledMultiplexedCSwap3D",
        ports,
        calls=(CostCall(child, bystander_qubits=3 * (num_bits_n - num_bits_p)),),
    )

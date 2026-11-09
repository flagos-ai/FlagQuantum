"""Periodic square-lattice Hubbard evolution from arXiv:2012.09238."""

from collections.abc import Iterator
from dataclasses import dataclass
from math import isfinite, pi

from ...circuit import Circuit
from ...compiler._component_trace.resource_model import ComponentCost
from ...compiler.registers.bits.hamming_weight import hamming_weight_requirements
from ...compiler.registers.layout.partition import unpartition_register
from ...compiler.registers.phase.hamming_weight_phasing import hamming_weight_phasing_ir
from ...compiler.registers.primitives.global_phase import scalar_phase_ir
from ...core._composition import inverted_instruction
from ...core.classical_control import validate_classical_control
from ...core.ir import CircuitIR, Instruction
from ...dynamic import DynamicCircuit
from .hubbard_plaquette import _plaquette_basis
from .hubbard_trotter import _suzuki_schedule, source_hubbard_suzuki_cost


def _plaquettes(length: int, offset: int) -> tuple[tuple[int, ...], ...]:
    return tuple(
        (
            row * length + col,
            row * length + (col + 1) % length,
            ((row + 1) % length) * length + (col + 1) % length,
            ((row + 1) % length) * length + col,
        )
        for row in range(offset, length, 2)
        for col in range(offset, length, 2)
    )


def _gather(wires: tuple[int, ...], order: tuple[int, ...]) -> Iterator[Instruction]:
    remaining = list(range(len(wires)))
    for target, site in enumerate(order):
        position = remaining.index(site, target)
        while position > target:
            pair = wires[position - 1 : position + 1]
            yield Instruction("swap", pair)
            yield Instruction("cz", pair)
            remaining[position - 1], remaining[position] = (
                remaining[position],
                remaining[position - 1],
            )
            position -= 1


@dataclass(frozen=True)
class _DiagonalLayer:
    basis: tuple[Instruction, ...]
    groups: tuple[tuple[int, ...], tuple[int, ...]]
    theta: float

    def emit(
        self, work: tuple[int, ...], bits: tuple[int, ...]
    ) -> Iterator[Instruction]:
        yield from self.basis
        for group in self.groups:
            if work:
                # Rz(theta) = exp(-i theta/2) * diag(1, exp(i theta)).
                yield from scalar_phase_ir(group[0], -len(group) * self.theta / 2)
                yield from hamming_weight_phasing_ir(
                    group,
                    work[: len(bits)],
                    work[len(bits) :],
                    bits,
                    exponent=self.theta / pi,
                )
            else:
                for wire in group:
                    yield Instruction("rz", (wire,), {"theta": self.theta})
        for gate in reversed(self.basis):
            yield inverted_instruction(gate)


def _hopping_layer(
    up: tuple[int, ...], down: tuple[int, ...], length: int, offset: int, theta: float
) -> _DiagonalLayer:
    order = tuple(site for square in _plaquettes(length, offset) for site in square)
    blocks = tuple(
        wires[first : first + 4]
        for wires in (up, down)
        for first in range(0, len(wires), 4)
    )
    return _DiagonalLayer(
        (
            *_gather(up, order),
            *_gather(down, order),
            *(gate for block in blocks for gate in _plaquette_basis(block)),
        ),
        (tuple(block[1] for block in blocks), tuple(block[2] for block in blocks)),
        theta,
    )


def _interaction_layer(
    up: tuple[int, ...], down: tuple[int, ...], theta: float
) -> _DiagonalLayer:
    middle = len(down) // 2
    return _DiagonalLayer(
        tuple(Instruction("cx", (a, b)) for a, b in zip(up, down, strict=True)),
        (down[:middle], down[middle:]),
        theta,
    )


@dataclass(frozen=True)
class _BoundHubbardStep:
    layers: tuple[_DiagonalLayer, ...]
    work: tuple[int, ...]
    slot_count: int
    required_qubits: int
    cost: ComponentCost

    def resource_model(self) -> ComponentCost:
        return self.cost

    def append_to(self, circuit: Circuit) -> Circuit:
        if circuit.n_qubits < self.required_qubits:
            raise ValueError("Hubbard binding exceeds circuit width")
        bits: tuple[int, ...] = ()
        if self.slot_count:
            if not isinstance(circuit, DynamicCircuit):
                raise ValueError("Hubbard phasing requires a DynamicCircuit")
            start = validate_classical_control(
                circuit.to_ir().instructions, n_qubits=circuit.n_qubits
            )
            bits = tuple(range(start, start + self.slot_count))
        return circuit.compose(
            CircuitIR(
                circuit.n_qubits,
                tuple(
                    gate
                    for layer in self.layers
                    for gate in layer.emit(self.work, bits)
                ),
                metadata={"batch_size": circuit.bsz},
            ).validate()
        )


class HubbardSuzukiStep:
    """A periodic plaquette step with row-major modes in separate spin sectors."""

    @staticmethod
    def make_on(
        *,
        length: int,
        up: tuple[int, ...],
        down: tuple[int, ...],
        hubb_u: float,
        timestep: float,
        hubb_t: float = 1,
        hamming_weight: bool = False,
        workspace: tuple[int, ...] = (),
        strip_layer: bool = False,
        adjoint: bool = False,
    ) -> _BoundHubbardStep:
        """Bind all modes and optional clean workspace to one Suzuki step.

        Hamming-weight phasing reserves clean work of size `junk + count` for
        a batch of `length**2 // 2` modes and reuses it between batches.
        At length two the periodic colors count parallel bonds separately.
        """
        if type(length) is not int or length < 2 or length % 2:
            raise ValueError("Hubbard hopping requires a positive even lattice length")
        if not all(isfinite(value) for value in (hubb_u, timestep, hubb_t)):
            raise ValueError("Hubbard parameters must be finite")
        sites = length**2
        junk, count = (
            hamming_weight_requirements(sites // 2) if hamming_weight else (0, 0)
        )
        wires = unpartition_register(
            {"up": up, "down": down, "work": workspace},
            (("up", sites), ("down", sites), ("work", junk + count)),
        )
        schedule = _suzuki_schedule(strip_layer)
        if adjoint:
            schedule = tuple(reversed(schedule))
            timestep = -timestep
        layers = []
        for term, coefficient in schedule:
            time = coefficient * timestep
            if term == "interaction":
                layer = _interaction_layer(up, down, -time * hubb_u / 2)
            else:
                layer = _hopping_layer(
                    up, down, length, int(term == "gold"), -2 * time * hubb_t
                )
            layers.append(layer)
        return _BoundHubbardStep(
            tuple(layers),
            workspace,
            junk,
            max(wires) + 1,
            source_hubbard_suzuki_cost(
                length,
                hubb_u,
                timestep,
                hubb_t,
                hamming_weight=hamming_weight,
                strip_layer=strip_layer,
            ),
        )

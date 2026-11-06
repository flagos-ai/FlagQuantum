"""Executable four-mode hopping from Appendix E of arXiv:2012.09238."""

from ...compiler.registers.partition import unpartition_register
from ...core._composition import inverted_instruction
from ...core.ir import Instruction
from .fermionic_fourier import two_bit_fermionic_fourier_ir
from .hubbard_resource_model import source_hubbard_hopping_plaquette_cost
from .trotter_product import _bind_evolution, _BoundEvolution, _validate_angle


def _routed_fourier(
    modes: tuple[int, ...], first: int, second: int
) -> tuple[Instruction, ...]:
    """Apply the real Fourier pair with Jordan–Wigner routing restored."""
    low, high = sorted((first, second))
    route = tuple(
        (modes[position - 1], modes[position]) for position in range(high, low + 1, -1)
    )
    incoming = tuple(
        Instruction(name, pair) for pair in route for name in ("swap", "cz")
    )
    x, y = modes[low + 1], modes[low]
    if first > second:
        x, y = y, x
    return (
        *incoming,
        *two_bit_fermionic_fourier_ir(x, y, 0, 1),
        *reversed(incoming),
    )


def _plaquette_basis(wires: tuple[int, ...]) -> tuple[Instruction, ...]:
    # V† uses the first two Fourier pairs from Eq. E10. The third pair
    # and the occupation phases combine into the XX and YY block.
    a, b = wires[1:3]
    return (
        *_routed_fourier(wires, 2, 0),
        *_routed_fourier(wires, 1, 3),
        Instruction("cx", (a, b)),
        Instruction("h", (a,)),
        Instruction("cx", (a, b)),
        Instruction("x", (b,)),
    )


class HubbardPlaquette:
    """Positive-time hopping on four consecutive modes in cyclic site order."""

    @staticmethod
    def make_on(
        *, modes: tuple[int, ...], timestep: float, adjoint: bool = False
    ) -> _BoundEvolution:
        """Bind exp(i timestep K), where K is the unit-coupling cycle hopping."""
        _validate_angle(timestep)
        wires = unpartition_register({"modes": modes}, (("modes", 4),))
        a, b = wires[1:3]
        basis = _plaquette_basis(wires)
        theta = -2 * timestep
        instructions = (
            *basis,
            Instruction("rz", (a,), {"theta": theta}),
            Instruction("rz", (b,), {"theta": theta}),
            *(inverted_instruction(gate) for gate in reversed(basis)),
        )
        return _bind_evolution(
            wires,
            instructions,
            source_hubbard_hopping_plaquette_cost(theta),
            adjoint=adjoint,
        )

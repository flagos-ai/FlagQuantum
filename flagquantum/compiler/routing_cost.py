"""The device facts a routing cost model reads, and the coercion that builds one.

Routing prices a program against a device before it emits anything: how many
physical qubits the device has, which couplings it declares, whether a placed
gate can execute where it stands, and -- when it cannot -- which physical path to
move it along. Those four questions are the same on an undirected and on an
ordered device, so they are answered once here and read by the estimators and by
the planners alike.

This is a module of its own because the questions are the model's rather than the
router's: ``flagquantum/compiler/routing.py`` emits instructions and decides where
they stand, and it is long enough that the device introspection it prices against
has to be able to grow without displacing the routes it sits beside.
"""

from __future__ import annotations

from collections.abc import Iterable

from ..core.ir import Instruction
from .coupling import CouplingMap
from .directed_topology import DirectedCouplingMap
from .operand_semantics import _operand_symmetry


class _TopologyCostModel:
    """The device facts a routing cost model reads, ordered or unordered.

    A cost model asks its device four things: how many physical qubits it has,
    which couplings it declares, whether a two-qubit instruction can execute where
    it stands, and -- when it cannot -- which physical path to move it along. On an
    undirected ``CouplingMap`` the third question is the second one, because
    either direction of a link satisfies every two-qubit instruction. On a
    ``DirectedCouplingMap`` they differ: an operand-symmetric opcode such as
    ``swap`` may use either direction of a link, while a control/target opcode
    such as ``cx`` may use only the direction its operands are written in. That
    rule is read from
    ``flagquantum.compiler.operand_semantics._operand_symmetry``, which is also
    what direction legalization reads, so the two cannot drift apart.

    A wrong-way placement is repaired with the router's own vocabulary: one SWAP
    across the offending link exchanges the two operands, so the gate executes in
    the direction the device declares. The SWAP is symmetric, so it needs only the
    weak link, and it is layout-neutral when the surrounding strategy restores the
    mapping afterwards. That is why direction belongs to a routing cost model at
    all: on an ordered device it is priced in SWAPs, not merely reported.

    Two questions are deliberately not asked of the ordered edges. Paths and the
    SABRE planners search the weak projection, because inserting a SWAP across a
    link is legal in either direction; and multi-qubit locality is checked on weak
    edges, because a three-qubit instruction has no operand order to satisfy on the
    couplings it spans. Direction decides whether a placed gate is executable, not
    whether the router may cross the link.
    """

    __slots__ = ("_device", "_weak")

    def __init__(self, device: CouplingMap | DirectedCouplingMap) -> None:
        self._device = device
        self._weak = (
            device
            if isinstance(device, CouplingMap)
            else CouplingMap(device.n_qubits, device.edges)
        )

    @property
    def n_qubits(self) -> int:
        return self._device.n_qubits

    @property
    def edges(self) -> tuple[tuple[int, int], ...]:
        """Return the couplings as the device declares them, direction included."""

        return self._device.edges

    @property
    def is_ordered(self) -> bool:
        return isinstance(self._device, DirectedCouplingMap)

    @property
    def direction_semantics(self) -> str:
        return "directed_cx" if self.is_ordered else "logical_wire_order_preserved"

    @property
    def weak_coupling(self) -> CouplingMap:
        """Return the undirected graph the planners search."""

        return self._weak

    def path_cache_info(self) -> dict[str, int]:
        return self._weak.path_cache_info()

    def weak_edge(self, left: int, right: int) -> bool:
        return self._weak.has_edge(left, right)

    def path(self, wires: tuple[int, ...]) -> tuple[int, ...]:
        """Return the qubit sequence a SWAP series must follow to join two qubits."""

        return self._weak.shortest_path(*wires)

    def executes(self, instruction: Instruction, wires: tuple[int, ...]) -> bool:
        """Report whether the device runs this instruction on these qubits.

        Raises:
            ValueError: A directed device is asked about a two-qubit opcode with no
                recorded operand semantics, so no direction rule applies and any
                answer would be a guess. Routing such a device is not a fallback
                decision: the opcode has to be classified before either the device
                or the program can be priced.
        """

        if not self.is_ordered:
            return self._weak.has_edge(*wires)
        control, target = wires
        symmetric = _operand_symmetry(instruction)
        if symmetric is None:
            raise ValueError(
                f"two-wire instruction {instruction.name!r} has no recorded operand "
                "semantics, so an ordered device cannot price it: record the opcode "
                "in flagquantum.core.operator_schema.OPERATOR_SCHEMAS"
            )
        if symmetric:
            return self._weak.has_edge(control, target)
        return self._device.has_edge(control, target)

    def needs_orientation_swap(
        self,
        instruction: Instruction,
        wires: tuple[int, ...],
    ) -> bool:
        """Report whether a placed gate is on its link the wrong way round.

        True exactly when the device declares the link but not the direction these
        operands are written in, so one SWAP across the link makes the gate
        executable. An opcode with no recorded operand semantics, and a pair with
        no link to run along at all, are not this case: the first is refused
        earlier by ``executes``, and the second is unrouted rather than reversed.
        """

        if not self.is_ordered:
            return False
        if _operand_symmetry(instruction) is not False:
            return False
        return self._weak.has_edge(*wires) and not self._device.has_edge(*wires)


#: The device forms a routing entry prices: a declared coupling map, an ordered
#: one, an edge sequence, or the cost model this module builds from them. The last
#: member is what makes one weak projection and one path cache reusable across the
#: strategies an automatic choice prices.
_PricedDevice = (
    CouplingMap | DirectedCouplingMap | Iterable[tuple[int, int]] | _TopologyCostModel
)


def _topology_cost_model(
    coupling_map: object,
    wire_count: int,
    entry: str,
) -> _TopologyCostModel:
    """Coerce the device a cost-model entry was handed.

    A directed device is a device the cost model reads, not one it refuses: the
    ordered edges decide which placed gates need a reversal, and the weak
    projection decides where the SWAPs go.

    A model is returned unchanged so that a caller pricing several strategies --
    ``select_routing_strategy`` prices three -- builds one weak projection and one
    path cache, and so measures a later candidate's cache traffic against the
    earlier candidates' work rather than against a cold device.
    """

    if isinstance(coupling_map, _TopologyCostModel):
        return coupling_map
    if isinstance(coupling_map, (CouplingMap, DirectedCouplingMap)):
        return _TopologyCostModel(coupling_map)
    if not isinstance(coupling_map, Iterable):
        raise TypeError(
            f"{entry} takes a CouplingMap, a DirectedCouplingMap, or an iterable "
            f"of (control, target) edges, not a {type(coupling_map).__name__}"
        )
    return _TopologyCostModel(CouplingMap(wire_count, coupling_map))

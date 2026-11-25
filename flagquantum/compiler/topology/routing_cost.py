"""Adapt topology and calibration into routing legality and cost calculations.

``link_error_rates.py`` owns calibration validation, weighted path search and its
cache. This module adapts that calibration and connectivity for route estimation
and instruction emission, and counts source operations and emitted link faults.
Existing LinkErrorRates imports remain direct aliases here.
"""

from __future__ import annotations

from collections.abc import Iterable

from ...core.ir import CircuitIR, Instruction
from ..lowering.operand_semantics import _operand_symmetry
from ..optimization.ordering_barrier import is_ordering_barrier
from .coupling import CouplingMap
from .directed_topology import DirectedCouplingMap
from .link_error_rates import LinkErrorRates as LinkErrorRates
from .link_error_rates import _quantized as _quantized


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
    ``flagquantum.compiler.lowering.operand_semantics._operand_symmetry``, which is also
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

    A device that also states a per-link error rate answers the path question
    differently, and the model carries that calibration rather than replacing
    itself: ``path`` returns the lowest-error route when the device is calibrated
    and the shortest route when it is not, so one planner serves both devices and a
    caller can read which question was answered from ``link_error_rates``.
    """

    __slots__ = ("_device", "_weak", "_link_error_rates")

    def __init__(
        self,
        device: CouplingMap | DirectedCouplingMap,
        link_error_rates: LinkErrorRates | None = None,
    ) -> None:
        self._device = device
        self._weak = (
            device
            if isinstance(device, CouplingMap)
            else CouplingMap(device.n_qubits, device.edges)
        )
        if link_error_rates is not None and (
            link_error_rates.n_qubits != device.n_qubits
            or tuple(link_error_rates.edges) != tuple(device.edges)
        ):
            raise ValueError(
                "link error rates describe a different device than the couplings "
                "being priced"
            )
        self._link_error_rates = link_error_rates

    @property
    def n_qubits(self) -> int:
        return self._device.n_qubits

    @property
    def link_error_rates(self) -> LinkErrorRates | None:
        """Return the device calibration this model routes against, if any."""

        return self._link_error_rates

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
        if self._link_error_rates is not None:
            return self._link_error_rates.path_cache_info()
        return self._weak.path_cache_info()

    def weak_edge(self, left: int, right: int) -> bool:
        return self._weak.has_edge(left, right)

    def path(self, wires: tuple[int, ...]) -> tuple[int, ...]:
        """Return the qubit sequence a SWAP series must follow to join two qubits.

        The shortest such sequence when the device states only its couplings. When
        it also states a per-link error rate, the sequence that pays the least
        summed link error, which may be a longer sequence than the shortest one:
        that substitution is the whole difference between pricing a route in hops
        and pricing it in faults.
        """

        if self._link_error_rates is not None:
            return self._link_error_rates.lowest_error_path(*wires)
        return self._weak.shortest_path(*wires)

    def link_error(self, left: int, right: int) -> float:
        """Return one link's two-qubit error probability.

        Raises:
            ValueError: If the device states no link error rate, so there is no
                such quantity to read. Returning zero would price every link as
                faultless and silently turn a weighted route into a shortest route.
        """

        if self._link_error_rates is None:
            raise ValueError(
                "this device states no link error rate, so it cannot be priced in "
                "faults: pass a LinkErrorRates device to weigh a route"
            )
        return self._link_error_rates.error(left, right)

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
#: one, a calibrated one, an edge sequence, or the cost model this module builds
#: from them. The last member is what makes one weak projection and one path cache
#: reusable across the strategies an automatic choice prices.
_PricedDevice = (
    CouplingMap
    | DirectedCouplingMap
    | LinkErrorRates
    | Iterable[tuple[int, int]]
    | _TopologyCostModel
)


def require_weight_reading_strategy(
    strategy: str,
    topology: _TopologyCostModel,
    entry: str,
    reading_strategies: tuple[str, ...],
) -> None:
    """Refuse a strategy that would route a calibrated device without reading it.

    A caller who calibrates a device has stated which of its links are clean and
    which are not. A strategy that chooses routes by hop count answers a different
    question, so running it would report a plan the calibration had no part in and
    leave the caller to assume otherwise. The refusal names both sides and the
    strategies that can be used instead, because the repair is a choice the caller
    makes rather than one this function can make for it.

    `reading_strategies` is the caller's vocabulary rather than this module's: the
    router owns which strategies it has and which of them read a weight, and this
    function owns the check.
    """

    if topology.link_error_rates is None:
        return
    if strategy in reading_strategies:
        return
    raise ValueError(
        f"{entry} cannot route a device that states a per-link error rate with "
        f"strategy {strategy!r}: that strategy chooses its routes by hop count and "
        f"would ignore the calibration. Use one of "
        + ", ".join(repr(name) for name in reading_strategies)
        + "."
    )


def emitted_link_faults(
    routed: Iterable[Instruction],
    topology: _TopologyCostModel,
) -> float | None:
    """Sum the declared link error of every two-qubit operation a routed program runs.

    Read off the program rather than accumulated while emitting it, so the number
    describes what a device would have to execute: every SWAP the strategy inserted,
    every routed gate on the link it landed on, and nothing that is not a
    two-qubit device operation. A channel instruction spans two qubits without
    being a gate, and an ordering barrier carries no qubits of its own, so both are
    excluded -- the same exclusions the source counts make, which is why the two
    totals describe the same program.

    ``None`` when the device states no per-link error rate, in which case there is
    no such quantity to report.
    """

    if topology.link_error_rates is None:
        return None
    total = 0.0
    for instruction in routed:
        if len(instruction.wires) != 2:
            continue
        if instruction.metadata.get("is_channel"):
            continue
        if is_ordering_barrier(instruction):
            continue
        total += topology.link_error(*instruction.wires)
    return total


def _topology_cost_model(
    coupling_map: object,
    wire_count: int,
    entry: str,
) -> _TopologyCostModel:
    """Coerce the device a cost-model entry was handed.

    A directed device is a device the cost model reads, not one it refuses: the
    ordered edges decide which placed gates need a reversal, and the weak
    projection decides where the SWAPs go.

    A calibrated device is the same couplings with a per-link error rate beside
    them, so the model keeps both: the couplings answer executability, and the
    calibration answers which route to take.

    A model is returned unchanged so that a caller pricing several strategies --
    ``select_routing_strategy`` prices three -- builds one weak projection and one
    path cache, and so measures a later candidate's cache traffic against the
    earlier candidates' work rather than against a cold device.
    """

    if isinstance(coupling_map, _TopologyCostModel):
        return coupling_map
    if isinstance(coupling_map, LinkErrorRates):
        return _TopologyCostModel(
            coupling_map.device,
            link_error_rates=coupling_map,
        )
    if isinstance(coupling_map, (CouplingMap, DirectedCouplingMap)):
        return _TopologyCostModel(coupling_map)
    if not isinstance(coupling_map, Iterable):
        raise TypeError(
            f"{entry} takes a CouplingMap, a DirectedCouplingMap, a LinkErrorRates "
            f"device, or an iterable of (control, target) edges, not a "
            f"{type(coupling_map).__name__}"
        )
    return _TopologyCostModel(CouplingMap(wire_count, coupling_map))


def _is_routable_two_wire_gate(instruction: Instruction) -> bool:
    """Report whether a routing strategy inserts SWAPs for this instruction's qubits.

    The SWAP estimate and a planner's placements have to answer this the same way,
    or the price and the plan disagree about which gates the strategy moved. A
    channel moves with its qubits and an ordering barrier moves nothing at all, so
    neither is a gate a strategy prices.
    """

    if is_ordering_barrier(instruction):
        return False
    return len(instruction.wires) == 2 and not instruction.metadata.get("is_channel")


def _source_routing_counts(
    ir: CircuitIR,
    topology: _TopologyCostModel,
) -> tuple[int, int, int]:
    """Return the source program's topology, routed, and skipped-channel counts.

    These three describe a program against one device, and every strategy reports
    them the same way: a two-qubit operation is a topology gate, one the device
    cannot execute as written is also a routed gate, and a two-qubit channel is
    neither because routing may not move a channel -- and an ordering barrier is
    excluded at every width because it is not a gate. On an ordered device "as
    written" is a claim about the operands, so a CX written against its link's
    declared direction counts as routed work rather than as a gate the device
    already carries. Counting them in one place keeps a strategy's routing
    metadata and its cost estimate from disagreeing about the same program.
    """

    topology_gate_count = 0
    routed_gate_count = 0
    skipped_channel_count = 0
    for instruction in ir:
        if is_ordering_barrier(instruction) or len(instruction.wires) != 2:
            continue
        if instruction.metadata.get("is_channel"):
            skipped_channel_count += 1
            continue
        topology_gate_count += 1
        if not topology.executes(instruction, instruction.wires):
            routed_gate_count += 1
    return topology_gate_count, routed_gate_count, skipped_channel_count


def _count_sabre_orientation_swaps(
    ir: CircuitIR,
    topology: _TopologyCostModel,
    placements: tuple[tuple[int, ...], ...],
) -> int:
    """Count the gates one planned placement leaves on their link the wrong way.

    A planner reports where it places every source instruction, so the count is
    read off that placement rather than re-derived, and the price of a planned plan
    describes the plan the caller will get. Each counted gate costs two SWAPs: the
    materializer wraps it in a SWAP sandwich, so the plan's own layout bookkeeping
    survives the repair and the rest of the plan stays valid.
    """

    return sum(
        1
        for index, instruction in enumerate(ir)
        if _is_routable_two_wire_gate(instruction)
        and topology.needs_orientation_swap(instruction, tuple(placements[index]))
    )

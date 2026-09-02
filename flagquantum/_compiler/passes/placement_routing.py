"""Private deterministic placement and routing for directed CX topologies."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..diagnostics import Diagnostic, DiagnosticCode
from ..ir.modules import Block, QuantumModule, Region
from ..ir.operations import Operation
from ..ir.schemas import circuit_ir_v1_schema_registry
from ..ir.values import ValueId, ValueRef
from ..ir.verifier import verify_module
from .base import PassDescriptor, PassResult

_NATIVE = frozenset({"quantum.rx", "quantum.ry", "quantum.rz", "quantum.cx"})


@dataclass(frozen=True)
class DirectedCouplingGraph:
    """Deterministic directed physical-qubit connectivity."""

    n_qubits: int
    edges: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        n_qubits = int(self.n_qubits)
        edges = tuple(sorted((int(left), int(right)) for left, right in self.edges))
        if n_qubits <= 0:
            raise ValueError("coupling graph requires at least one physical qubit")
        if len(edges) != len(set(edges)):
            raise ValueError("coupling graph edges must be unique")
        for left, right in edges:
            if left == right:
                raise ValueError("coupling graph cannot contain self edges")
            if min(left, right) < 0 or max(left, right) >= n_qubits:
                raise ValueError("coupling graph edge is outside the device")
        object.__setattr__(self, "n_qubits", n_qubits)
        object.__setattr__(self, "edges", edges)

    def canonical(self) -> tuple[object, ...]:
        return ("directed_coupling_graph_v1", self.n_qubits, self.edges)

    def has_edge(self, control: int, target: int) -> bool:
        return (control, target) in self.edges

    def shortest_path(self, start: int, goal: int) -> tuple[int, ...]:
        if min(start, goal) < 0 or max(start, goal) >= self.n_qubits:
            raise ValueError("coupling path endpoint is outside the device")
        if start == goal:
            return (start,)
        adjacency = {wire: set() for wire in range(self.n_qubits)}
        for left, right in self.edges:
            adjacency[left].add(right)
            adjacency[right].add(left)
        queue = deque([(start,)])
        visited = {start}
        while queue:
            path = queue.popleft()
            for neighbor in sorted(adjacency[path[-1]]):
                if neighbor in visited:
                    continue
                candidate = (*path, neighbor)
                if neighbor == goal:
                    return candidate
                visited.add(neighbor)
                queue.append(candidate)
        raise ValueError(f"no coupling path between physical qubits {start} and {goal}")


def _failure(message: str) -> Diagnostic:
    return Diagnostic(
        DiagnosticCode.ATTRIBUTE_TYPE_MISMATCH,
        message,
        notes=("Batch C placement/routing never falls back or repairs topology",),
    )


class _Router:
    def __init__(
        self,
        block: Block,
        graph: DirectedCouplingGraph,
        initial_layout: tuple[int, ...],
    ) -> None:
        self.block = block
        self.graph = graph
        self.logical_to_physical = list(initial_layout)
        self.physical_to_logical = [0] * len(initial_layout)
        for logical, physical in enumerate(initial_layout):
            self.physical_to_logical[physical] = logical
        self.current = list(block.arguments)
        self.source_logical = {
            argument.id: logical for logical, argument in enumerate(block.arguments)
        }
        self.output: list[Operation] = []
        self.next_index = (
            max(
                (
                    value.id.index
                    for operation in block.operations
                    for value in (*operation.operands, *operation.results)
                    if value.id.scope == "routing"
                ),
                default=-1,
            )
            + 1
        )
        self.swap_count = 0
        self.reversed_cx_count = 0

    def _result(self, operand: ValueRef) -> ValueRef:
        result = ValueRef(ValueId(self.next_index, "routing"), operand.type)
        self.next_index += 1
        return result

    def _emit(
        self,
        name: str,
        physicals: tuple[int, ...],
        attributes: Mapping[str, Any],
        source: Operation,
    ) -> None:
        operands = tuple(self.current[physical] for physical in physicals)
        results = tuple(self._result(operand) for operand in operands)
        self.output.append(
            Operation(name, operands, results, attributes, location=source.location)
        )
        for physical, result in zip(physicals, results, strict=True):
            self.current[physical] = result

    def _h(self, physical: int, source: Operation) -> None:
        self._emit("quantum.rz", (physical,), {"theta": 3.141592653589793}, source)
        self._emit("quantum.ry", (physical,), {"theta": 1.5707963267948966}, source)

    def _cx(self, control: int, target: int, source: Operation) -> None:
        if self.graph.has_edge(control, target):
            self._emit("quantum.cx", (control, target), {}, source)
            return
        if not self.graph.has_edge(target, control):
            raise ValueError(f"physical edge {control}->{target} is unavailable")
        self._h(control, source)
        self._h(target, source)
        self._emit("quantum.cx", (target, control), {}, source)
        self._h(control, source)
        self._h(target, source)
        self.reversed_cx_count += 1

    def _swap(self, left: int, right: int, source: Operation) -> None:
        self._cx(left, right, source)
        self._cx(right, left, source)
        self._cx(left, right, source)
        left_logical = self.physical_to_logical[left]
        right_logical = self.physical_to_logical[right]
        self.physical_to_logical[left], self.physical_to_logical[right] = (
            right_logical,
            left_logical,
        )
        self.logical_to_physical[left_logical] = right
        self.logical_to_physical[right_logical] = left
        self.swap_count += 1

    def _route_cx(self, control: int, target: int, source: Operation) -> None:
        left = self.logical_to_physical[control]
        right = self.logical_to_physical[target]
        path = self.graph.shortest_path(left, right)
        for index in range(max(0, len(path) - 2)):
            self._swap(path[index], path[index + 1], source)
        self._cx(
            self.logical_to_physical[control],
            self.logical_to_physical[target],
            source,
        )

    def route_operation(self, operation: Operation) -> None:
        try:
            logicals = tuple(
                self.source_logical[item.id] for item in operation.operands
            )
        except KeyError as exc:
            raise ValueError("operation consumes an unknown logical value") from exc
        if operation.name not in _NATIVE:
            raise ValueError(
                f"operation {operation.name!r} is outside the target profile"
            )
        if operation.name == "quantum.cx":
            self._route_cx(logicals[0], logicals[1], operation)
        else:
            self._emit(
                operation.name,
                (self.logical_to_physical[logicals[0]],),
                operation.attributes,
                operation,
            )
        for result, logical in zip(operation.results, logicals, strict=True):
            self.source_logical[result.id] = logical

    def restore_identity(self, source: Operation) -> None:
        while True:
            logical = next(
                (
                    item
                    for item, physical in enumerate(self.logical_to_physical)
                    if item != physical
                ),
                None,
            )
            if logical is None:
                return
            path = self.graph.shortest_path(self.logical_to_physical[logical], logical)
            edges = tuple(zip(path, path[1:]))
            for left, right in (*edges, *reversed(edges[:-1])):
                self._swap(left, right, source)


class PlacementRoutingPass:
    """Place and route a decomposed module, restoring logical output order."""

    def __init__(
        self,
        graph: DirectedCouplingGraph,
        *,
        initial_layout: tuple[int, ...] | None = None,
    ) -> None:
        self.graph = graph
        self.initial_layout = (
            tuple(range(graph.n_qubits))
            if initial_layout is None
            else tuple(int(item) for item in initial_layout)
        )
        self.descriptor = PassDescriptor(
            "directed_placement_routing",
            "1.0",
            options={
                "graph": graph,
                "initial_layout": self.initial_layout,
                "final_layout": tuple(range(graph.n_qubits)),
                "precondition": "universal_rx_ry_rz_cx_v1",
            },
            program_identity_policy="transform",
        )

    def _validate(self, block: Block) -> str | None:
        logical_count = len(block.arguments)
        if self.graph.n_qubits != logical_count:
            return (
                "Batch C v1 requires physical-qubit count to equal logical-qubit count"
            )
        if len(self.initial_layout) != logical_count:
            return "initial layout length must equal logical-qubit count"
        if set(self.initial_layout) != set(range(logical_count)):
            return "initial layout must be a permutation of physical qubits"
        return None

    def run(self, module: QuantumModule) -> PassResult:
        if len(module.body.blocks) != 1:
            return PassResult(module, diagnostics=(_failure("one block is required"),))
        block = module.body.blocks[0]
        invalid = self._validate(block)
        if invalid is not None:
            return PassResult(module, diagnostics=(_failure(invalid),))
        if not block.operations:
            return PassResult(module, statistics={"physical_swap_count": 0})
        router = _Router(block, self.graph, self.initial_layout)
        try:
            for operation in block.operations:
                router.route_operation(operation)
            router.restore_identity(block.operations[-1])
        except (IndexError, ValueError) as exc:
            return PassResult(module, diagnostics=(_failure(str(exc)),))
        changed = (
            router.swap_count > 0
            or router.reversed_cx_count > 0
            or self.initial_layout != tuple(range(self.graph.n_qubits))
        )
        statistics = {
            "initial_logical_to_physical": self.initial_layout,
            "final_logical_to_physical": tuple(router.logical_to_physical),
            "directed_edges": self.graph.edges,
            "physical_swap_count": router.swap_count,
            "reversed_cx_count": router.reversed_cx_count,
            "operations_emitted": len(router.output),
        }
        if not changed:
            return PassResult(
                module, preserved_analyses=frozenset({"def_use"}), statistics=statistics
            )
        candidate = QuantumModule(
            Region((Block(block.arguments, tuple(router.output)),)),
            revision=module.revision + 1,
        )
        verification = verify_module(candidate, circuit_ir_v1_schema_registry())
        if not verification.ok:
            return PassResult(
                candidate,
                changed=True,
                diagnostics=verification.diagnostics,
                statistics=statistics,
            )
        return PassResult(candidate, changed=True, statistics=statistics)


__all__ = ["DirectedCouplingGraph", "PlacementRoutingPass"]

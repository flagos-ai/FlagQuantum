"""Bind ordinary and controlled block snapshots through the existing LCU boundary."""

from collections.abc import Mapping
from dataclasses import dataclass

from ...compiler._component_trace.resource_model import ComponentCost
from ...core.ir import CircuitIR
from ...core.ir.program.signatures import Signature
from ...core.value_math import SymbolicReal
from ._block_binding import _BoundBlockOperation
from ._block_metadata import _packed_block_signature
from ._circuit_snapshot import _emit, _model, _storage
from .black_box_prepare import BlackBoxPrepare, _identity_prepare


@dataclass(frozen=True)
class CircuitBlockEncoding:
    """Own a zero-signal block's four explicitly supplied execution snapshots.

    Order is forward, adjoint, positive-controlled forward, positive-controlled
    adjoint. The controlled pair may be explicitly absent. Each snapshot packs
    optional control, system, ancilla, resource,
    then clean work. Auxiliary logical ports remain through ports; they are not
    scratch. Models, when supplied, are independent declarations in the same
    order, with an unavailable direction represented by None. Missing models
    cannot be queried as zero costs.

    Zero logical widths admit scalar blocks. A scalar phase snapshot can own
    a clean carrier in its workspace, including an empty identity body; its
    source model remains independent of that physical representation. No
    zero-qubit Circuit or IR is constructed.
    """

    system_bitsize: int
    ancilla_bitsize: int
    resource_bitsize: int
    alpha: SymbolicReal
    epsilon: SymbolicReal
    snapshots: tuple[CircuitIR, CircuitIR, CircuitIR | None, CircuitIR | None]
    models: (
        tuple[
            ComponentCost | None,
            ComponentCost | None,
            ComponentCost | None,
            ComponentCost | None,
        ]
        | None
    ) = None
    binding_swaps: tuple[int, int, int, int] = (0, 0, 0, 0)

    def __post_init__(self) -> None:
        from sympy import Expr

        if any(
            not isinstance(value, (int, float, Expr))
            for value in (self.alpha, self.epsilon)
        ):
            raise TypeError(
                "Block normalization and precision must be native scalar expressions"
            )
        widths = (self.system_bitsize, self.ancilla_bitsize, self.resource_bitsize)
        if any(type(n) is not int or n < 0 for n in widths):
            raise ValueError("Block snapshots require concrete nonnegative port widths")
        if any(len(values) != 4 for values in (self.snapshots, self.binding_swaps)):
            raise ValueError(
                "Block snapshots require four directional/control variants"
            )
        if self.models is not None and (
            len(self.models) != 4
            or any(
                model is not None and not isinstance(model, ComponentCost)
                for model in self.models
            )
        ):
            raise TypeError(
                "Block snapshot models must be independent ComponentCost values"
            )
        if any(type(n) is not int or n < 0 for n in self.binding_swaps):
            raise ValueError("Block binding swap counts must be nonnegative integers")
        if (self.snapshots[2] is None) != (self.snapshots[3] is None):
            raise ValueError(
                "Controlled block snapshots must be supplied as a directional pair"
            )
        if self.snapshots[2] is None and any(self.binding_swaps[2:]):
            raise ValueError("Uncaptured controls cannot declare binding swaps")
        for controlled in (False, True):
            if controlled and self.snapshots[2] is None:
                continue
            self.requirements(controlled=controlled)
            for index in (2 * int(controlled), 2 * int(controlled) + 1):
                model = self.models[index] if self.models is not None else None
                if model is not None and model.qubits < sum(widths) + int(controlled):
                    raise ValueError("Block model omits its logical ports")

    @property
    def signature(self) -> Signature:
        return _packed_block_signature(
            self.system_bitsize, self.ancilla_bitsize, self.resource_bitsize
        )

    @property
    def signal_state(self) -> BlackBoxPrepare:
        return BlackBoxPrepare(
            (("ancilla", self.ancilla_bitsize),), (), _identity_prepare
        )

    def _directions(self, controlled: bool) -> tuple[CircuitIR, CircuitIR]:
        if controlled:
            forward, backward = self.snapshots[2:]
            if forward is None or backward is None:
                raise ValueError("Controlled block snapshots were not captured")
            return forward, backward
        return self.snapshots[0], self.snapshots[1]

    def requirements(self, *, controlled: bool = False) -> tuple[int, int]:
        """Return reusable clean work and result slots for the selected interface."""
        if type(controlled) is not bool:
            raise ValueError("Block snapshot control mode must be Boolean")
        return _storage(
            self._directions(controlled),
            self.signature.n_qubits() + int(controlled),
        )

    def resource_model(
        self, *, controlled: bool = False, adjoint: bool = False
    ) -> ComponentCost:
        if type(controlled) is not bool or type(adjoint) is not bool:
            raise ValueError(
                "Block snapshot direction and control mode must be Boolean"
            )
        index = 2 * int(controlled) + int(adjoint)
        model = self.models[index] if self.models is not None else None
        if model is None:
            raise ValueError(
                "No independent resource model was supplied for this block direction"
            )
        return _model(
            model,
            self._directions(controlled)[int(adjoint)],
            self.binding_swaps[index],
        )

    def make_on(
        self,
        *,
        system: tuple[int, ...],
        ancilla: tuple[int, ...] = (),
        resource: tuple[int, ...] = (),
        control: tuple[int, ...] = (),
        workspace: tuple[int, ...] = (),
        classical_bits: tuple[int, ...] = (),
        adjoint: bool = False,
    ) -> _BoundBlockOperation:
        """Bind explicit ports for circuit.append(op), preserving all logical outputs."""
        if (len(system), len(ancilla), len(resource)) != (
            self.system_bitsize,
            self.ancilla_bitsize,
            self.resource_bitsize,
        ) or len(control) > 1:
            raise ValueError("Block snapshot operands must match its declared ports")
        if type(adjoint) is not bool:
            raise ValueError("Block snapshot direction must be Boolean")
        through = (*control, *system, *ancilla, *resource)
        gates = _emit(
            self._directions(bool(control))[int(adjoint)],
            through,
            workspace,
            classical_bits,
            self.requirements(controlled=bool(control)),
            adjoint,
        )
        return _BoundBlockOperation(gates, max((*through, *workspace), default=-1) + 1)

    def bind_child(
        self,
        predicate: int,
        ports: Mapping[str, tuple[int, ...]],
        work: tuple[int, ...],
        bits: tuple[int, ...],
        adjoint: bool,
    ) -> _BoundBlockOperation:
        """Implement the existing LCU positive-controlled child factory."""
        if set(ports) - {"system", "ancilla", "resource"}:
            raise ValueError("Block child received undeclared ports")
        return self.make_on(
            system=(
                ports.get("system", ()) if not self.system_bitsize else ports["system"]
            ),
            ancilla=ports.get("ancilla", ()),
            resource=ports.get("resource", ()),
            control=(predicate,),
            workspace=work,
            classical_bits=bits,
            adjoint=adjoint,
        )

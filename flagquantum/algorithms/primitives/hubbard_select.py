"""Square-lattice Hubbard SELECT with source-ordered bounded addresses."""

# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 SelectHubbard and its two address consumers.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/hubbard_model/qubitization/select_hubbard.py

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import pi

from ...circuit import Circuit
from ...compiler._component_trace.resource_model import (
    ComponentCost,
    CostCall,
    GateCost,
)
from ...compiler.registers.bits.bitwise import _validate_wires
from ...compiler.registers.layout.partition import RegisterLayout, unpartition_register
from ...compiler.registers.lookup.bounded_majorana_select import (
    bounded_majorana_ir,
    bounded_majorana_workspace,
)
from ...compiler.registers.lookup.unary_iteration import (
    append_nested_bounded_unary_tree,
)
from ...compiler.registers.primitives.and_compute import (
    and_compute_ir,
    and_uncompute_ir,
)
from ...compiler.registers.primitives.controlled_swap import controlled_swap_ir
from ...compiler.registers.primitives.global_phase import scalar_phase_ir
from ...core.classical_control import validate_classical_control
from ...core.ir import CircuitIR, Instruction
from ...dynamic import DynamicCircuit
from .hubbard_prepare import hubbard_selection_layout
from .hubbard_qubitization_resource_model import (
    source_hubbard_majorana_formula,
    source_hubbard_spin_up_formula,
)


@dataclass(frozen=True)
class SelectHubbard:
    """Bind the source SELECT with an optional single-qubit control."""

    dimension: int
    controlled: bool = False
    control_value: int = field(default=1, kw_only=True)

    def __post_init__(self) -> None:
        if type(self.dimension) is not int or self.dimension < 2:
            raise ValueError("Hubbard SELECT requires square side length at least two")
        if type(self.controlled) is not bool:
            raise ValueError("Hubbard SELECT control flag must be boolean")
        if type(self.control_value) is not int or self.control_value not in (0, 1):
            raise ValueError("Hubbard SELECT control value must be zero or one")
        if not self.controlled and self.control_value == 0:
            raise ValueError("Hubbard SELECT control on zero needs a control wire")

    @property
    def layout(self) -> RegisterLayout:
        control = (("control", 1),) if self.controlled else ()
        return (
            *control,
            *hubbard_selection_layout(self.dimension),
            ("target", 2 * self.dimension**2),
        )

    def requirements(self) -> tuple[int, int]:
        """Return the clean Majorana work and reusable feedback slots."""
        d = self.dimension
        work = bounded_majorana_workspace(
            2 * d**2, controlled=self.controlled, axis_extents=(d, d, 2)
        )
        return work, work - 1

    def source_resource_model(self) -> ComponentCost:
        """Report the source wrapper's declared costs and its decomposed width."""
        sites = self.dimension**2
        controls = int(self.controlled)
        coordinate = (self.dimension - 1).bit_length()
        width = sum(size for _, size in self.layout)
        majorana_ports = 2 * sites + 2 * coordinate + 1 + controls
        majorana = source_hubbard_majorana_formula(
            self.dimension, controlled=self.controlled
        )
        majorana_cost = GateCost(
            and_bloq=int(majorana.and_bloq),
            clifford=int(majorana.clifford),
            measurement=int(majorana.measurement),
        )
        majorana_peak = int(majorana.logical_qubits)
        spin_up = source_hubbard_spin_up_formula(
            self.dimension, controlled=self.controlled
        )
        spin = ComponentCost(
            "HubbardSpinUpZ",
            int(spin_up.logical_qubits),
            GateCost(
                and_bloq=int(spin_up.and_bloq),
                clifford=int(spin_up.clifford),
                measurement=int(spin_up.measurement),
            ),
        )
        swap = ComponentCost("CSwap", 2 * coordinate + 1, GateCost(cswap=coordinate))
        swap_spin = ComponentCost("TwoBitCSwap", 3, GateCost(cswap=1))
        scalar = (
            ComponentCost("SGate", 1, GateCost(clifford=1))
            if self.controlled
            else ComponentCost("GlobalPhase", 0)
        )
        sign = ComponentCost(
            "CZ" if self.controlled else "ZGate", 1 + controls, GateCost(clifford=1)
        )
        select = ComponentCost(
            "SelectHubbard",
            width,
            calls=(
                CostCall(
                    ComponentCost(
                        "HubbardMajorannaOperator(Y)", majorana_peak, majorana_cost
                    ),
                    bystander_qubits=width - majorana_ports,
                ),
                CostCall(swap, 2, bystander_qubits=width - swap.qubits),
                CostCall(swap_spin, bystander_qubits=width - swap_spin.qubits),
                CostCall(
                    ComponentCost(
                        "HubbardMajorannaOperator(X)", majorana_peak, majorana_cost
                    ),
                    bystander_qubits=width - majorana_ports,
                ),
                CostCall(swap_spin, bystander_qubits=width - swap_spin.qubits),
                CostCall(swap, 2, bystander_qubits=width - swap.qubits),
                CostCall(scalar, bystander_qubits=width - scalar.qubits),
                CostCall(sign, bystander_qubits=width - sign.qubits),
                CostCall(spin, bystander_qubits=width - majorana_ports),
            ),
        )
        if self.control_value == 0:
            return ComponentCost(
                "ControlledViaAnd",
                select.qubits,
                GateCost(clifford=2),
                calls=(CostCall(select),),
            )
        return select

    def make_on(
        self, ports: Mapping[str, tuple[int, ...]], *, workspace: tuple[int, ...]
    ) -> "_BoundSelectHubbard":
        """Bind SELECT for circuit.append on a DynamicCircuit."""
        logical = unpartition_register(ports, self.layout)
        _validate_wires((*logical, *workspace))
        if len(workspace) != self.requirements()[0]:
            raise ValueError("Hubbard SELECT requires matching clean work")
        return _BoundSelectHubbard(
            self, dict(ports), workspace, max((*logical, *workspace)) + 1
        )

    def _emit(
        self,
        ports: Mapping[str, tuple[int, ...]],
        work: tuple[int, ...],
        slots: tuple[int, ...],
    ) -> tuple[Instruction, ...]:
        d = self.dimension
        px, py, qx, qy = (ports[name] for name in ("p_x", "p_y", "q_x", "q_y"))
        alpha, beta = ports["alpha"], ports["beta"]
        (u,), (v,) = ports["U"], ports["V"]
        target = ports["target"]
        control = ports["control"][0] if self.controlled else None
        axes = (d, d, 2)
        swap_x = controlled_swap_ir(v, px, qx)
        swap_y = controlled_swap_ir(v, py, qy)
        swap_spin = controlled_swap_ir(v, alpha, beta)
        gates = [
            *bounded_majorana_ir(
                (*px, *py, *alpha), target, control, "Y", work, slots, axis_extents=axes
            ),
            *swap_x,
            *swap_y,
            *swap_spin,
            *bounded_majorana_ir(
                (*qx, *qy, *beta), target, control, "X", work, slots, axis_extents=axes
            ),
            *swap_spin,
            *swap_y,
            *swap_x,
        ]
        start = int(control is not None)
        predicate = v
        if control is None:
            gates.extend(scalar_phase_ir(u, -pi / 2))
            gates.append(Instruction("z", (u,)))
        else:
            gates.extend(
                (Instruction("sdg", (control,)), Instruction("cz", (control, u)))
            )
            predicate = work[0]
            gates.extend(and_compute_ir(v, control, predicate))

        def phase(index: int, enabled: int) -> None:
            gates.append(Instruction("cz", (enabled, target[d**2 + index])))

        # The source spin-up leaf deliberately traverses y before x.
        append_nested_bounded_unary_tree(
            gates, predicate, (qy, qx), (d, d), work[start:], slots[start:], phase
        )
        if control is not None:
            gates.extend(and_uncompute_ir(v, control, predicate, slots[0]))
            if self.control_value == 0:
                flip = Instruction("x", (control,))
                return (flip, *gates, flip)
        return tuple(gates)


@dataclass(frozen=True)
class _BoundSelectHubbard:
    select: SelectHubbard
    ports: Mapping[str, tuple[int, ...]]
    workspace: tuple[int, ...]
    required_qubits: int

    def append_to(self, circuit: Circuit) -> Circuit:
        if circuit.n_qubits < self.required_qubits or not isinstance(
            circuit, DynamicCircuit
        ):
            raise ValueError(
                "Hubbard SELECT needs a DynamicCircuit of sufficient width"
            )
        start = validate_classical_control(
            circuit.to_ir().instructions, n_qubits=circuit.n_qubits
        )
        _, count = self.select.requirements()
        gates = self.select._emit(
            self.ports, self.workspace, tuple(range(start, start + count))
        )
        return circuit.compose(
            CircuitIR(
                circuit.n_qubits, gates, metadata={"batch_size": circuit.bsz}
            ).validate()
        )

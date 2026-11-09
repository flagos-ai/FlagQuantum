# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 QubitizationWalkOperator.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/qubitization/qubitization_walk_operator.py

"""Compose a SELECT block and reflection about its prepared signal."""

from collections.abc import Mapping
from dataclasses import dataclass

from ...circuit import Circuit
from ...compiler._component_trace.resource_model import (
    ComponentCost,
    CostCall,
    GateCost,
)
from ...compiler.registers.bits.bitwise import _validate_wires
from ...compiler.registers.control.control_predicate import (
    ControlRegister,
    bind_control_registers,
)
from ...compiler.registers.control.controlled_component import controlled_component_ir
from ...compiler.registers.layout.partition import RegisterLayout, unpartition_register
from ...compiler.registers.preparation.prepare_reflection import (
    prepare_reflection_ir,
    source_prepare_reflection_cost,
)
from ...core.ir import CircuitIR, Instruction
from .lcu_block_encoding import LCUBlockEncoding
from .select_block_encoding import ControlledSelectEmitter, SelectBlockEncoding


@dataclass(frozen=True)
class QubitizationWalk:
    """Walk in the signal basis of a SELECT or zero-signal LCU block."""

    block_encoding: SelectBlockEncoding | LCUBlockEncoding

    def __post_init__(self) -> None:
        block = self.block_encoding
        if isinstance(block, LCUBlockEncoding) and block.control_value is not None:
            raise ValueError(
                "The source walk signature omits the controlled LCU ctrl port"
            )

    @property
    def junk_layout(self) -> RegisterLayout:
        """The block's logical junk passes through the walk."""
        block = self.block_encoding
        return block.junk_layout if isinstance(block, LCUBlockEncoding) else ()

    @property
    def selection_layout(self) -> RegisterLayout:
        """Return the shared named selection ports in a deterministic order."""
        block = self.block_encoding
        select = dict(block.selection_layout)
        signal = dict(block.signal_state.selection_layout)
        if any(select[name] != signal[name] for name in select.keys() & signal.keys()):
            raise ValueError(
                "Walk selection ports with a shared name must have equal widths"
            )
        if (select.keys() | signal.keys()) & {name for name, _ in block.target_layout}:
            raise ValueError("Walk selection and target port names must be distinct")
        return tuple((select | signal).items())

    @property
    def alpha(self) -> float:
        """Return the L1 normalization of the underlying block."""
        return self.block_encoding.alpha

    def resource_model(
        self,
        select_cost: ComponentCost,
        prepare_cost: ComponentCost,
        unprepare_cost: ComponentCost,
        *,
        adjoint: bool = False,
        control_registers: tuple[ControlRegister, ...] = (),
    ) -> ComponentCost:
        """Count ordered calls and live bystanders.

        For an external control pattern, select_cost is the directional cost
        of the specialized positive-controlled SELECT.
        """
        pattern = tuple(
            bit for register in control_registers for bit in register.bits()
        )
        if control_registers and not pattern:
            raise ValueError("Controlled walk requires a nonempty control")
        controlled = bool(pattern)
        block = self.block_encoding
        width = (
            sum(size for _, size in self.selection_layout)
            + sum(size for _, size in self.junk_layout)
            + block.system_bitsize
            + int(controlled)
        )
        signal = block.signal_state
        if isinstance(block, LCUBlockEncoding):
            encoding = block.resource_model(
                select_cost,
                prepare_cost,
                unprepare_cost,
                adjoint=adjoint,
                controlled=controlled,
            )
            identity = ComponentCost("PrepareIdentity", signal.selection_width)
            reflect_prepare = reflect_unprepare = identity
        else:
            encoding = block.resource_model(select_cost, controlled=controlled)
            reflect_prepare, reflect_unprepare = prepare_cost, unprepare_cost
        reflection = source_prepare_reflection_cost(
            signal.selection_width,
            signal.junk_width,
            reflect_prepare,
            reflect_unprepare,
            controlled=controlled,
        )
        calls = (
            CostCall(encoding, bystander_qubits=width - encoding.qubits),
            CostCall(
                reflection,
                bystander_qubits=width - signal.selection_width - int(controlled),
            ),
        )
        model = ComponentCost(
            "QubitizationWalkOperator†" if adjoint else "QubitizationWalkOperator",
            width,
            calls=calls[::-1] if adjoint else calls,
        )
        if not pattern or pattern == (1,):
            return model
        prefix = len(pattern) - 1
        # CtrlSpecAnd uses a direct And at width two. Its wider MultiAnd
        # declaration charges the polarity flips in both directions.
        cliffords = (prefix or 2) + (4 * pattern.count(0) if prefix > 1 else 0)
        return ComponentCost(
            "ControlledViaAnd",
            width + prefix,
            GateCost(and_bloq=prefix, clifford=cliffords, measurement=prefix),
            (CostCall(model, bystander_qubits=2 * prefix),),
        )

    def make_on(
        self,
        *,
        selection: Mapping[str, tuple[int, ...]],
        target: tuple[int, ...],
        block_junk: tuple[int, ...] = (),
        signal_junk: tuple[int, ...] = (),
        select_workspace: tuple[int, ...] = (),
        select_classical_bits: tuple[int, ...] = (),
        prepare_workspace: tuple[int, ...] = (),
        prepare_classical_bits: tuple[int, ...] = (),
        reflection_workspace: tuple[int, ...],
        reflection_classical_bits: tuple[int, ...] = (),
        adjoint: bool = False,
        controls: tuple[tuple[int, ...], ...] = (),
        control_registers: tuple[ControlRegister, ...] = (),
        controlled_select: ControlledSelectEmitter | None = None,
        control_workspace: tuple[int, ...] = (),
        control_classical_bits: tuple[int, ...] = (),
    ) -> "_BoundQubitizationWalk":
        """Bind one walk step for circuit.append using caller-owned clean work.

        Block junk is logical and stays live through reflection. Signal junk
        is temporary and belongs to the reflection. Clean work can be reused
        between the block and reflection. External controls use one shared
        predicate around both stages and a specialized positive SELECT emitter.
        """
        if type(adjoint) is not bool:
            raise ValueError("Walk direction must be boolean")
        block = self.block_encoding
        signal = block.signal_state
        logical = (
            *unpartition_register(selection, self.selection_layout),
            *block_junk,
            *target,
        )
        select_wires = tuple(
            wire for name, _ in block.selection_layout for wire in selection[name]
        )
        signal_wires = tuple(
            wire for name, _ in signal.selection_layout for wire in selection[name]
        )
        if logical or select_workspace:
            _validate_wires((*logical, *select_workspace))
        _validate_wires((*logical, *signal_junk, *reflection_workspace))
        outer_wires, pattern = bind_control_registers(controls, control_registers)
        reserved = (*outer_wires, *control_workspace)
        if not pattern and (
            control_registers
            or controls
            or controlled_select is not None
            or reserved
            or control_classical_bits
        ):
            raise ValueError("Controlled walk requires a nonempty control")
        if pattern:
            if controlled_select is None:
                raise ValueError(
                    "Controlled walk requires a specialized SELECT emitter"
                )
            for workspace in (
                select_workspace,
                prepare_workspace,
                reflection_workspace,
            ):
                _validate_wires((*reserved, *logical, *signal_junk, *workspace))

        def emit(control: int | None) -> tuple[Instruction, ...]:
            control_wires = () if control is None else (control,)
            phase_wire = reflection_workspace[:1] if control is None else control_wires
            prepare: tuple[Instruction, ...]
            unprepare: tuple[Instruction, ...]
            if isinstance(block, LCUBlockEncoding):
                if signal_junk:
                    raise ValueError(
                        "The zero signal has no temporary preparation junk"
                    )
                lcu_encoding = (
                    LCUBlockEncoding(block.select_block, 1, controlled_select)
                    if controlled_select is not None
                    else block
                )
                select = lcu_encoding.make_on(
                    control=control_wires,
                    selection=select_wires,
                    junk=block_junk,
                    target=target,
                    select_workspace=select_workspace,
                    select_classical_bits=select_classical_bits,
                    prepare_workspace=prepare_workspace,
                    prepare_classical_bits=prepare_classical_bits,
                    adjoint=adjoint,
                ).instructions
                prepare = unprepare = ()
            else:
                if block_junk:
                    raise ValueError("The SELECT block has no logical junk ports")
                _validate_wires(
                    (*logical, *signal_junk, *phase_wire, *prepare_workspace)
                )
                encoding = (
                    block.with_control(1, controlled_select)
                    if controlled_select is not None
                    else block
                )
                select = encoding.emit(
                    (*control_wires, *select_wires),
                    target,
                    select_workspace,
                    select_classical_bits,
                    adjoint=adjoint,
                )
                prepare = signal.emit(
                    signal_wires, signal_junk, prepare_workspace, prepare_classical_bits
                )
                unprepare = signal.emit(
                    signal_wires,
                    signal_junk,
                    prepare_workspace,
                    prepare_classical_bits,
                    adjoint=True,
                )
            reflection = prepare_reflection_ir(
                signal_wires,
                reflection_workspace,
                reflection_classical_bits,
                prepare=prepare,
                unprepare=unprepare,
                global_phase=-1,
                control=control,
            )
            return (*reflection, *select) if adjoint else (*select, *reflection)

        gates = (
            controlled_component_ir(
                controls,
                control_registers,
                emit,
                control_workspace,
                control_classical_bits,
            )
            if pattern
            else emit(None)
        )
        return _BoundQubitizationWalk(
            gates,
            max(
                (
                    *logical,
                    *reserved,
                    *signal_junk,
                    *select_workspace,
                    *prepare_workspace,
                    *reflection_workspace,
                )
            )
            + 1,
        )


@dataclass(frozen=True)
class _BoundQubitizationWalk:
    instructions: tuple[Instruction, ...]
    required_qubits: int

    def append_to(self, circuit: Circuit) -> Circuit:
        if circuit.n_qubits < self.required_qubits:
            raise ValueError("Walk binding exceeds circuit width")
        return circuit.compose(
            CircuitIR(
                circuit.n_qubits,
                self.instructions,
                metadata={"batch_size": circuit.bsz},
            ).validate()
        )

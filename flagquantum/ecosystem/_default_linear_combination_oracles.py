# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 LinearCombination default oracle construction.

"""Replace explicit default LCU oracle objects with native directional Programs."""

from collections.abc import Callable
from math import pi
from typing import Any, cast

from ..compiler.formats.legacy_alias_preparation_factories import (
    alias_preparation_factories,
)
from ..compiler.formats.legacy_component_library import deserialize_component_library
from ..compiler.formats.legacy_component_resolution import resolve_component_program
from ..compiler.formats.legacy_fixed_gate_factories import fixed_gate_factories
from ..compiler.formats.legacy_linear_combination_factories import ExactPair
from ..compiler.formats.legacy_registers import deserialize_registers
from ..compiler.registers.component_builder import ComponentProgramBuilder
from ..compiler.registers.component_gate_function import bit_gate_function
from ..compiler.registers.control.indexed_selection_component import (
    indexed_selection_component_program,
)
from ..compiler.registers.layout.auto_partition import AutoPartition
from ..compiler.registers.layout.controlled_auto_partition_component import (
    controlled_auto_partition_pair,
)
from ..compiler.registers.layout.packed_oracle_component import (
    packed_oracle_component_program,
)
from ..compiler.registers.phase.controlled_block_phase_component import (
    controlled_block_phase_program,
)
from ..core.ir import CircuitIR, Instruction
from ..core.ir.program import FunctionRecord, ProgramRecord
from ..core.ir.program.register_types import quantum_bit_type
from ..core.ir.program.registers import Register
from ..core.ir.program.signatures import Signature
from ..errors import SerializationError


def _signature(registers: Any) -> tuple[Register, ...]:
    from qualtran.serialization.registers import registers_to_proto

    return deserialize_registers(registers_to_proto(registers).SerializeToString())


def default_linear_combination_prepare_pair(source: object) -> ExactPair:
    """Reuse the complete alias constructor and packed PREPARE owners."""
    from qualtran.bloqs.state_preparation.black_box_prepare import BlackBoxPrepare
    from qualtran.bloqs.state_preparation.state_preparation_alias_sampling import (
        StatePreparationAliasSampling,
    )
    from qualtran.serialization.bloq import bloqs_to_proto

    if not isinstance(source, BlackBoxPrepare) or not isinstance(
        source.prepare, StatePreparationAliasSampling
    ):
        raise SerializationError(
            "Default LCU PREPARE requires an explicit packed dense alias oracle"
        )
    foreign = cast(Any, source)
    oracle = foreign.prepare
    pair = tuple(
        resolve_component_program(
            deserialize_component_library(
                bloqs_to_proto(child, max_depth=0).SerializeToString()
            ),
            0,
            alias_preparation_factories(),
        )
        for child in (oracle, oracle.adjoint())
    )
    groups = (
        ("selection", _signature(oracle.selection_registers)),
        ("junk", _signature(oracle.junk_registers)),
    )
    return cast(
        ExactPair,
        tuple(
            packed_oracle_component_program(
                Signature(_signature(foreign.signature)),
                groups,
                pair[0],
                inverse=pair[1],
                adjoint=direction,
            )
            for direction in (False, True)
        ),
    )


def _positive_pair(
    source: Any, child_owner: Callable[[object], ExactPair] | None = None
) -> ExactPair:
    from qualtran.bloqs.basic_gates import Hadamard, TGate, XGate, ZGate
    from qualtran.bloqs.block_encoding.phase import Phase
    from qualtran.bloqs.block_encoding.unitary import Unitary
    from qualtran.bloqs.bookkeeping.auto_partition import (
        AutoPartition as SourcePartition,
    )
    from qualtran.bloqs.bookkeeping.auto_partition import (
        Unused,
    )

    if isinstance(source, SourcePartition):
        child = _positive_pair(source.bloq, child_owner)
        data = _signature(source.bloq.signature)
        widths = {reg.name: reg.total_qubits() for reg in data}
        layout = AutoPartition(
            tuple(
                (
                    outer.name,
                    tuple(
                        (
                            (None, part.bitsize)
                            if isinstance(part, Unused)
                            else (part, widths[part])
                        )
                        for part in parts
                    ),
                )
                for outer, parts in source.partitions
            ),
            left_only=source.left_only,
        )
        return controlled_auto_partition_pair(
            *child, _signature(source.signature), layout, data_registers=data
        )
    if isinstance(source, Phase):
        child = _positive_pair(source.block_encoding, child_owner)
        # Controlled.adjoint() controls the source child's adjoint. Its visitor
        # still visits the block before the disconnected scalar. This differs
        # from the generic Adjoint(Controlled(Phase)) call decomposition.
        return (
            controlled_block_phase_program(child[0], source.phi),
            controlled_block_phase_program(child[1], -source.phi),
        )
    if isinstance(source, Unitary):
        child = _positive_pair(source.U, child_owner)
        data = _signature(source.U.signature)
        from qualtran import Controlled, CtrlSpec

        ports = _signature(Controlled(source, CtrlSpec()).signature)
        auto = controlled_auto_partition_pair(
            *child,
            (ports[1],),
            AutoPartition(
                (("system", tuple((r.name, r.total_qubits()) for r in data)),)
            ),
            data_registers=data,
        )
        result = []
        for record in auto:
            builder = ComponentProgramBuilder(Signature(ports))
            state = dict(builder.inputs)
            values = {
                r.name: state.pop(r.name) for r in record.functions[0].registers or ()
            }
            state.update(builder.call_program(record, values))
            result.append(builder.finalize(state))
        return result[0], result[1]
    for original, suffix in (
        (Hadamard(), "hadamard.CHadamard"),
        (XGate(), "cnot.CNOT"),
        (ZGate(), "z_basis.CZ"),
    ):
        if source == original:
            function = fixed_gate_factories()["qualtran.bloqs.basic_gates." + suffix](
                "main", None, {}
            )
            assert isinstance(function, FunctionRecord)
            record = ProgramRecord((function,))
            return record, record
    if source == TGate():
        pair = tuple(
            ProgramRecord(
                (
                    bit_gate_function(
                        "main",
                        tuple(Signature.build(ctrl=1, q=1)),
                        CircuitIR(
                            2,
                            (
                                Instruction(
                                    "cphase", (0, 1), params={"theta": sign * pi / 4}
                                ),
                            ),
                        ),
                    ),
                )
            )
            for sign in (1, -1)
        )
        return pair[0], pair[1]
    if child_owner is not None:
        return child_owner(source)
    raise SerializationError(
        "Default LCU SELECT requires an explicit native owner for this child"
    )


def default_linear_combination_select_pair(
    source: object, *, positive_child_owner: Callable[[object], ExactPair] | None = None
) -> ExactPair:
    """Own original default indexed SELECT and its positive controlled wrapper."""
    from qualtran import Controlled, CtrlSpec
    from qualtran.bloqs.multiplexers.apply_lth_bloq import ApplyLthBloq
    from qualtran.bloqs.multiplexers.black_box_select import BlackBoxSelect

    foreign = cast(Any, source)
    positive = isinstance(source, Controlled)
    if positive:
        if foreign.ctrl_spec != CtrlSpec():
            raise SerializationError(
                "Default packed SELECT requires a single positive control"
            )
        foreign = foreign.subbloq
    if not isinstance(foreign, BlackBoxSelect) or not isinstance(
        foreign.select, ApplyLthBloq
    ):
        raise SerializationError(
            "Default LCU SELECT requires an explicit packed indexed oracle"
        )
    raw = cast(Any, foreign.select)
    if raw.control_val is not None:
        raise SerializationError(
            "Already controlled indexed oracle requires an explicit owner"
        )
    leaves = tuple(
        _positive_pair(child, positive_child_owner) for child in raw.ops.flat
    )
    selectors, targets = _signature(raw.selection_registers), _signature(
        raw.target_registers
    )
    indexed = tuple(
        indexed_selection_component_program(
            selectors,
            targets,
            tuple(raw.ops.shape),
            tuple(p[0] for p in leaves),
            inverses=tuple(p[1] for p in leaves),
            control=Register("control", quantum_bit_type()) if positive else None,
            adjoint=d,
        )
        for d in (False, True)
    )
    if positive:
        auto = controlled_auto_partition_pair(
            indexed[0],
            indexed[1],
            _signature(foreign.signature),
            AutoPartition(
                (
                    ("selection", tuple((r.name, r.total_qubits()) for r in selectors)),
                    ("system", tuple((r.name, r.total_qubits()) for r in targets)),
                )
            ),
            data_registers=(*selectors, *targets),
        )
        result = []
        for record in auto:
            builder = ComponentProgramBuilder(
                Signature(_signature(cast(Any, source).signature))
            )
            result.append(
                builder.finalize(builder.call_program(record, builder.inputs))
            )
        return result[0], result[1]
    groups = (("selection", selectors), ("system", targets))
    return cast(
        ExactPair,
        tuple(
            packed_oracle_component_program(
                Signature(_signature(foreign.signature)),
                groups,
                indexed[0],
                inverse=indexed[1],
                adjoint=d,
            )
            for d in (False, True)
        ),
    )

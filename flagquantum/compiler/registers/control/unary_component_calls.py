# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Typed lifetime calls follow Qualtran v0.7.0 UnaryIterationGate decomposition.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/multiplexers/unary_iteration_bloq.py

"""Build bounded unary calls with graph-owned predicate lifetimes."""

from collections.abc import Callable

import numpy as np

from ....core.ir import CircuitIR, Instruction
from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.registers import Register
from ..component_builder import ComponentProgramBuilder
from ..component_gate_function import bit_gate_function
from ..lookup.unary_iteration import _walk_bounded_unary_tree, unary_iteration_workspace
from .conjunction_program import _and_storage_program, and_component_program


def bounded_unary_adjoint_pairs() -> tuple[tuple[ProgramRecord, ProgramRecord], ...]:
    """Own exact compute/erase and involutory primitive pairs for typed loops."""
    pairs: list[tuple[ProgramRecord, ProgramRecord]] = []
    for left in (0, 1):
        for right in (0, 1):
            compute = and_component_program((left, right), name="compute")
            erase = and_component_program((left, right), name="erase", uncompute=True)
            pairs.extend(((compute, erase), (erase, compute)))
    for gate, ports in (("x", ("q",)), ("cx", ("ctrl", "target"))):
        function = bit_gate_function(
            gate,
            tuple(Register(port, quantum_bit_type()) for port in ports),
            CircuitIR(len(ports), (Instruction(gate, tuple(range(len(ports)))),)),
        )
        primitive = ProgramRecord((function,))
        pairs.append((primitive, primitive))
    return tuple(pairs)


def append_bounded_unary_calls(
    builder: ComponentProgramBuilder,
    selection: tuple[RegisterBinding, ...],
    extent: int,
    emit_leaf: Callable[[int, RegisterBinding], RegisterBinding],
    *,
    control: RegisterBinding | None = None,
    lower: int = 0,
    break_early: Callable[[int, int], bool] | None = None,
    reserve_workspace: bool = False,
) -> tuple[tuple[RegisterBinding, ...], RegisterBinding | None]:
    """Append a forward typed traversal and return successor selector/control bits.

    The leaf returns the successor positive predicate and owns its other ports
    and private allocation/free calls. Predicates have directional And ports.
    And calls allocate predicate storage in the logical graph. The Program adjoint owner
    reverses these calls using exact caller-supplied And and leaf adjoints.
    All supplied bindings are one-bit quantum elements from the same builder.
    reserve_workspace retains the source generator's ordered scratch word for
    the whole loop; erased target slots stay owned until that loop returns.
    """
    if type(reserve_workspace) is not bool:
        raise ValueError("Unary workspace reservation must be boolean")
    if not selection:
        raise ValueError("Bounded unary calls require a nonempty selector")
    count = len(selection)
    control_id = count if control is not None else None
    bindings = dict(enumerate(selection))
    if control is not None:
        bindings[count] = control
    start = count + int(control is not None)
    work_ids = tuple(range(start, start + unary_iteration_workspace(count, control_id)))
    storage = (
        builder.allocate(len(work_ids)) if reserve_workspace and work_ids else None
    )
    if storage is not None:
        bindings.update(zip(work_ids, builder.split(storage), strict=True))

    def emit_gate(gate: Instruction) -> None:
        size = len(gate.wires)
        name = "__fq_unary_" + gate.name
        ports = ("q",) if size == 1 else ("ctrl", "target")
        function = bit_gate_function(
            name,
            tuple(Register(port, quantum_bit_type()) for port in ports),
            CircuitIR(size, (Instruction(gate.name, tuple(range(size))),)),
        )
        outputs = builder.call_program(
            ProgramRecord((function,)),
            dict(zip(ports, (bindings[wire] for wire in gate.wires), strict=True)),
            entry=name,
        )
        for wire, port in zip(gate.wires, ports, strict=True):
            value = outputs[port]
            assert isinstance(value, RegisterBinding)
            bindings[wire] = value

    def emit_and(
        left: int, right: int, target: int, values: tuple[int, int], erase: bool
    ) -> None:
        name = f"__fq_unary_and_{values[0]}{values[1]}" + ("_dg" if erase else "")
        program = (
            _and_storage_program(values, name=name + "_storage", uncompute=erase)
            if reserve_workspace
            else and_component_program(values, name=name, uncompute=erase)
        )
        entry = program.functions[0].name
        arguments: dict[str, object] = {
            "ctrl": np.array([bindings[left], bindings[right]], dtype=object)
        }
        if erase or reserve_workspace:
            arguments["target"] = bindings[target]
        outputs = builder.call_program(program, arguments, entry=entry)
        controls = np.asarray(outputs["ctrl"])
        bindings[left], bindings[right] = controls[0], controls[1]
        if erase and not reserve_workspace:
            del bindings[target]
        else:
            output = outputs["target"]
            assert isinstance(output, RegisterBinding)
            bindings[target] = output

    def visit(index: int, predicate: int) -> None:
        bindings[predicate] = emit_leaf(index, bindings[predicate])

    _walk_bounded_unary_tree(
        control_id,
        tuple(range(count)),
        extent,
        work_ids,
        visit,
        emit_gate,
        emit_and,
        lower=lower,
        break_early=break_early,
    )
    if storage is not None:
        # Return one ordered loop frame before releasing its scratch. These
        # existing storage views couple SSA successors without a quantum gate,
        # preventing physical scheduling from freeing erased slots before a
        # later nested leaf completes on the loop's selector/control outputs.
        frame_ids = (
            *range(count),
            *((control_id,) if control_id is not None else ()),
            *work_ids,
        )
        frame = builder.join([bindings[index] for index in frame_ids])
        bindings.update(zip(frame_ids, builder.split(frame), strict=True))
        restored = builder.join(
            [bindings[index] for index in work_ids], storage.register.dtype
        )
        builder.free(restored)
    return tuple(bindings[index] for index in range(count)), (
        bindings.get(control_id) if control_id is not None else None
    )

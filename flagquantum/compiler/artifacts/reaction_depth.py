# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix compute_reaction_depth and compute_num_touched_qubits.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/circuit/circuit_util.cc

"""Compute source reaction depth and touched-wire counts on reversible artifacts."""

from collections import defaultdict
from collections.abc import Iterable

from .reversible_phase import ReversiblePhaseOperation, ReversiblePhaseProgram


def reaction_depth(program: ReversiblePhaseProgram) -> int:
    """Propagate source X-frame, Z-frame and classical reaction depths."""
    return reaction_depth_for_operations(program.operations)


def reaction_depth_for_operations(
    operations: Iterable[ReversiblePhaseOperation],
) -> int:
    """Analyze physical records without requiring a bound executable program."""
    x: defaultdict[int, int] = defaultdict(int)
    z: defaultdict[int, int] = defaultdict(int)
    c: defaultdict[int, int] = defaultdict(int)
    scopes = [0]

    for op in operations:
        depth = scopes[-1]
        if op.condition_bit is not None:
            depth = max(depth, c[op.condition_bit])
        guarded = len(scopes) > 1 or op.condition_bit is not None
        opcode = op.opcode
        if opcode in {
            "neg",
            "info_tag",
            "debug_print_empty",
            "debug_print_q",
            "debug_print_c",
        }:
            continue
        if opcode == "push_condition":
            scopes.append(depth)
            continue
        if opcode == "pop_condition":
            if len(scopes) > 1:
                scopes.pop()
            continue
        if opcode in {"bit_store0", "bit_store1", "bit_invert"}:
            assert op.result_bit is not None
            bit = op.result_bit
            c[bit] = max(c[bit], depth) if guarded or opcode == "bit_invert" else 0
            continue
        target = op.qubits[-1]
        if opcode == "x":
            x[target] = max(x[target], depth)
        elif opcode == "z":
            z[target] = max(z[target], depth)
        elif opcode == "cx":
            control = op.qubits[0]
            x[target] = max(x[target], depth, x[control])
            z[control] = max(z[control], depth, z[target])
        elif opcode == "cz":
            control = op.qubits[0]
            z[target] = max(z[target], depth, x[control])
            z[control] = max(z[control], depth, x[target])
        elif opcode == "swap":
            left = op.qubits[0]
            if guarded:
                x[left] = x[target] = max(x[left], x[target], depth)
                z[left] = z[target] = max(z[left], z[target], depth)
            else:
                x[left], x[target] = x[target], x[left]
                z[left], z[target] = z[target], z[left]
        elif opcode == "r":
            if guarded:
                x[target] = max(x[target], depth)
            else:
                x[target] = z[target] = 0
        elif opcode == "hmr":
            assert op.result_bit is not None
            bit = op.result_bit
            if guarded:
                c[bit] = max(c[bit], depth, z[target] + 1)
                x[target] = max(x[target], depth)
                z[target] = max(z[target], depth)
            else:
                c[bit] = z[target] + 1
                x[target] = z[target] = 0
        elif opcode == "ccx":
            first, second, target = op.qubits
            x[target] = max(x[target], 1 + max(depth, x[first], x[second]))
            z[first] = max(z[first], 1 + max(depth, z[target], x[second]))
            z[second] = max(z[second], 1 + max(depth, z[target], x[first]))
        elif opcode == "ccz":
            first, second, target = op.qubits
            z[target] = max(z[target], 1 + max(depth, x[first], x[second]))
            z[second] = max(z[second], 1 + max(depth, x[first], x[target]))
            z[first] = max(z[first], 1 + max(depth, x[second], x[target]))
        elif opcode == "cswap":
            control, left, right = op.qubits
            dx = max(x[left], x[right])
            dz = max(z[left], z[right])
            x[left] = x[right] = 1 + max(dx, x[control], depth)
            z[left] = z[right] = 1 + max(dz, x[control], depth)
            z[control] = max(z[control], 1 + max(dx, dz, depth))
        elif opcode == "z_pow":
            assert op.angle is not None
            increment = (
                depth
                if op.angle.is_multiple_of_180_degrees()
                else 1 + max(depth, x[target])
            )
            z[target] = max(z[target], increment)
        else:
            raise ValueError(f"Unsupported reaction-depth operation {opcode!r}")

    return max(max(depths.values(), default=0) for depths in (x, z, c))


def touched_qubit_count(
    program: ReversiblePhaseProgram, *, debug_wires: Iterable[int] = ()
) -> int:
    """Count operation and debug wires, independent of declared registers."""
    wires = {wire for op in program.operations for wire in op.qubits}
    for wire in debug_wires:
        if type(wire) is not int or not 0 <= wire < program.n_qubits:
            raise ValueError("Debug wire is outside the reversible artifact")
        wires.add(wire)
    return len(wires)

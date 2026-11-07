# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix comparator generators and CircuitBuilder gate dispatch.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/gen/comparators/gen_cmp_qq.cc
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/gen/comparators/gen_cmp.cc
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/build/circuit_builder.cc

"""Fixed-wire quantum comparison in the locked Kickmix carry schedule."""

from math import ceil, inf, isnan
from typing import cast

from ...core.classical_dataflow import ClassicalXor
from ...core.ir import Instruction
from ._kickmix_comparison_gates import _flip, _parity_controlled_flip
from .finite_field import ClassicalFieldCoefficient
from .kickmix_low_space_comparison import kickmix_low_space_less_than_ir
from .measurement import measure_x_clear_ir
from .polynomial import _validate_registers


def kickmix_quantum_compare_ir(
    left: tuple[int, ...],
    right: tuple[int, ...],
    target: int | None,
    workspace: tuple[int, ...] = (),
    *,
    or_equal: int | bool | ClassicalFieldCoefficient = False,
    control: int | bool = True,
    measurement_bit: int | None = None,
    minimize_qubits: bool = False,
) -> tuple[Instruction, ...]:
    """XOR the controlled comparison predicate into a qubit or negate its phase.

    The equal-width words are big-endian. Workspace order matches the
    source's carry allocation and begins clean. The equality operand is a
    Boolean, quantum wire, or defined classical slot. A measured carry
    ladder reuses one caller-owned classical slot. A three-control
    operation without clean scratch borrows the lowest wire ID outside its
    controls and target. Include that wire in the circuit width, even when
    its ID is above every operand. The borrowed wire is restored.
    A None target selects the source's phase mode. Scalar negations use
    wire zero as an arbitrary-state phase carrier in native IR.
    """
    width = len(left)
    if len(right) != width:
        raise ValueError("Quantum comparison requires equal word widths")
    quantum_inputs = tuple(q for q in (or_equal, control) if type(q) is int)
    output = () if target is None else (target,)
    _validate_registers(left, right, output, workspace, quantum_inputs)
    if measurement_bit is not None and (
        type(measurement_bit) is not int
        or measurement_bit < 0
        or (
            isinstance(or_equal, ClassicalFieldCoefficient)
            and or_equal.slot == measurement_bit
        )
    ):
        raise ValueError("Comparison requires a separate nonnegative result slot")
    if type(minimize_qubits) is not bool:
        raise ValueError("Comparison minimize_qubits must be boolean")
    if control is False:
        return ()
    if width == 0:
        return _flip(target, control, or_equal)
    if width == 1:
        link = Instruction("cx", (right[0], left[0]))
        parity = _flip(right[0], or_equal)
        return (
            link,
            *parity,
            *_flip(
                target,
                right[0],
                control,
                left[0],
                scratch=workspace[0] if workspace else None,
                measurement_bit=measurement_bit,
            ),
            *parity,
            link,
            *_flip(target, control, or_equal),
        )

    gates: list[Instruction] = []
    if type(or_equal) is int:
        carry, clean = or_equal, workspace
        load_equal: tuple[Instruction, ...] = ()
    else:
        if not workspace:
            raise ValueError("Quantum comparison requires a clean carry wire")
        carry, *remaining = workspace
        clean = tuple(remaining)
        load_equal = _flip(carry, or_equal)
        gates.extend(load_equal)

    # Use the source's little-endian walk while preserving caller wire IDs.
    lhs, rhs = left[::-1], right[::-1]
    split = 0 if minimize_qubits else max(0, min(len(clean) - 1, width - 1))
    handoff = clean[split] if split else carry
    invert_rhs = tuple(Instruction("x", (wire,)) for wire in rhs)
    gates.extend(invert_rhs)
    initialize = (
        (Instruction("x", (clean[0],)), *_flip(clean[0], or_equal))
        if split
        else (Instruction("x", (carry,)),)
    )
    gates.extend(initialize)

    for i in range(split):
        gates.extend(
            (
                Instruction("cx", (clean[i], rhs[i])),
                Instruction("cx", (clean[i], lhs[i])),
                Instruction("ccx", (rhs[i], lhs[i], clean[i + 1])),
                Instruction("cx", (clean[i], clean[i + 1])),
            )
        )

    def carry_step(
        i: int, incoming: int, controls: tuple[int, int]
    ) -> tuple[Instruction, ...]:
        return (
            Instruction("cx", (rhs[i], lhs[i])),
            Instruction("cx", (rhs[i], incoming)),
            Instruction("ccx", (*controls, rhs[i])),
        )

    reversible: list[Instruction] = []
    if split < width - 1:
        reversible.extend(carry_step(split, handoff, (handoff, lhs[split])))
    for i in range(split + 1, width - 1):
        # Kickmix's loop emitter serializes its CCX controls in reverse order.
        reversible.extend(carry_step(i, rhs[i - 1], (lhs[i], rhs[i - 1])))
    gates.extend(reversible)

    hi_left, hi_right = lhs[-1], rhs[-1]
    if split == width - 1:
        high_carry = clean[width - 1]
        before = (
            Instruction("cx", (high_carry, hi_left)),
            Instruction("cx", (high_carry, hi_right)),
        )
        gates.extend(
            (
                *_flip(target, control, high_carry),
                *before,
                *_flip(target, control, hi_left, hi_right),
                *before,
            )
        )
    else:
        incoming = rhs[-2]
        before = (
            Instruction("cx", (hi_right, hi_left)),
            Instruction("cx", (hi_right, incoming)),
        )
        gates.extend(
            (
                *before,
                *_flip(target, control, hi_left, incoming),
                *_flip(target, control, hi_right),
                *before,
            )
        )
    gates.extend(reversed(reversible))

    if split:
        if measurement_bit is None:
            raise ValueError("Measured comparison requires a classical result slot")
        feedback = {"conditions": ((measurement_bit, 1),)}
        for i in reversed(range(split)):
            gates.extend(
                (
                    *measure_x_clear_ir(clean[i + 1], measurement_bit),
                    Instruction("z", (clean[i],), metadata=feedback),
                    Instruction("cz", (lhs[i], rhs[i]), metadata=feedback),
                    Instruction("cx", (clean[i], lhs[i])),
                    Instruction("cx", (clean[i], rhs[i])),
                )
            )
    gates.extend((*initialize, *invert_rhs, *_flip(target, control), *load_equal))
    return tuple(gates)


def kickmix_quantum_equal_ir(
    left: tuple[int, ...],
    right: tuple[int | bool | ClassicalFieldCoefficient, ...],
    target: int | None,
    workspace: tuple[int, ...] = (),
    *,
    control: int | bool = True,
    measurement_bit: int | None = None,
) -> tuple[Instruction, ...]:
    """Apply the locked equality predicate to a target bit or its phase.

    Words are big-endian. The right word may mix quantum wires, classical
    coefficients, and constants. At width two and above, the source uses
    one initially clean wire per input bit and reuses one result slot.
    """
    width = len(left)
    if len(right) != width:
        raise ValueError("Equality comparison requires equal word widths")
    output = () if target is None else (target,)
    quantum_control = (control,) if type(control) is int else ()
    quantum_right = tuple(dict.fromkeys(q for q in right if type(q) is int))
    _validate_registers(left, quantum_right, output, workspace, quantum_control)
    if control is False:
        return ()
    if width == 0:
        return _flip(target, control)

    lhs, rhs = left[::-1], right[::-1]
    difference = tuple(
        gate
        for wire, coefficient in zip(lhs, rhs, strict=True)
        for gate in _flip(wire, coefficient)
    )
    invert = tuple(Instruction("x", (wire,)) for wire in lhs)
    if width == 1:
        return (
            *difference,
            *invert,
            *_flip(target, control, lhs[0]),
            *invert,
            *difference,
        )
    if len(workspace) < width:
        raise ValueError("Equality comparison requires one clean wire per input bit")
    if (
        type(measurement_bit) is not int
        or measurement_bit < 0
        or any(
            isinstance(coefficient, ClassicalFieldCoefficient)
            and coefficient.slot == measurement_bit
            for coefficient in rhs
        )
    ):
        raise ValueError(
            "Measured equality requires a separate nonnegative result slot"
        )

    clean = workspace[:width]
    gates = [*difference, *invert, *_flip(clean[0], control, lhs[0])]
    for i in range(1, width):
        gates.append(Instruction("ccx", (clean[i - 1], lhs[i], clean[i])))
    gates.extend(_flip(target, clean[-1], lhs[-1]))

    outcome = ClassicalFieldCoefficient(measurement_bit)
    for i in reversed(range(1, width)):
        gates.extend(measure_x_clear_ir(clean[i], measurement_bit))
        gates.extend(_flip(None, clean[i - 1], lhs[i], outcome))
    gates.extend(measure_x_clear_ir(clean[0], measurement_bit))
    gates.extend(_flip(None, control, lhs[0], outcome))
    return (*gates, *invert, *difference)


def kickmix_less_than_ir(
    left: tuple[int, ...],
    right: tuple[int | bool | ClassicalFieldCoefficient, ...],
    target: int | None,
    workspace: tuple[int, ...] = (),
    *,
    or_equal: int | bool | ClassicalFieldCoefficient = False,
    control: int | bool = True,
    measurement_bit: int | None = None,
    minimize_qubits: bool = False,
    btol: float = inf,
    low_space_bits: tuple[int, ...] = (),
) -> tuple[Instruction, ...]:
    """Emit the locked comparison for equal-width mixed words.

    All words are big-endian. The full carry ladder requires width minus
    one clean wires and one reusable result slot. Homogeneous quantum words
    with fewer than width work wires use the dedicated quantum schedule.
    A None target selects a phase output.
    A bit tolerance below the input width compares only the most significant
    `ceil(btol) - 1` coefficients. A budget of one or less skips the
    comparison when below the word width. Infinity selects exact dispatch.
    The three-clean-wire classical schedule uses the caller's
    `low_space_bits` instead of the full ladder's single result slot.
    """
    width = len(left)
    if len(right) != width:
        raise ValueError("Comparison requires equal word widths")
    if isnan(btol):
        raise ValueError("Comparison bit tolerance must be ordered")
    quantum_rhs = all(type(coefficient) is int for coefficient in right)
    classical_slot_rhs = all(
        isinstance(coefficient, ClassicalFieldCoefficient) for coefficient in right
    )
    if btol < width:
        if btol <= 1:
            return ()
        width = ceil(btol) - 1
        left, right = left[:width], right[:width]
        btol = inf
        # The source's subspan retains the original RHS dispatch tag.
    quantum_right = tuple(q for q in right if type(q) is int)
    if width and quantum_rhs and btol == inf and len(workspace) < width:
        return kickmix_quantum_compare_ir(
            left,
            quantum_right,
            target,
            workspace,
            or_equal=or_equal,
            control=control,
            measurement_bit=measurement_bit,
            minimize_qubits=minimize_qubits,
        )
    output = () if target is None else (target,)
    quantum_controls = tuple(q for q in (or_equal, control) if type(q) is int)
    _validate_registers(
        left, tuple(dict.fromkeys(quantum_right)), output, workspace, quantum_controls
    )
    classical_inputs = {
        q.slot for q in (*right, or_equal) if isinstance(q, ClassicalFieldCoefficient)
    }
    if measurement_bit is not None and (
        type(measurement_bit) is not int
        or measurement_bit < 0
        or measurement_bit in classical_inputs
    ):
        raise ValueError("Comparison requires a separate nonnegative result slot")
    if type(minimize_qubits) is not bool:
        raise ValueError("Comparison minimize_qubits must be boolean")
    if control is False:
        return ()
    if width == 0:
        return _flip(target, control, or_equal)

    lhs, rhs = left[::-1], right[::-1]
    difference = tuple(
        gate
        for wire, coefficient in zip(lhs, rhs, strict=True)
        for gate in _flip(wire, coefficient)
    )
    invert = tuple(Instruction("x", (wire,)) for wire in lhs)
    if width == 1:
        scratch = workspace[0] if workspace and not minimize_qubits else None
        return (
            *difference,
            *_flip(
                target,
                control,
                lhs[0],
                rhs[0],
                scratch=scratch,
                measurement_bit=measurement_bit,
            ),
            *invert,
            *_flip(
                target,
                or_equal,
                control,
                lhs[0],
                scratch=scratch,
                measurement_bit=measurement_bit,
            ),
            *invert,
            *difference,
        )

    if not quantum_right and (minimize_qubits or len(workspace) < width - 1):
        return kickmix_low_space_less_than_ir(
            left,
            cast(tuple[bool | ClassicalFieldCoefficient, ...], right),
            target,
            workspace,
            measurement_bits=low_space_bits,
            or_equal=or_equal,
            control=control,
        )
    if len(workspace) < width - 1:
        raise ValueError("Comparison requires at least width minus one clean wires")
    if measurement_bit is None:
        raise ValueError("Measured comparison requires a classical result slot")

    clean = workspace[: width - 1]
    scratch = (
        workspace[width - 1]
        if len(workspace) >= width and not minimize_qubits
        else None
    )
    gates = [
        *invert,
        *difference,
        *_parity_controlled_flip(clean[0], or_equal, rhs[0], lhs[0]),
        *_flip(clean[0], rhs[0]),
        *_flip(clean[0], rhs[1]),
    ]
    loop_dispatch = quantum_rhs or classical_slot_rhs
    for i in range(1, width - 1):
        controls = (clean[i - 1], lhs[i]) if loop_dispatch else (lhs[i], clean[i - 1])
        gates.extend(
            (
                Instruction("ccx", (*controls, clean[i])),
                *_flip(clean[i], rhs[i]),
                *_flip(clean[i], rhs[i + 1]),
            )
        )
    for carry, coefficient in zip(clean, rhs[1:], strict=True):
        gates.extend(_flip(carry, coefficient))

    restore = _flip(clean[-1], rhs[-1])
    gates.extend(
        (
            *_flip(target, control, rhs[-1]),
            *restore,
            *_flip(
                target,
                control,
                lhs[-1],
                clean[-1],
                scratch=scratch,
                measurement_bit=measurement_bit,
            ),
            *restore,
        )
    )
    outcome = ClassicalFieldCoefficient(measurement_bit)
    for i in reversed(range(width - 1)):
        incoming = clean[i - 1] if i else or_equal
        gates.extend(
            (
                *measure_x_clear_ir(clean[i], measurement_bit),
                *_flip(None, rhs[i], outcome),
                *_flip(None, lhs[i], incoming, outcome),
                *_flip(None, lhs[i], rhs[i], outcome),
            )
        )
    return (*gates, *difference, *invert)


def kickmix_greater_than_ir(
    left: tuple[int, ...],
    right: tuple[int | bool | ClassicalFieldCoefficient, ...],
    target: int | None,
    workspace: tuple[int, ...] = (),
    *,
    or_equal: int | bool | ClassicalFieldCoefficient = False,
    control: int | bool = True,
    measurement_bit: int | None = None,
    minimize_qubits: bool = False,
    btol: float = inf,
    low_space_bits: tuple[int, ...] = (),
) -> tuple[Instruction, ...]:
    """Complement the locked less-than comparison and restore the equality input.

    The big-endian input, output and caller-owned workspace follow less-than.
    True equality selects greater-or-equal; true equality on less-than selects
    less-or-equal. A mutable equality input must be separate from the other
    operands and is restored after the output flip.
    """
    if control is False:
        return ()
    invert: tuple[Instruction, ...] = ()
    inverted_equal = or_equal
    if isinstance(or_equal, ClassicalFieldCoefficient):
        if (
            or_equal in right
            or or_equal.slot in low_space_bits
            or (measurement_bit == or_equal.slot)
        ):
            raise ValueError("Comparison equality must use a separate classical slot")
        slot = or_equal.slot
        invert = (ClassicalXor((slot,), slot, constant=True).instruction(),)
    elif isinstance(or_equal, bool):
        inverted_equal = not or_equal
    else:
        quantum_rhs = tuple(
            coefficient for coefficient in right if type(coefficient) is int
        )
        operands = (*left, *quantum_rhs, *workspace)
        if (
            or_equal in operands
            or (type(control) is int and control == or_equal)
            or target == or_equal
        ):
            raise ValueError("Comparison equality must use a separate quantum wire")
        invert = _flip(or_equal)
    compare = kickmix_less_than_ir(
        left,
        right,
        target,
        workspace,
        or_equal=inverted_equal,
        control=control,
        measurement_bit=measurement_bit,
        minimize_qubits=minimize_qubits,
        btol=btol,
        low_space_bits=low_space_bits,
    )
    return (*invert, *compare, *_flip(target, control), *invert)

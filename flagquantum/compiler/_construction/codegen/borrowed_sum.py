"""Recognize affine root-input sums eligible for a work-free XOR writer."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ....core.ir import Instruction
from ..._hybrid.model import INDEX, ValueId
from ...registers._fourier_kernels import borrowed_sum_xor
from ..analysis.lifetimes import LifetimeAnalysis
from ..verification.profile import _fail


@dataclass(frozen=True)
class _InputWord:
    bits: tuple[int, ...]
    shift: int = 0
    signed: bool = False


def borrowed_sum_instructions(
    analysis: LifetimeAnalysis,
    inputs: Mapping[ValueId, Sequence[int]],
    addends: tuple[ValueId, ValueId],
    output: Sequence[int],
    *,
    max_operations: int,
) -> tuple[Instruction, ...] | None:
    """Match only root words, full masks and lossless unsigned static shifts.

    Binding verification remains the caller's obligation. No temporary SSA
    allocation may be borrowed: it could be erased or remapped by recomputation.
    Signed extension is admitted only at the final modular result width.
    """
    definitions = {
        op.results[0].id: op for op in analysis.program.body.blocks[0].operations[:-1]
    }

    def word(key: ValueId, depth: int = 0) -> _InputWord | None:
        if key in inputs:
            return _InputWord(tuple(inputs[key]))
        if depth >= 64 or key not in definitions:
            return None
        op = definitions[key]
        if op.name not in {"qint.bitwise", "qint.multiply"}:
            return None
        for position in (0, 1):
            variable, constant = op.operands[position], op.operands[1 - position]
            if constant.type != INDEX:
                continue
            number = analysis.values[constant.id].upper_bound
            width = op.results[0].type.parameters[0]
            if op.name == "qint.bitwise":
                if (
                    op.attributes["operator"] != "&"
                    or number != (1 << width) - 1
                    or variable.id not in inputs
                ):
                    continue
                bits = tuple(inputs[variable.id])
                signed = op.attributes.get("signed", (False, False))[position]
                if width < len(bits) or (signed and width != len(output)):
                    continue
                return _InputWord(bits, signed=signed)
            if number < 1 or number & (number - 1):
                continue
            source = word(variable.id, depth + 1)
            if source is None or source.signed:
                continue
            shift = source.shift + number.bit_length() - 1
            if len(source.bits) + shift <= width:
                return _InputWord(source.bits, shift)
        return None

    words = word(addends[0]), word(addends[1])
    if any(value is None for value in words):
        return None
    for source, borrowed in (words, words[::-1]):
        assert source is not None and borrowed is not None
        # Only the unsigned word may be shifted; the other supplies one complete
        # root word and the final target supplies its single overflow position.
        if (
            source.signed
            or borrowed.shift
            or len(output) != len(borrowed.bits) + 1
            or source.shift >= len(borrowed.bits)
            or len(source.bits) + source.shift > len(borrowed.bits)
            or set(source.bits) & set(borrowed.bits)
        ):
            continue
        instructions: list[Instruction] = []
        for gate in borrowed_sum_xor(
            source.bits,
            borrowed.bits,
            tuple(output),
            shift=source.shift,
            signed=borrowed.signed,
        ):
            if len(instructions) >= max_operations:
                _fail("lowering.operations", "borrowed sum exceeds operation budget")
            instructions.append(gate)
        return tuple(instructions)
    return None

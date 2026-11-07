"""Immutable operations for a restricted reversible-phase target."""

from dataclasses import dataclass

from .phase_angle import PhaseAngle128

REVERSIBLE_OPCODE_ARITIES = {
    "x": 1,
    "cx": 2,
    "ccx": 3,
    "swap": 2,
    "cswap": 3,
    "z": 1,
    "z_pow": 1,
    "cz": 2,
    "ccz": 3,
    "neg": 0,
    "hmr": 1,
    "r": 1,
    "bit_store0": 0,
    "bit_invert": 0,
    "push_condition": 0,
    "pop_condition": 0,
}


@dataclass(frozen=True)
class ReversiblePhaseOperation:
    """A lowered opcode with ordered qubits and optional classical operands."""

    opcode: str
    qubits: tuple[int, ...]
    result_bit: int | None = None
    condition_bit: int | None = None
    angle: PhaseAngle128 | None = None


@dataclass(frozen=True)
class ReversiblePhaseProgram:
    """A validated operation stream using logical qubit order."""

    n_qubits: int
    n_bits: int
    operations: tuple[ReversiblePhaseOperation, ...]
    input_bits: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.n_qubits < 1 or self.n_bits < 0:
            raise ValueError("Invalid reversible-phase register widths")
        if any(
            type(bit) is not int or not 0 <= bit < self.n_bits
            for bit in self.input_bits
        ) or self.input_bits != tuple(sorted(set(self.input_bits))):
            raise ValueError("Invalid reversible-phase classical inputs")
        definitions: list[set[int]] = [set(self.input_bits)]

        def is_defined(bit: int | None) -> bool:
            return bit is not None and any(bit in scope for scope in definitions)

        for op in self.operations:
            if op.opcode == "z_pow":
                if not isinstance(op.angle, PhaseAngle128):
                    raise ValueError("Z_POW requires an exact PhaseAngle128")
            elif op.angle is not None:
                raise ValueError("Only Z_POW may carry an angle")
            if (
                REVERSIBLE_OPCODE_ARITIES.get(op.opcode) != len(op.qubits)
                or len(set(op.qubits)) != len(op.qubits)
                or any(not 0 <= wire < self.n_qubits for wire in op.qubits)
            ):
                raise ValueError(f"Invalid reversible-phase operation {op.opcode!r}")
            if op.opcode == "push_condition":
                if (
                    op.result_bit is not None
                    or op.condition_bit is None
                    or not is_defined(op.condition_bit)
                ):
                    raise ValueError("Condition scope requires a defined classical bit")
                definitions.append(set())
                continue
            if op.opcode == "pop_condition":
                if (
                    len(definitions) == 1
                    or op.result_bit is not None
                    or op.condition_bit is not None
                ):
                    raise ValueError("Invalid reversible-phase condition pop")
                definitions.pop()
                continue
            if op.condition_bit is not None and not is_defined(op.condition_bit):
                raise ValueError("Condition read before definition")
            if op.opcode in {"bit_store0", "bit_invert"}:
                if len(definitions) > 1:
                    raise ValueError(
                        "Condition scopes cannot write classical bits with XOR or store"
                    )
                if op.result_bit is None or not 0 <= op.result_bit < self.n_bits:
                    raise ValueError("Invalid classical result bit")
                if op.opcode == "bit_store0" and op.condition_bit is not None:
                    raise ValueError(
                        "Conditional classical initialization is unsupported"
                    )
                if op.opcode == "bit_invert" and not is_defined(op.result_bit):
                    raise ValueError("Classical inversion reads an undefined bit")
                definitions[-1].add(op.result_bit)
                continue
            if op.opcode == "hmr":
                if op.result_bit is None or not 0 <= op.result_bit < self.n_bits:
                    raise ValueError("Invalid HMR operands")
                if op.condition_bit is not None and not is_defined(op.result_bit):
                    raise ValueError("Inline guarded HMR requires a defined result")
                definitions[-1].add(op.result_bit)
            elif op.result_bit is not None:
                raise ValueError("Only HMR may write a classical bit")
        if len(definitions) > 1:
            raise ValueError("Unclosed reversible-phase condition scope")

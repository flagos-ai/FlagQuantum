"""Immutable operations for a restricted reversible-phase target."""

from dataclasses import dataclass

REVERSIBLE_OPCODE_ARITIES = {
    "x": 1,
    "cx": 2,
    "ccx": 3,
    "swap": 2,
    "cswap": 3,
    "z": 1,
    "cz": 2,
    "ccz": 3,
    "hmr": 1,
    "bit_store0": 0,
    "bit_invert": 0,
}


@dataclass(frozen=True)
class ReversiblePhaseOperation:
    """A lowered opcode with ordered qubits and optional classical operands."""

    opcode: str
    qubits: tuple[int, ...]
    result_bit: int | None = None
    condition_bit: int | None = None


@dataclass(frozen=True)
class ReversiblePhaseProgram:
    """A validated operation stream using logical qubit order."""

    n_qubits: int
    n_bits: int
    operations: tuple[ReversiblePhaseOperation, ...]

    def __post_init__(self) -> None:
        if self.n_qubits < 1 or self.n_bits < 0:
            raise ValueError("Invalid reversible-phase register widths")
        written: set[int] = set()
        for op in self.operations:
            if (
                REVERSIBLE_OPCODE_ARITIES.get(op.opcode) != len(op.qubits)
                or len(set(op.qubits)) != len(op.qubits)
                or any(not 0 <= wire < self.n_qubits for wire in op.qubits)
            ):
                raise ValueError(f"Invalid reversible-phase operation {op.opcode!r}")
            if op.opcode in {"bit_store0", "bit_invert"}:
                if op.result_bit is None or not 0 <= op.result_bit < self.n_bits:
                    raise ValueError("Invalid classical result bit")
                if op.opcode == "bit_store0" and op.condition_bit is not None:
                    raise ValueError(
                        "Conditional classical initialization is unsupported"
                    )
                if op.opcode == "bit_invert" and op.result_bit not in written:
                    raise ValueError("Classical inversion reads an undefined bit")
                if op.condition_bit is not None and op.condition_bit not in written:
                    raise ValueError("Condition read before definition")
                written.add(op.result_bit)
                continue
            if op.opcode == "hmr":
                if (
                    op.result_bit is None
                    or not 0 <= op.result_bit < self.n_bits
                    or op.condition_bit is not None
                ):
                    raise ValueError("Invalid HMR operands")
                written.add(op.result_bit)
            elif op.result_bit is not None:
                raise ValueError("Only HMR may write a classical bit")
            if op.condition_bit is not None and op.condition_bit not in written:
                raise ValueError("Condition read before measurement")

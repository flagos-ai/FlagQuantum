"""Private Compiler trial of exact result cleanup and physical storage reuse.

Run python -m examples.synthesis.tutorials.composition.register_cleanup.main. Ordinary users keep the
public Python predicate workflow; these owned records are contributor evidence.
"""

import flagquantum as fq
from flagquantum.compiler._construction.frontend_python import capture
from flagquantum.compiler._construction.register_profile import register_type
from flagquantum.compiler._hybrid.model import (
    QUANTUM_EFFECT,
    Block,
    HybridProgram,
    Operation,
    Region,
    Value,
    ValueId,
)
from flagquantum.services._synthesis_registers import synthesize_registers


def less(x: int, y: int) -> bool:
    return x < y


def equal(x: int, y: int) -> bool:
    return x == y


def copy_reference(x: int) -> int:
    return x


def flag_value(flag: int) -> int:
    return flag


def cleanup_composition(
    *, basis: tuple[int, int] | None = None, direct: bool = False
) -> tuple[HybridProgram, tuple[HybridProgram, ...]]:
    """Reuse one flag for two predicates, preserving an external reference."""
    functions = (
        capture(less, {"x": 2, "y": 2}),
        capture(equal, {"x": 2, "y": 2}),
        capture(copy_reference, {"x": 2}),
        capture(flag_value, {"flag": 1}),
    )
    scope = "cleanup_composition"
    entry = Value(ValueId(scope, 0), QUANTUM_EFFECT)
    registers = {
        name: Value(ValueId(scope, index), register_type(width))
        for index, (name, width) in enumerate(
            (("x", 2), ("y", 2), ("flag", 1), ("reference", 2)), 1
        )
    }
    operations = [
        Operation("qreg.declare", results=(value,), attributes={"name": name})
        for name, value in registers.items()
    ]
    effect = entry

    def emit(name: str, *actual: str, **attributes: object) -> None:
        nonlocal effect
        following = Value(ValueId(scope, len(operations) + 1), QUANTUM_EFFECT)
        operations.append(
            Operation(
                name,
                (effect, *(registers[key] for key in actual)),
                (following,),
                attributes,
            )
        )
        effect = following

    for index, name in enumerate(("x", "y")):
        if basis is None:
            emit("qreg.allocate", name)
            for bit in range(2):
                emit("qreg.gate", name, gate="h", index=bit)
        else:
            emit("qreg.initialize", name, value=basis[index])
    if basis is None:
        emit("qreg.gate", "x", gate="s", index=0)
    emit("qreg.compute", "reference", "x", callee="copy_reference")
    for predicate in ("less", "equal"):
        if direct:
            emit("qreg.phase", "x", "y", callee=predicate)
        else:
            emit("qreg.compute", "flag", "x", "y", callee=predicate)
            emit("qreg.phase", "flag", callee="flag_value")
            emit("qreg.uncompute", "flag")
    emit("qreg.allocate", "flag")
    operations.append(Operation("qreg.return", (effect, *registers.values())))
    return (
        HybridProgram(scope, Region((Block((entry,), tuple(operations)),))),
        functions,
    )


def main() -> None:
    model, functions = cleanup_composition()
    program = synthesize_registers(model, functions, memory_limit_bytes=64 * 1024**2)
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=tuple(program.registers))))
    print(program.report["resources"])
    print("Internal work qubits:", program.report["workspace_qubits"])


if __name__ == "__main__":
    main()

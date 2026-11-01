"""Contributor-only local storage and exact zero-slot reuse through Program/run.

Run python -m examples.synthesis_local_registers. Public Python and native source
syntax are unchanged; these owned records exercise the Compiler dependency.
"""

import flagquantum as fq
from examples.synthesis_register_cleanup import copy_reference, equal, flag_value, less
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


def local_marker(name: str, predicate: str) -> HybridProgram:
    """One verified Const helper with a temporary that cannot escape its return."""
    effect = Value(ValueId(name, 0), QUANTUM_EFFECT)
    x, y = (Value(ValueId(name, i), register_type(2)) for i in (1, 2))
    flag = Value(ValueId(name, 3), register_type(1))
    steps: tuple[tuple[str, tuple[Value, ...], dict[str, object]], ...] = (
        ("qreg.compute", (flag, x, y), {"callee": predicate}),
        ("qreg.phase", (flag,), {"callee": "flag_value"}),
        ("qreg.uncompute", (flag,), {}),
        ("qreg.release", (flag,), {}),
    )
    operations = [
        Operation("qreg.local", results=(flag,), attributes={"name": "temporary"})
    ]
    entry = effect
    for index, (opcode, operands, attributes) in enumerate(steps, 4):
        following = Value(ValueId(name, index), QUANTUM_EFFECT)
        operations.append(
            Operation(opcode, (effect, *operands), (following,), attributes)
        )
        effect = following
    operations.append(Operation("qreg.return", (effect, x, y)))
    return HybridProgram(name, Region((Block((entry, x, y), tuple(operations)),)))


def local_composition(
    *,
    basis: tuple[int, int] | None = None,
    overlapping: bool = False,
    direct: bool = False,
    helpers: bool = False,
) -> tuple[HybridProgram, tuple[HybridProgram, ...]]:
    """Keep only x/y/reference public while temporary condition storage is reused."""
    functions: tuple[HybridProgram, ...] = (
        capture(less, {"x": 2, "y": 2}),
        capture(equal, {"x": 2, "y": 2}),
        capture(copy_reference, {"x": 2}),
        capture(flag_value, {"flag": 1}),
    )
    scope = "local_composition"
    entry = Value(ValueId(scope, 0), QUANTUM_EFFECT)
    registers = {
        name: Value(ValueId(scope, index), register_type(2))
        for index, name in enumerate(("x", "y", "reference"), 1)
    }
    outputs = tuple(registers.values())
    operations = [
        Operation("qreg.declare", results=(value,), attributes={"name": name})
        for name, value in registers.items()
    ]
    effect = entry
    counter = 4

    def local(name: str) -> None:
        nonlocal counter
        value = Value(ValueId(scope, counter), register_type(1))
        counter += 1
        registers[name] = value
        operations.append(
            Operation("qreg.local", results=(value,), attributes={"name": name})
        )

    def emit(name: str, *actual: str, **attributes: object) -> None:
        nonlocal effect, counter
        following = Value(ValueId(scope, counter), QUANTUM_EFFECT)
        counter += 1
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
    if direct:
        for predicate in ("less", "equal"):
            emit("qreg.phase", "x", "y", callee=predicate)
    elif helpers:
        borrowed = tuple(local_marker("mark_" + p, p) for p in ("less", "equal"))
        functions = (*functions, *borrowed)
        for helper in borrowed:
            emit("qreg.call", "x", "y", callee=helper.name)
    else:
        for predicate in ("less", "equal"):
            local(predicate)
            emit("qreg.compute", predicate, "x", "y", callee=predicate)
            if not overlapping:
                emit("qreg.phase", predicate, callee="flag_value")
                emit("qreg.uncompute", predicate)
                emit("qreg.release", predicate)
        if overlapping:
            for predicate in ("less", "equal"):
                emit("qreg.phase", predicate, callee="flag_value")
                emit("qreg.uncompute", predicate)
                emit("qreg.release", predicate)
    operations.append(Operation("qreg.return", (effect, *outputs)))
    return (
        HybridProgram(scope, Region((Block((entry,), tuple(operations)),))),
        functions,
    )


def main() -> None:
    model, functions = local_composition(helpers=True)
    program = synthesize_registers(model, functions, memory_limit_bytes=64 * 1024**2)
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=tuple(program.registers))))
    print(program.report["resources"])
    print("All internal work qubits:", program.report["workspace_qubits"])


if __name__ == "__main__":
    main()

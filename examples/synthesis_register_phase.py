"""Private Compiler trial of ordered phase borrows and local workspace cleanup.

Run python -m examples.synthesis_register_phase. The owned IR producer below is
for contributors; the public high-level predicate entry remains unchanged.
"""

import flagquantum as fq
from flagquantum.compiler._construction.constraints import CircuitLimits
from flagquantum.compiler._construction.frontend_python import capture
from flagquantum.compiler._construction.register_profile import register_type
from flagquantum.compiler._construction.selection import Constraints, ConstructionBudget
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


def marked(x: int, y: int) -> bool:
    return x * x + y < 2


def product(x: int, y: int) -> int:
    return x * y


def identity(x: int) -> int:
    return x


def phase_composition() -> tuple[HybridProgram, tuple[HybridProgram, ...]]:
    """Build one owned-record fixture with a reference outside phase-call borrows."""
    functions = (
        capture(marked, {"x": 1, "y": 1}),
        capture(product, {"x": 1, "y": 1}),
        capture(identity, {"x": 1}),
    )
    scope = "phase_composition"
    entry = Value(ValueId(scope, 0), QUANTUM_EFFECT)
    names = ("x", "y", "product", "reference")
    registers = {
        name: Value(ValueId(scope, index), register_type(1))
        for index, name in enumerate(names, 1)
    }
    operations = [
        Operation("qreg.declare", results=(value,), attributes={"name": name})
        for name, value in registers.items()
    ]
    steps: tuple[tuple[str, tuple[str, ...], dict[str, object]], ...] = (
        ("qreg.allocate", ("x",), {}),
        ("qreg.gate", ("x",), {"gate": "h", "index": 0}),
        ("qreg.gate", ("x",), {"gate": "s", "index": 0}),
        ("qreg.allocate", ("y",), {}),
        ("qreg.gate", ("y",), {"gate": "h", "index": 0}),
        ("qreg.compute", ("reference", "x"), {"callee": "identity"}),
        ("qreg.phase", ("x", "y"), {"callee": "marked"}),
        ("qreg.gate", ("x",), {"gate": "x", "index": 0}),
        ("qreg.compute", ("product", "x", "y"), {"callee": "product"}),
        ("qreg.phase", ("product", "y"), {"callee": "marked"}),
        ("qreg.gate", ("y",), {"gate": "h", "index": 0}),
    )
    effect = entry
    for index, (name, actual, attributes) in enumerate(steps, len(registers) + 1):
        following = Value(ValueId(scope, index), QUANTUM_EFFECT)
        operations.append(
            Operation(
                name,
                (effect, *(registers[n] for n in actual)),
                (following,),
                attributes,
            )
        )
        effect = following
    operations.append(Operation("qreg.return", (effect, *registers.values())))
    return (
        HybridProgram(scope, Region((Block((entry,), tuple(operations)),))),
        functions,
    )


def main() -> None:
    model, functions = phase_composition()
    for objective in ("width", "depth"):
        program = synthesize_registers(
            model,
            functions,
            constraints=Constraints(CircuitLimits(max_qubits=24)),
            budget=ConstructionBudget(qubits=24),
            objective=objective,
            memory_limit_bytes=256 * 1024**2,
        )
        result = fq.run(program)
        print(
            f"{objective} probabilities:",
            dict(program.probabilities(result, registers=tuple(program.registers))),
        )
        print(f"{objective} resources:", program.report["resources"])
        print(f"{objective} workspace:", program.report["workspace_qubits"])


if __name__ == "__main__":
    main()

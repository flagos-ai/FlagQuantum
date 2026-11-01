"""Contributor-only owned-IR Const calls through the existing Program/run path.

Run python -m examples.synthesis_const_calls. Public native Const syntax is not
admitted by this dependency slice; ordinary Python synthesis remains unchanged.
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


def energy(x: int, y: int) -> int:
    return x * x + y


def marked(x: int, y: int) -> bool:
    return x < y


def const_composition() -> tuple[HybridProgram, tuple[HybridProgram, ...]]:
    """Two Const parameters, one fresh Output and a nested value/phase call."""
    arithmetic = (capture(energy, {"x": 2, "y": 2}), capture(marked, {"x": 2, "y": 2}))
    helpers: list[HybridProgram] = []
    for name in ("value_call", "phase_call", "combined_call"):
        entry = Value(ValueId(name, 0), QUANTUM_EFFECT)
        x = Value(ValueId(name, 1), register_type(2))
        y = Value(ValueId(name, 2), register_type(2))
        out = Value(ValueId(name, 3), register_type(4))
        first = Value(ValueId(name, 4), QUANTUM_EFFECT)
        second = Value(ValueId(name, 5), QUANTUM_EFFECT)
        declarations = (
            ()
            if name == "phase_call"
            else (
                Operation(
                    "qreg.declare", results=(out,), attributes={"name": "energy"}
                ),
            )
        )
        actions: tuple[Operation, ...]
        if name == "value_call":
            actions = (
                Operation(
                    "qreg.compute", (entry, out, x, y), (first,), {"callee": "energy"}
                ),
            )
        elif name == "phase_call":
            actions = (
                Operation("qreg.phase", (entry, x, y), (first,), {"callee": "marked"}),
            )
        else:
            actions = (
                Operation(
                    "qreg.call", (entry, x, y, out), (first,), {"callee": "value_call"}
                ),
                Operation(
                    "qreg.call", (first, x, y), (second,), {"callee": "phase_call"}
                ),
            )
        returned = (x, y) if name == "phase_call" else (x, y, out)
        helpers.append(
            HybridProgram(
                name,
                Region(
                    (
                        Block(
                            (entry, x, y),
                            (
                                *declarations,
                                *actions,
                                Operation(
                                    "qreg.return", (actions[-1].results[0], *returned)
                                ),
                            ),
                        ),
                    )
                ),
            )
        )
    scope = "const_root"
    entry = Value(ValueId(scope, 0), QUANTUM_EFFECT)
    registers = tuple(
        Value(ValueId(scope, i), register_type(w)) for i, w in enumerate((2, 2, 4), 1)
    )
    names = ("x", "y", "energy")
    operations = [
        Operation("qreg.declare", results=(v,), attributes={"name": name})
        for name, v in zip(names, registers, strict=True)
    ]
    x, y, out = registers
    steps: tuple[tuple[str, tuple[Value, ...], dict[str, object]], ...] = (
        ("qreg.allocate", (x,), {}),
        ("qreg.gate", (x,), {"gate": "h", "index": 0}),
        ("qreg.gate", (x,), {"gate": "h", "index": 1}),
        ("qreg.gate", (x,), {"gate": "s", "index": 0}),
        ("qreg.allocate", (y,), {}),
        ("qreg.gate", (y,), {"gate": "h", "index": 0}),
        ("qreg.gate", (y,), {"gate": "h", "index": 1}),
        ("qreg.call", (x, y, out), {"callee": "combined_call"}),
        ("qreg.gate", (x,), {"gate": "x", "index": 0}),
    )
    effect = entry
    for index, (opcode, args, attrs) in enumerate(steps, 4):
        following = Value(ValueId(scope, index), QUANTUM_EFFECT)
        operations.append(Operation(opcode, (effect, *args), (following,), attrs))
        effect = following
    operations.append(Operation("qreg.return", (effect, *registers)))
    return HybridProgram(scope, Region((Block((entry,), tuple(operations)),))), (
        *arithmetic,
        *helpers,
    )


def main() -> None:
    model, functions = const_composition()
    for objective in ("width", "depth"):
        program = synthesize_registers(
            model, functions, objective=objective, memory_limit_bytes=256 * 1024**2
        )
        result = fq.run(program)
        print(
            objective,
            dict(program.probabilities(result, registers=("x", "y", "energy"))),
        )
        print(
            program.report["resources"], "workspace", program.report["workspace_qubits"]
        )


if __name__ == "__main__":
    main()

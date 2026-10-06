"""Optimize a circuit without changing its numerical result.

This is the Compiler domain's ten-minute golden path. It answers, in order: what
an optimization level is, that the default is level 2, where level 2 earns its
place over level 1, that `compile` forwards the same level through routing, and
that a level this release has not implemented is refused rather than silently
downgraded. The level vocabulary is read from the library rather than restated
here, so a reader who adds a level changes one file.

Run it with::

    python -m examples.compiler_optimize
"""

from __future__ import annotations

import torch

import flagquantum as fq
from flagquantum.compiler import CouplingMap, optimize
from flagquantum.compiler import compile as compile_program
from flagquantum.compiler.optimization_levels import (
    DEFAULT_OPTIMIZATION_LEVEL,
    IMPLEMENTED_OPTIMIZATION_LEVELS,
    RESERVED_OPTIMIZATION_LEVELS,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.errors import CompilationError

#: The CPU statevector path. `complex128` with no fallback makes the comparison
#: below exact for every opcode this example emits.
OPTIONS = fq.ExecutionOptions(
    mode="statevector",
    backend="pytorch",
    device="cpu",
    precision="complex128",
    allow_backend_fallback=False,
)

#: An optimization level changes which gates are emitted, never the state the
#: program prepares. This is the tolerance the Compiler's own tests use.
ATOL = 1.0e-12


def _statevector(program: CircuitIR) -> torch.Tensor:
    return fq.run(program, options=OPTIONS).state


def main() -> None:
    circuit = (
        fq.Circuit(n_qubits=2, dtype=torch.complex128)
        .x(0)
        .x(0)
        .rx(1, theta=0.1)
        .h(0)
        .rz(0, theta=0.4)
        .h(0)
        .rx(1, theta=0.2)
        .cx(0, 1)
    )

    original = circuit.to_ir()
    reference = circuit.state()

    # Every implemented level has to reproduce the state the source prepared.
    # That is the whole claim of an optimizer, so it is checked at each level
    # rather than only at the default one.
    counts: dict[int, int] = {}
    for level in IMPLEMENTED_OPTIMIZATION_LEVELS:
        candidate = optimize(original, optimization_level=level)
        torch.testing.assert_close(
            _statevector(candidate), reference, atol=ATOL, rtol=0
        )
        assert candidate.metadata["optimization"] == {"level": level}
        counts[level] = len(candidate.instructions)

    # Level 0 is the request to leave the program as submitted, so its opcode
    # stream is the source's. Levels 1 and 2 both reach three gates here: what is
    # left to do is cancellation and the `h rz h` fold, which are level-1 passes.
    assert counts == {0: 8, 1: 3, 2: 3}
    level_zero = optimize(original, optimization_level=0)
    assert [item.name for item in level_zero.instructions] == [
        item.name for item in original.instructions
    ]

    # `optimize` with no level runs the default, and the default is level 2.
    optimized = optimize(original)
    assert DEFAULT_OPTIMIZATION_LEVEL == 2
    assert optimized == optimize(original, optimization_level=2)

    # `x x` cancels, the two `rx` on qubit 1 add up, and the `h rz h` that mixes
    # two opcodes on qubit 0 folds into one `u3`. The last of those is what makes
    # this check a guard on the fold: `u3` reproduces the run only if the emitted
    # angles carry the phase the run had, and the statevector comparison above
    # would see a difference if they did not.
    assert optimize(optimized) == optimized
    assert len(original.instructions) == 8
    assert len(optimized.instructions) == 3
    assert [item.name for item in optimized.instructions] == ["rx", "u3", "cx"]

    # Level 2 adds the passes that commute a gate past another, and those are the
    # ones level 1 is not allowed to use. `cz` is diagonal, so the two `rz` on
    # qubit 0 commute past it and add up: a saving only a level-2 pass can see.
    commuting = (
        fq.Circuit(n_qubits=2, dtype=torch.complex128)
        .rz(0, theta=0.3)
        .cz(0, 1)
        .rz(0, theta=0.4)
    )
    commuting_ir = commuting.to_ir()
    commuting_reference = commuting.state()
    level_one = optimize(commuting_ir, optimization_level=1)
    level_two = optimize(commuting_ir, optimization_level=2)
    assert len(level_one.instructions) == 3
    assert len(level_two.instructions) == 2
    for candidate in (level_one, level_two):
        torch.testing.assert_close(
            _statevector(candidate), commuting_reference, atol=ATOL, rtol=0
        )

    # `compile` takes the same level and forwards it to both points it optimizes:
    # before routing and again after, because routing inserts the SWAPs the second
    # pass cancels. This device hosts the circuit's `cz` edge, so no SWAP is
    # inserted and the counts agree with `optimize`.
    routed = compile_program(
        commuting_ir,
        coupling_map=CouplingMap.line(2),
        optimization_level=2,
    )
    assert len(routed.instructions) == len(level_two.instructions)
    assert routed.metadata["optimization"] == {"level": 2}
    torch.testing.assert_close(
        _statevector(routed), commuting_reference, atol=ATOL, rtol=0
    )

    # Level 3 is declared and reserved rather than an alias for level 2, so asking
    # for it fails closed and says why. A refused request must never be handed a
    # weaker program silently.
    refusal: str | None = None
    assert RESERVED_OPTIMIZATION_LEVELS
    try:
        optimize(original, optimization_level=RESERVED_OPTIMIZATION_LEVELS[0])
    except CompilationError as error:
        refusal = str(error)
    assert refusal is not None, "a reserved level must be refused, not run"

    result = fq.run(optimized, options=OPTIONS)

    print("FlagQuantum compiler optimization check passed")
    print(f"  original: {[item.name for item in original.instructions]}")
    print(f"  optimized: {[item.name for item in optimized.instructions]}")
    print(
        f"  instructions: {len(original.instructions)} -> {len(optimized.instructions)}"
    )
    print(
        "  gate count by level: "
        + ", ".join(f"{level} -> {counts[level]}" for level in sorted(counts))
    )
    print(
        "  level 2 commutes past the cz where level 1 cannot: "
        f"{len(level_one.instructions)} -> {len(level_two.instructions)} gates"
    )
    print(f"  level {RESERVED_OPTIMIZATION_LEVELS[0]}: refused ({refusal})")
    print(f"  execution path: {result.runtime['execution_path']}")


if __name__ == "__main__":
    main()

# Comparison gate cost tutorial audit

Q-0222 covers the complete 60-cell tutorial at Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389`. The source Git blob is
`9ff6e0d10cc925f687fee6e2edbe0fe26a6b4151`. The original author and DOI
10.5281/zenodo.8384491 are retained in the native asset. The original and
native versions both contain 36 executable cells.

## Ownership and source disposition

Compiler owns comparison, predicate reduction and measured And erasure.
The teaching module binds those existing components to concrete wires
and retains the tutorial's illustrative circuits at their original
abstraction levels. Operations use ordinary `circuit.append(op)`.
Inputs are big-endian. Predicate bits are XOR outputs. Scratch starts
and ends in zero. Three-way outputs are ordered less, equal, greater.
The example uses trajectory simulation for checks without extracting
the outputs by a separate classical simulation path.

| Original section and helper | Native disposition |
|---|---|
| Introduction and equality oracles | Preserve the arithmetic/depth discussion and the two MultiAnd-based examples. Use the existing native equality emitters, each followed by all basis checks and the uniform check. |
| Comparator definitions and one-bit construction | The notebook's illustrative coherent sequence retains its two exact CCX leaves and compute/copy/reverse order. The original helper reads the global equality output `z` even though it accepts `lz`. Native code explicitly binds that output. The literal gate test supplies the original notebook binding. |
| Two-bit mixer and comparison | Retain the illustrative XOR control bit and exact CSWAP leaves, then reuse the measured SingleQubitCompare output and adjoint. The example's control wire is distinct from its three XOR outputs. |
| Quantum–quantum comparison tree | Retain the floor split, post-order BiQubitsMixer calls, final single-bit comparison and reversed cleanup order, with distinct work and feedback. The original optional 64-input basis cell is retained, and tests cover every basis input coherently. |
| Quantum–classical mixer and two-bit comparison | Keep constant low/high bit handling and the original three work flags. All four constants are compared to source gates, including both parity cases. Preserve the original asymptotic argument alongside the actual two-bit 8 T example. |
| Linear prefix comparison, constant improvement and citation | Preserve the finite-state derivation, three-bit comparison to 5 and the original unoptimized prefix ladder. The zero-threshold-bit step preserves the source operand order and negative AND polarity. The unused first conjunction improvement remains a separate discussion. |

The exact gate checks execute function definitions from the frozen notebook
without rewriting them. The source gate operations are expanded with named
scratch bound to native indices. Native comparisons check primitive
multiplicities, per-wire dependency order, operand order, measurement slots
and conditional correction. Complete small-branch tests exercise 67
histories across the examples, including all 32 histories for the
three-bit tree. Each history checks the unnormalized full complex vector
over every input and arbitrary XOR target, including all work wires.

## Separate resource accounts

| Original example | Legacy source T estimate | Emitted T/TDG | Other exact leaves | Measurements | Clean work |
|---|---:|---:|---|---:|---:|
| Three-bit equality with 5 | 8 | 8 | — | 2 | 2 |
| Two-bit quantum equality | 4 | 4 | — | 1 | 1 |
| Illustrative one-bit comparison | — | 0 | 2 CCX | 0 | 2 |
| Illustrative two-bit quantum comparison | — | 4 | 4 CSWAP | 1 | 3 |
| Three-bit quantum tree | 20 | 20 | — | 5 | 8 |
| Two-bit mixed comparison with 2 | 8 | 8 | — | 2 | 5 |
| Three-bit linear comparison with 5 | 12 | 12 | — | 3 | 4 |

Legacy T estimates come from the original notebook's measured-And cost
model. Retained coherent CCX and CSWAP would incur additional cost under a
chosen physical lowering. Their zero explicit T count is a statement
about the emitted IR opcode set. Asymptotic T and depth arguments are
separate from this finite comparison.

The full original notebook executes unmodified in an independent kernel.
The native kernel runs all 36 code cells while blocking reference imports.
The native asset reports circuits as IR instructions and validates
basis and uniform outputs in place of printing foreign Dirac notation.
Q-0222 closes as a complete teaching asset.

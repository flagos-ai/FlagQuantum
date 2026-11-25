# Stored programmed phase constructor audit

## Scope and ownership

The complete frozen `rotations/programmable_ancilla_rotation.py` file declares
`ZPowProgrammedAncilla` and `ZPowUsingProgrammedAncilla`, including their original
examples and failure-probability constructor. The reference remains Qualtran
v0.7.0 commit `bcb17bb25360ed68ce3bc084645db3e929799389` without modifications.
Classification is `single_device_fast_path`.

The format adapter owns the constructor metadata and typed declarations. The
shared phase component owns Allocate/H/ZPow graph construction. Independent
resource-state and adaptive model owners retain precision, round counts and
worst-case synthesis estimates. Numerical execution remains outside Compiler.

The resource state's single QBit is a RIGHT output. Its allocation starts clean
and transfers ownership to the caller. It is the logical prepared output, so it
remains live. There is no private workspace or classical slot. Physical binding
requires the existing explicit workspace-output ownership transfer. The adaptive
rotation has one caller-owned THRU QBit.

## Complete source and direct dependency disposition

The resource state preserves all three original calls and their operands. Finite
and nonlinear symbolic exponents reuse existing exact Z-power and parameter
expression owners. Epsilon stays in the original declaration and independent
model. The literal pre-synthesis output is exact.

The adaptive source supplies a worst-case call graph and no executable gate
body. Its native constructor retains a typed declaration. Physical lowering of
that declaration refuses the missing body. The existing separately certified
native adaptive channel retains early-success behavior, padded idle
measurements, all-failure behavior and optional final correction. It does not
become a claimed upstream primitive decomposition.

Concrete and symbolic round models reuse `ProgrammedRotationModel`, including
failure-probability sizing and precision splitting. The original model remains
separate from actual predicated primitive counts and actual outcome probabilities.

## Boundaries and verification

Source serialization succeeds at depth zero for both constructors. At depth one,
the adaptive source reaches its existing CBit serialization failure. The tests
preserve that exception. Source allocation and RIGHT output lifetime are covered
at both depths zero and one for the prepared state.

Prepared-state tests compare exact root connections, full complex amplitudes
with arbitrary complex spectators, emitted H/phase operands, symbolic nonlinear
binding, IR round trips and ordinary `circuit.append(op)`. The sole preparation
measurement history is empty. Existing adaptive tests cover all small histories,
including the exact branch-dependent complex phase, restored work and distinct
failure channel without correction. Unknown source bodies are not synthesized
from aggregate counts.

Arbitrary real non-integer round metadata and nonfinite numeric values remain
outside the typed constructor admission. Negative integer rounds may retain an
original declaration, but the existing native adaptive numerical/model owners
reject negative rounds. Large adaptive history enumeration, whole-repository,
GPU and performance validation remain outside this batch.

## Acceptance receipt, 2026-11-24

The concentrated regression passes 204 cases, including every original example
constructor. Affected local integration passes 145 cases. Unchanged original
file tests pass eight cases and skip five source-owned checks. Three frozen blobs,
Qualtran 0.7.0 and actual Kickmix 0.1 Simulator are verified. Strict mypy checks
all three new Python files. Black, Ruff, Compiler scope, language, architecture,
multi-level IR and API checks pass. Dependency analysis retains exactly the same
720 existing findings.

| Path | Actual native primitive inventory | Independent source model |
| --- | --- | --- |
| Prepared resource state | One H and one phase, one live logical output, no private work or measurement | One Clifford and one precision-sensitive rotation |
| Native adaptive channel, two rounds, no correction | Two each of H, phase, CX, measure and conditional X, one restored work QBit, two outcome slots | Worst-case six Cliffords, two precision-sensitive rotations and two measurements |

The second row reuses the previously certified adaptive implementation. The
source class itself has no primitive stream. Failed histories without final
correction retain their original different net rotation and branch-dependent
complex phase. Symbolic round counts remain declared models until concretized.

## Composed symbolic call identities, 2026-11-26

The complete file and all five examples were rechecked against the existing
owners. One source-success gap remained in resource graph composition: the
native symbolic-round declaration always used the literal placeholder
`_\phi0`. The original obtains a fresh angle from the containing graph's
`SympySymbolAllocator`. Reusing the native declaration in that graph could
therefore incorrectly combine distinct resource-state nodes.

`ProgrammedRotationModel.direct_calls` and `terminal_calls` now accept the
existing shared `SymbolAllocator`. The same allocator reaches the symbolic
ancilla, including when earlier graph nodes have already allocated angles.
An independent query still starts with the original deterministic first name.
Tests compare three successive direct declarations and two terminal expansions
with the unmodified source allocator, with and without final correction.
Precision division, multiplicities and correction angles remain unchanged.

No emitted instruction, physical binding, public Core behavior or source file
changes. Existing full complex branch amplitudes, zero work, preparation call
wiring, all five constructors and source resource formulas are reused in the
affected regression. The allocator fix itself does not provide conditional
measurement. The related physical early-exit consumer now reuses the already
existing internal guarded effect and local executors; see
`PROGRAMMED_PHASE_FILE_AUDIT.md`. The source adaptive constructor remains a
declaration without an invented upstream body.

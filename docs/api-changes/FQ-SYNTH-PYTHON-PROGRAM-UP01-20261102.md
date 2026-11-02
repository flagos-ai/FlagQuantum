# UP-01: Python qfunc preparation, arithmetic and composition

Date: 2026-11-02. Scope: unreleased `flagquantum.synth` domain extension,
`single_device_fast_path`, exact CPU complex128. The user explicitly requested
checking/merging #1003, then implementing UP-01 and submitting a tested PR.
Prerequisite #1003 merged as ef95c04f08ddeb05c1c8469b51f4cfccb7c4fc62 after
57 successful checks, three configured skips and CLEAN status.

This additive candidate follows the
[UP-00 semantic contract](FQ-SYNTH-UNIFIED-PROGRAM-UP00-20261102.md).
It is not a Stable Core freeze, trial approval, release or hardware promotion.
Existing run/phase/Qmod candidate contracts and proposals remain byte-identical.
A separate successor contract records the six authorized namespace additions;
the checker protects that exact delta instead of regenerating old snapshots.

## User task and public delta

The same Python qfunc prepares input registers and invokes arithmetic helpers.
No separate InputState, physical lines, gate design, scratch allocation or
bitstring decoding is required. Existing pure integer functions remain supported.

Add only `Output`, `Const`, `QNum`, `QBit`, `allocate`, `hadamard_transform` to
the existing seven `flagquantum.synth` exports, through the same Services facade.
Star import exposes exactly those thirteen names. No root export, synthesize
runtime parameter/default/mode, fq.run or Program.probabilities signature changes.
One additive typing overload admits a captured qfunc root without scalar inputs;
plain Python functions still require their explicit scalar input declarations.

- QNum denotes unsigned integer computational-basis encoding. `QNum[width]` and
  `QNum[width, False, 0]` accept literal widths 1..256; signed/fractional forms
  fail. Generic `QNum` infers widths through preparation, arithmetic and calls.
- QBit has fixed width one. It is not interchangeable with a numeric qnum port
  in a typed helper signature merely because both occupy one qubit.
- Output/Const are annotation roles, not allocated wrappers. Output establishes
  fresh initialized ownership; Const borrows an initialized basis value, with
  the existing allowance for phase/entanglement.
- `allocate(out)` requires known width. `allocate(size, out)` accepts one plain
  literal width 1..256, consistent with an explicit declared width. This is
  the familiar Qmod allocation overload, not another state-preparation language.
- `hadamard_transform(out)` requires initialized storage and applies H on each
  bit. It is a preparation operation in the program, not an external mode.
- Fresh `out |= expression` admits declared quantum names, nonnegative integer
  constants, +, *, and literal powers 1..8 through existing scalar capture.
  Literal basis initialization `out |= 3` is also allowed. This is fresh
  assignment, not in-place XOR or arbitrary replacement of a live state.
- Static typed qfunc calls accept positional quantum-port actuals; infer generic
  widths per call specialization. Verify exact kinds/widths, initialized Const,
  fresh/disjoint Outputs, and acyclic call graphs. Root ports are Output only.
- Function body/annotation/width/initializer calls are never evaluated during
  synthesis. Retain AST, resolve imported symbols/closures by identity, and
  snapshot the reachable typed call graph at synthesis. No general registry.

Numeric literal type parameters are an AST-captured annotation spelling; strict
static checkers do not generally accept value arguments as Python type arguments.
The executable, strictly checked example uses `Output[QNum]` with `allocate(2,x)`;
it expresses the same familiar type/size information without type-check ignores.
Fixed-width spelling is covered by source-retaining execution tests.

## Runnable target after implementation

```python
import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import Const, Output, QNum, allocate, hadamard_transform


@synth.qfunc
def energy(x: Const[QNum], y: Const[QNum], result: Output[QNum]) -> None:
    result |= x*x+y


@synth.qfunc
def main(x: Output[QNum], y: Output[QNum], result: Output[QNum]) -> None:
    allocate(2, x)
    allocate(2, y)
    hadamard_transform(x)
    hadamard_transform(y)
    energy(x, y, result)


program = synth.synthesize(main, max_qubits=24, memory_limit_bytes=2**30)
result = fq.run(program)
print(dict(program.probabilities(result, registers=("x", "y", "result"))))
```

Each of the sixteen `(x,y,x*x+y)` outcomes has probability 1/16. Result width
infers to four bits from the full domain 0..12. The root's Output names and order
are preserved; default reading remains the existing `result` marginal only.
The fixed-width variant uses `Output[QNum[2, False, 0]]` and `allocate(x)`.
Use the example only after owning-domain implementation; this contract commit
alone does not make it executable.

## Compatibility, ownership and implementation

Compiler adds typed source capture and annotation operations, reuses the shared
Python source reader, and emits existing qreg allocation/gate/compute/call/return
with existing quantum-effect tokens. Arithmetic expressions reuse scalar capture,
ranges and kernels. The authoritative register verifier checks initialization,
roles, aliases and cleanup before existing finite selection/target lowering.
No Python-to-Qmod string translation, second semantic IR, numerical kernel or
executor is introduced. Program roots are preparation isometries, not arbitrary
invertible/controllable unitaries.

Services routes the typed root once to existing register selection and checked
completion; the private source label may be python. Existing scalar/phase paths,
Qmod imports and run/read identity remain unchanged. Typed roots reject scalar
inputs/input_state/constants/functions/preparation and the Python predicate
output_mode=phase instead of preparing twice or inventing an interpretation.
Default planned-state memory for a typed root is 512 MiB, with explicit overrides
validated before construction. No process-peak-memory promise.

Reachable typed definitions are the Python program unit; unrelated module
functions are not implicitly imported into that graph. All captured definitions
and specializations are verified. Limits: 32 specializations/call depth, 4096
source/effect/model nodes under existing checks and 256-bit type/storage guards;
existing construction/candidate/target/execution budgets remain authoritative.
Semantic identities freeze resolved definitions and widths, not object addresses.

Current Python program scope excludes Input/consume/free/drop, mutable borrowing,
arrays/views/structs, local quantum declarations, explicit phase/control/inverse/
within-apply, measurement/reset, host callbacks, loops/branches, arbitrary numeric
expressions, runtime classical parameters and serialization. Existing broader
Qmod subset and pure-function capabilities are preserved; this addition does not
claim parity with all their syntax or any of the 216 original files.

## Acceptance gates

- Actual native/Qmod/Circuit comparisons: in-program preparation and composition,
  all 16 basis inputs, full complex coherent state and clean workspace. Retain
  existing nonzero-output XOR kernel tests; fresh Output never admits overwriting.
- Generic/fixed types, inferred output widths, nested and repeated helper calls,
  Const aliasing, source imports/aliases/closures, deterministic source identities
  and namespace rebinding before/after synthesis have explicit tests.
- Reject uninitialized use, duplicate preparation/fresh assignment, Const writes,
  Output overlap, insufficient width, recursive/unsupported host calls, forged
  symbols, unsupported types/effects, source/budget excess and duplicate state
  configuration. No function body or attribute property is called as fallback.
- One Program -> fq.run -> named probabilities, including final resource/memory
  budgets and an admitted routed named-read case. No required execution unwrap.
- Protect exact new namespace and signatures with successor-contract checks and
  deliberate tamper tests. Previous contracts, signatures, integer predicates and
  source/license pins remain intact. No assertion or skip weakened for green CI.

Official allocation overload reviewed on 2026-11-02:
[Classiq quantum variables](https://docs.classiq.io/qmod-reference/language-reference/quantum-variables).
Keep UP-00's strict free/drop and role boundaries; this reference is not SDK or
full-language certification. Trials remain 0/2 and release_authorized=false.

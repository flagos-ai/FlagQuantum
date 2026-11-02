# UP-01: Python quantum programs with preparation and composition

Follows #1003 merged as ef95c04f08ddeb05c1c8469b51f4cfccb7c4fc62 after
57 successful checks and three configured skips. The
[UP-01 contract](../api-changes/FQ-SYNTH-PYTHON-PROGRAM-UP01-20261102.md)
fixes this additive unreleased profile. It does not implement the later phase,
inverse or coherent-control stages.

## User workflow

Run `python -m examples.synthesis_python_program`, or save this in a `.py` file:

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

Each of the sixteen `(x,y,x*x+y)` outcomes has probability approximately 1/16.
Preparation is authored in the same program as calculation and function calls;
it is no longer required as external InputState configuration. The example also
factors preparation into its own qfunc to demonstrate nested function composition.
The result's full-domain width is four bits (0..12), even for a fixed basis input.
No gate sequence, physical line, workspace allocation or bitstring decoding is
required in this user path. Existing pure-function/InputState usage still works.

The fixed-width variant is `Output[QNum[2, False, 0]]` with `allocate(x)`.
`QNum[2]` means the same unsigned integer encoding. Generic QNum plus explicit
allocation size is also the spelling ordinary strict type checkers accept; no
type-check suppression is needed in the executable example. Star import exposes
exactly the approved thirteen public names; it does not expose Compiler internals.

## Equivalent Qmod

```qmod
qfunc energy(const x:qnum<2>,const y:qnum<2>,output result:qnum) {
    result=x*x+y;
}
qfunc main(output x:qnum<2>,output y:qnum<2>,output result:qnum) {
    allocate(x);allocate(y);
    hadamard_transform(x);hadamard_transform(y);
    energy(x,y,result);
}
```

Pass this to the existing `synth.synthesize(source, source_format="qmod")`, then
use the same Program/run/named-read consumer. It remains an independent frontend;
Python is not translated into Qmod text. No original source file is rewritten.

## Independent Circuit reference

The runnable example supplies a hand-derived two-bit circuit, using one clean
scratch wire and explicit MSB-first physical locations:

```python
import torch

reference = fq.Circuit(9, dtype=torch.complex128)
for wire in range(4):
    reference.h(wire)
reference.cx(1, 7).cx(3, 7)
reference.cx(2, 6).ccx(1, 3, 6)
reference.x(1).ccx(0, 1, 5).x(1)
reference.ccx(1, 2, 8).ccx(8, 3, 5).ccx(1, 2, 8)
reference.ccx(0, 1, 4)
```

Here x=2a+b, y=2c+d. The result bits, from low to high, are b XOR d,
c XOR bd, a(1-b) XOR bcd, and ab. The temporary bit stores bc and is restored.
This reference does not reuse the arithmetic kernel or synthesis implementation.
Tests compare every complex amplitude and clean workspace to the mathematical
oracle. Applying Hadamard to the result also checks all 256 signed coherent
terms, not merely the original uniform probabilities.

This specialized reference is cheaper than generic conservative synthesis.
Current native preparation+compute: 18 allocated/used qubits, 10 workspace,
166 operations (4 H, 106 CX, 56 CCX), depth 124 and 4 MiB planned complex128
state. The specialized reference uses nine qubits, one scratch, 15 operations
(4 H, 3 CX, 6 CCX, 2 X), depth 11 and an 8 KiB state buffer. These are actual bounded
CPU plan/circuit costs, not optimality, speed or process peak-memory claims.
UP-01 saves manual logic derivation and cleanup work; it does not claim to match
this domain-specific hand optimization. Compiler kernel optimization is separate.

## Ownership and reuse

- Compiler: common retained source reader and qfunc snapshots; typed
  annotation/call parsing; full-domain scalar arithmetic capture; existing qreg
  storage/effect records, verifier, dependency/cleanup analysis, finite
  selection, pure kernels, target conversion and routing.
- Services: six explicit re-exports and one typed-root admission branch into
  the existing checked register Program completion/run/read flow. Existing
  pure arithmetic and pi-predicate behavior remain; no effect-specific engine.
- Ecosystem/Core/Runtime/Simulation: no production changes. Qmod still emits the
  same Compiler-owned representation; CircuitIR and execution remain canonical.
- Integration: an authenticated successor API contract protects the authorized
  six exports and typing overload. Historical contracts/proposals are not edited.

Source scopes distinguish early-bound annotation symbols from body bindings
resolved at synthesis. Capture only the annotation names from the lexical
scope, release frame references, and never evaluate annotation strings or host
properties. Closures, qualified imports and primitive/type import aliases work
within the admitted profile. A sealed Program is unaffected by later namespace
rebinding; resynthesis sees the new helper and changes complete source identities.

Calls specialize generic Const/Output widths with a 32-specialization/depth cap.
Existing 4096-node/source/effect and 256-bit/storage bounds remain. Core verifier
owns fresh/initialized/Const/alias checks after capture. Default typed-root memory
is a 512 MiB planned-state ceiling; explicit limits and final target budgets are
checked before execution. Result names/order come from Output ports.

## Current acceptance evidence

Focused workflow/Compiler/API/retained-XOR run: 105 passed in 12.67 seconds.
It includes all 16 basis inputs, full uniform states, all 256 signed interference
terms, generic helpers, Const aliases, width specializations, lexical/quoted
annotations, star import, before/after binding identity, memory/operation/qubit
limits, routed named readout and refusal of invalid roles, types, effects,
source symbols, host calls, duplicate preparation and recursive/budget excess.
Existing nonzero-target XOR multiplication tests are retained; a fresh public
Output is never treated as permission to overwrite nonzero data.

Two source-overload typing fixtures pass in 10.52 seconds, including the new
root form and refusal of four invalid typed-root configurations in addition to
the original ten negative cases. Strict mypy passes all 881 production/example
sources. The related regression passes 2810 cases in 723.47 seconds across
87 files. Source-typing fixtures run separately rather than being hidden in
that count. Black26.10.1/Ruff, architecture, dependencies, public API successor,
generated-doc consistency, maturity, English-language, local links and whitespace
checks pass. No original contract, source/license pin or test assertion is weakened.

The final checker-only typing cleanup adds explicit dictionary/callable narrowing
and removes an obsolete ignore in the existing snapshot tool. Strict mypy passes
that tool as well; all 61 API/synthesis contract unit cases pass afterward in
0.85 seconds. No snapshot data, valid public behavior or numerical code changes.
Broader/default results above precede only this checker typing cleanup and
user-documentation updates.

### Full default tier and baseline

The complete default smoke/unit run records 17479 passed, 527 skipped, 27 failed,
6 errors and 6054 deselected in 1012.65 seconds. This is not a green default tier.
The failure/error node set is identical to UP-00's 33-node baseline. An isolated
archive of merged main ef95c04f08ddeb05c1c8469b51f4cfccb7c4fc62, with its own Git
metadata, local ref snapshot and read-only object alternates, is used to reproduce
that set in the same dependency/sandbox/history context. No implicated test,
Simulation, plugin, evidence-validator or network implementation is changed.
The same 33-node set reproduces on that base: 27 failed and six errors in
17.59 seconds. All nine implicated test files are byte-identical. This is a
local dependency/sandbox/history comparison, not a claim that main CI failed.

Logs: `/private/tmp/fq-up01-focused.log`, `fq-up01-regression.log`,
`fq-up01-source-typing.log`, `fq-up01-mypy.log`, `fq-up01-default.log`,
`fq-up01-baseline-default-failures.log`, `fq-up01-baseline-comparison.json`.
Remote PR CI must complete independently; local baseline reproduction is not
fresh-runner certification.
The original corpus stays 8/216 paths, 7/208 distinct blobs; this native frontend
addition increases no original Qmod coverage count.

Development findings: the first closure example exposed annotation-only local
imports being absent from function closure cells. Capturing only lexical type
bindings fixes that without executing body code or using get_type_hints/eval.
The first strict typing check exposed the missing no-input qfunc overload and
two untyped builder lists; both were fixed under the explicit successor contract.
An initial edit missed a resolver keyword parameter; source/lint checks caught
it before final conformance. Failed attempts are not counted as successful tests.

## Remaining scope

This is the bounded preparation/arithmetic/typed-call slice, not all Python or
all Qmod: no consuming Input/free/drop, mutable helper ports, arrays/views,
quantum locals, explicit phase/control/inverse/within-apply, ordinary host calls,
loops/branches, runtime real parameters or serialization in this new profile.
Unrelated existing pure-function and Qmod capabilities retain their documented
scope. Whole-program phase is UP-02, inverse/cleanup composition UP-03, and
coherent control UP-04. Trials remain 0/2 and release_authorized=false. There is
no GPU/QPU, scalability or release promotion.

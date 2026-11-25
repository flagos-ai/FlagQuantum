# UP-00 contract and retained RZ compatibility

The [unified-program contract](../api-changes/FQ-SYNTH-UNIFIED-PROGRAM-UP00-20261102.md)
supersedes the prototype's native-entry decision below. RZ import, decimal-angle
validation, original source/license pins and exact Core phase conventions remain.
The unmerged float-return evaluator, Compiler flag and Services dispatch are
withdrawn. This document's historical results are bound to head
4c923bf111ace9c29bf0d784bbfabfc0fee940d9, not the revised runtime admission.

## Current runnable user workflow

Run `python -m examples.synthesis.tutorials.qmod.native_and_qmod_rz.main`. It demonstrates:

- Python high-level `x*x+y` with named uniform preparation and reading;
- equivalent Qmod scalar helper with Const inputs and fresh output;
- coherent Qmod RZ import, independent Circuit reference and absolute-phase oracle.

The first two agree on all sixteen `(x,y,x*x+y)` outcomes. The native output is
named `result` by the existing API; the Qmod example uses that same output port.
Named joint reading explicitly requests `(x,y,result)`; default reading is the
result marginal. Full complex amplitudes and workspace cleanup are checked in
the integration suite, not inferred from matching probabilities.

RZ's native mathematical program counterpart is an explicit gap for UP-02,
not a reason to force a Qmod user to translate into a float-return function.
Unified typed Python preparation/phase/control examples are design candidates,
not runnable APIs. See the UP-00 contract for grammar, ownership, source safety,
resource limits, staged acceptance and the three current/target distinctions.

## Revised validation

Current focused run: 163 passed in 3.86 seconds, covering the new Compiler and
integration boundary suites, Python source regression, RZ register contract,
and Qmod RZ/retained PHASE. This includes both objectives, basis/coherent and
reference-entangled phases, exact Decimal bounds and final-plan budget refusal.
The frontend replacement test reads identical named full states and proves
workspace zero over all sixteen basis terms of the coherent two-bit example.

Initial replacement fixtures incorrectly assumed a `z` native result and an
all-register default marginal. They were corrected to the existing `result`
name and explicit registers, without changing production result semantics.
The failed fixture runs are not reported as passing. The related regression passes 2688 cases in 645.27 seconds; default-tier
results for the revised head are recorded in the
[UP-00 delivery](SYNTH_UNIFIED_PROGRAM_UP00_DELIVERY.md): 17417 passed, 527 skipped,
27 failed and 6 errors. The same 33 failure/error nodes reproduce on the isolated
main base in the same local environment/history context; the default tier is not
green. No unrelated assertions or implementation are changed.
Original pins and licenses are unchanged. Trials remain 0/2 and
release_authorized=false; corpus coverage remains 8/216 paths, 7/208 blobs on
the retained RZ candidate. This PR is not merged by this task.

---

## Historical prototype record (superseded, not current user instructions)

The following text is preserved verbatim as evidence for head
4c923bf111ace9c29bf0d784bbfabfc0fee940d9. Its native phase example is no longer
admitted. Commands, counts and resource figures below must not be attributed to
the adjusted UP-00 implementation.

# Native high-level radians intent and original RZ migration

Follow #1002 merged at 6df1f2ae6296334d087fdf39918f8b1e46559a19 after
57 successful checks and three configured skips. See the
[contract](../api-changes/FQ-SYNTH-LITERAL-RZ-20261102.md).

## FlagQuantum-native high-level workflow

```python
import flagquantum as fq
from flagquantum import synth


def phase(q: int) -> float:
    return 1.9 * (q - 0.5)


program = synth.synthesize(
    phase,
    inputs={"q": synth.UInt(1)},
    input_state=synth.InputState.uniform(),
    output_mode="phase",
    max_qubits=1,
    max_workspace=0,
    memory_limit_bytes=1024**2,
)
result = fq.run(program)
print(dict(program.probabilities(result, registers=("q",))))
print(result.statevector())
```

Users write a phase function in radians, not an RZ call, circuit, physical wire,
workspace assignment or Qmod string. The function specifies the action
|q> -> exp(i*phase(q))|q>. Users still declare quantum domain/encoding, initial
state, the phase effect and budgets. An explicit float return selects radians;
existing bool/int/unannotated phase functions retain pi-predicate semantics.

Compiler checks both values of the complete UInt(1) domain from the retained
AST and selects existing gates. Here phases are -0.95 and +0.95, so it chooses
RZ(1.9); uniform preparation adds H automatically. Amplitudes are approximately
0.4113-0.5752i and 0.4113+0.5752i, with probabilities 1/2 each. The complete
CPU complex128/Torch 2.13.0 program uses one allocated/used qubit, zero workspace,
two operations (H and RZ), depth two and a 32-byte planned state buffer.
This is resource accounting, not a process peak-memory or performance claim.

## Migration and gate reference are separate levels

Run `python -m examples.synthesis.tutorials.qmod.native_and_qmod_rz.main` for an executable three-way
comparison. The migration equivalent is source containing H(q);RZ(1.9,q),
passed through synth.synthesize with source_format=qmod. The existing native
gate reference is fq.Circuit(1,dtype=torch.complex128).h(0).rz(0,theta=1.9).
Circuit remains a gate-level baseline; it is not presented as high-level intent.

The unmodified original is independently retained at
`tests/fixtures/classiq_qmod/RZ.qmod` and accepts Path/text inputs at its original
one-bit size. Its basis-zero amplitude is exp(-0.95i), not 1. It measures one
allocated/used qubit, one RZ operation, depth one, zero workspace and a 32-byte
planned state. RZ(theta)=diag(exp(-i*theta/2),exp(i*theta/2)); PHASE(theta)
differs by a global phase. Tests compare absolute complex amplitudes, including
reference entanglement for the gate/migration baselines, without normalizing away
global phase. The first high-level profile does not accept unknown entangled
preparation or custom preparation IR.

## High-level semantic and syntax boundary

The native profile is one plain source-retaining Python function, one UInt(1),
one return expression, optional docstring and explicit float return annotation
in output_mode=phase. Allowed nodes are real numeric literals, the input name,
+, -, *, and unary signs. Integer arithmetic remains exact; float operations
preserve ordinary float64 evaluation order. Literals, intermediate results and
final phases must be finite with magnitude <=2**53, and expression depth <=32.
Both domain values are checked even for basis-zero preparation; no measurement,
sampling, state narrowing or host function invocation is used.

Constant/global phases are valid and retained. Exact antisymmetric phases select
RZ; a zero phase on basis zero selects PHASE; other phase pairs use existing
X/PHASE gates. No tolerance-based equality or hidden approximation is added.
The initial finite-domain procedure is deliberately one-bit; it must not be
expanded by exponential enumeration of wider domains. Helpers, constants
bindings, multiple/wider inputs, assignments, loops, calls, data comparisons,
general real quantum-data computation and custom preparation IR remain refused.
No qfunc decorator is required. Existing unsigned data and bool/int predicate
profiles, including helpers and cleanup, retain their previous semantics.

## Architecture and source literal boundary

Compiler extends the shared Python source reader/capture authority, reuses the
existing register/effect HybridProgram and register selection/lowering, and
inserts validated known X/H preparation after allocation. Services composes the
same capture, preparation checks, constraints, selection and checked completion.
The output is the existing Program with input names only, result_width=0 and
scope=actual_cpu_preparation_real_phase; fq.run executes its sealed plan.
There is no new public export, argument, mode, IR type, quantum real-data type,
numerical kernel, allocator, executor, SDK or dependency.

The register gate's closed single-angle set now includes phase and rz, with
exact gate/index/angle and finite Python float validation. Other gate records
remain gate/index. Existing Core opcodes receive the named theta parameter.

Ecosystem reuses existing tokens/expression records and parses contiguous fixed
Decimal literals only in the first PHASE/RZ argument. Literal length is <=78
characters; Decimal compares the source value exactly to 2**53 before float64
conversion, so a rounded overshoot such as 9007199254740992.1 is refused.
The grammar remains nonnegative integer/fixed decimal only: no negative,
exponent-form, leading/trailing point, split decimal, symbolic, computed or
quantum angle. PHASE now intentionally admits decimals; its previous decimal
refusal is replaced by positive complete-state tests. Parenthesized integer
PHASE literals remain accepted. General decimal data arithmetic, new qnum
kinds, Const gate targets and within/apply actions are not admitted. RZ and
PHASE are reserved intrinsics; helpers cannot shadow them.

## Original counts and provenance

RZ.qmod retains original bytes at Classiq commit
2bfd835e73b6882fcc515a725098b04766c238e7, pinned by upstream path, Git blob,
SHA-256 and shared MIT license in rz_sources.json plus independent test assertions.
Old manifests, fixtures and license bytes are unchanged.

Original-size file/text execution increases verified paths from 7/216 to
8/216 (3.70%), distinct blobs from 6/208 to 7/208 (3.37%). Native high-level
examples do not increase original-source counts. A fresh full 216-file frontend
scan admits eight and refuses 208, with only RZ changing admission; first
qmod.operator blockers decrease from eleven to ten. Admission is distinct from
execution, and file counts are not counts of independent algorithm families.

## Validation and remaining scope

The focused 118-case run passes in 1.41 seconds: 85 new Compiler/RZ/native cases
and 33 retained PHASE integration cases. It covers native intent/Qmod/gate/
analytic agreement, basis/uniform and both objectives, absolute/global and
reference phases, constant phase, host-body nonexecution, complete-domain bounds,
source/IR malformed data, exact decimal overshoot and tiny nonzero angles,
old predicate semantics, role/type/grammar and real physical memory/gate budgets.

The first native range test exposed premature conversion of integer literals to
float during AST arithmetic: q+2**53 rounded instead of being rejected. The
interpreter now preserves int intermediates and float64 operation order and
converts only validated final phases. A helper-refusal fixture used an already
invalid float helper annotation; it now uses a valid integer helper so it checks
the intended profile boundary. An initial cached-source probe used a different
name from the helper's required main name; this fixture was corrected. These
failed runs are not reported as passing or used to weaken existing contracts.

The broader Compiler/Ecosystem/synthesis/API regression passes all 2689 cases
in 612.40 seconds. The final focused rerun uses distinct cache filenames for
three rejection fixtures, so each validates its intended retained source rather
than a later cached definition; production is unchanged between runs. Two
source-overload mypy-cache fixtures are excluded. Strict mypy 2.4 passes 879
production/example sources. Black 26.10.1, Ruff, architecture, unchanged API
snapshot, dependency policy, generated docs, maturity, language, local links,
whitespace and original source/license checks pass.

Required default smoke/unit records 5064 passed, 121 skipped and the same three
existing local Simulation QFT fusion assertion failures in 212.67 seconds,
stopped at its configured three-failure limit. They expect 28 state applies and
observe 44; Simulation and assertions are unchanged. The default tier is not
green. Logs: /private/tmp/fq-rz-regression.log and /private/tmp/fq-rz-default.log.
Independent developer trials remain 0/2 and release_authorized=false. General
real/multi-input phase synthesis, unknown input states, signed/fixed-point data,
trainable high-level coefficients, full Qmod and GPU/QPU admission remain gaps.

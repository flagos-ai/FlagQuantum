# UP-00: unified quantum program semantics and compatibility boundary

Date: 2026-11-02. Owner: Compiler semantics; Services facade; Ecosystem import.
Scope: `single_device_fast_path`, exact local CPU construction and execution.
Status: user-requested revision-seven contract baseline for follow-on UP work.
This is not Stable Core API freeze, a release approval, or implementation of the
candidate Python program language. No protected snapshot or root export changes.

## Decision and user journey

A Qmod user must not have to learn another language to use FlagQuantum. Keep
original `.qmod` files executable unchanged within an explicitly tested subset.
For Python, prefer familiar `qfunc`, quantum types and roles, `allocate`, calls,
`phase`, `control` and `within_apply` concepts. New spelling needs a demonstrated
user benefit, semantic comparison and migration evidence; renaming is no benefit.

One whole program can prepare states, evaluate expressions, compose quantum
functions and apply coherent controls. Compiler owns the shared checked semantics;
frontends translate into that model. Ordinary pure-function convenience forms
remain adapters, not separate phase/value/control execution engines. Users specify
encoding, roles, preparation, effects and budgets; Compiler handles gates,
workspace, reversible computation and cleanup. Circuit remains expert interop and
an independent reference, not a substitute for high-level native semantics.

The first follow-on language deliberately captures retained AST rather than
executing arbitrary Python. This differs from a generative Python DSL: closures,
side effects and arbitrary host expressions are not implicitly supported.

## Runtime admission in this PR

| Surface | Accepted now | Not admitted by UP-00 |
| --- | --- | --- |
| Python `qfunc` | Existing source snapshot of bounded integer/predicate definitions; optional for plain functions | Output/Const/QNum program annotations and general effectful bodies |
| Python value | Existing unsigned expressions, assignments, bounded helpers/loops and fresh result | General overwriting, quantum real data or dynamic shapes |
| Python `output_mode="phase"` | Existing one-bit bool/int/unannotated predicate, phase `(-1)**predicate` | Float-return radians functions; multi-bit numeric phase |
| Qmod | Existing checked scalar subset, helpers, local/within-apply subset and one-bit `phase(expression, pi)` | General full-language or standard-library compatibility |
| Qmod `PHASE`/`RZ` | Initialized qbit target; nonnegative integer/fixed-decimal literal angle, exact source bounds `0..2**53`, 78-character cap | Negative/exponent/symbolic/computed/quantum angles, arbitrary numeric/array/Const targets |
| Return/execution | Existing `Program`, `fq.run(program)`, named `program.probabilities` | Another executor, public IR builder or required `program.execution` |

The uppercase gate restriction is frontend admission, not a restriction of Core
RZ or its private finite-float Compiler contract. No restriction is inferred from
a gate name alone. Existing Python integer/bool behavior, exception categories,
source checks and result identity remain unchanged.

The unmerged #1003 float-return prototype is withdrawn, not deprecated as a
released API. Its recorded tests/resources describe that historical head only.
There is no `real_phase` Compiler capture flag or Services dispatch left in the
revised implementation. `float` annotations fail the ordinary source checker.
Do not expand finite one-bit enumeration to wider domains.

## Follow-on program syntax: fixed design, staged admission

The following names/spellings are the first-program design baseline, not imports
available today. `qfunc` already exists for the narrow profile above. Additions
need the owning-domain implementation, conformance tests and API review before
admission; this document does not authorize a snapshot update to bypass review.

| Qmod | Python program design | Meaning / first owner |
| --- | --- | --- |
| `qfunc f(...) { ... }` | `@synth.qfunc def f(...): ...` | Captured callable boundary; UP-01 |
| `qbit` | `QBit` | Single binary quantum value; UP-01 |
| `qnum<2, False, 0>` | `QNum[2, False, 0]` | Explicit two-bit unsigned integer; UP-01 |
| Unspecified `qnum` output | `Output[QNum]` | Infer fresh result width over full domain; UP-01 |
| `output` / `const` / `input` | `Output[T]` / `Const[T]` / `Input[T]` | Roles below; consuming Input remains refused initially |
| `allocate(q)` | `allocate(q)` | Fresh zero-state ownership; UP-01 |
| `hadamard_transform(q)` | `hadamard_transform(q)` | Uniform computational-basis preparation; UP-01 |
| `z = x*x+y` | `z |= x*x+y` | Fresh assignment, not in-place XOR; UP-01 |
| `phase(expr, coefficient)` | `phase(expr, coefficient=1.0)` | Explicit diagonal effect; UP-02 |
| `within { U } apply { V }` | `within_apply(lambda: U(), lambda: V())` | Same chosen `U`, then `V`, then `U` adjoint; UP-03 |
| `control (p) { B }` | `control(p, lambda: B())` | Coherent quantum condition; UP-04 |

The initial candidate `phase` expression grammar is constants, initialized
unsigned quantum scalars, `+`, `-`, `*` and literal powers 1..8, with a
finite binary64 construction-time real coefficient. Reject nonfinite binding/lowered angles
rather than silently reducing, overflowing or approximating them. This is not
a general real quantum data type. Negative intermediate phase expressions are
valid in that future profile; this does not enlarge today's Qmod literal gate parser.

Root outputs use signature ports, not a second mandatory return dictionary.
The first root has Output ports only and prepares them inside the program;
combining that root with scalar-profile inputs/input_state/custom preparation
must fail rather than prepare twice. Existing pure-function inputs/InputState
stay supported through their adapter. External live-state Circuit composition
has a separate explicit port/ownership contract, not an implicit root input.
`from flagquantum.synth import *` remains limited to the existing `__all__` until
the additions pass their own export review. No `.fq` format is required.
Aliases such as `with quantum_if`, a new angle-first phase API or mandatory
`uniform()` constructor are not part of this contract.

## Semantic invariants

### Definition, encoding and ownership

A pure expression has no quantum effects. A reusable quantum function has typed
ports and an effect summary. A root allocates/prepares outputs and yields an
executable Program; a preparation isometry is not automatically a unitary on
arbitrary inputs. A decorator alone proves neither controllability nor inversion.

For the future program frontend, resolve statically named qfunc calls at synthesis
into a bounded definition snapshot. Later host rebinding cannot change a sealed
Program. No public function registry is required. Existing pure-function helper
configuration remains supported without an implicit arbitrary-call fallback.

Width, signedness, fractional bits, numeric bit weights and source indexing are
explicit. First program admission is unsigned integer only. Do not infer QNum
semantics from UInt spelling or initial basis values. Fresh `x*x+y`, with both
inputs in `0..3`, needs four result bits for the full range `0..12`. Classical
parameters occupy no quantum wires; construction-time binding changes program
identity. Runtime parameter training is a separate capability.

For future source indexing, preserve Qmod numeric significance: unsigned integer
bit i has weight 2**i. An ordered MSB-first w-wire FlagQuantum arithmetic port
therefore maps source bit i to port[w-1-i]. Do not reverse Core's wire convention
or extend this rule to array positions without a view contract. Numeric indexed
gates/views remain unadmitted today; UP-01 or a later view slice must prove the
mapping with basis-state and routed readout tests before exposing them.

| Role | Entry | Exit | Contract |
| --- | --- | --- | --- |
| Ordinary quantum | Initialized | Initialized | Declared in-place effects, future broader admission |
| Const | Initialized | Initialized | Preserve computational-basis value, allowing phase/entanglement |
| Output | Uninitialized | Initialized | Establish new ownership, never overwrite arbitrary unknown state |
| Input | Initialized | Uninitialized | Consume reference; not a read-only input |

Roles and initialization are separate dimensions. Repeated Const arguments may
alias when no basis write occurs; fresh Outputs must not alias each other or a
live borrowed input. Extend checking to slices before admitting views. Preserving
basis values is not preserving each input's reduced density matrix.

### Assignment and cleanup

Fresh assignment implements the zero-target case of the checked reversible
relation `|x,y,t> -> |x,y,t XOR f(x,y)>`. Nonzero-target tests exercise the
underlying kernel, not permission to reinitialize a live Output. Public general
XOR width truncation and modular addition are separate contracts.

Last use is permission to schedule cleanup, never permission to erase information.
Record producer implementation and dependency versions; uncompute only while
those dependencies remain valid. A changed dependency requires proven earlier
cleanup or refusal. Retain/recompute schedules preserve the same full quantum
map, including relative/global phases. Exported outputs remain live.

Initial `free` admission requires a clean-state proof; general caller assertions
remain unsupported. Upstream `free` trusts a zero-state assertion, while `drop`
retains possibly dirty wires without reuse. Do not silently translate either to
reset, partial trace or reclamation. No new public `drop` operation is added;
removing a logical binding does not discharge cleanup obligations. This stricter
boundary is documented compatibility scope, not upstream semantic equivalence.

### Phases, control and inverse

Future explicit `phase(f, theta)` multiplies a basis amplitude by
`exp(i*theta*f)`. A constant applies a global phase, which becomes relative under
control. Preserve it through calls and inverses. `RZ(theta)` has diagonal
`exp(-i*theta/2), exp(i*theta/2)`; PHASE has `1, exp(i*theta)`. Do not replace one
with the other by ignoring a scalar factor. Probability agreement is insufficient.

Upstream's documented phase-expression grammar is polynomial. Today's accepted
comparisons inside `phase(expression, pi)` are an explicit FlagQuantum extension,
not evidence that upstream admits them or that all coefficients are supported.
Keep the extension's existing behavior; require explicit evidence for expansion.

`control` is coherent, never a host bool conversion or measurement. Initially
condition and body cannot reference the same quantum variable, including aliases
or disjoint slices of the same variable. All external ports stay initialized;
local scratch is statically provisioned and clean on both branches. Refuse branch
outputs, user-visible conditional allocation/consumption, reset, measurement and
unsupported control depth. First admission is single-level control without an
else branch or general higher-order callbacks. Static calls remain acyclic and
bounded by the existing definition/node/expansion limits. Classical Python `if`
is construction-time only.

Only validated unitary bodies are invertible/controllable. Inverse must use the
same chosen implementation, including its phase and scratch contract. In
`within_apply`, V must preserve dependencies needed to invert U. Core's existing
instruction transforms are reusable after admission, not proof of these source
conditions. Dynamic hybrid measurement regions are a different effect family.

### Failure, resources and execution

Keep one Compiler-owned checked representation: existing HybridProgram values,
operations, regions and quantum-effect tokens with added operation verification
as needed; no second canonical IR. Lower into existing CoreIR/CircuitIR and use
existing finite-candidate selection, target conversion, routing and execution.
Having a region field does not establish any of the new semantic invariants.

The existing private register operations remain the lowering vocabulary:

| Operations | Required admission invariant |
| --- | --- |
| qreg.declare / local | Unique typed logical storage; declaration does not initialize state |
| qreg.allocate / initialize | Fresh target only; establish initialized ownership |
| qreg.compute | Checked unsigned expression and full-domain result fit; inputs stay live |
| qreg.phase | Checked borrowed expression; preserve basis values and phase convention |
| qreg.gate | Closed verified gate attributes; effects respect port roles |
| qreg.call | Verify reachable/unused definitions, exact ports and acyclic bounded expansion |
| qreg.uncompute | Same producer inverse with valid dependencies; restore clean ownership |
| qreg.release | Compiler-proven clean local storage, invalidate reference before reuse |
| qreg.return | Export only live declared outputs and terminate the current effect |

Use the existing linear quantum-effect chain: each stateful operation consumes
the current token once and produces its successor. A source-local expression
value is immutable and is not itself a storage reference. A caller-supplied
"unitary" or "clean" attribute cannot establish either property. Future coherent
regions need checked dependency/alias/effect summaries and token joins; they
cannot enter through arbitrary qreg.gate records or measured-feedback regions.
No new control opcode or region schema is admitted by UP-00. This is a mapping
to current authoritative operations, not a duplicate registry or public schema.

Reject unsupported source/roles/effects at the earliest knowable point. Do not
run host bodies or silently narrow the domain, relax budgets, truncate outputs,
discard phases or substitute backends. Include source location and an actionable
supported alternative; cost failures refer to actually checked candidates.

Budget the final submitted preparation/computation/control/cleanup/target plan.
Distinguish logical estimates, final resources and state-buffer bytes from
process memory or optimality. Program identity and readout verify source/bindings,
output map and actual execution identity. Keep named results independent of
physical layout. No GPU/QPU, scalability or release claim follows from CPU tests.

## Acceptance and dependency gates

| Gate | UP-00 evidence | Follow-on proof still required |
| --- | --- | --- |
| Source safety | Float program rejection, retained integer source not executed | General program capture and call effects: UP-01 |
| Numeric meaning | Existing Python/Qmod arithmetic replacement, full-state/name/cleanup checks | New role-annotated syntax: UP-01 |
| Role/ownership | Current fresh/Const acceptance plus role violations refused | Aliases/views and general mutable ports: separate admission |
| Phase compatibility | Int/bool/unannotated pi semantics; Qmod RZ/PHASE absolute amplitudes | Explicit multi-bit polynomial phase: UP-02 |
| Cleanup | Existing bounded within/apply and dirty-use refusal | Cross-call dependency versions and inverse bodies: UP-03 |
| Control | Unsupported blocks fail closed | Full coherent body/control/global-phase tests: UP-04 |
| Compatibility | Original RZ file/text and license/hash pins, no source rewriting | Subsequent named originals at unchanged sizes |
| User experience | Runnable current Python/Qmod/Circuit comparisons with explicit gaps | Two independent trials, including a Qmod user: UP-06 |

Tests must execute real existing paths, not a toy semantic engine or assertions
that a document contains expected words. Replace one frontend with another while
keeping the same consumer and compare named complete states. No contract fake
pretends the unimplemented program language is admitted. No new runtime contract
class is necessary for this documentation/withdrawal slice.

## #1003 adjustment and migration

1. Land this contract commit before owning-domain implementation changes.
2. Remove the unreleased float evaluator and capture flag from Compiler and the
   real-phase branch/scope from Services. Keep existing Python int/bool paths.
3. Preserve RZ/PHASE register-angle validation, existing Core lowering and all
   Qmod decimal/source/license pins and negative tests.
4. Replace positive float-profile tests with explicit rejection and compatibility
   probes; preserve the prototype's historical evidence under its original head.
5. Show runnable current native high-level arithmetic, equivalent Qmod and a
   separate Circuit reference. For RZ, provide unchanged-source execution and a
   Circuit reference, with the unified native mathematical phase marked a gap.
6. Update current capability/docs without promoting maturity. No general qfunc
   program API is exported until UP-01. No deprecation/version bump is needed
   for the withdrawn unmerged/unreleased prototype. Do not merge #1003 in this task.

## Sources and version boundaries

Original corpus is pinned to Classiq/classiq-library commit
`2bfd835e73b6882fcc515a725098b04766c238e7`: 216 paths, 208 distinct blobs.
The RZ fixture's source/license manifest is separately verified. A declared
source version is not authentication. No all-corpus admission is implied.

Official pages reviewed on 2026-11-02, read rather than executed against Classiq:
[roles/free/drop](https://docs.classiq.io/qmod-reference/language-reference/quantum-variables),
[assignment](https://docs.classiq.io/qmod-reference/language-reference/statements/assignment),
[phase](https://docs.classiq.io/qmod-reference/language-reference/statements/phase),
[control](https://docs.classiq.io/qmod-reference/language-reference/statements/control).
These are living docs, not pinned SDK compatibility certification. Preserve the
[earlier review](../roadmap/CLASSIQ_SEMANTICS_REVIEW_20261029.md) and all existing
corpus pins. Follow-on additions must record their actual grammar and tests.

This contract corresponds to the local revision-seven plan dated 2026-11-02.
UP-01 prepares/computes/outputs; UP-02 adds explicit phases; UP-03 proves inverse
and cleanup composition; UP-04 adds coherent control; UP-05 normalizes existing
frontends; UP-06 validates resources, P1-P4 and independent user tasks. None is
completed by merely fixing names in this document. Trials remain 0/2 and
`release_authorized=false`.

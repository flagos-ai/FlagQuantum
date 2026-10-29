# Classiq semantics review for constrained construction

Reviewed on 2026-10-29 against the live official documentation. This is a
targeted review of modeling, language effects, synthesis, and result semantics,
not a claim to have audited the entire site or the proprietary synthesis engine.
The [documentation index](https://docs.classiq.io/llms.txt) was used to discover
the linked pages. Examples were read, not executed against Classiq services.

## Modeling and Python construction

Qmod describes quantum transformations and leaves gate implementation and
resource decisions to synthesis. Its Python embedding and native syntax expose
the same language semantics. Python can also perform ordinary computation while
building a model; construction-time Python values are distinct from symbolic
classical expressions. This means an AST-only frontend would be a FlagQuantum
design choice, not a requirement imposed by Classiq.
Sources: [language overview](https://docs.classiq.io/qmod-reference/),
[generative descriptions](https://docs.classiq.io/qmod-reference/language-reference/generative-descriptions).

`qfunc` declares a transformation with quantum, classical, and possibly function
parameters. `qperm` promises a computational-basis permutation, potentially with
phases; it does not mean every internal gate must literally be X/CX/CCX.
Higher-order function signatures describe both argument types and effects.
Consequently, ordinary pure integer functions are a useful first frontend but
are not a complete replacement for general quantum function composition.
Sources: [functions](https://docs.classiq.io/qmod-reference/language-reference/functions),
[operators](https://docs.classiq.io/qmod-reference/language-reference/operators).

## Parameter roles and resource lifetime

An ordinary quantum argument is initialized on entry and remains available.
`Output` starts uninitialized and is initialized by the callee. `Input` is
initialized on entry and becomes uninitialized afterward: it denotes consumption
of a reference, not a generic read-only input. `Const` restricts computational-
basis mutation while allowing phase changes; it is not a classical constant.

`free` is a caller assertion that the object is zero; the compiler does not
verify that state precondition. It reclaims the wires and bypasses automatic
uncomputation restrictions. `drop` abandons the reference and excludes it from
uncomputation while retaining its possibly dirty, entangled wires; those wires
cannot be reused. A dropped subsystem can change the interpretation of the
remaining state, so it cannot be translated into zero-workspace reclamation.
Source: [quantum variables](https://docs.classiq.io/qmod-reference/language-reference/quantum-variables).

## Uncomputation is an effect and dependency contract

Automatic cleanup relies on permutation/const classification and restrictions
on uses and dependencies. A declaration alone does not generally prove an
arbitrary circuit's cumulative behavior. The documented check-disabling options
are trusted escape hatches for implementations whose overall effect is known.
FlagQuantum's initial closed arithmetic profile therefore keeps proof by
construction separate from caller assertions and does not accept arbitrary
cleanup claims as established facts.
Source: [uncomputation](https://docs.classiq.io/qmod-reference/language-reference/uncomputation).

`within_apply` executes the chosen preparation, application, and the exact
inverse of that same preparation. Its rules allow reclamation, constrain uses
of computed values in the application, and avoid redundant outer control on
the preparation/inverse pair. Re-running synthesis to obtain a supposedly
equivalent inverse would not establish this guarantee.
Source: [within-apply](https://docs.classiq.io/qmod-reference/language-reference/statements/within-apply).

## Arithmetic, initialization and bit significance

Out-of-place assignment requires an uninitialized target and infers its numeric
representation. In-place XOR requires an initialized target and updates matching
bits; in-place addition has separate wraparound and fractional-bit semantics.
FlagQuantum's full-width XOR evaluator is a deliberate initial primitive. Its
zero-target case can implement fresh assignment, but accepting arbitrary Qmod
XOR widths or modular addition requires additional explicit rules.
Source: [assignment](https://docs.classiq.io/qmod-reference/language-reference/statements/assignment).

Qmod numeric attributes include width, signedness and fractional bits. The
reference specifies that the highest-index bit is most significant. FlagQuantum
uses wire 0 as the leftmost basis-label bit and its arithmetic maps are MSB
first. The future adapter must explicitly map source index/weight to destination
wires, including slices, arrays and named readout. Matching printed integers is
insufficient to certify indexed gate behavior.
Source: [quantum types](https://docs.classiq.io/qmod-reference/language-reference/quantum-types).

`bind` transfers references and rewires views without copying a quantum object
or adding gates. Sources are consumed, destination sizes must agree, and layout
stability has scoped rules. Physical alias tracking is therefore separate from
repeated reads of a logical arithmetic expression.
Source: [bind](https://docs.classiq.io/qmod-reference/language-reference/statements/bind).

## Coherent transformations and algorithm structure

Quantum `control` acts coherently, can entangle condition and target, and has
restrictions on overlapping variables and scoped allocations. It is not a
classical branch or an implicit measurement. `invert` applies the adjoint in
reverse order and has its own initialization constraints.
Sources: [control](https://docs.classiq.io/qmod-reference/language-reference/statements/control),
[invert](https://docs.classiq.io/qmod-reference/language-reference/statements/invert).

An expression phase applies exp(i theta f(x)); a constant phase becomes relative
when controlled. A single-qubit PHASE gate and a whole-block global phase are
not interchangeable. Full Core instructions and explicit matrix payloads must
survive the boundary, rather than being reduced to a one-angle trace record.
Source: [phase](https://docs.classiq.io/qmod-reference/language-reference/statements/phase).

Numeric expressions are motivated by predicates and oracles, not just arithmetic
demonstrations. Block-encoding combinators further carry normalization and
register-size information. These support keeping domain algorithm contracts in
Algorithms while Compiler owns general construction and resource planning.
Sources: [numeric modeling](https://docs.classiq.io/user-guide/modeling/quantum-numbers-arithmetics),
[block-encoding composition](https://docs.classiq.io/user-guide/applications/block-encoding/composing-block-encodings).

## Constraints, compilation and execution

The documented interface distinguishes a maximum-width constraint from an
optimization parameter. Preferences cover formats, timeouts and optimization,
while hardware settings supply gates and connectivity. These are different from
semantic correctness. The documentation does not establish the search algorithm,
global optimality, deterministic tie-breaking, or a performance guarantee.
FlagQuantum's finite-candidate policy must be evaluated on its own evidence.
Sources: [constraints](https://docs.classiq.io/user-guide/synthesis/constraints),
[preferences](https://docs.classiq.io/user-guide/synthesis/preferences),
[hardware-aware synthesis](https://docs.classiq.io/user-guide/synthesis/hardware-aware-synthesis).

The executable `main` has output quantum arguments; classical execution cannot
pass an arbitrary quantum state as such an argument. Execution parameters can
remain symbolic without becoming register-size parameters. This is distinct
from a reusable arithmetic subroutine that borrows quantum inputs.
Source: [entry point](https://docs.classiq.io/qmod-reference/language-reference/quantum-entry-point).

Sampling returns measurement statistics. State-vector calculation exposes
amplitudes and phases on supported simulators and is not a hardware output.
FlagQuantum should retain its existing execution/results rather than copying
Classiq's service workflow, and keep distribution inspection separate from
coherent correctness and differentiable expectation values.
Sources: [sampling](https://docs.classiq.io/user-guide/execution/sample),
[state vectors](https://docs.classiq.io/user-guide/execution/calculate-state-vector).

## CS-01 decisions

- Reuse the existing private structured SSA records with a closed integer
  profile, not a third generic IR or public DSL.
- Keep integer inputs basis-preserving borrows; do not claim full Input/Output,
  drop/free, bind, qfunc or Qmod support in this milestone.
- Track repeated expression use separately from physical port overlap.
- State zero-workspace preconditions explicitly; no state inspection claim.
- Preserve the complete existing CircuitIR for opaque blocks and phases.
- Reserve source-index translation tests for the Qmod adapter; never alter
  FlagQuantum's canonical bit order to mimic source spelling.
- Treat public decorator/frontend choices as separate from these semantics.

# Structured controlled-constructor dependency closure

## Ownership and interfaces

The locked reference is Qualtran v0.7.0 commit
bcb17bb25360ed68ce3bc084645db3e929799389. The complete
mcmt/controlled_via_and.py source and tests are the reference for control
orchestration. The original source remains unchanged.

Compiler formats/legacy_controlled_factories.py owns constructor arguments,
resolved child identity and per-catalogue nested control recipes. The injected
positive-control provider returns an existing ProgramRecord and ControlAdder.
The child owner retains target semantics, helper definitions and private work.
Existing component_control and component_predicate owners retain control naming,
ordered packing, predicate computation and measured erasure. Simulation owns
numerical execution. Ordinary execution continues through circuit.append(op).

Controls precede data ports, with row-major arrays of MSB-first words. Both
polarities, scalar and singleton controls, signed words and grouped controls
retain the source layout. Nested wrappers combine new controls before previous
clauses when obtaining the child's positive control system. Control names are
chosen against the original signature, including ctrl/ctrl2 collisions.
Symbolic children remain logical declarations and require concrete binding.

A temporary typed declaration call provides the entry expected by the existing
predicate graph owner. Its explicit child adder replaces that call. The unused
declaration is removed before publication. It introduces no primitive gates,
private work or executable fallback. Failed providers never publish a recipe.

## Verification and resources

The fixed X/Y/Z/H child providers match complete resolved definitions, including
their bodies. Unknown children require their registered positive-control owner.
The replacement provider test controls a stored H/phase composite with a coherent
spectator, its own helpers and private work. Every small raw measurement history
is compared against the unchanged original complex tensor, including probability
and phase, with all private work restored to zero. Gate comparisons include
operands and feedback. Both complete original source examples are compared as
gate streams, including the 32-control-bit grouped integer example. Large
example statevectors are outside this batch's exhaustive tensor coverage.

| Operation | Actual primitive inventory | Independent original model |
|---|---|---|
| n-bit predicate around a child | 4(n-1) T/TDG and n-1 measurement/reset pairs for n >= 1, plus the positive child | n-1 And calls and n-1 measurements, plus the child model |
| Negative scalar control | Two X surrounding the positive child, no predicate work | Two Clifford operations plus the child model |
| Controlled Hadamard child | One CH primitive | Two rotations and the source Clifford categories. Rotation-synthesis T estimates remain separate |
| Nested wrappers | New-first combined predicate inside the source outer wrapper, with exact erasure | Original nested source call graph, independently expanded |

Focused constructor checks pass 117 cases, including complex64/128 append-path
execution. Concentrated dependency regression passes 2786 cases, local
integration passes 145, and all ten unchanged original source cases pass. Six
frozen source blobs and the native Kickmix 0.1 Simulator extension are verified.
Four touched Python files pass strict typing, Black and Ruff. Compiler scope,
language, architecture, IR and API checks pass. Dependency findings retain the
identical 720-line baseline.

Full physical acceptance uses the existing bound Program and user append path.
No reference object is altered to force wire serialization to succeed.

## Remaining integration boundaries

This batch completes the owned control orchestration and provider replacement
path. The default catalogue recognizes exact X/Y/Z/H leaves. Automatic generic
positive-control dispatch for arbitrary serialized child graphs remains open,
so ControlledViaAnd remains excluded from complete built-in identifier counts.
The count stays 64/303, with 239 remaining. Generic Controlled/Adjoint/Power,
arbitrary-target MultiControlPauli, partial OnEach and permissive non-signed
sign-conversion constructors retain their independently recorded gaps.

Original random MatrixGate tensor fixtures do not supply a physical primitive
decomposition. They are reference tensor evidence only. Original
numpy.uint8/Shaped/HasLength codec failures, wider control-type admission,
pending Proposals 116-119 and whole-repository/GPU/performance acceptance remain
explicit. Batch evidence is stored under
ftqc-reference-env/g2-controlled-constructor-*.

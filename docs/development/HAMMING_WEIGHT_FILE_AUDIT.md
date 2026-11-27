# Hamming whole-file audit

## Current complete-file disposition (2026-11-28)

The historical open decisions below predate the explicitly approved
[faithful-migration closure rule](G2_FAITHFUL_MIGRATION_CLOSURE_PROPOSAL.md).
The complete file now has one additional defined SDK path: concrete SymPy
integer widths. `source_hamming_weight_program` owns that path alongside the
existing compression tree; the constructor and scoped exact-adjoint factories
reuse it. No Core contract, external SDK dependency or public entry is added.

Unlike Python integers, `Integer(n)` declares `n-1` junk bits. The actual SDK
generator uses only `n-popcount(n)` of them, leaving the trailing
`popcount(n)-1` bits untouched. These are explicit zero logical outputs, not
private work or discarded ports. The inverse consumes them with zero-bit
deallocation and measurement-erases only the used And targets. The input and
all words remain MSB first; caller ownership lasts until inverse consumption.

The original direct SDK generator succeeds at literal widths 1, 2, 3, 4, 5,
7 and 8. Native primitive operands, each wire's complete ordering, untouched
junk, output roles and actual instructions compare against those unmodified
generators. The source generic Bloq decomposition refuses those same inputs as
parameterized. That refusal is retained independently. Source serialization
still converts `Integer(5)` to Python `5` and loses the junk-width distinction;
the importer continues rejecting that inconsistent encoded signature. Direct
constructor metadata retains the distinction and now executes. Nonliteral
symbolic widths remain declarations rather than invented finite programs.

Widths one through four validate the actual scoped forward/inverse factories,
ordinary `circuit.append(op)`, complete complex states entangled with a
spectator, every raw measurement history and all logical work returning to
zero. Historical source-call, symbolic-model, drawing, compression/And,
width-cache and concrete constructor certificates are reused. Their ownership
and dependency boundaries do not change.

Resource inventories remain separate. At literal width three the source model
counts two And (eight T-equivalents), while actual forward emission uses one
And (four T/TDG instructions). The extra junk bit is untouched and its inverse
needs no measurement. This source lower-bound-model discrepancy is preserved,
not optimized away or presented as equal primitive counts. Python-integer
programs retain their previous bodies and resource semantics.

Q-0198 is eligible for complete-file acceptance after this implementation's
local checks and publication. Its one-bit empty-junk generic/SDK adjoint,
symbolic-literal generic decomposition, width-query and lossy-wire failures
remain explicit source limitations owned by the generic representation and
transport follow-ups. They are not claims of successful upstream execution.
Q-0199's accepted original test coverage remains unchanged. No other parent
or Hamming-phase file is closed by this disposition.

Local batch validation: 768 concentrated Hamming/count/phase/catalog cases and
32 integration cases pass. After test-import typing repairs, 201 affected
primary cases pass; the independent Cirq 1.6.1 lane passes 206 cases including
all five unchanged original Hamming tests. Five changed Python files pass
strict mypy, Black and Ruff. Compiler scope, architecture, IR, required-check
mapping, language, Markdown links, public API baseline and whitespace checks
pass. Five complete source/dependency blobs match the immutable inventory;
Qualtran 0.7.0 and Kickmix 0.1 Simulator are verified. Evidence is retained as
`g2-hamming-literal-closure-*20261128` in the reference workspace.

The concentrated run exposed a pre-existing count-model replacement fixture
that omitted the owner's existing `legacy_shims` argument. The fixture now
checks that argument explicitly and uses the actual typed source owners. Its
original test and both production model owners were byte-identical to the
published baseline before repair. The first catalog attempt also lacked the
original notebook marker; rerunning through the existing reference wrapper
registered it. Neither failure was excluded or converted into an expected pass.

Batch G2-FILE-HAMMING audits Q-0198 and Q-0199 from Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389`. It adds the missing complete original
test mapping and reuses accepted implementation evidence. Q-0199 is accepted as
a complete test file. Q-0198 remains open for its recorded empty-junk boundary.

## Complete source disposition

`qualtran/bloqs/arithmetic/hamming_weight.py` contains one class, one constructor
field, four cached properties and four methods. It declares no separate exports,
module constants, examples or additional functions. Its source header and paper
reference are retained through the native attribution and existing documentation.

| Source definition | Native owner and interface | Evidence or remaining boundary |
|---|---|---|
| `HammingWeightCompute` | bits-domain Hamming emitter, requirements, resource identity and model | `8477975b`, `0121b3ac`, `b780407a`, `5acb209b`; callers bind explicit words through `circuit.append(op)` |
| `bitsize` | `hamming_weight_requirements(width)` and existing `ArithmeticCall` width | Concrete gates and unchanged Python/Sympy width objects have separate admission |
| `signature` | Explicit source/junk/output words, requirements and connected width owner | THRU input and RIGHT logical outputs, inverse consumption, big-endian binding and all port connections are covered; source empty junk remains open |
| `junk_bitsize` | `hamming_weight_resource_widths` and concrete requirements | Python population count and symbolic lower-bound behavior covered by existing model tests |
| `out_bitsize` | Same width declarations with shared `bit_length` | Concrete and symbolic output widths covered by existing model tests |
| `bit_count_of_bitsize` | Shared resource-width calculation | Python and Sympy integer distinction retained; symbolic literals preserve source diagnostic counts |
| `_three_to_two_adder` | `bit_compression_structure.compression_components` and `compress_three_bits_ir` | Shared six logical components, ordered primitive expansion and measured AND erase |
| `_decompose_using_three_to_two_adders` | `hamming_weight_structure.hamming_weight_stages` | Source little-endian traversal and reversed-junk pop, shared by emission and width graph |
| `decompose_from_registers` | `hamming_weight_ir(source, junk, output, ...)` | Bound big-endian words, ordered gates/feedback, full complex amplitudes and clean inverse work; existing native validation remains explicit |
| `build_call_graph` | `arithmetic_callees`, QEC query and Hamming resource model | Ordered AND/CNOT declarations, modern categories, historical budget and symbolic diagnostics |

The existing immutable arithmetic identity replaces source attrs construction.
Native instructions and explicit words own compilation. Cirq and NumPy objects
stay in reference tests. `And` and CNOT use the existing conjunction and XOR
owners. Generic resource adjoint and connection scheduling remain with their
existing owners. No additional dependency implementation was required for this
batch's test-file closure.

## Complete original test mapping

`qualtran/bloqs/arithmetic/hamming_weight_test.py` has exactly two functions and
five parameter cases. The new test fixture verifies both source Git blobs and
the complete set of top-level test definitions against the frozen inventory.

| Original function | Full original parameter set | Native assertion mapping |
|---|---|---|
| `test_hamming_weight_t_complexity` | 3, 4, 5 bits | Shared QEC query with historical budget equals every original T/Clifford/rotation field and its formula; expanded primitive counts independently match |
| `test_hamming_weight_compute` | 3, 4 bits | Every basis input through normal `Circuit.append` yields the original formatted count assertion; native inverse restores every input for every forced measurement history, including unnormalized complex amplitude and clean logical outputs |

All five unchanged original test cases also run directly from the frozen test
module. Their `assert_valid_bloq_decomposition` calls remain source graph checks.
Native acceptance reuses complete weighted graph comparisons and successful
normal-entry emission/execution, with explicit register ownership. The original
Cirq simulator and helper remain reference infrastructure. Native execution uses
FlagQuantum Circuit and the existing forced-branch helper.

The original compute test checks count bits and a complete compute/inverse
round trip. Existing `test_hamming_weight.py` adds full coherent complex inputs
for widths one through five, every measured erasure branch and exact gate/
feedback comparisons over nine widths. Those accepted assertions are reused.

## Reused evidence and resource accounts

| Existing evidence | Reused acceptance |
|---|---|
| `8477975b` | Ordered compute/erase source streams, complete complex amplitudes, small exhaustive measurement branches, logical cleanup and runnable append example |
| `0121b3ac` | Complete direct calls, symbolic widths, both QEC profiles, phasing model replacement and diagnostic source boundaries |
| `b780407a` / `5acb209b` | Recursive width cache/order, seeded child peaks, fourteen weighted connection graphs, DOT output and forward/inverse ownership |

| Input width | Expanded forward primitives | Source modern category model | Source historical model |
|---|---|---|---|
| 3 | CX13, H2, T2, TDG2, S1 | AND1, Clifford7 | T4, Clifford16, rotations0 |
| 4 | CX34, H6, T6, TDG6, S3 | AND3, Clifford16 | T12, Clifford43, rotations0 |
| 5 | CX35, H6, T6, TDG6, S3 | AND3, Clifford17 | T12, Clifford44, rotations0 |

The new batch adds test coverage and audit disposition. Production gate emission,
quantum scratch and measurement requirements receive zero additions.

## Whole-file outcomes and remaining work

Q-0199 closes because every original function, parameter and assertion has an
explicit native mapping and all five original cases pass unchanged. Imports and
fixtures are accounted for as reference infrastructure. This acceptance covers
the original test file's three-to-five-bit scope.

Q-0198 stays open. Its existing Q-0198.02 records that the frozen one-bit generic
Bloq adjoint cannot bind the empty `QAny(0)` junk. Native one-bit compute/erase is
already supported, while successful upstream empty-junk Bloq conversion parity
remains unavailable. Existing failed decomposition and width-query witnesses are
retained. Compiler bits and generic register/adjoint owners retain the boundary.
Closure requires accepted coverage of that representation boundary under the
existing integration criteria. This batch preserves its previous failure evidence.

General dtype-specific width-cache seeds and dirty/array allocation belong to
the existing graph/resource parent files. They remain in the centralized ledger
with their own closure requirements. The next complete-file candidate is
Q-0174 addition and its original tests, using existing native arithmetic modules.

## Batch validation

All ten original/native mapping cases pass. Concentrated batch regression passes
422 tests in 8.74 seconds, with zero skips. It includes the existing Hamming
primitive, complex-branch, resource, width, phasing and gradient evidence plus
shared historical budget and QEC-category consumers. The previous batch's
8,646-test regression and 87 real-kernel/CPU integration acceptance are reused
for unchanged production paths. Strict typing, Black and Ruff pass for the new
test module. Team scope, architecture, internal IR and protected API checks pass.
The normal append example verifies all input/count complex amplitudes and six
clean temporary wires. Its full copy-count path emits CX53, H9, T6, TDG6, S3,
measurement3, feedback CZ3 and reset3, while its source model reports T12 and
measurement3 separately.

The initial standalone contract-check subprocess imported an installed package
instead of this checkout. The shared repository wrapper corrected import
precedence and the contract/API checks then passed. Reference files and
production interfaces retain their accepted versions.

## Symbolic wire and stored execution batch (2026-11-20)

The actual Hamming source object exposed a missing private wire-reader case.
Its unsigned output width is serialized as ceiling(log2(n + 1)), or the
floor-bearing form when n has no integer assumption. The existing safe AST
reader now admits a one-argument log2 call using the same SymPy codegen function
as the frozen source math owner and native Core math. It preserves deterministic
wire bytes, expression text, base-two numerical meaning and the existing decode
budgets. Empty, multiple, keyword, attribute and executable Python calls remain
rejected. No wire fields or public parameter operators change.

The original generic argument decoder uses SymPy sympify and reconstructs an
undefined function named log2. It preserves text but does not recover the
codegen function's numerical evaluation. This source decoding difference is
asserted separately. The native reader uses the original math function rather
than replacing base-two logarithms with natural logarithms or editing reference
objects. Actual symbolic Hamming forward and adjoint libraries now roundtrip
exact bytes at depths zero and one. Concrete width-three objects also roundtrip
and resolve caller factories into executable native Programs, including the
adjoint's referenced child. Symbolic resolution remains declaration-only.

Compiler bits hamming_weight_component.py now owns the missing typed tree.
It reuses hamming_weight_stages, compression_components, the existing CX gate
function and and_component_program. The forward signature is x THRU, junk RIGHT
opaque(n - popcount(n)), out RIGHT uint(bit_length(n)). The inverse keeps x THRU
and consumes junk/out as LEFT ports. All bit order is MSB first. Forward may
change x during compression. It exposes every allocated junk/count bit as a
caller-owned logical output. Inverse accepts only a correctly computed state,
restores x, measurement-erases each junk in reverse logical order and frees the
zero count. The typed root owns no hidden workspace. Width one carries an
explicit opaque zero-width junk port through existing Split/Join operations.
Native forward/inverse binding and execution work for that representation.

The root stores actual CX and And calls. Input/output rebinding stays with the
shared builder and physical placement with the existing binder. Accepted widths
1, 2, 3, 4, 5, 7, 8, 9 and 16 cover both directions, reversed input placement,
noncanonical clean pools, complete output aliases and serialized CircuitIR.
Every leaf count agrees with n-popcount(n) And calls and
5*(n-popcount(n))+popcount(n) CX calls. Original source primitive streams and
measurement feedback compare independently from the analytical source QEC
categories. The shared graph scheduler can reorder independent operations at
larger widths. Tests retain equal total gates and every wire's complete ordered
operations, operands and feedback. This is not claimed to preserve the direct
source generator's arbitrary global order. Flattened and retained-call Programs
are compared under that same explicit criterion.

Widths one through four verify complete complex input vectors entangled with a
spectator and every raw inverse measurement history. All logical work/count
outputs return to zero with unnormalized branch amplitudes retained. The existing
PopulationCount example now uses typed binding in both directions, copies the
actual returned count wires, and retains circuit.append(PopulationCount()). Its
normal DynamicCircuit execution proves all four-bit input/count amplitudes and
all six temporary wires zero. The existing emitter remains the independent
primitive reference and other consumers retain their existing behavior.

The sided SDK importer is also verified at widths one through five. Forward
placement derives fresh output aliases from its binder. Both directions preserve
every wire's complete gate order and primitive count against the native emitter.
Widths up to three retain complete complex amplitudes and every erasure branch.
The one-bit source inverse SDK import still raises the original empty QAny(0)
KeyError. Both one-bit forward and inverse source libraries roundtrip at depth
zero and fail at depth one. That source failure and the native successful empty
port execution remain distinct. Q-0198 and Q-0198.02 stay open because successful
original empty-junk Bloq conversion has not been demonstrated. Q-0199's previous
complete original-test acceptance is retained. The next shared representation
work must retain this evidence and satisfy that original closure condition.

Measured primitives remain separate from models. The four-bit complete copy/
erase example emits 53 CX, nine H, six T, six T-dagger, three S, three
measurements, three feedback CZ and three resets. The original modern source
model reports three And for computation and three measurements for erasure.
The separate T-equivalent count is twelve. Three tree-junk and three temporary
count bits remain visible logical outputs, with three retained result bits.
No matrix atom or extra physical gate is introduced by typed orchestration.

Batch validation passed 716 concentrated native/reference cases, 17 local
integration cases, all five original source tests, and 84 focused cases in each
of the Cirq 1.7 and 1.6.1 lanes. Six changed Python files passed strict mypy,
Black and Ruff. Compiler scope, architecture, multi-level IR and protected API
checks passed. Four frozen source blobs, Qualtran 0.7.0 and Kickmix 0.1 with its
Simulator were verified. Nine shared implementation owners and the ten-file
previous default-docstring failure baseline were byte-identical to the batch
base. The argument test's existing NumPy imports now handle the optional-extra
absence explicitly. Whole dependency findings decrease from 687 to 685, with
zero findings in the changed files. This evidence does not certify the full
repository default suite. See g2-hamming-foreign-evidence.json and its named logs
in the external reference evidence directory.


## Source-ordered catalog dependency closure

The stored forward count now allocates output bits on first use and gathers
existing typed word views through the shared layout owner. At width four the
original and native forward bodies each contain 26 ordered calls. The former
whole-output allocation/Split body contained 25 and was not graph-equal.

The inverse uses the shared component adjoint over that complete forward graph,
with specialized measurement-erased And and inverse Split/Join/Cast/Free
definitions. The width-four inverse also has 26 source-equal ordered calls.
The existing named And boundary remains available to resource consumers.
Unused forward helper definitions retained by the shared adjoint are excluded
from reachable catalog-child inventories. No primitive schedule is inferred
from graph-call totals.

All 27 focused count-component cases pass, including the existing operand,
feedback, complete complex-amplitude and raw-history checks. The Hamming phase
closure additionally checks fresh recursive drawings and full coherent forward
and inverse operators. The original one-bit empty-junk refusal remains under
Q-0198; this graph correction does not close that source boundary.

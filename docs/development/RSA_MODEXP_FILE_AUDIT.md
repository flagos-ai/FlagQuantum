# RSA modular exponentiation

The locked reference is Qualtran v0.7.0,
`bcb17bb25360ed68ce3bc084645db3e929799389`. This batch covers
`qualtran/bloqs/cryptography/rsa/rsa_mod_exp.py`,
`rsa_mod_exp_test.py` and `factoring-via-modexp.ipynb`.

## Binding and reference comparison

The native declaration preserves the big-endian through exponent, fresh
right-side result word, source initializer and symbolic child counts.
The circuit prepares 1 on the low bit of the initially clean result.
It visits exponent wires from low to high significance, using the
same squared constants, including repeated constants of value 1.
Every child keeps the same physical result binding and cleans the
shared work. The native factory also matches the source symbolic
expression and the seeded concrete rejection sampling.

Three complete gate comparisons bind the actual locked child Soquets
in their original order. They cover moduli 3, 15 and 6, each with
permuted physical wires, locally assigned measurement slots and
full primitive comparison of initialization, controlled multiply,
inverse accumulation and exchange. The channel certificate covers
all 32 basis exponents across the three cases and every balanced
measurement-erasure history in each circuit. Two independent full
sparse runs per case check complex amplitudes and all clean work.

| Base, modulus, exponent bits, result bits | Native emitted | Source model |
|---|---|---|
| 2, 3, 3, 2 | 618 T/T†, 144 measurements, 18 wires | 144 AND, 6 CSwap, 144 measurements, peak 18 wires |
| 4, 15, 4, 4 | 2,928 T/T†, 704 measurements, 27 wires | 704 AND, 16 CSwap, 704 measurements, peak 27 wires |
| 5, 6, 3, 3 | 1,287 T/T†, 306 measurements, 22 wires | 306 AND, 9 CSwap, 306 measurements, peak 22 wires |

The native exact exchange stream uses seven T/T† per controlled swap.
The source default T estimate discounts that leaf. The per-stage
comparison includes its underlying gates and feedback.

## Source boundary and test disposition

For modulus 8 the factoring factory produces three result bits. The
source high-level callback computes the value, and the lowered source
fails because a modular-add constant cannot fit. The native executable
binding rejects that configuration at admission. The user can supply an
explicit result width that accommodates the modulus. A zero-width
exponent has a mathematical result of 1 on the admitted domain. The
native binding emits only that initializer; the source composite
classical simulator fails in its empty-bit conversion.

The native callback matches the original 100 seeded factory cases.
The full frozen test file reports 14 passes, five source skips and one
expected failure. The frozen 18-cell tutorial passed in an independent
source kernel. The [native tutorial](../../examples/notebooks/register_factoring_via_modexp.ipynb)
repeats the twenty values of (8^emod 221), determines period 8,
recovers factors 13 and 17, and binds a small modular exponentiation
circuit. Its six code cells and two SVGs passed in a separate native
kernel. Q-0469 and Q-0472 close. Q-0471 retains the recorded source
boundary and the unverified full-width demonstration stream.


## Shared Program constructor and dependency closure

The source implementation blob is
`280bc11da4d7a5adef3382989b6e9bfed95a83ff`, also pinned in the original inventory.
`compiler/registers/modular/modular_exponentiation_component.py` now owns the
source-ordered typed root. Its exact constructor key is registered by
`compiler/formats/legacy_modular_exponentiation_factories.py` without an upstream
runtime dependency. The existing component tutorial delegates to this owner;
five complete published tutorial Programs remain exactly equal, including the
1024-bit logical example and empty exponent.

| Complete source member group | Native owner and disposition |
| --- | --- |
| Class fields, coprimality, signature | Existing Algorithms value object; exact Compiler constructor validates metadata and preserves exponent THRU / x RIGHT |
| make_for_shor | Existing seeded and symbolic Algorithms factory; original hundred-case evidence reused |
| _CtrlModMul, build_composite_bloq | Shared typed Program, original IntState(1), Split, reverse MSB-index traversal, every repeated squared multiplier, Join |
| build_call_graph, _K, _generalize_k | Existing source declarations and RSA generalized multiplier model; measured primitives remain separate |
| on_classical_vals, wire_symbol | Existing Algorithms callback, original label and generic register diagrams |
| Three examples and DocSpec | Actual source constructors and wire round-trips inspected; ordinary eight-bit result has a full native body, symbolic example remains a declaration, 2048-bit result remains a logical diagram witness |

Executable graphs import the already certified controlled modular multiplication
Programs. The result is a fresh logical word transferred to the caller, not
private work that should return zero. Every multiplication lease closes. The
teaching consumer explicitly requests one-level multiplier declarations, which
continue to refuse physical binding. It does not claim those declarations are
identities or that a large graph was physically executed.

The numerical `make_on` entry now delegates to the shared ModExp Program.
An explicit private classical-scratch pass follows ordinary fresh-SSA binding,
retaining the existing workspace and caller-slot budgets. The former flat
multiplier traversal is removed from this consumer. The default Program binder
and its fresh classical output contract are unchanged.

The new constructor checks depth-zero, one and two libraries against original
root calls, named operands and exact child primitives/feedback on moduli three,
six and fifteen, including repeated multiplier one and zero exponent. Primitive
reference comparisons bind each original multiplier's lifetime separately;
root Soquet connections are independently compared. No source reference is
patched and measurement bodies are not reversed as matrices.

Complete valid-input columns pass the existing all-history And-channel
certificate. Explicit raw histories preserve absolute complex amplitudes for
one exponent bit; larger histories verify each conditional half-probability
and use normalized sparse execution together with the certificate, avoiding
numerical underflow. All private work is zero and the logical result remains
owned. The exact primitive/model resource scan also checks live peak from
allocation events, rather than pool capacity. Unknown constructor fields,
wrong signature, noncoprime inputs and the original full-ring loader boundary
remain explicit rejections. The original empty-exponent classical conversion
failure is retained separately from its successful physical initializer.

Generic ModExp Adjoint now reconstructs the exact typed inverse through scoped
constructor recipes. The comparator, controlled modular add, constant load,
scaled traversal and modular multiply owners all reverse logical calls using
explicit specialized adjoints. And erasure is regenerated from those calls;
measured primitive bodies are not reversed. The comparator uses its literal
reverse, even though its mathematical operator is self-adjoint, because the
source inverse has a different private allocation and primitive order.

The source signed comparator inverse fails during decomposition with a
SignExtend/SignTruncate port-name mismatch in both control polarities. The new
native inverse refuses that same boundary explicitly. Unsigned and Montgomery
inverses and both modular control polarities retain literal source gate proofs.
Forward signed comparison remains unchanged. The ModExp inverse consumes x
only on the prepared image and returns the exponent; it is an adjoint isometry,
not an assertion of clean release for arbitrary result inputs.

Q-0471 remains open for the original full-width physical demonstration. The
numerical binding migration is implemented separately below; source constructors
and logical diagrams do not substitute for full-width physical evidence. No Core schema, backend selection, G3
or automatic PR merge is part of this batch.


## Current Program and source model scan

These are the Program route's actual counts. Both directions with one exponent
bit give the same row, with reversed source calls and different ownership of x.
Source default T estimates use four T per CSwap; the explicit seven-T setting
matches the unmodified source's literal exact swap circuit. They are distinct
cost settings, not a changed reference.

| Base/modulus, exponent/result bits | Direction | Actual T/Tdag | Actual CX | Actual measurements | Live peak | Source default T estimate | Source T with exact swap cost seven |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2/3, 0/2 | forward or inverse | 0 | 0 | 0 | 2 | 0 | 0 |
| 2/3, 1/2 | forward or inverse | 206 | 524 | 48 | 16 | 200 | 206 |
| 2/3, 3/2 | forward | 618 | 1572 | 144 | 18 | 600 | 618 |
| 5/6, 2/3 | forward | 858 | 2268 | 204 | 21 | 840 | 858 |
| 4/15, 2/4 | forward | 1464 | 4016 | 352 | 25 | 1440 | 1464 |
| 2/3, 2/2 | inverse | 412 | 1048 | 96 | 17 | 400 | 412 |

The zero-exponent row emits one X for the original basis ket/effect. Actual
live peaks equal the source QubitCount model in all eight scanned cases.
Ordinary native append executes both local strategies with noncontinuous
bindings, an entangled borrowed spectator, a minimal clean pool, copied Programs
and lowered JSON round-trips. Complete returned states preserve private zero
work. Independent original tensor contraction also matches unnormalized complex
amplitudes in three raw one-bit histories.

A direct binding intake found 1155 instructions in both the old explicit
numerical path and new Program path for base two, modulus three and one exponent
bit. Primitive inventories agree. The old path reuses four caller slots; the
Program uses 48 measurement SSA slots and a different private wire order, first
visible at instruction 41. No identity of the complete binding streams is
claimed. This is concrete evidence for the next allocation/binding adapter,
not a reason to alter the original source or add arithmetic gates.

## Numerical binding replacement

The existing numerical entry keeps its public signature, validation, MSB-first
operands, exact workspace requirements and `circuit.append(op)` protocol. It now
binds the shared typed Program and assigns its private measurement SSA values to
the caller's existing scratch slots. Private quantum leases use the authoritative
register workspace allocator. The scratch pass adds no gates and changes no
quantum operands, parameters, output aliases or lease events. Exported classical
values cannot be recycled. No Core contract or serialized vocabulary changes.

The old and shared schedules have equal primitive inventories, but are not
related by a single global wire permutation: independent work operations can
appear in a different order, and lease reuse changes physical identities. The
old schedule first differs in work placement at instruction 41 and gate order at
instruction 77 in the two-bit example. These differences are recorded rather
than hidden by a gate multiset comparison. Actual numerical instructions are
compared in order with untouched source child generators, using the new concrete
allocation lifetimes and caller feedback slots; original root Soquet edges are
checked independently. The previous hand-coded old-layout reference is removed.

Tests exercise repeated multiplier one, reversed physical bindings, all input
columns and the compositional every-history certificate, complex sparse states,
empty exponents, source tensor amplitudes and minimal work. The source-free
replacement test disables the former flat emitter and executes both native
strategies with an entangled spectator. Caller slots are the only destinations;
their values remain the latest assignments in the emitted schedule. This does
not promise equality to the old schedule's internal scratch identities or raw
measurement histories. The full 2048-bit physical graph remains unverified.

The replacement batch passes 56 concentrated cases in Cirq 1.7.0, three new
native/replacement/tensor cases and 18 affected full ordered source-stream cases
in Cirq 1.6.1. It also passes 145 local integration cases, strict typing of six
Python files, Black/Ruff, owner scope, language/links, architecture, IR and API
checks. Existing exact inverse dependency evidence is reused. The one-bit
numerical entry emits 1155 instructions, including 206 T/Tdag, 524 CX and 48
measurements on 16 wires using exactly four caller slots. Its source default
T estimate remains 200; explicitly pricing the exact swap at seven gives 206.
Zero exponent emits one X on two result wires with no measurement or scratch.

## Repeated component construction and bounded graph profile

Component import now has an explicit builder-owned operation returning the local
callee name without consuming handles or emitting a call. Ordinary call_program
still performs exactly import followed by call_dict. Scaled modular addition
imports a distinct reduced constant once inside each builder, while retaining
every original per-bit compute/add/erase operation and explicit inverse pair.
The mapping expires with the construction; there is no persistent cache or new
function registry. This removes repeated construction and validation, not gates.

Eleven complete Programs are unchanged from published 4551a7e11: eight scale
cases include zero, positive, negative and repeated residues, forward/inverse
and widths two/four/eight; three ModExp cases cover moduli three/fifteen and
widths two/four/eight. All eight scale cases also retain the complete bound
record, including instruction order, parameters, classical feedback, output
aliases and workspace events. Existing source and absolute-amplitude evidence
for these identical artifacts is reused. New explicit-import tests cover name
conflicts, failed imports, no-call ownership and every small coherent measurement
branch. The original import test file is retained unchanged.

The final affected regression passes 194 cases in Cirq 1.7 and 117 cases in
Cirq 1.6. Local integration passes 145 cases; the source/environment lock passes
separately. Three changed Python files pass strict typing, format, lint, owner
scope, language/links, architecture, IR and API checks. The earlier run before
restoring the existing test file is archived and is not the final regression.

The stored graph profile for exponent three, base four and modulus fifteen is:

| Result bits | Definitions | Unique structural bodies | Stored operations | Observed process RSS peak |
| --- | ---: | ---: | ---: | ---: |
| 16 | 3866 | 97 | 35770 | 259817472 bytes |
| 64 | 3866 | 97 | 106330 | 253837312 bytes |
| 256 | 3866 | 97 | 388570 | 445366272 bytes |

Structural fingerprints normalize private function and recursive call names;
all other fields remain present. These are observations of existing graph
representation, not physical primitive counts, source cost models, a benchmark
speedup claim or evidence of full-width execution. The first 2048-bit attempt
reached its 90-second observation limit with peak observed RSS 632569856 bytes;
it was terminated and reaped. A separate extended-budget attempt is recorded
independently. No complete 2048-bit graph or literal physical stream is claimed.
The next work must distinguish stored graph completion, physical stream
expansion and numerical simulation instead of treating them as one result.

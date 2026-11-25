# Hamming constructor and exact inverse context

The complete locked source is `arithmetic/hamming_weight.py` from Qualtran
v0.7.0, commit `bcb17bb25360ed68ce3bc084645db3e929799389`. Existing file audits
assign its full signature, widths, compression, decomposition and model to
native owners. This batch completes its previously missing outer Adjoint
constructor dependency using the already-accepted native exact inverse.
The reference implementation and its original failures remain unchanged.

## Interface and ownership

The source has one `bitsize` field. Forward ports are MSB-first unsigned THRU
`x`, opaque RIGHT `junk`, and unsigned RIGHT `out`. Inverse retains the same
ordered descriptors, consumes `junk` and `out` as LEFT ports, and restores `x`.
The caller must supply a valid computed image to the inverse; it is not an
arbitrary-input projection. Junk and count are owned logical outputs until
inverse consumption. Internal And cleanup retains existing measurement and
feedback semantics; no measurement instruction is reversed mechanically.

At width one, the zero-width junk word is still an explicit logical output
and input, even though its physical wire tuple is empty. Forward and inverse
each emit one CX. The inverse consumes that empty ownership value and restores
the count output to zero. No zero-width register is silently removed from the
signature, and no extra physical carrier is introduced.

Compiler Formats owns constructor dispatch and its compilation-scoped recipe
map. Register Hamming, compression, And, view, adjoint and binding owners remain
authoritative. The table records complete resolved FunctionRecords produced by
this owner, together with original width and direction. Adjoint lookup uses
those exact records, never a guessed callee name or matching signature.
Nested inverses toggle the exact existing recipe. Unknown children either
raise a context error or delegate to the explicitly supplied inverse owner.
Separate tables do not share inverse authorization.

Concrete positive widths call existing `hamming_weight_program` with the
requested direction. Zero and symbolic widths retain declarations and explicit
physical-body refusal. Requested inverse descriptors are checked against the
complete child's adjoint signature. The original symbolic-integer wire-loss
refusal remains intact. There is no new opcode, protected Core behavior,
backend choice, numerical execution or public user entry. This is a
`single_device_fast_path` constructor change; ordinary `circuit.append(op)`
continues through the existing native Program binding.

## Actual original metadata and exact execution

Before this batch, actual depth-zero original inverse libraries at widths one,
two and three failed with `No native factory for qualtran._infra.adjoint.Adjoint`.
They now resolve to the native exact inverse. Source-success original metadata
is tested at depths zero and one for widths two, three, five and eight. Width
one is tested at depth zero: the original generic depth-one empty-junk
construction failure is independently retained rather than counted as an
unexecuted passing case.

Native binding matches the accepted independent direct inverse Program,
including complete instruction order, operands, feedback, output ownership
and resources. Source model measurement counts match actual inverse measures;
opaque matrix instructions are absent. Double inverse returns the forward
body. Unknown same-signature bodies, a different compilation context and
incompatible inverse descriptors are refused. Explicit delegation and actual
native compilation/append with Qualtran and Cirq imports blocked both pass.

For widths one through four, actual original forward and inverse constructors
are resolved separately, bound, appended and copied through CircuitIR JSON.
A complex coherent input entangled with a spectator is restored on every raw
measurement history, retaining absolute phase. All count and junk work is zero
before inverse consumption. Wider actual streams reuse the already-accepted
Hamming component certificates instead of repeating exponential history grids.
Actual emitted instruction inventories remain separate from QEC model units.

## Evidence and retained boundaries

Concentrated constructor, direct component, foreign-object and Hamming-phase
regression: 119 passed. Independent Cirq 1.6 scope: 119 passed. Local integration:
145 passed. Complete changed-file strict typing, Black/Ruff, Compiler scope,
language, links, architecture and IR/API checks are recorded with receipts
`g2-hamming-inverse-constructor-*20261125` in the reference workspace. Original
source/model and primitive evidence is reused; no failed source is modified.

The one-bit original generic Bloq adjoint/SDK and width-query failures remain
original failures; the new native metadata route does not claim they pass.
Source Sympy Integer(5) serialization still loses the distinction that changes
junk width four to three. The native importer preserves the encoded signature
and rejects that inconsistent constructor. Direct symbolic-integer metadata
remains declaration-only. Q-0198 and its affected phase parents therefore remain
open under their existing complete-representation criteria; no catalogue or
whole-file count is inflated by this constructor dependency delivery.

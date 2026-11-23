# Proposal 120 — Symbolic array shapes in logical Program declarations

## Review status

Approved explicitly by the user on 2026-11-23 for implementation. The current Program verifier requires concrete
nonnegative integer array dimensions even for functions without a body. Proposal
103 records symbolic shape refusal separately from symbolic word-width metadata.
Proposals 107 through 115 do not change this array declaration contract.

## Requirement and frozen evidence

The unchanged Qualtran v0.7.0 revision is
`bcb17bb25360ed68ce3bc084645db3e929799389`. Its complete bookkeeping/partition.py
source admits Partition(n, (Register("part", QAny(1), shape=(n,)),)) for a symbolic
integer n. The original serialized register and constructor metadata retain n.
FlagQuantum already decodes that metadata and can construct the existing
partition signature. Reconstruction as a bodyless FunctionRecord fails in
Core with `executable array dimensions must be nonnegative integers`.

The fresh constructor witness is
`tests/unit/test_legacy_partition_factories.py` and its frozen intake receipt
`ftqc-reference-env/g2-partition-predicate-constructor-intake-20261123.json`.
The existing native partition signatures and symbolic width declarations remain
available. No reference source is modified.

## Proposed contract

Permit existing symbolic array shape expressions in logical register signatures
of bodyless Program functions. Preserve the original Register metadata, dtype,
side, array rank and ordered shape expressions. Reuse FunctionRecord,
ProgramRecord, register codecs and symbolic expressions. No new public entry,
record field, opcode, serialization vocabulary or numerical representation is
introduced.

Logical signature validation must validate each dtype and register side without
trying to enumerate an array whose extent is symbolic. Concrete dimensions
retain their current nonnegative integer validation. Existing name-conflict and
canonical dtype validation still apply. Mixed concrete/symbolic dimensions retain
their order. Symbolic numeric expressions follow the existing symbolic metadata
classification, including expressions without free symbols.

Finite SSA bodies, calls, connection enumeration, input/output binding and
numerical execution continue to require concrete element counts. They must fail
explicitly on unresolved symbolic shapes through the existing owning checks.
There is no implicit shape substitution, sampling, guessed element count or
zero-length interpretation. Existing symbolic widths remain logical metadata
under their current contract.

Core owns this logical declaration validation and finite execution boundary.
Compiler formats reuses the resulting bodyless declaration for the original
Partition constructor. Simulation, Runtime and Ecosystem gain no new numerical
responsibility. The normal consumer remains circuit.append(op) after concrete
register binding.

## Acceptance and compatibility

Execute the untouched original Partition symbolic-shape construction and wire
round trip, then the same native logical declaration and Program verification.
Cover scalar, multidimensional and mixed concrete/symbolic shapes, both
partition directions and dtype/side preservation. Check equivalent logical
signatures in structured predicate declarations where original serialization
exists. Preserve original Shaped and HasLength codec failures separately.

Verify explicit refusal of unresolved finite bodies, calls and physical binding.
Retain malformed concrete shapes, invalid dtypes, duplicate register names and
incompatible sides as failures. Reuse existing concrete partition, zero-width,
complex amplitude, alias ownership, cleanup and resource regressions. Run Core
and Compiler scope checks, strict typing, formatting, architecture, IR and public
API checks. Update this proposal and its contract evidence only after approval.

Core/AGENTS.md and PUBLIC_API_PROTECTION.md require explicit approval before
changing the current accepted Program declaration behavior. The approved declaration boundary is implemented independently of the
Compiler constructor adapters.

## Implementation evidence

The Core validator separates logical signature validation from finite port
enumeration. Tests in tests/unit/test_symbolic_array_declarations.py retain
symbolic multidimensional metadata and explicitly reject its use in bodies and
calls. Partition wire round trips and physical binding refusals are covered by
tests/unit/test_legacy_partition_factories.py. The complete dependency batch is
recorded in RESOLVER_PARTITION_PREDICATE_AUDIT.md.

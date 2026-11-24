# Proposal 121: fixed-width quantum storage with a symbolic unsigned bound

Status: pending explicit user approval. No protected implementation is changed.

## Observed requirement

Unchanged Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389` executes
`TopLeftRowColumnOracle(2, Symbol("s", positive=True, integer=True))` as the
complete 16-by-16 identity. The two quantum index words have concrete width
two; only the ordinal's iteration bound is symbolic. Its source signature
retains LEFT BQUInt(2,s), RIGHT QUInt(2) and THRU QUInt(2).

Native constructor resolution now preserves that exact signature, including
recovery of the private sparsity field from its explicit bounded ordinal
metadata. Program verification succeeds. Physical binding still rejects the
descriptor because `executable_register_width` requires every parameter value
to be a built-in integer. The observed diagnostic is `unsupported executable
register descriptor` for uint width two with parameter bound=s.

Evidence beside the worktree:
`ftqc-reference-env/g2-sparse-matrix-symbolic-literal-probe-20261125.log` and
`tests/unit/test_sparse_matrix_component.py::test_fixed_width_symbolic_ordinal_bound_retains_physical_storage_gap`.
The original source/test blobs are
`cacbdab733f0be4f0f28bf2a749e1bad79987bfd` and
`afcb0771b405ed26bfce7827d70d81c3d7f7a0d8`.

## Proposed contract

Admit existing canonical quantum uint descriptors whose quantum width is a
positive built-in integer, classical width is zero, and sole parameter is an
existing logical symbolic bound expression. Validate the descriptor through
the existing logical type owner. Preserve the expression, assumptions, side,
name, width, parameter order and serialized identity. Do not replace it with
2**width or discard the bound.

The quantum storage uses the concrete width. This does not resolve, evaluate,
sample, assume a value for or enforce a numeric iteration bound. Algorithms
that need a concrete iteration count keep their existing checks. Numeric value
encoding/decoding, classical descriptors, classical input/output validation,
control-literal interpretation and symbolic physical widths gain no admission.
Existing behavior for concrete bounded words is unchanged.

Symbolic widths, symbolic shapes, nonpositive widths in this new profile,
nonzero classical storage, extra/unknown parameters, symbolic modular or field
metadata, symbolic fixed-point fractions, malformed bounds and noncanonical
descriptors retain their existing failures. This does not expand other dtype
profiles or add a fallback interpretation.

Reuse RegisterType, Signature, FunctionRecord, ProgramRecord, existing register
binding, Split/Join and primitive lowering. Add no public entry, field, enum,
opcode, wire encoding, identity type or numerical representation. The user
workflow remains `circuit.append(op)` after ordinary concrete-wire binding.

Core owns executable descriptor validation. Compiler reuses that profile;
Runtime and Simulation receive existing finite wire operations. No external
framework object crosses the boundary.

## Acceptance and compatibility

- Compare the unchanged source and native sided identity signatures and every
  complex operator column for the fixed-width symbolic-bound counterexample.
- Cover quantum storage binding, Split/Join, primitive calls, exact inverse,
  bystanders, JSON identity and ordinary append. Retain zero private work and
  zero emitted gates for TopLeft identity.
- Prove that a genuinely symbolic sparse iteration count still prevents
  physical SparseMatrix decomposition, and that numeric codecs and classical
  symbolic-bound inputs retain their prior checks.
- Keep concrete bounded-word, malformed parameter, width, side, stale-handle,
  workspace, amplitude, source graph and resource regressions unchanged.
- Run Core/Compiler/Integration scopes, strict typing, formatting, language,
  architecture, IR and protected API checks. Update the approved contract and
  the currently explicit refusal test only after approval.

Core/AGENTS.md requires an integration contract before protected semantics
change. Root AGENTS.md and PUBLIC_API_PROTECTION.md require explicit user
authorization. Proposal 103 separates symbolic metadata from executable
admission; Proposal 120 only approved logical symbolic array declarations.
Neither authorizes this additional physical storage profile.

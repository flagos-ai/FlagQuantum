# Proposal 126 — Symbolic quantum field declarations

Status: explicitly approved by the user on 2026-11-27. Implementation and
acceptance remain tracked separately.

## Verified need

Frozen Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389` accepts logical identity graphs
with quantum finite-field and packed polynomial signatures whose characteristic,
field degree or polynomial degree is symbolic. Six unchanged-source probes pass
the original graph verifier; all six equivalent native Program declarations fail
with `program.register_type`. Evidence is
`ftqc-reference-env/g2-symbolic-quantum-field-declarations-intake-20261125.json`.

Examples include QGF(2,m), QGF(p,2), QGF(p,m) and polynomial declarations over
these fields. Native descriptors already preserve their source widths:
`m`, `ceiling(log2(p**2))`, `ceiling(log2(p**m))`, and coefficient extent times
the field width. The gap is that `logical_register_width` delegates directly
to physical execution admission for both field encodings.

## Narrow contract

Accept canonical symbolic quantum field and polynomial descriptors in logical
Program declarations, using the existing encodings and parameter vocabulary.
Preserve exact characteristic, degree, order, polynomial extent and width
expressions. Concrete parameters retain current validation; malformed metadata,
booleans, invalid concrete extents and inconsistent order/width remain refused.
Symbolic declarations express unresolved requirements, not physical feasibility.

Physical binding, allocation, primitive expansion and execution continue to
require concrete admitted widths and parameters. Proposal 125's concrete raw
storage scope is unchanged. No numeric field arithmetic, control predicate,
classical descriptor, codec or foreign object constructor is expanded.
GFPolySplit/Join retain their original symbolic polynomial-degree constructor
refusal; logical type declarations do not manufacture an executable view.

Core owns neutral declaration validation. Compiler retains concrete binding and
transforms. Simulation retains numerical codecs. No new dtype, opcode, schema
field or public entry point is introduced; concrete use remains
`circuit.append(op)`. Existing concrete Programs remain compatible.

## Acceptance after approval

* Match all six unchanged-source logical signature/graph successes through the
  native Program verifier, preserving exact widths and metadata.
* Exercise typed logical call/identity composition and existing record round
  trips where applicable; keep original QGF/QGFPoly codec failures separate.
* Preserve malformed/concrete-invalid descriptor refusals and prove that each
  unresolved profile still fails physical binding before numerical execution.
* Regress concrete field storage, binary arithmetic/control and polynomial
  views; reuse accepted complete-state and workspace evidence.
* Pass strict typing, format, scope, architecture, IR and protected API checks;
  synchronize all ledgers without prematurely closing implementation parents.

Root AGENTS.md rule 8 requires explicit authorization for accepted public type
and failure-behavior changes. Core/AGENTS.md states: "Changes to protected
artifacts, IR, public schemas, or Stable Core APIs require an integration
contract/ADR change before implementation." PUBLIC_API_PROTECTION.md protects
accepted public types and failure behavior. Approved Proposal 125 explicitly
retains symbolic physical refusal; Proposal 120 covers array shapes, not these
field parameter declarations. Neither approves this distinct logical extension.


## Implementation and evidence

Core field validation now has one focused owner,
`core/ir/program/field_register_validation.py`. Its concrete admission body is
moved unchanged from the shared register verifier. Logical field declarations
validate canonical parameter order, integral/symbolic dimensions, concrete
minimums, exact order and exact packed width. Physical and classical admission
still use the unchanged concrete validator. No foreign or numerical type enters
Core and no Program field or opcode is added.

All six original identity graphs and native equivalents pass. Native declarations
and nested typed identity calls cover scalar, shaped and empty arrays. Every
unresolved profile still fails physical binding; classical symbolic field ports
remain refused. Malformed order, width, storage, duplicate metadata, boolean
and invalid concrete dimensions retain rejection. The focused field/previous
classical-cast regression passes 164 cases; strict typing passes three changed
files. Physical primitive inventories and complete complex-state evidence for
concrete fields are reused because that validation body and all execution
owners are unchanged. Unresolved declarations have no physical gate or amplitude
execution claim.


The combined 124/126 acceptance batch passes 665 concentrated cases and 32
local integration cases. Five changed Python files pass strict typing; Black,
Ruff, scope, language/links, architecture, IR/public API, required-check mapping
and whitespace checks pass. The moved concrete field validator is AST-identical
to its published body. Three additional frozen field/OnEach source blobs are
verified; earlier Cast/QROAM source evidence is retained. Receipts are
`ftqc-reference-env/g2-storage-declarations-*` outside the repository.

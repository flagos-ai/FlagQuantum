# Hubbard packed-oracle constructor dependency ownership

The complete frozen state_preparation/black_box_prepare.py and
multiplexers/black_box_select.py source/test pairs retain their
[full constructor disposition](RESOLVER_PACKED_ORACLE_AUDIT.md). Their Hubbard
PREPARE/SELECT direct dependencies retain their existing shared signature,
preparation, selection, control and exact-adjoint owners.

The original scoped packed/Hubbard context is moved into Compiler formats under
legacy_hubbard_packed_context.py. It retains the same selection/junk/target
roles, deferred wrapper materialization, exact directional pairs and positive
or zero specialized-control adders. No duplicate registry is added. Contexts
remain local to a single library compilation. Ecosystem's existing entry points
now inject the coefficient converter and delegate Program parsing. Numerical
angles remain with the existing authoritative helper; Compiler receives only
owned finite angles or the existing symbolic declaration signal.

SELECT no longer constructs Algorithms PrepareHubbard or SelectHubbard objects
to obtain metadata or Programs. It reuses the same Compiler
hubbard_preparation_signature and hubbard_select_program owners that those
Algorithms entries already delegate to. Row-major packed words, MSB bit order,
original QAny property roles, specialized scalar carrier and source singleton
refusals remain unchanged. Logical junk/system words remain caller-owned;
private child work is cleaned by existing exact source-derived dependencies.

Before edits, 22 complete Program/bound-IR or original-failure records were
captured from published 9cff2b609. They cover both original packed examples and
bare Hubbard PREPARE/SELECT at dimensions one/two/three, both SELECT control
presence settings and both requested directions. Bare SELECT adjoint source
serialization fails on SpecializeOnCtrlBit enum values; those original failures
are recorded separately, not omitted or called successful Programs. Native
singleton body refusal also remains explicit. Replacement must match every
complete Program, primitive, operand, feedback, parameter, lifetime, output alias
and recorded refusal. Existing full complex/measurement-history/work-zero
proofs are reused only after this exact comparison.

Dedicated tests replace coefficient conversion on both original packed source
examples in both directions. Resolution with source/SDK/Algorithms/Simulation
imports blocked produces exactly the existing Program. SELECT does not invoke
coefficient conversion. Existing block-encoding consumers exercise nested role
materialization and both source control polarities through the same context.
No backend, numerical kernel, protected Core behavior or frozen source changes.

Primitive inventories remain separate from source resource estimates. The
existing four directional packed resource rows are rerun, including physical
lease peaks versus the entire supplied pool. The source controlled packed
SELECT omitted-control failure, enum codec defect, symbolic physical weights,
unknown native role owners and broader scalar boundaries remain explicit.
An ownership migration does not add a complete-parent acceptance count.

## Exact CY scalar/array source view

The Cirq 1.6 concentrated profile exposed the same pre-existing mismatch in
published 9cff2b609: source CYGate uses scalar ctrl/target ports, while native
Cirq-wrapper-derived leaves retain a two-element q array. The unchanged published
owner reproduces both packed SELECT failures. No production gate changed.

The comparison now uses an explicit bijective port view only when the actual
source child is CYGate and the complete native leaf equals the existing canonical
bit_gate_function for one CY on ordered wires (0,1). Every function field, gate,
parameter and condition is included in that equality. Scalar inputs and outputs
are viewed as the q array; every successor producer identity is mapped by the
same indexed bijection. All calls, operands, outputs and later connections are
still compared. Neither the frozen reference nor its objects are rewritten.

Negative fixtures prove that CX substitution, reversed CY operands and a wrong
following H operand are refused. This is not a numerical-unitary shortcut or a
skipped recursive subtree. The initial full Cirq 1.7 profile passed 265 cases.
The initial Cirq 1.6 profile had 219 passes and 46 failures at this source view;
that log is retained as superseded evidence, not called a passing full profile.
Final concentrated checks rerun all 46 failing nodes and both general trace
helpers, plus current constructor consumers in Cirq 1.7. Unchanged passing
physical/branch evidence is reused with the 22-record exact identity proof.

Final verification passes 117 repaired-node/general-helper cases in Cirq 1.6,
102 current consumer/general-helper cases in Cirq 1.7 and 145 local integration
cases. The earlier 265-pass Cirq 1.7 profile and unchanged passing numerical
branches are reused with the complete binding identity proof. After adding
optional NumPy collection guards, all thirteen CY-view/packed cases pass again
without skips. Strict typing covers six Python files; Black/Ruff, owner scopes,
English/links, architecture, IR/API and diff checks pass. Eight frozen source
blobs and real Kickmix 0.1 Simulator are verified. Repository dependency policy
remains nonzero at 733 findings, with none on this batch; two pre-existing
NumPy collection findings on touched helpers are removed. No repository-wide
dependency-policy pass is claimed.

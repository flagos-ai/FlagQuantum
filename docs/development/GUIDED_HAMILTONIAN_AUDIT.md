# Guided Hamiltonian source declarations

Qualtran v0.7.0 at `bcb17bb25360ed68ce3bc084645db3e929799389`
exports GuidedHamiltonian and GuidedHamiltonianPhaseEstimation from
`hamiltonian_simulation/__init__.py`. Native Algorithms owns the two
declarations in `guided_hamiltonian_resource_model.py`. Each uses a
caller-supplied block and guide shape. The block resource width must be
zero, and the guide selection and block system widths must agree.

The through register order is phase estimate, system, walk ancilla, and
guide ancilla. The phase word is unsigned binary fixed point with every
bit fractional. The others are opaque quantum words of the declared
widths. As in the source signature constructor, zero-width ports are
omitted. The declaration introduces no physical bindings or work slots.

## Source and native checks

- Both symbolic factories agree in every port, precision, failure
  probability, Kaiser parameters, and direct invocation multiplicity.
  The independent phase model also matches at two numerical parameter
  pairs. The amplification graph retains the initial phase estimate,
  both negative-phase reflection scopes, and the forward and inverse
  phase estimates in each modeled round.
- The nested qubitization QPE comparison includes the forward window,
  positive controlled walk, zero-controlled reflection, bare walk,
  and inverse textbook QFT with reversal. The adjoint uses the same
  self-adjoint reflection and the forward QFT.
- The native 18-cell notebook retains both symbolic examples, their
  signatures, and one-layer call graphs. Its independent kernel ran all
  nine code cells and produced four SVGs with reference imports blocked.
- The unmodified source suite ran with four passes, six skips, and two
  expected failures. Both source notebooks ran in their own kernels.
  The native call-graph, phase-estimation and qubitization regression
  passed 334 tests.

## Original closure review (superseded below)

Q-0535 and Q-0536 close the complete source initializer and module
notebook, respectively. Q-0537 remains open for the complete k-XOR
oracle and resource-leaf dependency chain. Its two modeled roots, native
signatures, and guided/QPE graph edges are now covered. Both original
roots raise DecomposeNotImplementedError. Their call and multiplicity
counts are resource declarations. There are no root-level emitted
primitive counts, statevectors, work-zero results, or measurement
trajectories to compare. Existing executable operations keep their own
Circuit.append entry points.

Q-0538 remains open for the native tutorial counterpart (Q-0542)
and source fixture serialization. The locked suite explicitly skips
decomposition-dependent count and dtype tests and expects serialization
failure for examples using KXorInstance. The source tutorial has been
recovered by blob and executes independently.

## Guided query-cost tutorial (Q-0542)

The native tutorial retains all 20 original cell positions, both oracle
signatures, the guide, and the amplified and standalone phase-estimation
resource tables. Its helper extends the existing guided call graph with
tutorial-specific generalized leaves. It keeps the position oracle
and its inverse distinct, for both bare and controlled calls. The
entry oracle is self-adjoint. The three identity preparations retain
their walk, guide, and whole-signature scopes.

The locked tutorial's 14 amplified and 10 phase-only leaves each agree
symbolically with the native exact tables and the separately computed
upper bounds. The rewrite uses the source's temporary integer-width
substitution before replacing the power-of-two window budget. This
preserves the simplifications in nested floor and logarithm
expressions. Controlled query counts remain exact. If the amplification
count is r, the controlled entry count is 4r+2 and the controlled
position counts are 2r+2 forward and 2r inverse.

The native tutorial independently executes 10 code cells and renders the
four-port guided signature. The three output tables contain 14 exact,
14 bounded, and 10 bounded phase-only rows. The original tutorial
kernel result is recorded above. Its oracles are declarations and emit
no primitives. Q-0538 now has both native notebook counterparts and
remains open for the original KXorInstance serialization expected
failures. Q-0537 still requires the distinct k-XOR oracle dependency
chain.

## Complete-file review under the approved faithful-migration rule

The current source owners are
`flagquantum/algorithms/primitives/guided_hamiltonian_resource_model.py` and
`examples/algorithms/guided_hamiltonian_models.py`. The latter now constructs
both original examples through native KXorInstance, GuidingState,
KikuchiHamiltonian and PackedPrepareDeclaration. The notebook consumes these
factories instead of copying register-size formulas. This closes the actual
example dependency-binding gap; it does not supply an unavailable oracle body.

| Complete original responsibility | Native owner and evidence |
| --- | --- |
| Phase-estimation fields, resource/system constraints and ordered signature | GuidedPhaseEstimationModel; source factory and numerical parameter comparisons |
| Walk, Kaiser parameters and QPE dependency | Existing qubitization/window owners; exact nested forward and adjoint call declarations, zero-controlled reflection and reversed QFT |
| Amplified root fields, eager signature, precision, failure probability and round count | GuidedHamiltonianModel; unchanged symbolic identities and constructor-refusal checks |
| Guide-ancilla and whole-signature negative reflections | Distinct scoped graph leaves; source phase, reflection signature and inverse identities checked |
| Root invocation inventories | Exact original Counter multiplicities including the initial QPE, both QPE directions and both reflections |
| Both symbolic factories and DocSpecs | Native factories and their notebook positions; guide/solve counts, block scale, entry precision, selection and junk widths are obtained from actual native dependency objects |
| Entire original test file | Both example protocols, both complete notebook counterparts and their original source fixtures; expected source skips and codec failures retained |

The source's third positional KikuchiHamiltonian argument is entry precision,
although the local variable is named `s`. The native factories preserve that
argument and leave sparsity at the original instance-derived default. The new
test checks both values separately; a hand-written shape-only example did not
exercise that distinction.

The applicable direct dependency chain is already owned by the separate
k-XOR modules: classical instance preprocessing; guide preparation/sorting/
duplicate/uncompute declarations; adjacency/index/RHS queries; sparse block
encoding; QPE, Kaiser, QFT and prepared reflections. Concentrated regression
exercises these actual owners, their source graph/cost comparisons and the
available typed/physical child paths. Their wider source parents keep independent
acceptance reviews. This root review does not certify an entire physical guide,
an arbitrary black-box oracle, or execution of an absent source decomposition.

Under the explicitly approved
[faithful-migration closure rule](G2_FAITHFUL_MIGRATION_CLOSURE_PROPOSAL.md),
Q-0537 and Q-0538 can close after publication and mandatory checks. The source
roots explicitly lack physical bodies, so their unavailable statevector,
measurement-history and clean-work experiments are inapplicable; no identity
substitute is used. The source KXorInstance serialization xfails remain original
codec limitations, rather than missing native numerical implementations.

### Resource and execution limits

| Layer | Actual primitives and workspace | Model evidence |
| --- | --- | --- |
| Both guided roots | No source-defined emitted circuit; no primitive total or private-work certificate claimed | Complete direct guided/QPE invocation graph, forward and adjoint |
| Query-cost tutorial | Declarative oracles; no emitted primitive inventory | All 14 amplified and 10 phase-only leaves, exact and separately bounded tables |
| Native k-XOR dependencies | Existing available child streams and applicable measurement/complex-state/work evidence are reused | Existing source query, control and adjoint categories remain separate from physical inventories |

The full guided physical algorithm, generic source-controlled adjoint failures,
missing oracle bodies and source transport defects remain explicit follow-ups.
They are not repaired by changing the locked source. Existing executable
operations continue to use `circuit.append(op)`.

Fresh closure evidence verifies 15 immutable source/direct-dependency blobs.
The original non-notebook protocols report 2 passes, 6 source-defined skips and
2 serialization xfails; the two original notebook kernel receipts are reused
at unchanged blobs. Both native notebooks execute independently with Qualtran
and Kickmix imports blocked: 9 and 10 code cells, respectively, and four and
one actual SVG outputs. No dependency installation or protected API change is
required by this batch.

The final batch passes 149 concentrated dependency cases, 32 local integration
cases and 32 direct consumers under the independent Cirq 1.6.1 profile. Both
changed Python files pass strict typing, Black and Ruff. Algorithms scope,
repository language, local links, architecture, multi-level IR, required-check
mapping, public API baseline and whitespace checks pass. The external receipts
use prefix `g2-guided-closure` and retain the mandatory actual Qualtran 0.7.0 /
Kickmix 0.1 Simulator preflight; no optional native comparison is silently skipped.

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

## Closure and boundaries

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

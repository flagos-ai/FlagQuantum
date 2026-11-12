# Classical-vector construction source audit

This audit covers all three functions of the unchanged Qualtran v0.7.0
`qualtran/simulation/xcheck_classical_quimb.py`, frozen at
`bcb17bb25360ed68ce3bc084645db3e929799389`. Q-0898 remains open for the
executable-admission and foreign-protocol boundaries below. The already accepted
notebook and five-function test asset are documented in
[CLASSICAL_TENSOR_NOTEBOOK.md](CLASSICAL_TENSOR_NOTEBOOK.md).

## Ownership and interface

| Source responsibility | Native owner and contract |
| --- | --- |
| `_add_classical_kets` | Compiler `component_classical_vectors.py`: traverse signature inputs and define/call integer-state functions |
| `_add_classical_bras` | Same Compiler owner: traverse outputs and define/call integer-effect functions |
| `flank_with_classical_vectors` | `flank_component_with_classical_vectors`: states, one child call, optional evaluator, effects, finalize |
| `IntState` / `IntEffect` imports | Compiler `component_basis_vectors.py`, existing canonical owned-word state/effect functions |
| `BloqBuilder`, signatures, registers and soquets | Existing native component builder and Core typed registers/bindings; no foreign object in IR |
| `as_composite_bloq().on_classical_vals` | Explicit caller evaluator; Simulation owns numerical evaluation |
| NumPy array operations | Existing local row-major array traversal; no broadcasting |
| Type-only classical values and `cast` | Native mapping types and concrete integer admission; no runtime protocol discovery |

The source has no classes, constants, registries or import-time execution beyond
imports and definitions. Imports local to each helper remain implementation
dependencies, not new public exports. Compiler constructs programs; Simulation
contracts their tensors. No backend selection or numerical kernel is added to
Compiler. Unsigned basis indices use MSB-first words; shaped registers traverse
row-major. State/effect functions own their words, the child retains its existing
workspace ownership, and physical binding uses existing clean allocation/release.
Users still append the resulting operation to their circuit.

## Actual correction and failure order

For a shaped output, source first converts values with `asarray`, looks up its
binding, and rejects a scalar soquet before iterating. Native now performs the
same lookup and scalar-binding rejection before defining effects or consuming
words. Previously a nonempty array could consume the first scalar word before
failing, and an empty array silently accepted it. Four shapes, including two
empty shapes, test missing and scalar bindings against the original helper and
prove that the original scalar remains available after refusal.

Short value arrays preserve the successful prefix in both implementations:
ket/effect traversal adds the first element and fails indexing the next one.
Neither path promises rollback. Scalar, vector, matrix and empty-array wrappers
check states/child/effects call order. Explicit outputs bypass the evaluator;
automatic outputs invoke it once with the original mapping object.

## Retained boundaries

- Source can construct `IntState` with float, symbolic or NumPy boolean values,
  zero width, or symbolic width. Five tests prove construction acceptance only;
  they do not claim source contraction/execution succeeds. Native executable
  loaders continue to reject those cases. This audit does not relax Core width
  or integer admission.
- Automatic classical outputs of the source empty-array identity fail with
  `KeyError` at `RightDangle`; native explicit evaluation succeeds. Explicit
  empty-array outputs construct on both sides. The source is not patched.
- Native evaluator injection replaces foreign composite/protocol discovery.
  Symbolic/classical/foreign dtype support and full diagnostic equivalence remain
  open. Native validation and function definition have their own observable
  failure boundary; this is not a claim of identical builder internals.
- Numerical wrong-output overlap is zero; physical effects express clean-release
  promises. These different consumer contracts remain explicit.

## Amplitudes, workspace and resources

The regression reuses the existing measured Add consumer: all 64 inputs and all
16 measurement histories (1,024 branches), including a spectator and an initial
imaginary amplitude. Every unnormalized branch equals the initial vector divided
by four; all consumed/work bits return to zero. Eight ordinary append trajectories
also finish at zero. Existing H/S shaped-vector contractions and gradients remain
covered. No replacement numerical engine or test-only production dispatcher is
introduced.

| Accounting layer | Verified result |
| --- | --- |
| This graph-construction correction | No added primitive gates, measurements, feedback or workspace |
| Physical vector wrapper | Existing child instruction counts plus X popcount of input/output values (0–11 in the Add fixture) |
| Unchanged source QEC wrapper model | Six additional Clifford units; all other model fields unchanged |
| Numerical vector graph | States/effects and contraction nodes; not physical gate counts |

Actual physical instruction counts and source analytical QEC costs are separate
quantities. The correction changes invalid construction behavior, not a supported
circuit's gate stream. Q-0898 is not marked closed by passing this bounded audit.

## Local acceptance

Implementation commit `405f8802` adds 25 source-boundary cases. The concentrated
run passes 2,365 tests in 45.31 seconds (27 retained warnings), including the
1,024 measured Add branches. CPU/tensor-network integration passes 84 tests in
5.60 seconds (two retained warnings). Strict typing covers both changed Python
files; Black, Ruff, Compiler scope, architecture, multi-level IR and protected API
checks pass. The touched helper has 73 statements and 30 branches, all covered.
The measurement-uncomputation example also runs successfully. Six relevant
source blobs, including the implementation and original test file, match the
frozen inventory. Evidence logs use `g2-classical-vector-source-` in the reference
workspace. The historical default-suite timeout remains open and was not rerun.

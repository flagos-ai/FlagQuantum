# Block encoding interface source files

Reference: Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`.
The frozen files under `qualtran/bloqs/block_encoding/` are Apache-2.0,
copyright Google LLC.

| Item | Source file | Frozen Git blob | Disposition |
| --- | --- | --- | --- |
| Q-0275 | `__init__.py` | `03c07b29829d3d6322c90faef4508f404e44cef1` | Closed. All twelve source exports have verified native destinations. |
| Q-0276 | `block_encoding.ipynb` | `b3f096d4123dd67e39f2d051b98256e868981e4c` | Closed by the native interface tutorial. |
| Q-0277 | `block_encoding_base.py` | `8e6acb38828863e959c0a8543d79bebc0bda1d2f` | Closed. Concrete/symbolic metadata and actual foreign subclass replacement are verified within existing physical admission boundaries. |
| Q-0278 | `block_encoding_base_test.py` | `534c9e329fdabd21cc005f0ca8ab12148a06110e` | Closed. Its sole scenario is independent execution of the interface notebook. |

## Interface and register ownership

The source base declares six properties and inherits the signature
requirement from Bloq. The private native `_BlockDeclaration` exposes
the six properties structurally. Physical compositions use the existing
`_BlockMetadata` specialization with concrete widths and a bound signal.

| Source property | Native owner |
| --- | --- |
| `system_bitsize` | Bound block system layout, with the declared child port order. |
| `ancilla_bitsize` | Signal register bound by the selected block implementation. |
| `resource_bitsize` | Additional declared through ports, distinct from temporary child workspace. |
| `alpha`, `epsilon` | Algorithm block metadata, with symbolic scalar expressions for owners that support them. |
| `signal_state` | Existing `BlackBoxPrepare` selection/junk partition and child emitter. |

The abstract upstream base has no gate recipe. The native demonstration uses
the existing unitary wrapper and keeps the independently bound ancilla and
resource qubit untouched. The system is bound to physical wire 1. Identity
signal preparation leaves the clean ancilla on wire 0 in zero. The example
checks the full eight-by-eight complex matrix, the selected two-by-two block
after scaling, and the exact inverse obtained through the same append path.
Its forward resource model declares one Clifford and peak width three.
The forward circuit emits one S and owns no temporary work or classical slots.

The symbolic view retains system, ancilla and resource expressions and
allows declaration-only signal preparation. The packed signature builder
is shared by Unitary, Phase and both Chebyshev owners. The test compares
all three symbolic widths, normalization, error and signal width with the
locked symbolic Unitary. The actual foreign subclass and physical admission
closure are documented below.

## Source package exports

| Source export | FlagQuantum status |
| --- | --- |
| `BlockEncoding` | Structural declaration with a concrete binding specialization. |
| `LCUBlockEncoding`, `SelectBlockEncoding` | Domain-owned LCU and SELECT compositions. |
| `LinearCombination` | `LinearCombinationBlockEncoding` in Algorithms. |
| `Phase`, `Product`, `TensorProduct`, `Unitary` | Existing phase, product, tensor product and unitary block owners. |
| `ChebyshevPolynomial`, `ScaledChebyshevPolynomial` | `ChebyshevBlockEncoding` and `ScaledChebyshevBlockEncoding` in their respective algorithm modules. |
| `SparseMatrix`, `SparseMatrixHermitian` | `SparseMatrixBlockEncoding` and `HermitianSparseMatrixBlockEncoding` in their respective algorithm modules. |

The export audit checks all twelve names from the locked package against
distinct native owners and the six metadata properties. Executable cases
remain covered by their family tests and append examples.

The source notebook has four cells. The six-cell native tutorial preserves
the metadata, block projection, normalization, error, SELECT/PREPARE and
literature topics, and adds the executable circuit example. The unchanged
source and native notebooks pass independent kernels. The nine directly affected native suites pass 389 cases without skips.

```bash
jupyter nbconvert --execute --to notebook --stdout examples/notebooks/block_encoding_interface.ipynb
```

## Generic external subclass and six-property closure

The existing Ecosystem importer admits a separately constructed immutable
subclass of the unchanged source base. The fixture uses the source's attrs
field pattern and independently implements its signature, signal property,
ordered OnEach(S) body and declared call graph. It is neither a known native
class nor a source Unitary. No new runtime registry or metadata type is needed.

`test_block_interface_source_binding.py` checks the exact six abstract
properties, inherited signature requirement and empty base DocSpec, then binds
one- and two-bit systems with absent or present ancilla/resource ports. All four
actual source forward, adjoint, controlled-forward and controlled-adjoint paths
pass through `import_block_encoding` into the owned snapshots. Their native
structural declarations preserve every width, alpha, epsilon and identity
signal preparation. A canonical source Unitary with the same OnEach(S) body
replaces the custom subclass through the unchanged append consumer and produces
the same complete instruction stream, operands and phase parameters.

For each variant the physical bindings are reversed and a final wire remains
an entangled spectator. Every complex output amplitude matches the actual
source tensor contraction. Applying the other direction restores the complete
input. Native JSON restoration preserves the same complete operator. There
are no work wires, measurement histories or cleanup branches in this family.
Previously accepted measured child and native block composition tests supply
their separate branch and cleanup evidence in the concentrated regression.

| Two-system-bit fixture with one ancilla and one resource | Emitted primitives | Independent source model |
|---|---|---|
| Ordinary forward | 2 S | 2 Cliffords, peak 4 |
| Ordinary adjoint | 2 Sdg | 2 Cliffords, peak 4 |
| Positive-controlled forward or adjoint | 2 CPhase, with the source direction's signed angles | 2 rotations, peak 5 |

The model is supplied independently of the captured instructions. Omitting it
retains an explicit unavailable-cost error. Symbolic system widths can remain
in source/native declarations; the physical importer refuses unresolved
widths. Resolving the immutable source width makes it executable. An actual
second source subclass with a nonidentity signal retains its metadata but is
rejected by this zero-signal snapshot owner. Existing PREPARE/SELECT and LCU
owners remain the executable paths for their own admitted signal contracts.
The abstract base defines no universal physical recipe for such subclasses.

Q-0277 closes. All definitions in the source base are the six abstract
properties and DocSpec; there are no hidden gate or cost implementations to
transfer. The original notebook/test/export evidence is reused, and the frozen
source base is verified before the new external subclass checks. No reference
or production implementation changes are made by this evidence batch.

The concentrated interface/dependency batch passed all 189 cases without
warnings or skips. Strict typing of the changed test, Black, Ruff, Algorithms
scope, architecture, multilevel IR and protected API checks passed. Earlier
accepted block-family and notebook evidence is reused. The fixture was corrected
to use source-style slotted immutable property fields and the source `bb`
keyword before acceptance; those are test-fixture corrections.

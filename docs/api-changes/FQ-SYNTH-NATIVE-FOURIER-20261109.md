# Native Fourier transform capture

The user authorized missing native capabilities needed to migrate the five
example categories. This unreleased extension exposes `qft(target)` through
`flagquantum.synth`, reusing the existing checked `qreg.qft` operation and shared
Fourier lowering. Stable Core, serialized IR and execution contracts are unchanged;
release authorization remains false. The path is `single_device_fast_path`.

The target is initialized writable unsigned QNum or QArray[QBit] storage.
Element zero remains the least significant bit. In logical most-significant-first
order the matrix is `F[j,k] = exp(2*pi*i*j*k/2**n)/sqrt(2**n)`, including the
final bit reversal. QFT preserves the register width and uses no workspace.
A mutable gate-only helper may apply QFT to a complete register. Partial array
views are refused because the existing register operation has no view attribute.
No new view opcode or implicit copy is introduced. Uninitialized/Const storage,
wrong arity, keyword/unpacked calls and resource-budget violations fail closed.
Existing adjoint/control admission boundaries remain in force: this extension
admits a direct transform, not a new arbitrary controlled or inverse helper.

## Acceptance

- Migrate the unchanged four-bit QFT source and its complete two-code-cell
  Notebook workflow to native Python without source imports in examples.
- Compare native and original programs with an independently constructed dense
  Fourier matrix Circuit; check all columns for widths 1..4, offset wires,
  nontrivial complex coherent inputs, absolute phase and hidden zero workspace.
- Exercise whole-register helper composition and reject partial views,
  uninitialized/Const targets, malformed calls and insufficient budgets.
- Preserve all 216 original identities and frozen resource records. Track native
  program coverage separately from complete Notebook workflows.

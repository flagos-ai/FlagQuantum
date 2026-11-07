# Linear-combination coefficient preparation

G2-BE05 ports the concrete `LinearCombination.rescaled_lambd` and default
`prepare` dependency from Qualtran v0.7.0, commit
`bcb17bb25360ed68ce3bc084645db3e929799389`. The unchanged source file
`qualtran/bloqs/block_encoding/linear_combination.py` has Git blob
`a3d8aa7fafa7a6f3fcaf095826751209427cd255` in both reference locations.
The source copyright and Apache-2.0 attribution remain in the implementation.

Ownership is algorithm composition. `LinearCombinationPreparation` accepts real
coefficients, constituent block normalizations and concrete coefficient bit
precision. It retains the original NumPy absolute product, L1 normalization,
trailing zero padding to a power of two, and fixed-bit alias preprocessing.
Negative signs belong to the subsequent SELECT. Preparation never drops zeros,
reorders terms, applies signs or replaces alias sampling by amplitude rotations.

The shared alias table, BlackBoxPrepare, uniform preparation, QROM, comparison
and exact controlled-swap decompositions remain authoritative. No compiler,
simulator or runtime implementation is copied. Packed selection and logical junk
are big-endian in the shared alias layout. Forward junk is meaningful output;
only caller-owned private workspace must clear. The original adjoint restores
selection and junk when given the prepared state. Workspace may be reused after
a component call, and measurement result slots remain explicit caller inputs.

The `resource_model()` method returns the shared forward alias call graph.
It does not emit IR or run a circuit. `estimate_ftqc_resources` accepts the
component directly and optionally reports emitted IR separately. No new public
Stable Core API or serialized report schema is introduced.

Run from the repository root:

```sh
python -m examples.algorithms.linear_combination_preparation
```

The example uses `circuit.append(op)` for both directions and verifies selection
probabilities (1/4, 3/4), private scratch zero, and absolute complex restoration.

| Quantity | Original model | Emitted IR |
|---|---:|---:|
| T equivalents / T and T-dagger | 8 | 11 |
| Measurements | 1 | 1 |
| Peak logical qubits / reservation | 7 | 7 |

The original abstract model charges four T equivalents per controlled swap.
The unchanged exact Fredkin decomposition emits seven T. Comparing using seven
T per controlled swap closes the emitted count; reporting the default model as
if it were the emitted count would hide that distinction.

Validation compares 18 coefficient/normalization configurations at 1, 3 and 8
bits against the original parent, including negative coefficients and padded
row counts. Four complete source PREPARE cases cover both directions at 1 and
8 bits; primitive dependencies, operands, measurements and feedback match.
The original component hierarchy is retained while exporting gates: flattening
bookkeeping across all scopes can change temporary allocation lifetimes.
Original QEC categories and peak width are compared independently. Two small
cases enumerate every measurement history in both directions, including all
private work amplitudes, and one original full matrix checks absolute phase.
The runnable example and a refusal-to-emit model test complete the evidence.

This accepts the coefficient PREPARE dependency only. The parent linear
combination's signed SELECT and full default composition are now consumed by
G2-BE06/07; see LINEAR_COMBINATION_SELECT.md and
LINEAR_COMBINATION_BLOCK_ENCODING.md for their separate acceptance evidence.
Symbolic coefficient precision is not executable in the original default
factory. Native concrete validation rejects nonfinite totals and negative block
normalizations explicitly; those rejected inputs are not certified source paths.

The pinned factory also fails in-place NumPy division when both coefficients
and block alpha values produce an integer array. The native path retains this
boundary and its regression; use floating-point inputs as in the example.
No dtype fix is applied to the reference or advertised as unchanged parity.

# Native bounded real polynomial phases

The user authorized successive native migrations of all examples in the five
categories, then explicitly approved this batch: native real array polynomial
phases, complete ADAPT-QAOA and MaxCut programs, independent Circuit states and
216-source replay. This extends unreleased source syntax only. Classification
is `single_device_fast_path`; implementation is authorized and
`release_authorized=false`. Existing public exports, callable entry signatures,
Stable Core snapshots and historical resource baselines remain unchanged.

## Semantics and ownership

Inside `phase(expression, coefficient)`, capture `+`, `-`, unary minus, `*`, `/`
by a nonzero classical scalar and integer literal powers 0..8. Expressions may
read unsigned numeric registers or scalar elements of `QArray[QBit]`, including
bounded writable-helper views and repeated reads of the same physical bit.
Array element zero is the least significant bit; a numeric register retains its
unsigned positional weights. Real scalar/array bindings retain the existing
finite synthesis-time snapshot semantics. No host function or quantum value is
executed to interpret an expression. Entire arrays and slices are not numbers.

A basis value receives `exp(i * coefficient * expression)` with absolute global
phase retained, including constant terms. The phase never changes stored basis
values. It does not admit real/signed fresh quantum assignment, quantum division,
measurement-dependent control or a new quantum numeric encoding. The annotation
class supplies capture-only float arithmetic, subtraction, reverse subtraction,
negation and division dunder methods so these expressions type-check; calling
them as host arithmetic still raises. Existing unsigned scalar expression phases
keep their established path and resource behavior. Real expressions and native
array/helper phases use the shared sparse path.

Compiler now owns the existing Qmod Boolean-polynomial algorithm in
`_phase_polynomial.py`. Syntax callbacks inspect frontend-owned nodes; they are
not another program IR. The Qmod `phase.py` is a format adapter retaining its
original call signature, constants, signed/fractional interpretation and source
diagnostics. Native `frontend_phase.py` supplies Python AST inspection, classical
numbers, checked physical views and native diagnostics. Both reuse the existing
`qreg.control_phase`, constant `qreg.phase`, verifier and lowering. No frontend
imports another frontend, and no execution kernel or dependency is added.

## Bounds and rejection

Reuse the established 24 distinct input bits, degree eight, 256 monomials and
20,000 intermediate monomial products. Native expression depth is at most 128;
source, loop, helper and effect limits remain shared. Coefficients and final
polynomial phase angles must be finite within ±2**53. Shared-bit multiplication
uses `b*b=b`. Sparse expansion never enumerates 2**24 basis states. Large
expressions beyond these limits fail closed; the native frontend does not claim
the Qmod factor-sharing or decision-diagram fallback in this batch.

Emit zero-angle read witnesses even when algebra cancels a value, so original
initialization and ownership checks cannot disappear. Aliased writable helper
arguments, Output escapes, invalid indices and resource-budget overruns retain
their existing refusal. Whole Const numeric helpers retain the existing
verification rules. Adjoint and arbitrary coherent control around writable helpers remain
outside the existing admitted boundary; phase predicates continue using their
previous checked path.

## Runnable user example

```python
import flagquantum as fq
from flagquantum.synth import (
    Output, QArray, QBit, RX, allocate, hadamard_transform, phase, qfunc, synthesize,
)


@qfunc
def cost(gamma: float, q: QArray[QBit]) -> None:
    phase(-0.3 + 0.75*q[0]*q[1] - q[0]/2, gamma)


@qfunc
def main(params: list[float], q: Output[QArray[QBit]]) -> None:
    allocate(2, q)
    hadamard_transform(q)
    cost(params[0], q)
    for i in range(len(q)):
        RX(params[1], q[i])


program = synthesize(main, classical_inputs={"params": [0.31, -0.27]})
print(dict(program.probabilities(fq.run(program), registers=("q",))))
```

## Acceptance

- Complete eight-bit ADAPT-QAOA cost and selected mixer, and five-bit, four-layer
  MaxCut. Preserve every original coefficient, parameter, source width and layer;
  scalar native helpers may replace redundant one-element source array slices.
- Both objectives and two nonzero parameter sets per original: compare raw
  complex128 statevectors against independent cost truth tables and Circuit
  rotations, without phase alignment. Check bit order, constant phase and clean
  workspace. Compare original and native physical resources.
- Replace frontend syntax inspection behind the shared expansion boundary and
  prove conformance; keep Qmod consumers unchanged. Existing signed, fractional,
  predicate, factor and sparse-phase regressions must still pass.
- Test read cancellation, repeated bits, helper scope, aliases/escapes, unsupported
  host/quantum operations and all expansion/resource limits. Verify a sparse
  24-bit construction without full-state enumeration.
- Replay all 216 original source identities, outcomes and resource rows; keep six
  approved revisions separate. Update native quantum-program counts independently
  of complete Notebook workflows. No remote CI or PR before full acceptance.

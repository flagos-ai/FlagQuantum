# Native full-image preparation capacity proposal

Status: approved for implementation by the user on November 21, 2026.
Approval: the user replied “Approved” (translated from Chinese) to the explicit request to approve both this
proposal and the companion capacity proposal. The approval covers the bounded
changes and acceptance requirements below; it does not declare acceptance or
production release readiness.

## Verified requirement

The accepted image-edge workflow loads all 16384 real amplitudes of its original
128-by-128 image. It currently constructs that preparation using Circuit gates,
then composes the native coherent difference program. The intended user journey
declares normalized image amplitudes and the coherent differences in one qfunc.
Image processing, normalization and final finite-shot interpretation remain
classical operations. The image size and sampling budget remain unchanged.

The Compiler's internal Walsh transform has been replaced with a butterfly and
its signed-state, port-order and target-basis checks pass. A private 14-qubit
preflight retained all 32765 instructions and checked every complex amplitude
without phase alignment. This does not authorize larger public input vectors.

## Proposed compatible extension

- Fresh native `prepare_amplitudes` and `prepare_state` accept power-of-two
  vector lengths from 2 through 16384, increasing their current 256-entry cap.
  Probability normalization and signed-amplitude conventions do not change.
- Root `classical_inputs` numeric arrays accept at most 16384 entries per
  array and 16384 total bound elements, increasing the current 4096-element
  limit. Scalars count toward the aggregate budget. The existing built-in
  list/tuple checks, finite real/integer validation and immutable snapshot
  semantics remain. This bound is not dynamic quantum allocation.
- Literal source, helper expansion, helper array binding, quantum register
  width, construction nodes, instruction budgets and memory ceilings retain
  their current caps. The full image uses a root bound array, not a generated
  16384-element Python literal.
- Existing Qmod vector admission and existing native inplace/controlled
  preparation limits remain unchanged. The verifier must distinguish fresh
  allocation capacity from those existing profiles rather than broadening
  every caller by changing one shared constant.
- Export names, signatures, result fields, serialized schemas, normalization,
  preparation ownership, alias refusal and failure categories remain unchanged.
  Oversized vectors and aggregate bindings still fail before lowering.

```python
import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import Output, QArray, QBit, prepare_amplitudes, qfunc


@qfunc
def image_state(amplitudes: list[float], image: Output[QArray[QBit]]) -> None:
    prepare_amplitudes(amplitudes, 0.0, image)


# amplitudes comes from the existing full-size, normalized image loader.
# This proposed example is not supported by the current 256-entry frontend.
program = synth.synthesize(
    image_state,
    classical_inputs={"amplitudes": amplitudes},
    max_qubits=14,
    max_workspace=0,
    max_operations=35000,
    memory_limit_bytes=64 * 1024**2,
)
result = fq.run(program)
```

## Acceptance before implementation closeout

1. Check both normalized nonnegative image data and a distinct signed vector
   at 16384 entries against independent mathematical/Circuit references in
   every amplitude, with absolute error at most 1e-12 and no phase alignment.
   Include normalization, bit order and zero hidden workspace.
2. Exercise 256, 512, 4096 and 16384 accepted lengths. Reject zero, non-power
   lengths, 32768 entries, nonfinite/bool values, bad norms and excessive
   aggregate root binding before emitting gates. Retain inplace, helper and
   Qmod refusal boundaries and host-list mutation isolation.
3. Preserve max-qubit, operation, memory, Const, alias and repeated-initialization
   refusals. Do not catch a budget failure and substitute Circuit construction.
4. Replace the image Notebook's manual preparation with the supported native
   call and complete its original 128-by-128 coherent-difference, workspace,
   200000-shot sampling and independent-reference checks. Keep the previous
   Circuit implementation solely as a clearly labeled independent reference.
5. Record capture, verification, lowering, planning, execution, named readout
   and peak-memory observations separately using the existing benchmark
   observers. Keep historical resource measurements intact; do not promise an
   end-to-end speedup from the Walsh microbenchmark alone.
6. Run preparation, arithmetic/control composition, protected API and relevant
   original-source regressions locally. Record exact failures and environment
   rechecks. No per-batch remote CI or PR is required.

Approval requested: admit only the two bounded capacity increases above and
their image-workflow implementation. Subsequent parameter reuse or QSVT APIs
are separate decisions.


## Implementation outcome on November 21, 2026

The approved fresh-native/root-binding extension is implemented in `5d85ee53a`;
helper, Qmod and in-place boundaries retain their earlier limits. The example
above describes the original proposal and is now supported within the approved
ceilings. The complete original image executes native preparation and coherent
differences in `ca9a528ec`, with twenty total qubits, three work bits and 33130
logical instructions. All seventeen code cells and complete complex-state,
cleanup and 200000-draw checks pass. Focused capacity, binding and protected
contract tests preserve the specified failure and snapshot semantics.
Local observations are retained in the delivery ledger; process RSS is not the
Runtime planned-state buffer limit, and no speedup or hardware claim is made.

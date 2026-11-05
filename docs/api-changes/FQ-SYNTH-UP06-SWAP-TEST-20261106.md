# UP-06 checked Qmod swap test

The private adapter admits swap_test(state1, state2, test) for two equally sized,
nonempty, initialized, writable quantum array/register views and one fresh scalar
qbit output. Locations must be distinct within each input and disjoint across
both inputs and the test bit. The output must be unallocated: this is a fresh
Output contract, not an XOR update to an arbitrary marker. Const inputs refuse
because controlled swaps temporarily change their storage. Dropped data states
retain their physical state and are not certified zero auxiliary registers.

Reuse existing amplitude preparation, qreg.allocate, qreg.gate, qreg.pair_gate and
indexed qreg.control_x. Compose H(test), the existing three-controlled-X Fredkin
identity for each matched low-to-high input bit, and H(test). No IR opcode,
physical kernel, stable export or approximation is added. Bound existing source
expansion and actual physical width/operations/depth/memory as before.

Accept byte-identical swap_test.qmod with every original amplitude and its stated
zero preparation bound. Compare the complete complex128 state against an
independent public Circuit controlled-swap permutation and an algebraic sum and
difference of the prepared tensor products, without phase alignment. Check the
readout overlap formula, actual workspace, input bit order, basis/coherent and
complex states, equal/orthogonal states, unequal widths, aliases, Const mutation,
uninitialized inputs, fresh Output reuse/escape, helper calls and all resource
budgets. Provide a runnable one-bit native phase-and-CX qfunc, Qmod and Circuit
comparison using existing public APIs; do not imply native array preparation or
a new native swap_test export. Recheck all 216 unchanged originals, preserve
prior source/license/resource evidence and source-semantic refusals. Continue
subsequent batches without a user confirmation gate; open the single integration
PR and run full CI only after all original-corpus acceptance is complete.

# Selected component dense contraction (GRAPH79)

`simulation/tensor_network/component_dense.py` completes numerical composition
for explicitly selected component graphs. It follows the numerical portion of
locked Qualtran v0.7.0 `bloq_to_dense`, retaining FlagQuantum ownership boundaries.
The source tree remains unchanged at
bcb17bb25360ed68ce3bc084645db3e929799389.

| Original call chain | Native owner | Input responsibility |
| --- | --- | --- |
| `as_composite_bloq` and `flatten_for_tensor_contraction` | Compiler `component_tensor_expansion` | Caller explicitly selects root/composite, classifier and expansion mode |
| `cbloq_to_quimb` then `quimb_to_dense` | `contract_component_tensors` | Selected Program, Compiler visitation positions, three-argument numerical supplier |
| `cbloq_to_superquimb` then `quimb_to_dense(..., superoperator=True)` | `contract_component_superoperator` | Selected Program, same order, four-argument supplier with fresh environment labels and explicit integer trace requests |

Simulation does not import Compiler. Its two entries receive the already selected
Core Program and visitation positions. They reuse the existing wiring and
register-contraction owners; they do not introduce a graph, contract, provider
registry, executor, type-based discovery or backend selector. Separate numerical
entries preserve the established pure and mixed callback signatures without a
union callback, runtime signature inspection or hidden adapter protocol.

## Interface and failure behavior

Both entries accept `(record, order, supply, like, entry="main")` and return a
PyTorch tensor. Pure output omits empty output/input bit groups, yielding a
scalar, vector or matrix. Mixed output groups right-forward, right-backward,
left-forward and left-backward axes, yielding a scalar, density/effect matrix or
rank-four channel. Classical axes are admitted only by the mixed graph owner;
pure-state classical-port refusal remains unchanged. Arrays use row-major word
order and bits use MSB-first order.

The existing numerical suppliers still own data, phases, dtype/device and any
trainable parameters. Like provides the idle-identity prototype. No conversion
through NumPy, dtype downgrade, normalization or sampling is added. Complete
Program/order validation precedes callbacks. Missing entries, invalid modules,
non-topological/incomplete orders, supplier exceptions (including generator
failures after yielding a tensor), explicit trace failures and contraction
failures propagate without retries or fallback. Supplier exception objects retain
their original identity.

An empty numerical graph is refused by the existing register contraction owner,
with `ValueError("register contraction requires at least one tensor node")`.
This is separate from a valid scalar graph containing an explicit rank-zero
node. On the reference stack, unchanged Qualtran 0.7.0 with Quimb 1.15.0 also
fails on an empty graph in `quimb_to_dense`'s `contraction_width` call, with
`ValueError("not enough values to unpack (expected 2, got 0)")`. Both original
pure/mixed failures are preserved. The native diagnostic is explicit; this is
not exception-message parity and no empty-graph scalar-one convention is added.

Explicit scalar suppliers are tested against original GlobalPhase tensors,
including the source's phase cancellation after conjugate doubling and native
parameter gradients. The test's selected zero-port descriptor supplies numerical
semantics only; it does not claim a physical GlobalPhase lowering from an empty
body. Likewise, a LEFT zero effect is a projection, not a discard or reset.

Direct input trace effects retain GRAPH77's source-unsupported extension:
original direct Discard/DiscardQ fails with missing input KeyError. The dense
entry reproduces the complete executable Cast-wrapped reference channel while
preserving that negative source case. It does not turn this into raw-tensor or
node-count parity with the broken direct graph.

## Source-backed and physical acceptance

Forty-six new cases exercise the complete explicit root-selection/numerical
chain. Both modes either retain a custom H-H identity root or expand it to H
leaves; source supplier precedence and the complete dense tensor/channel agree.
Relative phase, complex entangled spectators and input gradients are preserved.
Directional zero-state/effect roots compare vector and density/effect results.
Phaseful H/S array graphs compare every complex matrix/channel entry, and a
zero-prepared phaseful graph compares its complete state vector. The retained
complex scalar checks supplier parameter gradients, not just input gradients.

Mixed Z/X-measurement graphs compare complete original channels in complex64 and
complex128, with results retained or discarded and with both full/partial modes.
Replacing the numerical supplier by unchanged original leaf tensors preserves
the same consumer and graph. Both classical idle inputs and all four measurement
histories are checked, totaling 64 physical branch executions across these
configurations. After restoring the X-measurement environmental basis frame,
complete complex branch amplitudes match the original instrument. Tracing each
branch environment and summing the result reproduces the new entry's complete
output density, including spectator coherences and classical-result order.
An unused clean bit remains zero. Consumed measured inputs are not claimed reset.

Actual `circuit.append(op)` instructions, operands, measurement slots and feedback
remain identical to the existing binder output. Pure H-H physical execution also
preserves spectator amplitudes and clean storage. Compiler selection and
numerical evaluation do not change physical lowering.

## Actual costs and source estimates

The numerical entry adds zero physical gates and zero work bits. Eight retained
resource scenarios separately record full/partial mode, result shape, returned
bytes, actual primitive counts and source model values:

| Scenario | Actual primitives | Original QEC model | Complex128 returned dense bytes |
| --- | --- | --- | ---: |
| Phaseful three-bit graph | H3/S2 | Clifford5/T0 | 1024 |
| Atomic MeasureZ channel | measurement1 | measurement1/T0 | 256 |
| Mixed retained results | H1/measurement2/classical entry XOR1 | measurement2/Clifford0/T0 | 1048576 |
| Mixed discarded results | H1/measurement2/classical entry XOR1 | measurement2/Clifford0/T0 | 65536 |

The source measurement abstraction does not charge the physical basis H, and
classical entry initialization is a native binding cost. Neither is hidden in a
claim of identical primitive counts. Source QubitCount is respectively 3, 1 and
4 for these scenarios; its mixed count includes a classical port, whereas caller
physical storage separately includes a spectator and unused clean bit. These
numbers describe different ownership categories.

Returned dense bytes are not peak construction, contraction or autograd memory.
Dense channels require exponentially growing storage; no capacity or scalability
claim follows. The source's Quimb `full_simplify` policy is not imported into the
existing native contraction engine.

## Existing user workflow and checks

Users continue to append their operation to the same circuit. The existing
binary-field example is executable without reference packages; source-backed
component tests require the pinned reference environment:

```bash
python -m examples.algorithms.binary_field_arithmetic
python -m pytest tests/unit/test_component_dense.py -q
```

The new internal entries are not stable public exports. G2 and whole parent-file
acceptance remain open. Automatic provider discovery, lazy foreign decomposition
error translation, pure supplier trace-token admission and source simplification
policy equivalence require separate audit. G3 is not entered.


## Verification evidence

Implementation: `442f8f39`. All 46 new source-backed cases pass. Related regression
passes 1459 tests in 17.98 s (41 existing/source warnings). Focused verification
passes 734 tests in 21.53 s (39 warnings); all 64 local integration tests pass
in 3.88 s. The new numerical owner covers all 19 statements; it has no separately
instrumented branch. Pure/mixed callback and failure behavior are verified by
scenarios rather than treating zero branch count as semantic evidence. Seventeen
related owners cover 530 statements and 206 branches at 99%, retaining the two
historical GRAPH60 profile-refusal misses.

Strict mypy passes 41 related files plus three source-following files. Black,
Ruff, Simulation scope, architecture, Program IR and public API checks pass.
Thirty-two locked source blobs match the source inventory, including the
GlobalPhase source and test file; the latter is hash-checked, not included in the
25 unchanged original tests run here. Those 25 tests pass in 1.13 s with five
existing/source warnings. Qualtran 0.7.0 and native Kickmix 0.1 Simulator were
checked; the existing binary-field append example passes. Resources and the
original empty-network failures are retained in the resource evidence.

Default smoke/unit stops at maxfail: six failed, 21485 passed, 300 skipped and
1037 warnings in 161.45 s. All six observed failure IDs belong to the independent
seven-failure baseline. The seventh baseline case was not reached; no new failure
ID was observed. This is not a green default suite or a claim that the unobserved
case passed. Logs, source hashes, resources and failure comparison are retained
under `ftqc-reference-env/g2-component-dense-*`.

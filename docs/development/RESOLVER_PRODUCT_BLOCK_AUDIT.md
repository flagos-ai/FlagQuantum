# Typed Product block call composition

Reference: unchanged Qualtran v0.7.0,
`bcb17bb25360ed68ce3bc084645db3e929799389`, Apache-2.0, Google LLC.
Source/test blobs are `0c806922e9d734f7a2d823c960369c1a443ebb4a`
and `389ffb997e324e16e3bb64e87e9bbd274188bdbe`.

## Ownership and source behavior

Compiler layout owns `product_block_component.py`: shared width signatures,
the existing unequal-resource admission diagnostic and typed Product root
composition. Existing Algorithms Product signature and resource-partition
consumers delegate there. Identity signal-state admission, numerical metadata
and independent cost formulas retain their existing Algorithms owners. Compiler
imports no Algorithms or source SDK and performs no numerical execution.

All external words are caller-owned, MSB first. Ancilla contains leading
`n-1` shaped QBit flags followed by the maximum child ancilla width. The system
and resource words are shared. Child owners retain private work, feedback and
exact inverse bodies. Unused portions retain their source AutoPartition names
and coherent state rather than becoming implicit clean work.

The root partitions ancilla, visits constituents in reverse tuple order,
then rejoins ancilla. Each constituent retains its source AutoPartition call
and Unused tail. Non-final children with nonzero ancilla split the shared word,
call all-zero MultiControlX on the original child-width prefix and flag, apply
X to that flag, then rejoin the shared word. Generic Adjoint reverses immediate
calls using exact constituent and Partition pairs. Original MultiControlX
returns itself as its adjoint; its existing predicate owner retains measured
cleanup. No equivalent opaque matrix or flattened flag circuit replaces this
call chain. Existing builder collections represent control arrays; no new
array type or public execution operation is introduced.

## Evidence

`tests/unit/test_product_block_component.py` compares original recursive calls,
named operands, shaped flag ports, ordered results and full complex operators
for both concrete original examples and both directions. Singleton, empty
inner ancilla, varying-width unused slices, dirty flags, shared resources and
three-child products are covered. Every combined raw measurement history of
each small forward/inverse pair restores the coherent input and zero private
work. Native build, binding and `circuit.append(op)` also run with source SDK
imports blocked. Existing Product and rewrite consumers verify the shared
signature and partition-owner replacement without changing their callers.

Receipts are `ftqc-reference-env/g2-product-block-constructor-*` beside the
worktree. Intake verifies both frozen blobs and all three original factories,
both directions and serialization depths zero through two. Resource receipts
verify Qualtran 0.7.0 and an actual Kickmix 0.1 Simulator; they keep emitted
primitives and live private leases separate from source QEC and logical-qubit
models. Pool capacity is not live peak, and lease release is not a substitute
for the complex-amplitude and raw-history cleanup evidence.

For the original properties example, actual forward primitives are one T,
one H, three X and one CX on five live wires, with no private work or
measurement. The source model reports one T and three Clifford calls on five
logical wires; it does not charge both open-control polarity flips at that
model boundary. For the measured-flag replacement, actual forward primitives
include five T-family gates, seven CX and one measurement with one private
lease and six live wires. Its independent source model reports one T, one And,
four Clifford calls and one measurement on six logical wires. Atomic And
estimates are not relabeled as emitted T counts, and the supplied nine-wire
pool is not reported as the six-wire live peak.

## Original failures retained

- Original tuple-of-Bloq serialization fails on object arrays or deeper ragged
  AutoPartition tuples. No original Product constructor wire is claimed to
  succeed, and no substitute tuple encoding is introduced.
- Symbolic Products retain their typed declaration and cost formula; original
  decomposition and native physical binding refuse execution.
- Unequal nonzero resource widths fail in the original shallow decomposition
  with `BloqError`: the shared QAny resource dtype differs from the constituent
  public dtype. The constituent's Unused tail also reflects this inconsistency.
  Native composition preserves the existing explicit refusal; it does not
  silently enlarge the original constituent port.
- Arbitrary child closure and identity-signal providers, zero-system physical
  execution and whole-repository/GPU/performance gates remain separate. Existing
  Algorithms source-file acceptance is not a claim that the generic constructor
  catalogue or original tuple wire is closed.

In the locked environment, preimport torch and run from the repository root:

```bash
python -m pytest tests/unit/test_product_block_component.py -q
```

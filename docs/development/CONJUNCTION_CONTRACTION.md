# Instance selection and native contraction information

The executable `examples/algorithms/conjunction_contraction.py` completes the
locked tensor notebook's `MultiAnd(...).as_composite_bloq().flatten(i != 2)`
workflow and exposes the existing native contraction profile. It composes
existing Compiler expansion and Simulation wiring/planning/contraction; there
is no new execution graph, model record, planner, backend or Core API.

```python
import torch
from examples.algorithms.conjunction_contraction import (
    contract_conjunction_with_info, select_conjunction_instances,
)

like = torch.empty((), dtype=torch.complex128)
record, instances = select_conjunction_instances((1, 1, 1, 1), like)
matrix, info = contract_conjunction_with_info((1, 1, 1, 1), like)
assert instances == (1, 2, 3)
assert matrix.shape == (128, 16)
print(info.summary())
```

The source's retained instance two is an **atomic And**. The source root expands
into three Ands with occurrence IDs 1, 2, 3 regardless of that retention. Native
predicate definitions contain executable physical bodies; explicitly registered
logical And suppliers establish the same stopping boundary. The example combines
the existing occurrence predicate with the provider's leaf classification.
Expanding these physical definitions would change the tensor decomposition and
is not equivalent to the notebook's atomic stopping rule. Retaining root zero
instead deliberately leaves an unprovided MultiAnd; both original and native
construction refuse rather than guessing a tensor.

## Cost conventions and path-controlled comparison

The example returns the existing `TensorNetworkContractionProfile`. Its cost is
the product of the dimensions in each pair's label union, summed across steps.
It is an arithmetic-work estimate, not a physical gate count or timing result.
The native greedy path executes through the existing native contractor.

Tests hold the reference path fixed using unmodified Quimb
`contraction_info(optimize="greedy")`, then reuse the existing dynamic-path
adapter, pair-step profiler and executor. They compare peak intermediate and
step/output sizes, every selected source leaf's data, the full isometry and
complex phase gradients. The reference opt_einsum cost counts multiply and add
as separate FLOPs. For these pairwise binary-sum networks it is exactly twice
the native estimate on that same path. This relation is tested for these
networks, not asserted for all tensor contractions or data types.

| Four-control source path | Native estimate | Unmodified Quimb/opt_einsum |
| --- | --- | --- |
| Pair-position path | `(0,1), (0,1)` | `(0,1), (0,1)` |
| Arithmetic cost | 4608 union-dimension products | 9216 FLOPs |
| Peak intermediate | 2048 elements | 2048 elements |
| Pair steps / output elements | 2 / 2048 | 2 / 2048 |

The peak includes the dense `(128,16)` output. At complex128, its storage alone
is 32768 bytes, not a process-memory prediction or a working-set upper bound.
Original/native optimizers are not promised to choose the same path for other
networks. Profile construction also works with meta tensors and agrees with
real-tensor shape metadata without reading amplitudes.

These operations add zero physical gates, work qubits, measurements or feedback
slots. Actual MultiAnd primitives and measurement-assisted cleanup retain their
separate evidence in [CONJUNCTION_TENSORS.md](CONJUNCTION_TENSORS.md); the inherited
full-branch/phase/work-zero regression is included rather than duplicated here.

## Reference environment and remaining boundaries

Reference commit: `bcb17bb25360ed68ce3bc084645db3e929799389` (Qualtran v0.7.0).
The notebook blob is `d91373e7d8eb21c48a0b870d99d1bb2973971861`.
Quimb's info path requires opt_einsum; the reference environment was restored
from the existing local opt_einsum 3.4.0 wheel. No repository dependency or
reference source was changed. Comparisons cover three-to-five-control concrete
ladders, positive/negative patterns and retained IDs 1/2/3/999, plus root refusal.
This is a local single-device diagnostic path, with no scaling claim.

This extends Q-0885's existing notebook evidence and adds no capability count.
The complete 18-cell original/native notebook audit, original index-object
presentation and Quimb layout differences still require final asset review.
Whole-notebook execution and whole-repository success are not claimed. The known
default-suite timeout remains open; G3 and PR merge remain outside this batch.

## Local validation

27 new focused cases pass, including both imported and direct-script workflows.
The 1,944-case related regression and 84-case CPU/TN suite pass. Strict typing
covers both changed Python files; Black, Ruff, Simulation-team scope,
architecture, multi-level IR and protected API checks pass. Final focused
coverage is 100% (39 statements, two branches); concentrated coverage before
adding script-entry execution was 95%. Only the test changed afterward, and all
27 focused cases were rerun. Reference numerical warnings remain visible. The
known default timeout was not repeated; these are bounded local checks rather
than a whole-repository or accelerator certification.

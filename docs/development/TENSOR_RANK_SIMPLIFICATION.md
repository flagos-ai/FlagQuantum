# Exact rank reduction for native tensor networks

Run the bounded numerical example:

```bash
python examples/algorithms/tensor_rank_simplification.py
# Rank pass: 3 -> 1 tensors; 8192 complex outputs
```

`simplify_tensor_ranks(nodes, output_labels=None, max_combinations=500)` belongs
to Simulation. It consumes and returns existing immutable `TensorNetworkNode`
objects. No graph class, registry, backend, new IR, public export or contraction
engine is introduced. Existing native einsum kernels execute the chosen pairs;
existing register contraction consumes the result without modification. When
outputs are omitted, labels occurring once in the original network are protected.
Explicit outputs can protect a hyperedge. Output order is still owned by the
consumer; no register bits, physical scratch or logical outputs are created.

## Source call-chain audit

Qualtran v0.7.0 `simulation/tensor/_dense.py` (pinned commit
`bcb17bb25360ed68ce3bc084645db3e929799389`, blob
`13aa81f50654ccaaa4293e8a63991686723022f5`) groups outputs, calls
`tn.contraction_width()`, and invokes `tn.full_simplify(inplace=True)` only when
width is **greater than 12**, before its dense contraction. The prior next-step
note mistakenly named `_quimb.py`; the numerical policy actually lives in
`_dense.py`, Q-0887.

Qualtran's metadata does not pin its Quimb dependency. This comparison explicitly
uses **Quimb 1.15.0**, rather than implying Qualtran locks that transitive version.
The unchanged `TensorNetwork.rank_simplify` source SHA256 is
`8053227b2b88d818ad4fe41b978ecb693e0270ce5a73ef9c3f88e8daee79d0a2`;
`full_simplify` is
`d2c3e546b58a2af55a33a2035b0cd9632d1ef4416bbf053d46bfde33e67842df`.
Reference versions and source hashes are asserted by the tests. Quimb is
Apache-2.0; original author attribution and the versioned source link are retained.
The reference installation and numerical source files are not modified.

Its default full simplification repeatedly applies A (antidiagonal gauge),
D (diagonal reduction), C (column reduction), and R (rank simplification), after
squeezing unprotected size-one axes. This batch implements **R**, not the other
passes or the complete width-trigger policy. In particular, register contraction
does not silently start using an incomplete `full_simplify` replacement.

| Source R behavior | Native ownership and verification |
| --- | --- |
| Remove floating scalars | Collect them explicitly; multiply into first surviving node |
| Collapse repeated tensor indices | Existing single-tensor einsum takes diagonals before the index census |
| Protect output indices | Add output occurrences to the same numerical census |
| Dimension/rank-weighted index queue | Stable insertion order, pop from right, requeue retained edges |
| Sum unprotected single occurrences | Existing einsum reduces them; no quantum discard is inferred |
| Limit large hyperedge pair search | Same combination cutoff, budget and rank-ordered candidates |
| Skip previously rejected pairs | Per-call occurrence/data-identity cache; no global state |
| Choose maximum rank gain, then occurrence IDs | Same tie breaks and ordered operands |
| Retain shared hyperedges | Remove an edge only at its last unprotected occurrence |
| Execute pair contraction | Existing native pair kernel, no NumPy/device transfer |

The pass reads shapes only for selection. It preserves device, dtype, PyTorch
autograd and input immutability; meta tensors support the same structural pass.
It rejects empty networks, missing/duplicate output labels, inconsistent edge
dimensions, mixed dtype/device and nonpositive/noninteger search budgets. Those
explicit native admission rules are not claimed to reproduce every foreign
exception or arbitrary Quimb option. Norm equalization, persistent external
caches, arbitrary tensor IDs, source tag unions and presentation metadata are
not new native contracts. Untouched/reduced single leaves retain metadata;
newly contracted pair nodes follow the existing native kernel convention.

## Scalar representation difference

Quimb spreads a scalar over up to eight leaves using fractional powers. Native
rank simplification multiplies it into the first surviving tensor, or returns
one scalar node for a closed network. This preserves the contracted function
while keeping zero-scalar gradients finite. It is an explicit representation
adaptation: scalar-bearing per-leaf arrays and normalization are not promised
identical. No reference patch is used to conceal this difference. Full complex
contractions and gradients are checked, including real and complex zero scalars.
This is not a claim of a new upstream defect or an upstream patch.

## Evidence and resource units

The reference comparison instruments only the source contraction call boundary;
the original method executes unchanged. Every pair's two operand-index tuples
and retained-index tuple agree across fixed graphs and randomized hypergraphs,
including low search budgets. Full complex results are compared to both the
reference and the original unsimplified native network. No gate sequence is
emitted by this numerical pass.

The 12/13-output-bit cases execute the original Qualtran `quimb_to_dense` and
verify the exact threshold and full-simplification invocation. These bounded
networks' R results equal its complete dense result; this does not establish
ADCR equivalence on general networks. Real Bell prepared/unitary graphs replace
only preprocessing before the same native binder/contraction, preserving phase
and gradients. Existing component regressions retain physical append/inverse,
measurement-branch and workspace evidence.

| Quantity | Result / meaning |
| --- | --- |
| Wide example | 3 input tensors -> 1 tensor, 2 rank-nonincreasing pair contractions |
| Dense output | 8,192 complex128 elements = 131,072 bytes; not process RAM |
| Tensor shape choices | No amplitude tolerance, truncation, SVD or numerical normalization |
| Physical primitives / measurements / extra work bits | 0 / 0 / 0 added by preprocessing |
| Source quantum gate-cost model | Not applicable to this numerical helper |

Q-0887 remains open for squeeze/A/D/C, full policy integration and other
source-wide boundaries. The existing Q-0887.03 evidence is extended without a
new capability checkbox ID. G3 and integration-PR merge are outside this batch.

Local validation: **71 new scenarios**, **2,016 related tests in 71.08 s**, and
**84 CPU/TN tests in 7.14 s** pass. The new numerical module plus example cover
**120 statements and 48 branches at 100%**. Strict mypy covers all three changed
Python files; Black, Ruff, scope, architecture, internal IR and protected API
checks pass. Reference autoray numerical warnings remain visible. The prior
default-suite timeout is not rerun or reported as passing. Implementation:
`f25503fe`. No parent is closed by this dependency-only acceptance.

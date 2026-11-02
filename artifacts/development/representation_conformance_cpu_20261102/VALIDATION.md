# Validation and #899 closure assessment

Base main: `ef95c04f08ddeb05c1c8469b51f4cfccb7c4fc62`.
Numerical producer: `71003d31521f6106feaacfe83fd46ef97fac9d44`.
The source pin is a committed ancestor of the delivery branch and must remain
obtainable when the changes are integrated. No Stable Core export, signature,
result schema or production execution path was changed.

## Focused verification

| Check | Recorded result |
| --- | --- |
| Scientific, artifact and launcher checks, isolated clone | 75 passed |
| Distributed placement regression, isolated clone | 3 passed |
| Benchmark contracts | 945 passed, 24 skipped |
| Public API baseline, capability matrix and generated documentation | Passed |
| Ruff, formatting and mypy on changed producers | Passed |
| Unmodified evidence-origin and citation checks in branch-isolated clone | Passed |

The commands below use Python 3.12.14, PyTorch 2.14.1 and pytest 9.1.1, with
`PYTHONPATH` pointing at the checkout and the thread limits recorded in README.

```bash
python -m pytest tests/integration/test_representation_agreement_sweep.py \
  tests/integration/test_representation_conformance_evidence.py \
  tests/unit/test_mps_truncated_gradient_policy.py \
  tests/unit/test_conformance_launcher.py -q
GLOO_SOCKET_IFNAME=lo0 python -m pytest \
  tests/distributed/test_mps_placement_conformance.py -q
python -m pytest tests/benchmark_contract -q
python tools/public_api_snapshot.py
python tools/check_capability_maturity.py
python tools/docs_source_of_truth.py --check
mypy --follow-imports=silent benchmarks/internal/evidence/representation_conformance.py \
  benchmarks/mps_placement_conformance.py benchmarks/representation_agreement_sweep.py
```

Use the actual local interface name on Linux; the recorded macOS loopback name
is `lo0`. Gloo requires local socket permission. The launcher has process and
collective deadlines, retains all nine rank logs, and tests rank failure and
timeout cleanup. Historical raw JSON was not rewritten.

Focused verification covers the same-forward negative control, independent
exact-map parameter shift, actual intermediate SVD boundaries, individual and
bucketed tied splits, low-precision/unstable oracle refusal, no representation
substitution, matched placement, provenance, per-case tolerances, multiple
observables and rank ownership/communication/memory metadata. Ruff and mypy
pass for the changed producer code. The public API baseline, capability matrix
and generated documentation checks pass.

## Repository-wide checks and baseline comparison

The ordinary checkout contains many historical development refs. Its full
evidence gate reports 98 historical-origin diagnostics both on base main and
on this branch: the diagnostic set has no additions. Those refs make commits
declared as unreferenced appear referenced to the checker. An isolated clone
containing only the delivery branch, its main ancestor and no unrelated local
tags passes the unmodified evidence-origin and citation checks. Neither the
checker nor the origin declarations were weakened to obtain this result.

The smoke/unit run also exposes failures already reproduced on base main in
the same environment:

- Three QFT Hadamard/phase-fusion assertions in
  `tests/team/simulation/test_controlled_phase_graph.py`.
- One terminal fused-rotation assertion in
  `tests/team/simulation/test_statevector_ops_numerics.py`.
- Six parameterized undisclosed-pin assertions in
  `tests/unit/test_evidence_provenance.py`.
- Six local-address/port/preflight assertions in
  `tests/unit/test_multinode_launch_plan.py`.
- One compact-CX threshold assertion in `tests/unit/test_native_cpu_adjoint.py`.
- Four batch-initialization assertions in
  `tests/unit/test_statevector_batch_chunking.py`.
- Six setup errors in `tests/unit/test_mps_distribution_accounting.py`.

These are separate from the new #899 scientific checks. They are recorded
as failed checks, not waived tests or passing results. The initial smoke/unit
invocation had a 600-second process-group deadline; completed files were
retained as its first partition, and failed/incomplete/unvisited files were
rerun in the isolated clone with the same marker selection.

The second partition completed with 7,954 passed, 309 skipped, 21 failed and
six setup errors in 391.30 seconds. Every failure/error group was reproduced
on base main; test function and parameter-case counts agree. The completed
file list, retry list, marker selection and result are retained in
`smoke_unit_partitions.json`. The two partitions cover the original 17,925
selected items; failed and interrupted files from the first partition were
included again, rather than counted as successful first-partition coverage.
No production source under `flagquantum/` differs from the base in this change.

## Closure decision

The corrected CPU corpus and placement evidence cover all six #899 acceptance
criteria in the README's stated scope. The remaining approximate-gradient error
and explicitly excluded numerical domains are measured limitations, not missing
conformance implementation. GPU certification and optimizer/checkpoint training
are outside this issue's bounded acceptance.

This is a closure-ready implementation for review, not a statement that the
GitHub issue has already been resolved on main. Integrate the branch while
retaining its source pin, and require the repository's CI/review outcome before
closing #899. The independently reproduced broader baseline failures remain
visible and prevent claiming a completely green repository-wide run here.

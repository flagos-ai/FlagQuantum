# Runtime readout sampling acceptance

Status: formal source and installed artifact qualified; this batch still
requires its own exact pushed-head CI. Parent #1337
`f019aafc634640c008b421468c6a50e4c50a840b` passed all applicable checks
(243 successful, 3 existing conditional skips). Main passed on
attempt 1; GPU passed on attempt 2 after one controlled same-head failed-job
retry. The original parent performance-variance failure remains preserved in
its PR record. No source, workload, sample or threshold change was made.

## Source evidence

- Six product files; five numerical helpers in three existing Simulation
  modules. Public signatures, result schema and pickle locators are unchanged.
- Four normalized body comparisons cover all moved statements. All existing
  destination definitions have identical ASTs. All other 11,906 tracked parent
  blobs remain byte-identical after the complete 16-path documentation/test batch.
- Original related tests: 383 passed. Four added replacement cases verify
  Runtime dispatch, normalization-before-generator ordering, validation and
  post-draw readout errors through the actual Simulation owners.
- Ninety-four original installed-parent scenarios match the source preview
  byte-for-byte, including 31 original refusals, public samples/statistics and
  generator states. The immutable original result SHA-256 is
  `8430814c229bbf01f949ec01833d735d11274abd845772c03b27c90963d10ee3`.
  Numerical tensor values are compared by shape, dtype and raw-byte hash.
  Dynamic wall-clock timing fields are excluded explicitly; deterministic
  execution statistics remain compared. No threshold or sampled result changes.
- The initial vocabulary check rejected new private `n_wires` parameters.
  New helper parameters now use `n_qubits`; the frozen contract is unchanged.
  The vocabulary gate passes without changing the contract. All eleven final
  preview quality gates pass, including strict mypy over 1,262 product modules,
  Black over 3,664 Python files and links over 1,913 Markdown files. The raw
  initial private-vocabulary diagnostic is retained.

## Complete formal source regression

All three phases ran in the actual Git checkout on the new branch, with four
workers. Existing optional/hardware skips and the credential-dependent Kaiwu
live-smoke exclusion are unchanged.

| Phase | Passed | Existing skips | Pytest seconds |
| --- | ---: | ---: | ---: |
| unit-api-team-final | 19414 | 957 | 401.87 |
| qec-full | 2801 | 329 | 75.62 |
| root-local-integration | 1592 | 223 | 89.83 |

All eleven formal quality gates and five additional gates pass. The fixed
five source hooks and four final documentation/diff checks are required by the
commit gate, together with an exact 1,262-product-blob/wheel comparison.
Original scientific assertions, frozen contracts, dependency policies and
recorded evidence are unchanged. The 102-entry facade inventory has no
addition; the existing public result and bound method retain their identities.

## Installed artifact

The native wheel SHA-256 is
`ec42656d3986203f77d2078b78ed1c20d12b22dbeb2c32e57aa5350623aad495`.
All 1,262 product Python file sets and bytes match source, wheel and the isolated
installation. Original dependency versions and native ABI remain unchanged.
The environment borrows the same third-party dependency path as the parent;
a task-local sibling Torch library link supplies the normal native loader
layout. This is not an independent dependency-resolution certification.

Cold catalog import loads neither Torch nor JAX. Native CPU and SparseBlossom
extensions load from the installation's own site, and the installed benchmark
CLI help exits zero. Ninety-four installed scenarios match the original parent
JSON bytes, including 31 original refusals. Original wall-clock statistics are
excluded as stated above; deterministic data and RNG states remain compared.

The original parent result and its bound public `sample` method restore from
pickle SHA-256
`1609154367aa12a1f23f2824a21e229a098ef06e17c17fd678503aa800747eb4`.
Class/field/annotation identities, summary metadata, subsequent bits/indices
and generator bytes match the original parent snapshot.

Outside a product checkout, **360 unchanged/replacement tests pass**. Two source
architecture/admission test files remain required in source regression. The
first installed test invocation reported a missing original example fixture;
that raw log is retained. Copying the original mathematical reference example
byte-for-byte completed the test fixture closure; no test or assertion changed.
There is no product source directory in this installed-test root.

## Published-head acceptance

The eventual child still requires its own exact-head combined CI. A green
parent, source regression and installed-artifact qualification do not replace
that gate. Formal publication additionally binds the commit to the qualified
artifact and enforces the final source/document checks.

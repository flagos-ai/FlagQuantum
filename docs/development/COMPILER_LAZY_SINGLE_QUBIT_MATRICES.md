# Defer matrices for fixed single-qubit singleton runs

The existing single-qubit fold now computes a parameter-free fixed gate's matrix
only when another readable gate joins its run. A singleton cannot become shorter,
so independent fixed gates retain their instructions without constructing unused
matrices. This is a Compiler-owned `single_device_fast_path` implementation change;
no public API, opcode vocabulary, numerical tolerance, target policy or evidence
contract changes.

## Semantics and ownership

The existing fold owns run membership, operand validation, product arithmetic,
exact global phase and replacement acceptance. This change remains inside that
implementation. The existing [upstream reuse record](COMPILER_UPSTREAM_REUSE.md)
describes the compiler's Qiskit/Cirq adaptations; this is a native allocation
improvement, not a new upstream algorithm port.

Only admitted fixed opcodes with no parameters or supplied matrix defer their
matrix. Parameterized, symbolic, trainable, batched and supplied-matrix inputs
retain their existing validation path. An unreadable angle still closes an open
run. When a second readable gate arrives, the original first instruction supplies
the deferred matrix and the exact existing left-associated product arithmetic
resumes, including multiplication by identity. Products exist only within this
pass invocation; no cache survives a call or source mutation. Completed runs are
still emitted at their original positions, preserving spectator and barrier order.

Three new test instances cover independent fixed gates without matrix reads and
trainable/batched angle boundaries followed by a new foldable run. Existing tests
check exact runtime matrices, retained global phases, trainable gradients,
nonunitary operands, vocabulary decline, randomized differential optimization and
wide disjoint circuits. Four pre-existing strict typing errors in the test module
are resolved by asserting that the tested numerical products exist before using
them; the numerical assertions remain unchanged. No casts or ignores are added.

## Paired acceptance and development measurements

The [exact paired record](evidence/lazy-single-qubit-matrices.json) loads the
original fold function from integrated main
`11745ab11070765974b90f5a8f512bf4c30576b1` into the changed source tree. Everything
else remains identical. The source digest records the uncommitted implementation
that was measured. The unchanged SDKBench checkout is pinned to
`9aa9260920984e2ca05bcb293abb885fccd9c144`; its adapters, evaluator, cases and
budgets are unchanged.

Widths 6/8/12, five applications and four topologies are compared in both ordinary
restored compilation and native target-artifact compilation: 120 pairs and 240
independent acceptances. Every pair retains identical complete normalized
operation/layout digests, quality metrics and adapter audit counters, including
target rewrite counts and serialized artifact sizes. Identity endpoint layouts
are required. Width six uses complete operators; larger widths retain three-probe
scope. These use the previously inspected nonce, not new prospective inputs.

Separate complete `optimize` observations use two warmups, nine alternating
retained timing samples per variant, one separate Python allocation observation
and separate profiles:

| Inspected program | Production / changed median | Python traced peak bytes, production / changed | Fold matrix reads, production / changed |
| --- | ---: | ---: | ---: |
| 4,000 independent H gates | 21.834 / 15.993 ms | 2,340,364 / 831,044 | 4,000 / 0 |
| 32,000 independent H gates | 236.175 / 163.108 ms | 19,499,644 / 7,017,368 | 32,000 / 0 |
| Four-wire mixed local circuit | 1.230 / 1.232 ms | 31,243 / 32,819 | 98 / 98 |

The wide cases reduce observed medians by approximately 27% and 31% and traced
peaks by approximately 64%. The mixed case shows no latency improvement and
allocates 1,576 more peak traced bytes; retain this cost. No local test run
overlapped these measurements. The shared macOS host is not qualified or reserved;
these are development observations, not process RSS, statistical significance,
general speedup, hardware, cross-SDK ranking or scalability evidence. One-shot
application elapsed fields are not substituted for repeated workflow latency.

## CI failure and verification boundary

PR #1286 head `200ddc2504a2c07346f96e2edcadbab626ad5b57` failed coverage job
`137059795969` in run `44973674426` at the existing wide-disjoint timing test:
large time `1.8286628443747759` seconds exceeded
`24 * 0.0733492523431778 + 0.05` seconds. This observation motivated inspecting
wide-circuit work. The failure does not establish its cause, and this change is
not evidence that the exact Linux shard is repaired. The original timing test
and threshold are unchanged; no blind retry, waiver or relaxed assertion follows.

The focused one-qubit/wire-local selection passes 42 tests locally. It also passes
with four pytest workers and in a separate direct coverage 7.17.2 sysmon run.
An earlier branch-coverage invocation warned that Python 3.12 sysmon cannot
measure branches and selected the default core; its passing result is not sysmon
evidence. Controller coverage around xdist does not prove workers were covered;
the direct run separately instruments the tests in their executing process.

Strict Linux-target mypy passes the implementation and complete test module with
imported modules followed silently and unavailable dependencies ignored. That is
two-file typing evidence, not whole-repository certification. The sparse default
tier passes 169 unit tests and runtime tier 42 integration tests. Full-source CI
remains required before integration. Formal resource-qualified SDKBench remains
user-deferred; no host change, reboot or service restart was required.

## Existing workflow and reproduction

```python
import flagquantum as fq
from flagquantum.compiler import optimize

source = fq.Circuit(3).h(0).h(1).h(2).to_ir()
assert optimize(source).instructions == source.instructions
```

Expected: independent gates remain unchanged. Normal same-wire folds continue
through the existing `optimize` entry point, with exact operator and phase rules.

```sh
python -m pytest tests/unit/test_compilation_one_qubit_optimization.py \
  tests/unit/test_compilation_wire_local_optimization.py -q
python tools/ci_tier.py pr-default
python tools/ci_tier.py pr-runtime
```

The archive retains the exact measurement driver and baseline source revision.
Its temporary harness/output paths describe the measured setup and must be
adjusted elsewhere. Run it with one numerical thread and a separate 600-second
process-group supervisor with TERM/KILL cleanup, keeping SDKBench's timers intact.

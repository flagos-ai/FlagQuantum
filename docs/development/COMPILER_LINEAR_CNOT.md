# Native linear CNOT synthesis

Status: private, explicitly admitted experiment; no default policy or new public
export. Owner: Compiler. Classification: `single_device_fast_path`.

## Verified requirement and upstream reuse

Parity relays and reversible changes of binary basis can span more than two
qubits. A two-qubit collector cannot generally reduce such a network. For example,
`CX(0,1) CX(1,2) CX(0,1) CX(1,2)` implements exactly `CX(0,2)`.
This identity is an inspected structural control, not a scientific workload or
an assertion that rewriting it is beneficial on every device.

Adapt the section-pattern elimination, pivot back reduction and transposed second
elimination from Qiskit 2.6.2's PMH implementation, pinned at
`0c26a5a158c4d62b2b303a8c2f3c6eb2b14653b4`:

- `crates/synthesis/src/linear/pmh.rs`, SHA-256
  `dbb99569a76e821a5069742a6bb3fbb65d440da87ca646932263e0fa5e4e581f`.
- `LICENSE.txt`, Apache-2.0, SHA-256
  `0a7558bc0c5a5bcac5e2bdc5e5367f3550b0c3b45b7e6f5a81b4f9fde6693374`.

The IBM copyright and modification notice accompany `_linear_cnot.py`.
Integer bit masks replace external Boolean array types. Gate output is native
FlagQuantum IR. Qiskit is a development conformance reference, not a production
dependency. The algorithm is a heuristic for finite circuits; no minimum-CNOT
claim follows from its asymptotic motivation.

## Admission and boundaries

Only consecutive unannotated, parameter-free CX instructions join a block.
Other operations, conditions, source evidence, custom matrices and barriers split
the run. At most 256 source gates join one block, so at most 512 distinct wires
need binary rows. Each proposal must strictly reduce CNOT count and match the
exact binary transformation. No numerical tolerance, global-phase quotient,
tensor conversion or mutable cross-call cache is involved. The complete output
must not increase two-body dependency depth.

A caller may supply an undirected `CouplingMap`. A proposal containing a missing
edge is declined; this pass does not route it. This protects the legal line
relay above from being replaced by a remote gate. It does not prove that a
mixed program requiring subsequent routing retains its delivered cost. Final
layout, target lowering, lineage verification and ordinary compiler defaults
remain with their existing owners. The pass is not registered automatically.

An eight-wire, ten-layer matching network at seed zero is a concrete depth
counterexample: raw PMH proposes 23 instead of 40 CNOTs but raises depth from 10
to 14. The whole-program guard retains the source. Tests use this real network,
not an injected cost, to exercise the rejection.

## Existing expert workflow

```python
import flagquantum as fq
from flagquantum.compiler import compile
from flagquantum.compiler.optimization._linear_cnot import resynthesize_cnot_blocks

source = fq.Circuit(3).cx(0, 1).cx(1, 2).cx(0, 1).cx(1, 2)
result = compile(
    source, optimization_level=0,
    passes={"example.linear": resynthesize_cnot_blocks},
)
assert len(result.instructions) == 1
assert result.instructions[0].wires == (0, 2)
```

The private import marks the development boundary. The supported pass-admission
entry remains `compiler.compile(..., passes=...)`. For a line, bind the same map
to this pass with `functools.partial` and pass it to `compile`; the relay remains
four local gates. Do not use the unguarded all-to-all form as an automatic
restricted-topology preprocessing policy.

## Fixed prospective development controls

Before complete workflow timing, retain `benchmarks/compiler_linear_cnot.py`
and freeze widths 4/8/16/32, seeds 0/1/2, serial binary mixing, ten parallel
matching layers, and disjoint parity relays, on all-to-all and line topologies:
72 pairs. The seed-zero depth witness and relay identity are already inspected;
none of these constructions is held-out scientific evidence. The unchanged
SDKBench task catalog and formal campaigns are not modified or replaced.

Both variants use the same existing explicit pass hook at optimization level
zero, one with an identity pass and one with PMH. They create a fresh topology
per compilation and run one warmup plus seven alternating retained samples.
GC is outside both timers. One separate full-call traced peak per variant
follows timing. Retain all rows and any delivered count/depth regression.
Default-admission gates: no count/depth regression in any pair, each median
time ratio at most 1.10, each Python traced peak ratio at most 1.05. These local
observations do not certify process RSS, host exclusivity, ranking, native CZ
delivery, or production latency. A 600-second owned process group bounds the run.
Do not tune these inputs or limits to obtain admission.

The same driver compares exact gate sequences against the pinned installed
Qiskit 2.6.2 on 120 fixed networks. Independent tests enumerate all 512 binary
three-wire matrices, accept precisely the 168 invertible ones, compare their
complete basis permutations, check larger seeded networks, sparse/reversed
wire labels, metadata changes, corrupt proposals, whole-program depth,
bounded splitting, complete operators and shared-angle gradients.

```bash
python -m pytest tests/unit/test_linear_cnot.py -q
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m benchmarks.compiler_linear_cnot --output /tmp/linear-cnot.json
```

An initial test collection used invalid barrier and dynamic-measurement
construction. The tests now use supported barrier metadata and the required
`is_dynamic` measurement marker. Those setup attempts are not passing evidence;
the compiler source was not changed to accept invalid fixtures.

## First complete observations

The declared implementation `b4f4a5bddb477235626a8f8ea6bba72122cafc18`
completed all 72 pairs and 120 exact upstream sequence comparisons. There were
zero delivered count/depth regressions. Twenty-one of 36 all-to-all pairs and
three of 36 line pairs reduced two-body count; all others retained that count.
These are inspected constructed controls, not application qualification.

Default admission failed: the median of case time ratios was 1.2060, the worst
case was 2.9988, and the largest Python traced peak ratio was 13.4596. No
fixture, limit or baseline was changed to rescue the result. In particular,
PMH can create a longer proposal than the incumbent and discard it only after
allocating that proposal. The full source, driver, every sample, upstream license,
setup failures and supervisor are retained in
[evidence/compiler-linear-cnot.json](evidence/compiler-linear-cnot.json).

A successor must stop synthesis as soon as the monotonically accumulated PMH
operation count cannot strictly beat the source block. It must preserve every
previously accepted proposal and retain a new published implementation before
repeating the same fixed controls. This is an algorithmic emission budget, not
a looser wall/memory gate or a smaller fixture. No default admission is granted.

## Bounded emission successor

The successor adds a strictly exclusive operation limit to both elimination
phases. The first phase stops as soon as its recorded operation count reaches
the source block count; the second receives only the remaining allowance.
PMH concatenates these operations without later cancellation, so an exceeded
allowance cannot become a strictly shorter proposal. Rejected candidates no
longer allocate an unbounded gate sequence relative to the 256-gate input cap.
Full binary-map and whole-program depth acceptance still apply. The same
72-pair inputs, samples and resource thresholds will be retained; the first
failed observations above remain authoritative for their implementation.

## Bounded successor observations and decision

Implementation `72d30b573b312d28c16e47193122885dd7d4a34c` completed the
same 72 pairs and 120 upstream sequence comparisons. Every delivered baseline
and candidate metric equals the first implementation's result. The median time
ratio was 1.1335; the worst time ratio was 3.0272, and the largest traced peak
ratio was 5.6534. The emission cap reduces the worst allocation ratio relative
to the first run, but these are different whole-run observations on a shared
host, not a paired speedup estimate between the two implementations. Both fail
the unchanged default-admission gates. The successor source, complete samples,
log and supervision are retained alongside the first run.

Keep this as a private explicit optimization for callers who choose its tradeoff;
do not add it to the default optimizer, change the protected pass catalog, or
claim that it improves the SDKBench routing result. It exposes a real exact
multiwire synthesis capability while preserving topology/depth rejection and
strict source operation budgets. Further default work needs a better admission
strategy and independently qualified real-application value, rather than
relaxed timing or memory limits. No claim of superiority over Qiskit follows:
this adapts its PMH implementation and currently matches its exact output on
the fixed conformance inputs.

Replacement conformance uses an independent breadth-first search over all 168
three-wire reversible permutations, substitutes that synthesizer at the same
internal boundary, and runs the unchanged compiler/pass consumer. This proves
the consumer does not require PMH-specific identity or orchestration machinery.

## Evidence validation correction

The first complete-corpus workflow verification passed 124 tests and failed one
origin-vocabulary assertion. The archive had stated `external_dependency` in an
inline field whose consumer accepts only repository/producing-host history.
The external Qiskit pin is now declared in `evidence-revision-origins.toml`, in a
separate integration-owned commit, and those two inline fields are removed.
The pre-correction document hash is retained. Numerical observations, source
pins, archived measured drivers and acceptance thresholds are unchanged. The
live driver's future output uses the same declared external origin. No checker
or supported origin vocabulary was weakened.

## CI benchmark import correction

Exact-head CI found one failure in the benchmark import inventory: the runner
imported the private module through its parent optimization namespace. Python's
`from` import could load the submodule, but the inventory correctly could not
resolve it as an exported parent attribute in a fresh process. Import the three
used functions from their owning private module directly. The source digest now
reads that same repository file explicitly. No parent export is added, no check
is relaxed, and the archived measured driver and numerical evidence are unchanged.
This is a future-runner import correction, not a repeat of the measured study.

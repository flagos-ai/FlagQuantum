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

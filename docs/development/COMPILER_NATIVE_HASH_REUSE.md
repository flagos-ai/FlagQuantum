# Reuse hashes within native result construction

This is a private compiler `single_device_fast_path` improvement. Public entry
points, signatures, defaults, result fields, identity payloads, serialized schemas,
target admission, and optimization decisions remain unchanged. No support
boundary or capability maturity level changes.

## Work removed and validation retained

Native gate legalization used to serialize and hash the same source separately
for its recorded fingerprint and identity payload. When legalization returned
that very source object, identity construction hashed it again as the output.
Native block construction repeated the same work.

Each constructor now calculates the source hash once for these two uses. It
reuses that value as the output hash only when output and source are the same
object. A distinct rewritten output is hashed separately. Values live only
within the current call; there is no global, object, or cross-call cache.

The result constructor still reads the source independently. Later block
verification and physical-plan identity validation still read fresh source and
output hashes. Circuit IR can contain mutable tensors despite its frozen outer
record: replacing these checks with a lifetime cache would be incorrect.

The block identity helper retains its existing IR-taking signature because
physical-plan validation consumes it. A separate private encoder accepts hashes
already read during construction; both paths produce the same JSON bytes and
SHA-256 identity. An initial development prototype changed the helper signature;
the target-artifact check exposed that cross-module consumer. The final change
preserves the call contract and passes physical-plan and v4 evidence tests.
No upstream code or new dependency is introduced by this optimization.

## Correctness evidence

The legalization regression covers unchanged and rewritten inputs, compares the
identity with the existing JSON specification, and changes a scalar tensor in
place. A subsequent call observes a different source hash and identity; rebuilding
a stale result fails. The block regression also checks that both reconstruction
and independent verification reject the stale source after mutation.

Legalization hashes four times on the parent for either case; the candidate
hashes twice for unchanged output and three times for rewritten output, including
the independent constructor check. Both added legalization cases fail their
hash-count assertions on the parent. The block no-rewrite case includes baseline
replay and independent checks, and performs seven source hash reads.

The expanded focused run passed 265 tests across gate/block legalization, target
legalization, emission, conformance, physical plans, and v4 evidence. These checks
preserve existing phase, mutation, tamper, and serialization behavior; they do not
replace the default and runtime CI tiers.

## Local engineering observation

The [raw samples and source fingerprints](evidence/native-hash-reuse-development.json)
compare the exact baseline functions from
`cf5fc36c29c3f9162b6b58054f8163c98944e33d` with the candidate, keeping result
classes and consumers unchanged. Two warmups precede 41 direct-pass samples or
21 workflow samples per variant. Variant order alternates each round. All
outliers remain in the raw artifact.

| Inspected workload | Parent median | Candidate median |
| --- | ---: | ---: |
| Legalize 768 native RZ instructions on 12 wires | 4.659 ms | 2.530 ms |
| Legalize 256 CX instructions to RX/RZ/CZ on 12 wires | 12.967 ms | 12.743 ms |
| Ordinary restored compilation, 12-wire line QFT | 37.886 ms | 36.747 ms |
| Target artifact compilation, same QFT | 885.956 ms | 864.685 ms |

Direct-pass complete output and identity agree between variants. Workflow
operations and final layouts agree exactly. The unchanged three-probe development
acceptance also passes, with maximum errors approximately 1.03e-14 and 2.51e-14;
three probes are not universal equivalence proof. The inspected workload, basis,
thread limits, environment, raw seconds, and source hashes are recorded in the
artifact.

This shared macOS observation establishes redundant work removal and suggests a
local benefit for unchanged legalization. The small whole-workflow median changes
do not establish statistically significant speedups. No isolated-host, process
memory, cold-start, held-out, cross-SDK ranking, or scalability claim is made.
Formal qualification still requires SDKBench's unchanged neutral protocol.

## Existing user workflow and regression reproduction

```python
import flagquantum as fq
from flagquantum.compiler.basis_conversion import convert_basis

source = fq.Circuit(2).rzz(0, 1, 0.37)
converted = convert_basis(source, gates=("rx", "rz", "cz"))
assert len(converted.program.instructions) == 9
assert sum(gate.name == "cz" for gate in converted.program.instructions) == 2
```

This existing local workflow retains its output. The savings apply when
constructing its native legalization evidence and, for eligible targets in the
artifact compilation workflow, native block evidence. Run the mutation and identity regressions with:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest tests/hybrid_compiler/test_native_gate_legalization.py \
  tests/unit/test_native_block_optimization.py -q
```

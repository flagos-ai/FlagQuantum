# Direct CZ lowering for parity and controlled-phase rotations

QFT, QAOA and Ising circuits repeatedly use controlled phases or parity
rotations. Translating every CX in their definitions independently to CZ
introduces an interior `H RZ H` sequence. That sequence is an RX rotation;
keeping it as three separately lowered gates spends unnecessary local pulses.

The existing equivalence table now includes three additional candidate words.
They use the original angle or its signed half and preserve metadata:

- RZZ: `H(target), CZ, RX(theta, target), CZ, H(target)`.
- CRZ: `RZ(theta/2, target), H(target), CZ, RX(-theta/2, target), CZ, H(target)`.
- Controlled phase: the CRZ word followed by `RZ(theta/2, control)`.

These are algebraic consequences of the existing CX definitions and
`H RZ(theta) H = RX(theta)`. They are FlagQuantum-owned identity compositions,
not a new Qiskit or Cirq code port. The change reuses the existing equivalence
search, target predicate, deterministic shortest-word selection, symbolic angle
arithmetic and decomposition records. No dependency or public API is added.
The original CX rules remain first, retaining equal-length ties.

The RZZ and CRZ words are operator-exact before further one-qubit lowering.
Controlled phase keeps the existing global-phase quotient; it must not be
substituted into a phase-sensitive controlled construction without a phase
correction. Full native words retain the established up-to-global-phase
contract of the existing one-qubit synthesis.

## Scope and boundaries

This is `single_device_fast_path` compilation, with no numerical execution in
the compiler. Native gate legalization and explicit named-basis conversion use
the same table. The direct word is an option, not an assumption that arbitrary
RX is available. Every expanded leaf still passes the caller's target predicate;
a fixed-pulse or different basis retains a runnable existing path when the new
word does not qualify. An incomplete decomposition is still refused by native
legalization. No target descriptor or hardware capability is inferred.

Angles are not sampled, converted to floats, detached or approximated. Numeric,
trainable, batched and symbolic values follow the same algebra. Metadata stays
on every generated leaf, including original conditions and routing ancestry.
Per-instruction operation-growth checks, target scheduling, layout restoration
and serialized evidence schemas remain unchanged. Strict CZ block reduction is
unchanged: these savings occur in the retained native baseline, not by relaxing
its acceptance rule. A tighter budget may now accept a shorter valid decomposition.

## Inspected application observations

The following source-control diagnostic compares the unchanged repository
QFT, QAOA-MaxCut and Ising examples against parent
`63f3ccfbae122954caefe452a596e57b01610f02`. Both controls receive the identical
already-routed IR and the exact RX/RZ/CZ vocabulary. Full operators are checked
against the original source with an independent simulation-matrix reference.
The largest phase-aligned entry residual was below `4.1e-15`.

| Example | Topology | Instructions before / after | CZ count, unchanged | Dependency depth before / after |
| --- | --- | ---: | ---: | ---: |
| QFT, 4 wires | none | 156 / 120 | 18 | 87 / 57 |
| QFT, 4 wires | line | 324 / 288 | 42 | 176 / 149 |
| QAOA-MaxCut, 5 wires | none | 462 / 318 | 48 | 232 / 130 |
| QAOA-MaxCut, 5 wires | line | 798 / 654 | 96 | 464 / 362 |
| Ising, 6 wires | none | 609 / 429 | 60 | 234 / 144 |
| Ising, 6 wires | line | 1239 / 1059 | 150 | 524 / 434 |

Dependency depth counts unit-duration instructions along shared wires. It is
not a calibrated physical schedule, runtime or fidelity measurement. These are
inspected development examples, not held-out SDKBench tasks or cross-SDK claims.
The ordinary compiler without a requested native basis does not acquire a new
implicit basis conversion.

A local macOS/Python 3.12.14 diagnostic used one warmup and seven paired
alternating-order batches over these six already-routed inputs, with OMP, MKL
and OpenBLAS threads set to one. Median translation-only batch time was
25.12 ms before and 29.42 ms after; exploring additional rules has a compilation
cost. Python tracemalloc peak was 716,436 / 612,056 bytes for one batch. These
are neither whole-compiler timings nor process-memory results. They do not
establish a qualified performance magnitude or hardware execution improvement.
Formal SDKBench qualification remains separate and unchanged.

## Existing user workflow

```python
import flagquantum as fq
from flagquantum.compiler.basis_conversion import convert_basis

circuit = fq.Circuit(2).rzz(0, 1, 0.37)
converted = convert_basis(circuit, gates=("rx", "rz", "cz"))
assert len(converted.program.instructions) == 9
assert sum(gate.name == "cz" for gate in converted.program.instructions) == 2
```

The same path is exercised through target artifacts, with and without routing,
including emission, scheduling, serialized evidence round trips and budget
rejection. The legacy evidence version remains in use when no collective
CZ-reducing rewrite occurs.

```bash
python -m pytest tests/unit/test_compilation_basis_translation.py \
  tests/hybrid_compiler/test_native_gate_legalization.py \
  tests/integration/test_compiler_application_cz.py \
  tests/integration/test_native_block_physical_evidence.py -q
```

Regressions cover zero, tiny, positive and negative angles; reversed and
nonconsecutive wires; symbolic binding; coherent phase-sensitive readout and
batched shared-angle gradients; annotation preservation; fixed-pulse constraints;
and exact operation-growth refusal. The sixty direct native-word regressions
failed on the parent revision because it emitted six unnecessary instructions
per parity or controlled rotation.

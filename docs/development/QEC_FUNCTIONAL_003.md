# QEC-FUNCTIONAL-003: Clifford circuit-location fault signatures

Owner: QEC maintainers with Simulation numerical ownership.
Status: implementation under review; no merge or release is authorized.
Decision A of [proposal #776](API_CHANGE_PROPOSAL_087_QEC_SCALABLE_EXPERIMENTS.md)
was explicitly approved by the API owner in the delivery chat on 2026-10-24.
Decisions B/C, new batch/experiment APIs and correlated-noise expansion remain
outside this implementation. The branch starts at main
`678cd02200435d9652142b63de33116d199b10e6`.

## Problem and resulting behavior

The phenomenological round-boundary route already derives responses from the
declared layouts without amplitudes. The gate-bound `NoiseModel` route still
spliced each fault into a lowered circuit and executed two dense trajectories.
This prevented circuit-location models from reaching patches whose data and
ancillas exceed the reference executor's statevector capacity.

The gate-bound route now enumerates exactly the same fault locations and
probabilities, but obtains every response through exact Clifford propagation.
Existing public constructors, noise grammar, model merge rules and output
ordering remain unchanged. The original `_spliced_fault`, `_forced_signature`
and `_executed_flip_set` remain independent small-system reference tools. There
is no runtime fallback, new mandatory dependency or public backend selector.

## Algorithm and ownership

Simulation's private `stabilizer/_fault_signatures.py` owns the numerical work.
It reuses the existing engine's Clifford images, adjoints and local Pauli
conjugation; it does not define another tableau simulator or gate-image table.
Each parity is propagated backwards through the actual lowered instruction
order with integer X/Z masks and a phase. A selected recorded measurement
contributes Z at its occurrence. An unconditional reset takes the expectation
against |0>. A nondeterministic parity is refused exactly rather than accepted
because two sampled trajectories happened to agree. Deterministic odd baseline
parities retain their sign.

At each post-instruction fault location, Pauli anticommutation determines that
fault's response. The QEC execution bridge translates detector references and
terminal logical supports into these numerical parities and maps response bits
back to existing detector/observable identifiers. It retains the reference
Runtime profile validation without running the numerical session. DEM
construction still owns fault enumeration and probability merging.

The original gate-bound grammar accepts bit-flip and phase-flip channels on one
wire. This PR does not add Y-channel, depolarizing, two-qubit correlated,
preparation or readout noise. Rules matching the last instruction retain their
existing refusal. Conditional instructions, custom matrices, channels in the
propagated program and unsupported gates fail closed. The numerical primitive
can test X/Y/Z responses without broadening the public noise grammar.

## User example

Run the registered example with an installed package, or `PYTHONPATH=.` from
the checkout. Neither optional QEC SDK is needed:

```bash
python examples/qec/circuit_noise_dem.py
```

It uses only supported public APIs:

```python
from flagquantum.noise import NoiseModel, bit_flip_channel
from flagquantum.qec import DetectorErrorModel, RotatedSurfaceCode, build_memory_circuit

memory = build_memory_circuit(RotatedSurfaceCode(distance=5), rounds=3)
noise = NoiseModel().add("cx", bit_flip_channel(0.01))
model = DetectorErrorModel.from_memory_circuit(memory, noise=noise)
sample = model.dem_sampling(shots=4, seed=0)
print(model.num_detectors, tuple(sample.detectors.shape))
```

The snippet prints `72 (4, 72)`. The complete example's expected stdout is:

```text
Physical qubits: 49
Detectors: 72
Logical observables: 1
Sample shapes: (4, 72) (4, 1)
```

This is construction and declared-model sampling, not a 49-qubit dense
simulation, a decoder benchmark or a logical-suppression experiment.

## Numerical and compatibility acceptance

- Simulation tests compare every supported Clifford gate and every X/Y/Z fault
  position with independent dense two-qubit evolution. Further identities cover
  correlated measurement records, reset erasure, odd baselines and refusals.
- QEC tests compare every enumerated location and the merged DEM against forced
  execution for repetition, rotated, Steane and mixed ZXXZ checks, including X
  readout where the existing code record declares it. They retain the existing
  last-instruction and unsupported-channel refusals.
- Distance-five and distance-seven models build with reference injection and
  execution entry points disabled. They contain 49 and 97 physical qubits.
- Keep all ordered 108 root exports, surviving class/pickle identities, single
  registered rotated class, registry behavior and original 15 frozen fixtures.
  The independent AI predecoder on current main is not changed.
- Run the complete QEC/retirement suite, Simulation tests, strict package type
  checking, source quality/architecture checks, installed examples and full
  required CI including single- and two-GPU gates at the reviewed head.

Focused reproduction:

```bash
PYTHONPATH=. python -m pytest tests/team/simulation/test_clifford_fault_signatures.py tests/qec/test_gate_fault_propagation.py -q
PYTHONPATH=. python -m pytest tests/api_contract/test_qec_retirement_contract.py tests/qec -q
```

## Resource evidence procedure and limits

`benchmarks/qec_clifford_dem_construction.py` measures one distance per fresh CPU
process. Run it from a committed source checkout:

```bash
PYTHONPATH=. python benchmarks/qec_clifford_dem_construction.py --distance 3
PYTHONPATH=. python benchmarks/qec_clifford_dem_construction.py --distance 5
PYTHONPATH=. python benchmarks/qec_clifford_dem_construction.py --distance 7
```

Each record includes the actual source commit, platform, Python/Torch versions,
instruction/measurement-parity/fault counts, elapsed construction time, traced
Python allocation peak and whole-process RSS high-water marks. Initial imports,
layout construction and the preliminary lowering used to count instructions are
outside the traced allocation interval. RSS includes those earlier costs and
must not be described as the primitive's exclusive memory. Tracemalloc timings
include instrumentation overhead and are not latency or speedup guarantees.

For I instructions, P parities and F faults, the implementation visits I*P
reverse instructions and F*P fault-response positions. It stores the instruction
sequence, fault positions and integer response masks rather than a 2^n amplitude
vector or a dense F-by-P tensor. Integer-mask cost still grows with wire and
parity counts. This is a polynomial construction path, not a constant-memory or
hardware scalability claim. Artifact measurements and final CI evidence are
recorded in the PR handoff; a pending run is not a passing result.

Measured local CPU results from the [resource artifact](../../benchmarks/results/comparison/qec_clifford_dem_cpu_arm64_20261024.json), produced by published implementation commit
`42ff10d4b0f081dd1465dd5338e6b26f89af7368`:

| Distance | Physical qubits | Reverse instruction visits | Fault responses | Traced Python peak (MiB) | Instrumented seconds |
| --- | --- | --- | --- | --- | --- |
| 3 | 17 | 3600 | 3600 | 0.434 | 0.140 |
| 5 | 49 | 33288 | 35040 | 1.303 | 0.695 |
| 7 | 97 | 135720 | 146160 | 2.456 | 2.382 |

Visit counts are exact loop counts derived from the measured I/P/F sizes, not
CPU instruction or FLOP counters. Each row is one instrumented observation, not
a latency distribution. The artifact retains the full process RSS measurements
and environment. Source QEC/retirement plus new Simulation regression: **1,894
passed, one existing inapplicable-fault skip**. Full-package strict type checking
passes for 810 source files; the architecture and all 499 QEC alignment checks
pass. Installed-profile and final required CI results belong to the PR handoff.

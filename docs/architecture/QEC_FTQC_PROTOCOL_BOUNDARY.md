# QEC and FTQC Protocol Ownership Proposal

Status: Proposed; architecture review only. No new API or package is admitted.

Prepared from main `2241e4b62c036d872706baf62afd6d17c5421853`, after the
[approved Stage A pilot](../development/FTQC_REFERENCE_WORKLOADS_AND_BUDGETS.md#supervisor-admission-and-stage-a-pilot-observations).
This is preparatory FTS-03/16 work under the
[master plan](../roadmap/FTQC_SIMULATION_MASTER_PLAN.md#appendix-b-conditional-sibling-ftqc-package-proposal).
FTS-02 and the complete FTS-03/16 contracts remain open. The pilot does not
establish mixed-circuit scalability or authorize a new execution contract.

## User problem and decision requested

A memory experiment already has code checks, measurement references, detectors,
logical observables, a compiled sampler and decoding. An encoded protocol also
needs to state which blocks participate, which logical operation a sequence
implements, which measurements determine a later operation, and when a pending
frame must be interpreted. Calling a memory experiment a logical gate would
leave these obligations unspecified.

Review the ownership and staged review sequence below. Do not approve an
imaginary API from a directory diagram. Public names, signatures, exception
behavior, serialization, numeric error budgets and supported execution matrices
must be specified in a later concrete contract before implementation. Merging
this document does not approve those contracts, admit `flagquantum.ftqc`, or
authorize new physical-device runs.

## Current authoritative boundaries

| Concern | Existing owner and reuse | What this proposal does not change |
| --- | --- | --- |
| Code definitions and logical operators | `qec.codes`, `qec.logical`, shared `pauli.py` and `gf2.py` | Existing code classes, registry and single `RotatedSurfaceCode` identity. |
| Syndrome extraction and observation order | `qec.circuits`; `MeasurementRef`, detector layout and logical observables | A round/qubit reference is not automatically a global dynamic measurement identifier. |
| Fault model and decoding | `qec.dem`, `qec.decoders`, `qec.adapters` | A declared DEM is not proof of an executed noisy protocol; predictions and complete decoder records remain distinct. |
| Memory execution and compilation reuse | `qec.execution`, including the existing compiled sampler | Sampling source, lifecycle, RNG ordering and batch semantics. |
| Measured-bit dependencies, reset and control | Existing DynamicCircuit, Core IR, Compiler and Runtime | The [ratified dynamic IR scope](decisions/IR_007_PHASE_4_HYBRID_DYNAMIC_SCOPE.md) and [dynamic Program IR contract](../development/API_CHANGE_PROPOSAL_076_DYNAMIC_PROGRAM_IR_CONTRACT.md). |
| Numerical state evolution | Existing Simulation and Kernels | No FTQC state simulator, backend manager, fallback policy or numerical registry. |
| Noise and device access | Existing noise, Compute, Runtime and remote boundaries | No protocol-owned channel implementation, provider stack or hardware certification. |
| Orchestration with no new domain semantics | Existing Services | A script composing stable calls does not by itself justify a new package. |
| Cross-block encoded protocol semantics | Candidate FTQC owner, only after admission | No new public class, frame type, result schema, root export or pickle path in this proposal. |

The existing QEC frame representation remains authoritative for its current
domain. A protocol-level frame must not silently reinterpret its coordinates or
reuse its class name with different semantics. The approved retirement of old
QEC modules did not authorize another class move or pickle break.

## Two concrete consumers required before package admission

| Candidate consumer | Distinct semantic obligation | Shared concept to justify, not an implemented feature |
| --- | --- | --- |
| Encoded two-block Clifford operation, initially a finite merge/split study | Track input/output block logical bases, gauge or parity measurements, and the resulting logical map; separate ideal logical action from fault tolerance. | Block identity, ordered logical dependencies and a checkable protocol outcome. |
| Magic-state injection with measurement-conditioned correction | Account for measurement branches and normalization, resource-state assumptions, and a frame that may cross a non-Clifford boundary. | The same block mapping and frame interpretation, without adding another execution engine. |

These are requirements for design evidence, not approved algorithms or executable
claims. Compare a minimal composition in QEC/Services with a protocol-owned
alternative. Admit FTQC only if both consumers demonstrate shared domain
semantics that cannot live clearly in the existing owners. Neither consumer is
implemented by the Stage A sampler, and no independent state engine is proposed.

## Contract questions that must be settled first

1. **Identity and mapping:** define physical-qubit, code-block and logical-wire
   coordinates independently; specify allocation/lifetime, preparation basis,
   measurement order and final readout. An output shape alone cannot prove the
   same logical operation.
2. **Round-to-dynamic lowering:** map every existing round/qubit measurement to
   an explicitly ordered live dynamic value, including terminal readout and
   reset. Reject missing definitions, duplicate ownership and unsupported
   schedules before execution; preserve existing detector parity semantics.
3. **Feedback versus frame tracking:** state which operation is physically
   applied and which is deferred. Give update direction, measurement-basis
   interpretation, correction timing, and non-Clifford boundary rules. A Pauli
   frame cannot simply pass through T as an unchanged Pauli correction.
4. **Branch semantics:** distinguish unconditional sampling, conditioning and
   postselection; preserve joint probabilities and report accepted and rejected
   shot denominators. Zero-probability branches and exhausted budgets need
   explicit contract behavior, not an invented exception in this proposal.
5. **Noise and fault tolerance:** declare channel placement and code/operation
   coverage. Compare ideal logical action separately from fault-injected
   detection/correction. A small noiseless truth table or one sampled DEM does
   not certify a fault-tolerant gate.
6. **Accuracy and refusal:** reuse exact supported paths first. Specify each
   numerical, approximation and sampling error source separately. Unsupported
   lowering, dynamic/noise combinations or resource limits must remain visible;
   no hidden flattening, fallback or approximation.
7. **Results and persistence:** first demonstrate consumption of existing
   results. Only propose a new field/type when an unmet user task requires it;
   review identity, module/pickle paths, ordered exports, plugins and serialized
   versioning before code. Do not activate the pending FTS-05 batch contract.

## Candidate layout and permitted dependency direction

The [master layout](../roadmap/FTQC_SIMULATION_MASTER_PLAN.md#appendix-b-conditional-sibling-ftqc-package-proposal)
remains conditional. Start with one reviewed protocol file if admission is
justified; do not create empty `frames`, `logical`, `protocols` or test packages.
Existing QEC code stays in place. Candidate FTQC code may consume public QEC
code/record/algebra contracts and approved shared execution contracts; QEC must
not import FTQC. Numerical kernels must not import either application domain.
Compiler and Runtime must not depend on protocol-specific classes to execute
ordinary circuits.

The concrete implementation review must map every proposed import against the
current architecture checker and team ownership policy. This document does not
grant new dependency exceptions. Protocol tests belong with their domain;
numerical evolution and dynamic def-use conformance stay with their current
owners. Benchmark registration, provenance and evidence use the existing
benchmarking infrastructure rather than a new FTQC harness.

## Reviewable delivery sequence

| Step | Deliverable and gate | User-visible evidence |
| --- | --- | --- |
| 1: this proposal | Ownership alternatives and missing semantic questions; preserve current behavior. | Both runnable examples below, including a current public refusal. |
| 2: concrete schedule contract | One finite existing QEC extraction schedule mapped onto existing dynamic control, with truth tables, measurement identities, frame rules, refusal matrix and explicit approval request. | The same small logical experiment expressed statically and dynamically, with an independent reference design. No API implementation before approval. |
| 3: conformance implementation | Only approved schedule semantics; existing IR/Runtime/Simulation paths and frozen compatibility checks. | Joint/conditional distributions, reset ordering, decoder-to-correction behavior and failure denominators, rather than matching a seed's bits. |
| 4: protocol admission decision | Two consumers and a demonstrated ownership need; separate public/serialization contract if needed. | Compare QEC/Services composition against FTQC ownership. Create the package only after specific approval. |
| 5: encoded protocol delivery | One bounded, independently verified protocol at its approved home, followed by a second consumer. | Logical action, representative faults, failure/refusal cases and resource evidence. Performance/hardware scope gets its own approval. |

FTS-10 algorithm selection can be designed in parallel as a separate review;
this proposal neither selects Clifford decomposition nor assumes MPS/TN support
for arbitrary adaptive noisy protocols. Stage B scale, domestic devices and
experimental data remain separate admissions. No total completion date or
performance gain follows from this sequence.

## Runnable current user workflows

Requirements: a supported Python/core FlagQuantum installation on CPU. These
examples require no SDK extras, credentials, hardware device or dataset. Save
each snippet as a Python file and execute `python /tmp/qec_boundary_example.py`
or `python /tmp/feedback_boundary_example.py`, respectively. Run outside the
checkout to verify the installed package; source validation uses `PYTHONPATH=.`.

### A. Code definition and an existing invalid-input refusal

```python
from flagquantum.qec import RotatedSurfaceCode

code = RotatedSurfaceCode(distance=3)
print("Code distance:", code.distance)
try:
    RotatedSurfaceCode(distance=1)
except ValueError:
    print("Distance below two rejected")
else:
    raise AssertionError("Expected the existing distance validation")
```

Expected stdout:

```text
Code distance: 3
Distance below two rejected
```

This refusal concerns the current constructor lower bound, not a hypothetical
protocol API or a new restriction on supported code distances.

### B. Existing measurement and classical feedback

```python
import flagquantum as fq
from flagquantum.dynamic import DynamicCircuit

circuit = DynamicCircuit(2).h(0).measure(0, classical_bit=0)
circuit.conditional("x", 1, classical_bit=0)
result = fq.run(circuit, shots=128)
assert result.samples is not None
assert result.samples.shape == (128, 2)
parity = bool((result.samples[:, 0] == result.samples[:, 1]).all())
assert parity
print("Dynamic samples:", tuple(result.samples.shape))
print("Feedback parity:", parity)
```

Expected stdout:

```text
Dynamic samples: (128, 2)
Feedback parity: True
```

This verifies the existing local measurement-dependent operation, not an
encoded-gate simulation or completed QEC feedback loop. No edit to PR #731 is
part of this proposal.

## Mission acceptance checklist

- [x] Map existing code, schedule, decoder, control and numerical owners; preserve
  current imports, class/pickle identities, ordered exports, registries and all
  original frozen fixtures.
- [x] Identify two concrete protocol consumers and an alternative in Services;
  package admission remains a decision rather than an assumed deliverable.
- [ ] R02/R05/R06: approve one concrete schedule/lowering/frame contract and its
  independent conformance matrix before implementation.
- [ ] R03/R07: justify package admission with both consumers and demonstrate
  complete protocols through existing IR and execution entry points.
- [ ] R04/R09/R10: select mixed algorithms and equal-semantics measurement
  targets separately; retain the user's baseline-only decision.
- [ ] R11/R12/R15: obtain actual device/data ownership, access and validation
  agreements. There is no hardware or experimental result in this proposal.
- [ ] Close R01-R15 only with execution, scientific and resource evidence under
  the master plan's milestone gates; this document closes no system requirement.

# QEC Schedule and Feedback Conformance Proposal

Status: The repository owner explicitly approved this finite Groups A-E matrix
and complex128 absolute error <=1e-12 / rtol=0 on 2026-10-25 (Asia/Shanghai) in
the delivery chat. Conformance implementation is under review. This approval
does not add a public API or authorize a generic feedback adapter, new frame
representation or FTQC package.

Prepared from main `1a5aa96891d56159de2a8273c2ddb2eb6c8d7fa8`, following the
[ownership proposal](../architecture/QEC_FTQC_PROTOCOL_BOUNDARY.md). This is a
bounded FTS-03/04/16 contract slice, not completion of those packages or FTS-02.
The [master plan](../roadmap/FTQC_SIMULATION_MASTER_PLAN.md) remains authoritative
for R01-R15, milestones and separate API/resource admissions.

## Existing delivery and the actual gap

`run_repetition_memory_experiment` already implements compiled lookup, runtime
physical correction, runtime frame tracking, offline frame analysis and temporal
decoder modes. It lowers through existing hybrid Compiler and Runtime sessions.
Existing tests already cover deterministic single-error correction, decoder
replacement, temporal decoding and seeded stochastic noise. These are delivered
capabilities, not proposed new features.

`build_memory_circuit(RepetitionCode(distance=3), rounds=2)` separately declares
check order, detector parities and terminal logical readout. Existing decoder
context tests already pin declaration numbering against sampling. Dynamic
execution already owns reset, measurement def-use and conditional gates.

The narrow addition requested here is an independent, explicit cross-path
conformance matrix: distinguish raw measurements from frame-adjusted histories;
compare the no-feedback code-driven schedule with a directly specified dynamic
extraction; and check that the same extraction/correction preserves an encoded
superposition. Do not create another production circuit builder, decoder,
sampler, state engine, registry or result type. Reuse existing tests rather than
copying their deterministic parameter grids.

## Approved finite domain

CPU only; five physical qubits, two rounds, Z checks, no stochastic noise.
Data wires are 0, 1, 2 and check ancillas are 3, 4. Ancillas start in zero and
are measured and reset after each check. Deterministic X faults occur only at
the beginning of a round, before its checks. A fault label is either no fault
or one of the three data wires. This is a conformance test domain, not a
large-scale benchmark or a fault-tolerance claim.

The fixed all-zero preparation uses existing QEC entry points. Encoded
`(|000> + |111>)/sqrt(2)` is prepared only in a test-owned public DynamicCircuit
composition; the current memory API receives no new preparation parameter.
Temporal decoders, stochastic channels, logical gates between rounds, Surface
feedback, generic code families, non-Clifford operations and hardware providers
remain outside this slice. Their existing behavior is preserved, not newly
qualified by this matrix.

## Measurement and schedule mapping

Let `m[r,j]` be the raw recorded ancilla bit for round r and check j. The fixed
schedule is:

| Order in each round | Operation | Meaning |
| --- | --- | --- |
| 1 | Inject the declared optional data X | A round-boundary error, not gate-location noise. |
| 2 | CX(0,3), CX(1,3), measure(3), reset(3) | Check 0 measures Z0 Z1. |
| 3 | CX(1,4), CX(2,4), measure(4), reset(4) | Check 1 measures Z1 Z2. |
| 4 | Optional feedback after both bits exist | No decision may read a half-complete syndrome. |

The dynamic classical values are 0/1 for round 0 and 2/3 for round 1. The
declaration references are `MeasurementRef(r, 3+j)`, not `MeasurementRef(r,j)`.
Their flat positions are `2*r+j`. Terminal data readouts append positions 4,
5, 6 and correspond to `MeasurementRef(None,0/1/2)`; they are final samples, not
extra mid-circuit measure instructions in the current memory source. Do not
silently overwrite these identities with the most recent ancilla register.

For the **no-feedback** code-driven experiment only, the six detector parities
are `m[0,j]`, `m[1,j] XOR m[0,j]`, and
`final_data[j] XOR final_data[j+1] XOR m[1,j]`, for j=0,1, in the existing
declared detector order. Its existing logical observable is the parity of all
three final data bits. Read actual layout order from the declaration, not an
assumed matrix row order. Check the two independent descriptions against each
other. Do not attach this unchanged memory detector layout to a physically
corrected circuit: inserted corrections change the raw trajectory.

## Raw bits, corrected histories and frame direction

For a three-bit mask f define its check syndrome as
`S(f) = (f[0] XOR f[1], f[1] XOR f[2])`. For an observed raw syndrome m and the
tracked X frame F **before** the current decision, the controller input is
`m XOR S(F)`. The existing repetition lookup is:

| Syndrome | Correction |
| --- | --- |
| (0,0) | none |
| (1,0) | X0 |
| (1,1) | X1 |
| (0,1) | X2 |

In physical modes the correction changes the quantum state and the pending
frame stays empty. In runtime frame mode the state is not physically corrected;
the frame update is parity/XOR, `F_after = F_before XOR correction_mask`.
Final interpreted data is `raw_final XOR F_after`. A frame is an interpretation
record, not evidence that a physical X occurred. Preserve the current meaning
of existing `executed_feedback`, raw and decoded record fields rather than
renaming or broadening them in this work.

At the end of round 1 there is no further gate in this slice. Do not infer rules
for moving an X frame across H, T, surgery or injection from this terminal
readout rule. `PauliFrame` remains the current three-data-bit X-only record;
it is not promoted to a general stabilizer/logical frame.

## Independent acceptance matrix

| Group | Cases and independently specified answer | Existing path to reuse |
| --- | --- | --- |
| A: no-feedback numbering | No fault and each one-X fault in either round: seven schedules. Propagate classical data bits and compute both checks directly with XOR; compare all four round bits, terminal data, six declared detector parities and logical parity. | Existing memory construction, sampling/decoder maps and a small directly written DynamicCircuit extraction. Do not inject feedback into the unchanged memory layout. |
| B: online decisions | Sixteen pairs `(e0,e1)`, each label in `{none,0,1,2}`. A test-only bit oracle applies the truth table after each round. Compiled and runtime physical paths end at raw/decoded 000. Runtime frame path preserves raw accumulated error bits, with frame parity equal to that accumulated mask and decoded 000. | Existing compiled/runtime lookup and runtime frame paths. Compare complete shot records, not only zero logical-failure counts. |
| C: offline limits | The seven single-total-fault schedules agree on final decoded 000. Also retain a two-distinct-fault example, X0 in round 0 then X1 in round 1: raw 110 has syndrome (0,1), terminal lookup recommends X2 and decoded data is 111. | Existing offline analysis. Do not require it to match online correction outside its correction radius or silently discard its logical failure. |
| D: coherence | For no fault or one X on each data wire before the first extraction, analytically project all measurement branches of the five-qubit encoded-plus state. Syndrome is fixed by the fault, reset disentangles ancillas, and physical correction restores the encoded-plus state. Undo its encoding and apply H to the logical input; final data is 000 with probability one. | Public DynamicCircuit composition and existing dynamic execution, checked against a test-only dense projector oracle. No new production state engine or preparation API. |
| E: negative cases | Undefined condition input, missing/duplicated measurement identity, non-binary records, wrong frame width and out-of-range deterministic events are explicit invalid inputs. Correct mapping must fail its assertion if round/check order or reset is deliberately perturbed in a test-only fixture. | Existing validation exceptions and verifier diagnostics; no new exception classes or changed public messages. |

The bit oracle must not call `_repetition_wire`, the production frame-adjustment
helper, decoder code, memory source generation or runtime feedback internals.
The dense projector oracle must explicitly encode qubit order and projectors;
it must not call the Simulation gate/measurement implementation under test.
Both remain small test references, not public APIs or a numerical backend.

Group D checks analytic branch weights and conditional states before sampling;
observed random frequencies alone cannot certify the joint distribution or
coherence. Zero-probability branches have zero weight and are never normalized;
the test must fail if an implementation invents a nonzero branch or loses the
normalization mass. The approved comparison for this small complex128 oracle is
absolute error at most `1e-12`, relative tolerance zero, separately applied to
branch weights, normalization and normalized density matrices. This is the
**approved value for this finite conformance slice**, not a general runtime
precision guarantee or the deferred 1.5x/2x performance target.
Density-matrix comparisons remove irrelevant global phase without ignoring
relative phase. Probability-one terminal outcomes must also satisfy the exact
bit invariant in executed shots; no seeded bit sequence or statistical
frequency tolerance is proposed.

Start by auditing existing cases and add only the missing independent reference
or cross-path assertion. A failure, unsupported path or reference disagreement
must be reported before fixing implementation; do not update frozen fixtures,
weaken tolerances or add skips to make a new conformance gate pass. Any necessary
public behavior change gets a separate proposal and explicit approval.

## Delivery and approval boundary

Approved decision: Groups A-E, their five-qubit/two-round domain,
independent reference boundaries, frame interpretation and approved complex128
comparison value for this independent conformance PR. Implementation
should change tests and minimal internal code only if a separately reported
defect requires it; no public additions or new serialized objects are approved.
Existing original frozen fixtures remain byte-identical. No GPU performance
experiment, Stage B campaign or new hardware/data access is requested.

Public imports, ordered exports, single `RotatedSurfaceCode`, class/pickle
identities, registration, stochastic behavior and prior decoder contracts stay
unchanged. FTS-05's pending complete-result batch operation stays inactive.
Follow the existing test markers/reachability policy, full required CI and
installed-user-example gates. This proposal does not modify or merge #731.

## Runnable current user example

Requirements: supported Python, core FlagQuantum, CPU only; no optional SDK,
credentials, device allocation or dataset. Save as `/tmp/qec_feedback_example.py`
and run `python /tmp/qec_feedback_example.py` outside the checkout, or use
`PYTHONPATH=. python /tmp/qec_feedback_example.py` for a source checkout.

```python
from flagquantum.qec import (
    ErrorEvent, ErrorSchedule, run_repetition_memory_experiment,
)

for mode in (
    "compiled_lookup", "runtime_decoder",
    "runtime_pauli_frame", "offline_pauli_frame",
):
    result = run_repetition_memory_experiment(
        error_schedule=ErrorSchedule((ErrorEvent(0, 1),)),
        rounds=2, shots=4, seed=53, strategy="trajectory", feedback_mode=mode,
    )
    assert result.shot_count == 4 and result.logical_failures == 0
    expected_raw = (0, 1, 0) if "frame" in mode else (0, 0, 0)
    for shot in result.shot_records:
        assert shot.raw_final_data_bits == expected_raw
        assert shot.decoded_data_bits == (0, 0, 0)
    print(mode, result.shot_count, result.logical_failures)

try:
    ErrorEvent(0, 1, pauli="z")
except ValueError:
    print("Non-X reference fault rejected")
else:
    raise AssertionError("Expected the existing X-only reference profile")
```

Expected stdout:

```text
compiled_lookup 4 0
runtime_decoder 4 0
runtime_pauli_frame 4 0
offline_pauli_frame 4 0
Non-X reference fault rejected
```

This is existing single-fault behavior, not evidence that Groups A-E are already
implemented or that the three-bit code protects against arbitrary quantum noise.

## Mission checklist

- [x] Reuse delivered repetition, decoder-feedback, frame and dynamic paths;
  identify precise independent-reference and measurement-mapping obligations.
- [x] Explicitly approve this bounded semantic/numerical acceptance matrix.
- [ ] Implement missing conformance and show independent branch/coherence
  comparisons plus meaningful negative cases without altering frozen fixtures.
- [ ] R05/R06: qualify this finite cross-path profile; broader QEC feedback,
  non-Clifford frame semantics and latency evidence remain open.
- [ ] R02/R03/R04/R07: obtain separate logical-protocol, algorithm and package
  admissions and executable evidence; this slice closes no system requirement.


## Conformance implementation under review

`tests/qec/test_schedule_feedback_conformance.py` adds 87 cases with no skips:
seven declaration/lowered/direct-extraction comparisons; the sixteen online
fault-label pairs for compiled trajectory/batched and runtime physical/frame
trajectory paths; eight offline single-fault/radius-boundary cases; four dense
joint-probability/coherence references; relative-phase, reset and mapping
mutations; invalid records/frames/dependencies/events; and two explicit existing
batched-runtime-feedback refusals. The independent references do not call the
production decoder, frame-adjustment or numerical gate/measurement helpers.
Existing compiler lowering and fault-injection test helpers are reused.

The runtime decoder paths deliberately retain their existing trajectory-only
restriction. Explicit batched requests fail with the existing diagnostic;
there is no fallback, skipped matrix entry or expanded execution support.
The complex128 oracle enumerates all sixteen syndrome branches, leaves zero
branches unnormalized, compares actual measurement/reset probability inputs,
checks the final joint data distribution and conditional density matrix, and
checks inverse-encoding readout. A relative-Z-phase mutation preserves basis
probabilities but fails density/coherence acceptance, demonstrating why final
sample counts alone cannot establish coherence.

Only tests and documentation change. No production defect has been identified
by these cases, no original frozen fixture changes, and no new public type,
export, pickle path, registry, noise, frame semantics beyond this approved
terminal profile, or FTQC package is introduced. This small-domain conformance
contributes to R05/R06; system requirements and non-Clifford feedback stay open.


## Accepted proposal mainline

The proposal merged in [#835](https://github.com/flagos-ai/FlagQuantum/pull/835).
Actual main `a660077023dde2d461d204e0bba6619d064f3939` passed
[all 54 complete-CI jobs](https://github.com/flagos-ai/FlagQuantum/actions/runs/40130667081),
[source, single GPU and required two GPUs](https://github.com/flagos-ai/FlagQuantum/actions/runs/40130662360),
and [smoke](https://github.com/flagos-ai/FlagQuantum/actions/runs/40130620338).
Three existing optional hardware pools were unselected. Package job
120981579481 passed the real QEC optional-installation step. Fresh isolated
sdist/wheel match 828 package Python members; a new Python 3.12.14 / Torch
2.14.1 core installation, dependency/native/Bell checks, master-plan examples,
H-T-H and the exact-reference CLI passed. The four-mode public example above
also passed outside the checkout with no SDK overlay. Original Stage A raw
observations were retained without recollection. The owner's later explicit
finite-contract approval, rather than these green checks or documentation merge,
authorizes the conformance work in this PR.

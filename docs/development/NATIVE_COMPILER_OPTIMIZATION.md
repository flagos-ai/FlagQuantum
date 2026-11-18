# Native compiler optimization

Status: implementation in progress; no performance parity or superiority claim.
Execution classification: `single_device_fast_path`. Compiler work partitions no
numerical workload and establishes no distributed scalability claim.

## Objective and ownership

Combine the strengths of mature quantum compilers, including Qiskit and Cirq,
within FlagQuantum-owned IR and compilation stages. Reuse existing algorithms,
implementation components, and dependencies before introducing new mechanisms.
Independent workload evidence, including regressions and unsupported cases,
determines whether an implementation improves the product.

Compiler owns program transformation. Core owns IR and target contracts;
Deployment owns executable packaging; Simulation supplies execution references.
SDKBench remains an independent evaluator with participant-neutral acceptance.
Changes to protected contracts require their own approved integration change.

## Existing components reused

| Requirement | Existing authority | Work in this slice |
| --- | --- | --- |
| Target-independent optimization | `compiler/pipeline.py` | No built-in optimization pass changes |
| SABRE search | `compiler/topology/sabre.py` | No second SABRE implementation |
| Automatic routing | `compiler/topology/routing.py` | Existing candidate selection reused as the approved native compile default |
| Numeric instruction matrices | `compiler/optimization/two_qubit_optimization.py` | Reuse matrix reading and multiplication; normalize reversed operands |
| Two-qubit KAK synthesis | `compiler/optimization/two_qubit_synthesis.py` | Reuse existing CZ synthesis |
| Basis cost | `compiler/lowering/basis_translation.py` | Price the actual per-instruction CZ decomposition |
| Explicit transformation admission | `compile(..., passes=...)` | Exercise the existing extension point without adding a public API |

No external code has been copied and no production dependency has been added
in this slice. The existing synthesis implementation retains its source history
and attribution. Any later component import requires a separate license,
maintenance, supply-chain, boundary, and replacement review. Reusing an external
compiler wholesale is not a substitute for integrating proven mechanisms into
FlagQuantum's own pipeline.

### Mature compiler review

Documentation reviewed on 2026-11-18:

| Source | Relevant mechanism | Adoption decision |
| --- | --- | --- |
| [Qiskit SabreSwap](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.transpiler.passes.SabreSwap) | Frontier candidates, lookahead and decay; documented LightSABRE enhancements and multiple trials | Keep the existing FlagQuantum-owned planner first. It is deterministic and is not an implementation-equivalent substitute for Qiskit's enhanced search. Evaluate search changes separately from the default-policy change. |
| [Qiskit ConsolidateBlocks](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.transpiler.passes.ConsolidateBlocks) | Collect local blocks, consolidate their operators, and resynthesize against a target | Reuse the existing native numeric reader, basis translation, and KAK implementation. Keep local block ownership in Compiler and verify each accepted candidate. |
| [Cirq CZTargetGateset](https://quantumai.google/reference/python/cirq/CZTargetGateset) | Full or partial CZ targets, two-qubit matrix decomposition, and optional moment-structure preservation | Admit full CZ only in this slice. Preserve instruction dependencies and treat total gate count/depth as separate outcomes. Do not assume this pass reproduces Cirq's target optimizer or scheduling policy. |

These are mechanism references, not performance evidence. No claim that the
implementations or defaults are identical follows from their shared use of KAK
or SABRE. Cirq's documented numerical tolerance and negligible-operation
postprocessing are not imported into generic FlagQuantum optimization; this
slice has its own explicit synthesis residual bound and preserves the existing
exact-identity policy outside this opt-in target transformation.

## Call-local routing plan reuse

Native `compile` retains the SABRE plan produced when pricing candidates and
materializes that same plan if SABRE wins. This removes a second identical
search within one automatic compilation. Selection objectives, candidate order,
calibration exclusions, and final-layout restoration are unchanged. There is no
cross-call plan cache: a different program or device is planned independently.
Separate public calls to `select_routing_strategy` and `route_to_topology`
continue to plan independently. Path-cache telemetry records the operations
actually performed; fewer planner calls are not a measured wall-time speedup.

Regression tests compare automatically routed instructions with the explicit
selected strategy, check statevector equivalence and deployment-plan replay,
and exercise changed programs, changed devices, and calibrated-device exclusion.

## Bounded CZ block transformation

`compiler/optimization/_cz_blocks.py` is a private, opt-in transformation for a
full-CZ target with arbitrary RZ and RX rotations. It collects static operations
over a pair, allowing commuting operations on disjoint qubits between members.
Two open pairs cannot overlap. A different pair sharing a qubit closes the old
block. Reversed operands are converted into the same ordered matrix basis.

Each candidate must strictly reduce CZ count relative to recursive native-basis
translation. Its reconstructed 4 by 4 operator must match the source within
`1e-11` per entry after accounting for one unit-modulus global phase. The
candidate may add at most 32 instructions relative to the translated baseline.
This objective can trade single-qubit gates and depth for fewer CZ gates; it
does not claim to minimize device duration, error, or total instruction count.

Each block has at most 64 source instructions. Matrix storage is constant per
block; output and replacement storage grow with program size. The number of open
blocks is at most half the qubit count. This bounds local numerical synthesis,
not whole-process wall time or memory. External evaluation must still enforce a
process timeout and memory budget. KAK work is attempted only for blocks whose
translated baseline contains at least two CZ gates.

Annotations, custom matrices, nonunitaries, trainable or batched angles, and
operations wider than two qubits form boundaries. Zero-qubit directives close
all blocks. Annotated routing SWAPs are retained, so layout evidence is never
invalidated by deleting a movement operation. Program metadata, observables,
measurements, and final assignment are retained; a logical SWAP is never
absorbed into an implicit final permutation.

### Shortest developer path

Run from the repository root with the project dependencies installed:

```python
from flagquantum.compiler import compile
from flagquantum.compiler.optimization._cz_blocks import resynthesize_cz_blocks
from flagquantum.core.ir import CircuitIR, Instruction

source = CircuitIR(2, (
    Instruction("swap", (0, 1)),
    Instruction("cx", (1, 0)),
))
result = compile(
    source,
    optimization_level=0,
    passes={"experiment.cz": resynthesize_cz_blocks},
)
assert sum(op.name == "cz" for op in result.instructions) == 2
```

This is a private developer experiment, not a new supported user API. The
unchanged per-instruction CZ translation costs four CZ gates for this source.
The two-CZ result preserves the identity final assignment and the full operator
up to global phase. It does not establish a workload-level speedup.

The focused checks are:

```bash
python -m pytest tests/unit/test_compilation_target_cz_resynthesis.py -q
```

The tests independently embed Simulation gate matrices into the complete
register, including reversed operands and disjoint spectators. They also check
incorrect candidate rejection, instruction-growth rejection, long blocks,
trainable parameter gradients, ordering boundaries, and the existing explicit
compiler-pass workflow. Compiler matrix reconstruction is not the test oracle.

## Limits and remaining integration

- The pass preserves semantics up to global phase. IR currently has no
  authoritative global-phase field. Do not admit it to generic `optimize`,
  phase-sensitive controlled construction, or an exact-amplitude contract.
- Conservative annotation boundaries mean it does not yet consolidate routed
  SWAPs with neighboring gates. This slice does not resolve the complete routed
  block optimization issue. Integration must construct correct layout and
  source-to-output evidence, rather than strip metadata to unlock a rewrite.
- Leading single-qubit instructions are not drawn into a new pair block.
- Only full CZ cost is implemented; arbitrary target gates, calibration-aware
  cost, symbolic synthesis, and approximate synthesis are not admitted.
- Automatic routing already considers deployable SABRE. The approved native
  compile default is now `auto`; explicit old-policy calls remain supported.
  See the separate API proposal and authorization record.
- `fq.compile` still rejects native target selection; the existing expert route
  is `flagquantum.compiler.compile`. No new root workflow is claimed here.

Next evidence must compare explicit existing routing strategies and the new
transformation on held-out tasks using unchanged SDKBench acceptance. Report
target basis, topology, initial and final layout policy, versions, seeds,
optimization settings, CZ count, dependency depth, compile time, memory,
timeouts, failures, and unsupported cases. Separate default-policy effects from
algorithm and adapter effects. Gate savings alone do not establish superiority.

## Development verification on 2026-11-18

Environment: macOS arm64, Python 3.12.14, PyTorch 2.14; CPU execution with
`OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1`. This is development correctness
evidence, not SDKBench resource-isolated performance evidence.

- Compiler unit files (`tests/unit/test_compilation*.py`), Compiler team tests,
  deployment routing evidence, routing candidate contracts, and the optimization,
  weighted-routing, synthesis, target-aware, and artifact-compilation integration
  golden paths: **2,752 passed** in 51.15 seconds after plan reuse.
- Vocabulary census and vocabulary contracts: **173 passed** after correcting a
  new comment's terminology and selecting this checkout's Git work tree.
- Public API migration baseline and the target-aware compilation example passed.
  Ruff lint and formatting checks passed for the changed Python files.
- The earlier full smoke/unit run was **not green**: 20,344 passed, 15 failed,
  6 errors, and 785 skipped. The routing assertion was migrated to the explicit
  approved previous policy and subsequently passed in the compiler suite.
  The terminology regression and incorrect Git work-tree selection were
  diagnosed and corrected, rather than updating expected vocabulary counts.
- Six launch-plan failures and six MPS process-group setup errors reproduced
  on the unchanged baseline `95f839f8fd43641143caaa5cca93f715cbf9183e`
  in the same sandbox: port binding was denied and Gloo could not resolve a
  local address. CPU process-group setup errors are not scalability evidence.
- The checked-in evidence revision gate still fails against the local clone's
  available history/origin declarations. It has not been waived or repaired in
  this compiler change. No full-suite passing claim is made.
- The broad runtime tier exceeded its earlier 600-second budget and was
  terminated. Targeted local integration checks passed; the full runtime tier
  and Linux SDKBench evaluation remain merge/release qualification work.

Target legalization's per-instruction decomposition records cannot truthfully
describe a many-to-many block rewrite. Connecting the private CZ pass to that
workflow requires a reviewed, versioned lineage/equivalence contract first.
Removing routing annotations or attributing a whole block to one source gate
would corrupt mapping and source evidence and is not an admissible shortcut.

# Complete qubitization phase-estimation source audit

The frozen source is `phase_estimation/qubitization_qpe.py`, blob
`39b608ea5708eba219a7fa4f4365176170c40bf5`, at Qualtran v0.7.0
`bcb17bb25360ed68ce3bc084645db3e929799389`. Its complete corresponding test
file has blob `301228e1f11e677993a2330a1c1dc64f2ec61c6b`. Both remain untouched.
Algorithms owns `primitives/qubitization_phase_estimation.py`. It reuses the
existing TextbookPhaseEstimation binding, phase-window declarations, packed
PREPARE, QFT, register partitioning and child capture. It adds no compiler,
public-export or numerical-engine implementation. Execution is a local
`single_device_fast_path` composition.

## Complete responsibility mapping

| Frozen responsibility | Native owner |
| --- | --- |
| Walk, window and optional inverse-QFT metadata | Owned signatures and explicit directional callbacks |
| Phase width, target and phase descriptors | Exact original descriptors, shared window and QPE metadata |
| Default inverse QFT and width assertion | Shared TextbookPhaseEstimation owner |
| Positive-controlled walk on the lowest phase bit | Existing ControlledSelectEmitter interface |
| Two zero-controlled reflections around each unconditional walk power | `_emit_power`, reflection subset and child ownership validation |
| Repeated powers 2,4,8,... | Reuse one directional walk emitter with source Power repetition semantics |
| Five direct child declarations and symbolic counts | `source_graph` |
| Concrete costs and live bystanders | `resource_model`, independent supplied child costs |
| Original string representation | FlagQuantum-named width label |
| All five original example factories | Register/precision declaration fixtures, detailed below |

The first phase wire is most significant. Only the final phase wire controls a
walk directly. For each earlier wire, the source emits a zero-controlled
reflection, an unconditional walk repeated a power of two times, and another
zero-controlled reflection. Reflection ports must be unchanged walk descriptors
and each reflection callback is restricted to that subset plus control and work.
Caller-owned shared scratch is reused sequentially after child cleanup. Children
receive result slots explicitly and emit both directions independently.
PREPARE and QFT retain their separate work partitions. All users append the
resulting bound operation with `circuit.append(op)`.

The original source explicitly notes differences from the cited paper's Fig.2
and leaves reconciliation unresolved. The native schedule retains those exact
source choices. It does not claim the paper's disputed first-bit discrimination.

## Calls, operands, feedback, phase and workspace

At rectangular widths 1,2,3 and measured LP widths 1,2, the complete original
stage sequence is checked directly. Source Power children have the exact 2,4,...
exponents. Captured complete Power streams match repeated walk bodies at every
wire, including angles and opaque unitary matrix values. The native composition
runs the original controlled-walk and zero-reflection snapshots through existing
owned callback interfaces. Those captures are reference-only inputs. Production
imports no Qualtran or Cirq.

Full complex output tensors match the untouched source QPE on coherent walk
inputs and a complex spectator, with every scratch bit zero. Every forward
measurement history is checked. Each independent forward/inverse history
restores the initial full state. Rectangular fixtures have one measurement
per direction, while LP fixtures have respectively three and four. Thus all
4,64,256 applicable combined histories are included. No global-phase alignment
is applied. A replacement test swaps the captured walk body for the already
accepted native QubitizationWalk and SelectPauliLCU owners, leaving the QPE
consumer unchanged, and checks both controlled-walk measurement branches,
absolute complex phase and clean work.

The shared textbook-QPE regression is strengthened to compare every emitted
parameter explicitly. Its earlier dependency comparison retained names and
operands, with angles indirectly covered by full complex tensors. Both checks
now run together. Measured LP and controlled-walk feedback is verified by the
complete branch histories and reused accepted child evidence.

## Actual execution and independent models

Actual primitive inventories in `g2-qubitization-qpe-inventory.log` are for the
explicit reference-captured child fixtures. Their matrix leaves and measurement
synthesis are retained. They are not a claim that a different native child
emits the identical SDK-default stream.

The declared graph separately uses PREPARE once, positive-controlled walk once,
zero-controlled reflection `2(m-1)` times, uncontrolled walk `2^m-2` times and
inverse QFT once. Zero-weight declared entries remain present, matching the
original dictionary at width one. Symbolic widths preserve the original counts.
Independent child costs use existing native bounded-Pauli SELECT, walk and
prepared-state-reflection models. All QEC categories and peak widths match the
original full QPE at widths 1,2,4. Resource evaluation uses emitters that fail
if called, proving model evaluation does not synthesize a circuit.

## Complete example and original-test scope

All five original factories execute unchanged. Their walk/reflection register
metadata is imported only into reference fixtures and retains type, bound,
shape, name, side and order. Native declarations use the existing rectangular
or LP accuracy factories. Original widths 4,14,21,40,21 match for Ising,
small Hubbard, large Hubbard, THC and sparse chemistry respectively. Existing
source normalizations are explicit child inputs. The fixtures are declaration
checks and reject physical binding without a bound PREPARE. They do not certify
large chemistry simulations or replace existing chemistry ownership boundaries.
Source factories import marked test helpers, so an isolated bounded process
executes their untouched metadata construction outside the native strict-marker
pytest session. No source marker or repository policy is changed.

The complete original non-notebook suite reports 19 passed, five original
strict-type skips and eight original xfails. The five skips retain the inverse
QFT QUInt to phase QFxp mismatch. Five xfails retain original disabled counts
checks, and three retain unsupported original serialization. Two notebook tests
are explicitly deselected and remain owned by Q-0629 and Q-0626. All original
slow walk cases and the original serialization-speed assertion run. Evidence is
`g2-qubitization-qpe-source.log`.

Q-0631 remains open for these original strict/serialization/count boundaries,
full SDK-default leaf equality, physical integration of the five large native
factory fixtures and the separately tracked complete notebooks. The existing
source LP adjoint and native chemistry typed-admission boundaries remain in
their owning records. Q-0630 covers the complete source composition and
responsibility mapping with these declared child integration boundaries.

## Verification

Focused native QPE and strengthened textbook tests pass. Concentrated regression,
strict mypy, Black, Ruff, Algorithms scope, architecture, multi-level IR and
protected API checks run before publication. Logs use `g2-qubitization-qpe-`.

For example, the two-bit rectangular reference-captured execution has 105
primitives and one measurement, using one work bit and one result slot.
Its independent source model has two And, 45 Clifford, two measurements and
peak five. The two-bit LP fixture has 227 actual primitives and four measurement
events, using six work bits and three reusable slots. Its source model has
five And, 92 Clifford, 18 rotations, five measurements and peak ten.
The actual and modeled measurement counts differ because the shared direct
QFT body and its measured declared model retain their established conventions.

The concentrated batch reports 231 passed. Strict mypy covers all three changed
Python files. Black, Ruff, Algorithms scope, architecture, multi-level IR and
protected API checks pass.

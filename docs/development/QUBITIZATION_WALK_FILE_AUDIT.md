# Qubitization walk source responsibility audit

The frozen reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The native source retains
Google LLC 2023 Apache-2.0 attribution.

| Parent | Frozen blob | Current disposition |
|---|---|---|
| Q-0656 `qubitization/__init__.py` | `d259f1657c3ca0e9d3ac08ed554667b9be79cd57` | Its sole relative export maps to native `QubitizationWalk` and ordinary append. |
| Q-0658 `qubitization_walk_operator.py` | `117515efb1e42c58b0fab451c454fbd756585f74` | All members audited below. Source/admission boundaries remain open. |
| Q-0659 `qubitization_walk_operator_test.py` | `a9dfd4398e3ec279e9a2a08a99d31c6ca22bc52b` | All original responsibilities mapped. Slow reference cases remain tracked. The complete native notebook is mapped in QUBITIZATION_WALK_NOTEBOOK_AUDIT.md. |

## Ownership and complete source mapping

`algorithms/primitives/qubitization_walk.py` owns concrete walk binding and
composition. Its block providers own SELECT and PREPARE. The shared compiler
reflection recipe owns zero marking, feedback and mathematical inverse. The
accepted `PreparedStateReflection` supplies the packed reflection declaration.
Core register descriptors and signatures retain metadata ownership. The path
is `single_device_fast_path`.

| Original member | Native responsibility |
|---|---|
| `block_encoding` | Existing `SelectBlockEncoding` / `LCUBlockEncoding`, explicit native child providers and named wire layouts. |
| `selection_registers` | Existing `selection_layout` gives a deterministic first-occurrence union. `source_walk_signature` independently unions complete typed descriptors, rejecting unequal same-name descriptors including dtype/shape mismatches. The original source set order varies across processes. |
| `target_registers`, `junk_registers` | Underlying target layout and existing through block-junk layout. |
| `signature` | Native scalar binding ports in selection/target/junk order. Opaque layout words retain no invented integer semantics. Full typed, shaped or symbolic metadata uses the separate shared Signature-based function. Physical typed graph admission remains its existing boundary. |
| `reflect` | `reflection` returns the accepted shared packed declaration with coefficient -1. Named execution uses existing block signal layouts. |
| `sum_of_lcu_coefficients` | Existing `alpha` delegates to block normalization, retaining symbolic values and missing-normalization errors. |
| `build_composite_bloq` | Existing `make_on` emits block then reflection, with adjoint reversing stages. PREPARE junk has explicit lifetime, and every borrowed scratch partition is restored under its child contract. |
| `build_call_graph` | Existing resource model retains ordered block/reflection calls, live bystanders and accepted external-control cost rules. |
| circuit diagram info | `wire_symbols` gives W per logical binding wire. |
| `prepare` | `prepare` returns the original packed preparation even when the LCU signal state is identity. The source inappropriate-generic-object branch is outside the native two-block admission. |
| `_walk_op` | Existing periodic Ising factory, original four sites / epsilon .2; normalization, signature and wire count checked directly. |
| `_thc_walk_op` | Existing THC chemistry native owners. Original 152-spin-orbital / 450-auxiliary / 10-bit PREPARE / 20-bit angle factory executes unchanged in original autotests. Existing THC walk tests provide native small-system composition and original resource/signature comparisons. Large factory physical execution remains part of its chemistry parents. |
| `_walk_op_chem_sparse` | Existing sparse chemistry native owners. Original eight-spin-orbital / 8-bit rotation / 12-bit PREPARE factory executes unchanged in original autotests. Existing sparse walk tests supply native physical, full-signal and resource evidence. |
| DocSpec | All three original examples are retained and their dependency ownership is recorded above. |

The new metadata properties reuse existing owners. They add no operation or
execution representation. The two-bit SELECT/LCU fixture checks every full
operator column, a coherent complex spectator, borrowed anchor zero, inverse
restoration, and IR JSON. Replacing its complete native SELECT emitter with a
frozen source capture preserves the same consumer and complex result. Comparable
primitive order and operands match the capture. The accepted walk suite already
covers shared and distinct selection words, logical junk, forward/adjoint order,
positive/open/multiple controls, all small measurement histories and feedback.

```python
from flagquantum.algorithms.primitives.ising_walk import ising_walk

walk, hamiltonian = ising_walk(4, 0.2)
assert walk.walk.alpha == 8
assert walk.walk.reflection.global_phase == -1
# The existing PauliHamiltonianWalk binding supplies the ordinary append path.
```

## Physical counts and declarations

The smallest two-row X/Z example has twelve explicit native instructions in
either SELECT or LCU basis. They comprise two Identity, six X, one CX, two CZ
and one anchor phase. There is no measurement and one clean phase anchor.
The original declaration has zero T equivalents. Scalar phase and identity
instructions keep their physical inventory independently from that QEC query.
Existing larger fixtures retain their separate physical/model tables.

## Original test responsibility and remaining gaps

| Original function | Disposition |
|---|---|
| three `*_autotest` functions | All three factories execute in the isolated reference environment. Original serialization xfails retain their source reason. Native constructor/provider ownership maps above. |
| parametrized eigenstate overlap simulation | The bounded original slow run emits one successful case before its 180-second deadline. The remaining original slow cases stay unverified. Accepted native Ising/Pauli walk tests retain eigenstate projection and complete branch evidence. |
| adjoint tensor comparison | The original slow three-site tensor comparison remains unverified after the bounded suite timeout. Native complete small operators and exact inverse are verified independently. |
| controlled/adjoint T complexity | Original assertion executes, with native walk control/resource cases already accepted. |
| four diagram scenarios | Original diagrams execute unchanged. Native metadata and shared reflection labels are checked independently. |
| notebook | Frozen blob a31dafeb427e4d7e13bfb0eb9bc903b38a15a316 is restored. The unchanged original notebook test passes in an isolated kernel. The complete native Q-0657 migration is verified in QUBITIZATION_WALK_NOTEBOOK_AUDIT.md. |

A controlled LCU supplied directly to the source walk omits its control from the
walk signature and fails. The native existing constructor rejects that case.
External control of a complete walk remains its independently verified path.
Source partial/generic metadata and chemistry-specific failures remain with their
own parent records. Q-0658/Q-0659 therefore retain open status.

OpenFermion 1.6.1 was restored only into the isolated reference environment,
with h5py 3.12.1, PubChemPy 1.0.4 and deprecation 2.1.0. The original source and
its assertions retain their frozen content. The non-slow, non-notebook source
run passes fourteen cases and retains three original serialization xfails,
four deselections and six upstream unknown-marker warnings.

The concentrated native regression passes 281 cases without skips or warnings.
Strict typing covers both changed Python files. Black/Ruff, Algorithms scope,
architecture, multilevel IR and protected API checks pass. The original notebook
test separately passes with twenty deselections and five upstream marker warnings.
The slow suite timeout is recorded as incomplete evidence. No repeated slow run
is used to inflate its disposition.


## Complete original test-file acceptance and remaining root (2026-11-28)

Q-0659 is accepted after closing the two previously unverified slow source
responsibilities. Each original slow node was executed unchanged in the same
Cirq 1.6 profile: three sites / epsilon 0.5 passes in 103.65 seconds, four sites /
epsilon 0.5 passes in 133.73 seconds, and the three-site adjoint tensor comparison
passes in 1.78 seconds. Both simulation tests loop over every Hamiltonian
eigenstate with the original assertions. These are actual complete test nodes,
not sampled replacements, renamed tests or assertions transplanted to a faster
implementation. The original 180-second whole-run timeout remains historical
incomplete evidence; it is superseded only for these exact newly passed nodes.

The unchanged non-slow factory/autotest, controlled-adjoint count, diagram and
notebook receipts above are reused, retaining source skips/xfails and SDK
profiles. Native migrated responsibilities continue to use existing complete
small operators, exact inverse, actual four-site physical context capture,
full complex joint states, applicable history certificates, clean private work,
source-free append and separately reported resource models. Actual source slow
simulation is not relabeled a dense native all-environment operator proof.

The current native concentrated batch passes 186 cases and the independent
Cirq 1.6 context/Ising batch passes 57. All 88 frozen source/dependency/asset blobs
are unchanged. No Python implementation, public contract or source test changes.
Strict production, 32 local integration and actual Kickmix Simulator evidence
from accepted predecessor batches is reused. Owner scope, language, links,
architecture, IR, protected API baseline and whitespace remain required checks.

Q-0658 remains open. Its generic two-child composition, exact inverse and
explicit context entry are implemented, but the complete native dependency
contexts for the original large THC and sparse chemistry examples are still
pending. Their metadata/producer diagnostics and source atomic/invalid-cleanup
boundaries are recorded in the constructor audit; descriptor admission alone
is not executable child-body proof. The next batch must address the complete
chemistry source owners and their actual contexts before claiming this root
closed. Retained signal junk remains live, and lossy source-ABI binding is
refused. Neither slow-test completion nor the small replacement fixture closes
that separate implementation obligation.

The THC factory now has a literal native dependency owner separate from the
corrected mathematical operation. Its original RIGHT-junk lifetimes, narrow
reflection and opaque physical leaves are retained. The specialized positive
forward graph inverse and the source's distinct generic Controlled.adjoint
constructor must not be conflated. The latter is an explicit remaining source
constructor gap under Q-0658; see `QUBITIZATION_WALK_CONSTRUCTOR_AUDIT.md`.

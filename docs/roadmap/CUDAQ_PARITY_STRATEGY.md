# CUDA-Q Parity Strategy

> **Future intent only.** This document states a direction and the rules that
> govern it. It is not a capability claim, not a schedule commitment, and not
> evidence that any capability named here exists. The authoritative statement of
> what exists and what is missing is the generated scoreboard
> [`docs/reference/CUDAQ_PARITY_MATRIX.md`](../reference/CUDAQ_PARITY_MATRIX.md),
> rendered from [`contracts/cudaq-parity-matrix.toml`](../../contracts/cudaq-parity-matrix.toml).

## 1. What this document decides, and what it does not

Three questions about the CUDA-Q parity programme recur, and each already has an
authoritative owner elsewhere in the repository. This document does not restate
them:

| Question | Authority |
| --- | --- |
| Which capabilities must exist, and their current status | `contracts/cudaq-parity-matrix.toml`, generated into `docs/reference/CUDAQ_PARITY_MATRIX.md` |
| What FlagQuantum may depend on, and how a dependency is adopted | [`dependency-policy.toml`](../../dependency-policy.toml) plus engineering decision principle 6 |
| Whether the programme is authorized to run beside the CPU vertical path | [ADR ARCH-012](../architecture/decisions/ARCH_012_CUDAQ_PARITY_CONTROL_SEQUENCE.md), **Approved** |

ARCH-012 is approved. The control sequence in
[`AGENTS.md`](../../AGENTS.md) § Current Strategic Priority has been replaced by the
one that ADR states, so the sequence that forbade opening additional horizontal
architecture tracks while the CPU vertical path is incomplete no longer binds as
written. The replacement sequence admits a second track under two conditions that
this document inherits rather than loosens: a new horizontal abstraction must
replace an existing implementation behind an existing boundary and carry a
replacement test, and two tracks may run concurrently only where they do not share
an unproven contract.

This document still carries no authorization of its own beyond the strategy layer:
it does not approve any contract, promote any capability maturity level, or complete
any Multi-Level IR phase gate. Each of those remains its own decision.

What remains is the strategy layer those three do not cover: how the goal is
stated so that progress is measurable rather than asserted, which dependencies are
a replacement obligation and which are not, what may be borrowed from outside the
programme and under what terms, and what the programme explicitly declines to do.
That is the content of this document.

## 2. The goal, stated so it can be falsified

The product requirement is that CUDA-Q capability coverage is a **floor, not a
ceiling**: every capability CUDA-Q exposes must have a FlagQuantum counterpart
with its own evidence, and FlagQuantum must additionally hold capabilities that
CUDA-Q structurally lacks. A second, independent constraint applies to the first:
the parity path must not depend on an NVIDIA-proprietary component.

Three properties of that statement matter for how the programme is run.

**Parity is measured per capability row, not per framework.** A single summary
percentage would let a large number of shallow rows hide a small number of deep
missing ones, which is exactly the failure mode that matters here. The programme
is therefore governed by row status in a generated document rather than by a
headline number. The current shape of the 95-row matrix, against the baseline
captured on 2026-09-30, is:

| Status | Rows |
| --- | ---: |
| `supported` | 10 |
| `partial` | 58 |
| `unsupported` | 27 |

`local_emulation` moved from `unsupported` to `supported` in wave 6, when
`flagquantum.remote.emulation.emulate` landed as a target-directed local entry
point. Its registry entry stays at its own level, which is the point of the
distinction above: a row closed by implementation does not raise the maturity of
the capability it names.

`third_party_target_sdk` moved from `unsupported` to `partial` in the same wave,
because the registration point it describes exists: a third party declares its
target in an extension manifest and FlagQuantum validates the declaration
fail-closed and turns it into a capability snapshot. It is deliberately not
`supported`, because a registered description is not yet a runnable backend, and
claiming the row closed would hide the remaining work behind an interface that
does exist.

`realtime_host_api` moved from `unsupported` to `partial` in the same wave, and it
is the only realtime row that moved. It is also the only row in the domain whose
`dependency_class` is `B_open_neutral`, so the vendor-neutral half of the realtime
surface -- a messaging protocol and a device-call abstraction -- is software this
programme can own and deliver, while the FPGA, the network interface, and every
timing property are not. The row is deliberately not `supported`: the dispatcher,
the ring buffer, and the host entry points that launch through them are absent by
design, and the three `A_nvidia_proprietary` rows beside it stay `unsupported` so
a protocol that measures nothing cannot be read as a transport.

Four rows moved from `unsupported` to `partial` in wave 6, when the matrix and
the registry were reconciled against the tree and four modules that already
existed turned out to have no entry and no row: `fermion_operator_algebra` now
names [fermion.py](../../flagquantum/observables/fermion.py), whose Jordan-Wigner
image is the `Observable` the existing execution path measures, with the
Bravyi-Kitaev, parity, and ternary-tree encodings still absent;
`boson_operator_algebra` now names
[boson.py](../../flagquantum/observables/boson.py), whose dense value is a
reference with no bosonic backend behind it; `chemistry_domain_library` now names
[chemistry.py](../../flagquantum/algorithms/chemistry.py), which is the ansatz
half of the row and none of the integral driver; and `resource_estimation` now
names
[resource_estimation.py](../../flagquantum/compiler/resource_estimation.py),
which reports the tally, the schedule depth, and the T family over a
straight-line sequence and refuses a data-dependent program, so loops and
functions stay unestimatable exactly as that row always said. All four were
registered in `capability-maturity.toml` at `development_evidence` in the same
change, because a row cannot be closed by evidence that no entry carries.

`commutation_aware_rewrite` moved from `unsupported` to `partial` in the same
wave, and it is the one move that required no new code at all. Its reason said the
pass was absent and needed the pass manager first; the tree said otherwise.
[commutation.py](../../flagquantum/compiler/commutation.py) answers whether two
instructions commute and which instructions form one commuting block on a wire,
from the operator schemas rather than from a generated table, and
[commutation_cancellation.py](../../flagquantum/compiler/commutation_cancellation.py)
consumes that analysis inside the optimizer's fixed-point loop, so
`cx(0, 1) rz(0) cx(0, 1)` reduces to a bare `rz(0)` -- a reduction the wire-local
self-inverse merge cannot reach, because the latest writer of wire 0 is the `rz`.
The row stays `partial` because it is named for CUDA-Q's pass rather than for the
reduction: CUDA-Q ships a *driver* that searches backward from an anchor operation
for the nearest endpoint a consumer will accept, so a second rewrite can cross a
commuting frontier without its author writing the commutation proof. FlagQuantum
has one consumer hard-coded against the analysis. Recording that as the gap, rather
than reading the reduction as the pass, is the whole content of this move.

`qir_code_generation` moved from `unsupported` to `partial` in the same wave on the
same evidence: its reason said FlagQuantum emits OpenQASM and QCIS text only, and
that was false. [qir.py](../../flagquantum/compiler/qir.py) emits QIR base-profile
LLVM IR -- the entry point with its `entry_point` attribute, the `__quantum__qis__*`
calls for the lowered program, the measurement and output-recording runtime calls,
and the module flags that declare the QIR version and the two dynamic-management
settings -- and
[translate.py](../../flagquantum/compiler/translate.py) exposes it as `"qir-2.0"`,
one of four `TRANSLATION_FORMATS`, so the surface is public and not a private
helper. QIR is vendor-neutral, which makes this the most valuable row of its class
to have been closed, and the two vendor-neutral rows it moved ahead of --
`cpp_kernel_frontend` and `mlir_dialect_stack` -- remain the expensive ones because
they are toolchain work rather than text emission. The row stays `partial` for the
half that is absent and it is the half the row is named for: CUDA-Q generates
*Profile-QIR*, and with both dynamic-management flags `false` this emitter cannot
express `__quantum__rt__qubit_allocate_array`, `qubit_release_array`, or
`read_result` at all, so no program it emits allocates a qubit at runtime. The
adaptive profile, pulse-level generation, and runtime library extension calls are
absent too, and this is text rather than an object file: no LLVM compilation was
run and no external QIR validator saw the output, so conformance is the repository's
own semantic check plus a host C parser accepting the text as LLVM IR syntax.

`classical_data_encoding` moved from `unsupported` to `partial` in the same wave,
and it is the first move in this document whose old reason was false rather than
incomplete. That reason said amplitude encoding "needs a normalized
state-preparation path" and that the only related surface was one data-reuploading
helper in the simulation domain reachable only as a module attribute; the first
half was already wrong when it was written, because
[state_preparation.py](../../flagquantum/algorithms/primitives/state_preparation.py)
has shipped `arbitrary_state` -- an exact preparation of a classical amplitude
vector, verified to `1e-8` after one free global phase -- and the second half
described a private helper rather than the public primitive beside it. What was
genuinely absent was the classical front end and the angular half, and
[data_encoding.py](../../flagquantum/algorithms/data_encoding.py) now supplies
both: `amplitude_encode` pads a feature vector to the next power of two with a
caller-chosen value, normalises it, and delegates the ladder to `arbitrary_state`
rather than growing a second preparation implementation, and `angular_encode` with
its `append_angular_encode` form emit one `rx`/`ry`/`rz` per wire on CUDA-Q's own
axis names, which is what a data-reuploading map composes from. The row stays
`partial` because the CUDA-Q contract is broader than the mathematics: CUDA-Q
returns a data-carrying `State` and re-normalises *after* padding, and its
`angular_encode` is a kernel-language call the compiler intercepts, while this
encoder returns a `Circuit`, pads before normalising, and has no kernel front end
to be called from. Recording that as the gap, instead of reading "an encoder
exists" as the row, is the whole content of this move.

`classical_control_flow` moved from `unsupported` to `partial` in the same
wave, and three reasons beside it were corrected in the same change, because the
`language_and_programming_model` domain was describing a repository that no longer
existed. [capture.py](../../flagquantum/compiler/_hybrid/capture.py) has been a
restricted Python AST bridge since before this document was written: it reads one
selected synchronous function, carries classical values with real def-use through
`Value`, `Operation`, `Block`, and `Region`, verifies them with an SSA pass that
fails closed on a use before definition or a duplicate definition, branches on `if`
both at compile time and on a measurement result, and unrolls `for` over `range`,
`enumerate`, and a statically ranked tensor under an explicit ceiling while
carrying outer names through region results. Three
[QEC](../../flagquantum/qec/sampling.py)
[entry](../../flagquantum/qec/repetition.py)
[points](../../flagquantum/qec/dem_construction.py) execute it in production, and
`while`, `break`, `continue`, and `for-else` are refused with an exact diagnostic.
So the row's old reason -- classical values "designed but not implemented" -- was
not incomplete, it was false, and the row is `partial` rather than `supported`
because the `while` form the baseline names is absent and no public kernel surface
exposes any of it: the capability is implemented Compiler machinery, not a
programming model a user can write.

The three reasons corrected in the same change are the ones that rested on the same
mistake. `python_ast_kernel` said "there is no AST layer", which is the false half
of the sentence above; it keeps `partial`, because the layer is private Compiler
machinery exported from neither `flagquantum` nor `flagquantum.compiler`, and its
reason now says that instead of denying the layer exists.
`kernel_first_class_functions` stays `unsupported` -- its capability is genuinely
absent, because the private operation vocabulary has no call or function op and a
hybrid program is a single-function container, so parameters, composition, and
nested calls are inexpressible rather than merely unexposed -- but it no longer
names
[MULTI_LEVEL_IR_ARCHITECTURE.md](../architecture/MULTI_LEVEL_IR_ARCHITECTURE.md)
5.2 as the planned home, because that document is now labelled a historical design
baseline and the level that exists is narrower than the design it describes.
`type_system_qubit_register_view` stays `unsupported` because a half-stale reason
is worse than a missing one: `IRType`, a `linear` flag, and verifier-enforced
linearity of the quantum-effect token do exist, so "its reference-versus-value
linearity rules are not implemented" was too broad, but the linear subject is the
effect token rather than a qubit, and nothing models a qubit, a register, or a view
over one. `runtime_qubit_allocation` was left untouched on purpose: its claim that
runtime allocation is not implemented is true, and only the name of the document
describing it would have changed.

The `pauli_tracking_sbe` row is the second reconciliation of this round, and it
moved on one named refusal disappearing rather than on a family being finished.
Its reason had already reduced its blocker to a single fact: the stabilizer engine
declined a noise instruction instead of tracking a frame through it. That refusal
is gone. A channel whose Kraus operators are a mixture of Pauli unitaries up to a
global phase is now tracked on as many wires as the channel acts on, so a noisy
Clifford program is sampled without a dense state, and a channel outside that
class is still refused rather than sampled as its nearest Pauli approximation. The
engine names no channel instruction wider than two wires, so a wider frame is
spelled as a chain of correlated-error instructions, one term per non-identity
frame, with each weight divided by the probability that the frames before it did
not fire - a chain term is conditional on no earlier term having fired, which is
the shape of a Kraus mixture, and emitting the caller's own weights would make
every frame after the first too rare. The row did not go to `supported`, because
only half of its name is delivered. The readout stays in the computational basis,
since a frame is what the engine tracks and the sampled bits are what it reports,
so a measurement is never expanded in a stabilizer basis; expectation values,
detection events, and observable flips are absent from this route; and the
trajectory selection and per-trajectory shot allocation that a stabilizer-basis
sampler would be paired with belong to the trajectory row and are still open. The
same change widened the package's public surface by one entry point:
`survey_stabilizer_program` reports what the sampling route would make of a
program without executing it, counting gates, resets, recorded measurements and
noise instructions and collecting one blocker per refused instruction. It reads
the IR and never imports the engine, so a caller decides between the stabilizer
regime and a dense one without the optional distribution installed. That is a
census and not a promise: an empty blocker list says every instruction
translated, not that the samples are correct.

`clifford_t_synthesis` and `angle_synthesis` moved from `unsupported` to `partial`
together, and they are the first two rows of the `B_open_neutral` class this
document had listed as a research expense rather than an integration one. The
Clifford+T half is exact and is now carried:
[angle_synthesis.py](../../flagquantum/compiler/angle_synthesis.py) classifies a
z-rotation by its residue modulo eight when its angle is an exact multiple of
`pi/4`, and `basis_translation` consults one short word per residue as an
additional candidate rewrite, so a target publishing only `h`, `s`, `t`, and `cx`
reaches a rotation it previously refused outright -- the measured failure the row
was written against. Both rows stay `partial` because the halves named in their
titles are still absent, and saying which half exists is the whole content of the
move. Nothing is approximated: the classification is an exact equality, and
`pi/8`, `pi/3`, and `0.3` all fail closed rather than being answered with a
rotation nobody asked for. That refusal is deliberate, because the Ross-Selinger
route an accuracy-directed synthesizer needs runs on exact arithmetic in
`Z[omega, 1/sqrt(2)]`, a Diophantine norm equation, and integer factorization,
and the GMP and MPFR dependency this document already records against the row is
still unadopted. Two properties of the table are worth naming because they are
what a fault-tolerant consumer reads. Each residue's first word carries the
minimum `t` count, and the test suite re-derives that claim by enumerating the
whole vocabulary rather than trusting the table. Each residue also carries a
fallback word, so a target that publishes `t` but not `tdg` still reaches
`rz(-pi/4)`; without it `{t, s}` is not a group and misses a residue, which is the
same class of silent hole the row was written to record. Trainable angles are
refused by the same check that refuses non-quarter turns, so a rotation never
leaves the autograd graph by being answered with a constant word.

`qec_stim_user_migration` is the first row in this document closed by a document
rather than by code, and writing it is what showed that the row's capability is
not the page it is named after. The route a Stim user takes into this package runs
from `stim.Circuit.generated` through the detector error model text to
[the guide](../guides/STIM_USER_MIGRATION.md), and every step of it already
existed; what was missing was the statement, in one place, of which steps this
package owns and which stay Stim's. The guide names the boundary rather than
crossing it -- nothing under `flagquantum/` imports `stim`, so the extra is needed
only by the half that generates circuits -- and then says what a migrating script
actually meets. A multi-round circuit's model text carries a `repeat` block that
[the reader](../../flagquantum/qec/dem.py) refuses by name instead of expanding,
`stim.DetectorErrorModel.flattened()` is the whole repair, and the three other
refusals are tabled with theirs. The row stays `partial` for the reason the
document exists, and the guide states it in the place a reader will hit it: the
Stim integration this route lands on is itself `partial`, so
`qec_stim_user_migration` cannot be `supported` while the thing it documents is
not. The `^` separator is the first half. The default reading states the line as
written and is not graphlike, so both registered decoders refuse it -- 113 of the
219 mechanisms of a distance-3, two-round rotated surface code flip more than two
detectors -- while the reading a matcher accepts,
`use_decomp_suggestions=True` followed by `merge_duplicate_mechanisms()`, is
measurably a different distribution: against stim's own sampler at 200000 shots
the default reading's observable marginal sits 2.1 standard errors away and the
graphlike reading's sits 56. Both numbers are in the guide, with the default
reading named as the model for a threshold or a marginal and the graphlike one as
the model for a matcher, because a decoded logical error rate at this distance
does not separate the two routes while the per-shot agreement does -- the in-tree
matcher and the PyMatching cross-check agree on every shot and belief propagation
with ordered statistics differs on 1.7 percent of them. The second half is the
registry: belief propagation with ordered statistics decodes a hyperedge model and
is reachable from the package, but `register_decoder` requires a
`from_detector_error_model` classmethod that class does not carry, so
`decoder_names()` holds the matcher and the cross-check alone. Naming only the
graphlike reading as the migration route would have been a silent numerical
downgrade of the model the caller handed over, so the guide states the tradeoff
instead, and the test that keeps it true executes every fence in one namespace and
compares each print against the transcript quoted under it.

**A row is closed by evidence at the maturity its registry entry requires**, never
by moving a status. `capability-maturity.toml` holds the maturity levels and
`docs/roadmap/CAPABILITY_MATURITY.md` explains them; a parity row cites a registry
entry but cannot raise it.

**Absence is a recorded, owned gap rather than silence.** A row that is
`unsupported` names its reason, its priority, and the evidence for the absence,
including the negative search when the absence was established by search.

## 3. The replacement obligation is narrower than "replace CUDA"

The programme's cost is dominated by a misconception worth stating plainly:
replacing CUDA-Q is not the same as replacing CUDA. Each matrix row declares a
`dependency_class`, and only one of the four classes is a replacement obligation.
The distribution over the 95 rows is:

| Class | Rows | Meaning | Obligation |
| --- | ---: | --- | --- |
| `A_nvidia_proprietary` | 9 | The CUDA-Q implementation rests on an NVIDIA-proprietary component | FlagQuantum must supply its own component. Depending on the original is not an option. |
| `B_open_neutral` | 16 | The CUDA-Q implementation uses a permissively licensed, vendor-neutral component | May be used directly. The work is integration, conformance, and ownership. |
| `C_flagos_replacement` | 3 | FlagQuantum satisfies the capability through a FlagOS-family component | The replacement target is named; the gap is maturity and evidence. |
| `none` | 67 | The capability is not dependency-bearing | The work is engineering, not replacement. |

The nine `A_nvidia_proprietary` rows are the whole of the CUDA replacement
obligation:

| Row | Domain | Status |
| --- | --- | --- |
| `backend_gpu_statevector` | simulation_backends | `supported` |
| `backend_multi_gpu_statevector` | simulation_backends | `partial` |
| `backend_tensor_network_exact` | simulation_backends | `partial` |
| `backend_dynamics` | simulation_backends | `partial` |
| `large_scale_statevector` | performance_and_scalability | `partial` |
| `production_sharded_statevector` | performance_and_scalability | `partial` |
| `realtime_feedback_loop` | realtime_control | `unsupported` |
| `realtime_transport` | realtime_control | `unsupported` |
| `realtime_sensor_bridge` | realtime_control | `unsupported` |

Six are numerical or scale rows and belong to the three owned cores described in
§ 4. The remaining three are realtime control, which § 6 addresses as a
non-goal for software alone.

The practical consequence is a scheduling one. 4 of the 16 `B_open_neutral` rows
are `unsupported`, and every one of them is closed by integration rather than by
research: a C++ front end and an MLIR dialect stack, a chemistry domain library,
a QEC dialect, and the arithmetic constructions a logical layer needs. That is the cheapest
capability per unit of effort available to the
programme, and § 5 governs it. The first of the family moved off `unsupported`
without new research, which is the shape the remaining rows are expected to
follow: the stabilizer backend landed by adopting the same permissively licensed
engine CUDA-Q uses behind its own stabilizer target, while the execution route
that would make it a planner-selectable backend is a separate change. The Trotter
slice of `algorithm_block_encoding_family` followed the same shape from the other
direction: it was written here rather than borrowed, because the plan's route -- adapt
Qualtran -- was ruled out when the repository owner weighed a 27-dependency
Apache-2.0 package against implementing the mathematics natively.

`logical_resource_estimation` moved the same way and one step further. It was
written here rather than adapted, so the Qualtran route this section's table lists
against it is no longer the route that was taken, and the row now names its own
module as evidence. It moved to `partial` rather than to `supported` because the
half that landed is the costing half and not the synthesis half: a rotated
surface-code cost over the compiler's own static estimate, with the gate-level
tally passed through rather than re-derived so the repository keeps one source of
truth for what a T-depth means. What is still absent beside it is the
approximation half of angle synthesis that would let an arbitrary program reach
it, and the
distillation, placement, and device model that would turn a patch count into a
compiled estimate. **The reuse is the part worth carrying forward**: the row moved
because a vertical path already existed through the compiler's resource estimate,
and the new unit extended it instead of starting a second one.

## 4. What must be owned

Three capabilities have no acceptable vendor-neutral substitute and are owned
outright, as admitted by ARCH-012 Decision 3:

- **Statevector core**, replacing `cuStateVec`-class device statevector execution,
  including multi-rank sharding.
- **Tensor-network core**, replacing `cuTensorNet`-class contraction and
  matrix-product-state execution.
- **Dynamics core**, replacing `cuDensityMat`-class time evolution. This is a
  maturation of the existing `flagquantum/simulation/lindblad.py` engine rather
  than a new implementation; its present CPU-only, dense, static-Hamiltonian
  limits are recorded in [`docs/reference/KNOWN_LIMITATIONS.md`](../reference/KNOWN_LIMITATIONS.md)
  and are lifted under their own evidence.

Each is admitted as a **replacement** of an existing backend implementation behind
an existing boundary, per control-sequence clause 2: admission requires a
replacement test in which at least one implementation is swapped without
modifying its consumers.

Two constraints on these cores are architectural rather than numerical.

**Device and operator selection does not live in the core.** All three reach
hardware through the Compute layer and FlagOS. Numerical implementation stays in
Simulation; planning and one execution attempt stay in Runtime. A core that
selects its own device, or that plans its own execution, has crossed a boundary
and is a defect regardless of its numerical correctness.

**A core that cannot be swapped is not finished.** Because clause 2 admits each
core as a replacement, a core is only complete when the implementation it
replaces can be removed without changing its callers. This is the operational
meaning of architecture principle 10 for this programme.

Performance is measured per workload against a named reference, never as a
framework-wide multiple. The first such measurement is recorded in
[`docs/development/CUDAQ_PARITY_BASELINE.md`](../development/CUDAQ_PARITY_BASELINE.md)
and its payload, and it covers one workload shape on one device per host. It is
explicitly not scalability evidence, and neither is any later single-node result.

## 5. What may be borrowed

Borrowing a vendor-neutral project is permitted, and is the intended route for the
`B_open_neutral` rows. It is not unrestricted: every adoption remains subject to
`dependency-policy.toml` and to engineering decision principle 6, which requires a
documented need, an ownership boundary, a licence and supply-chain review, a
replacement interface, and an exit plan.

No project below is currently a FlagQuantum dependency. The table records
ownership of the **licence check**, not approval to adopt. Licences were read from
each project's own repository metadata on 2026-09-30; the licence column must be
re-verified inside the adoption review, because an upstream licence can change and
because several of these projects ship bundled components under different terms.

| Project | Upstream licence | What it would close | Note |
| --- | --- | --- | --- |
| LLVM / MLIR | Apache-2.0 with LLVM exceptions | `mlir_dialect_stack`, compiler infrastructure |  |
| QIR specification | Community Specification License 1.0 | `qir_code_generation` | A specification licence. Implementing to it is the route; it grants no code. |
| Stim | Apache-2.0 | `backend_stim_stabilizer`, `qec_stim_integration`, `detector_error_model` | Also recorded as Apache-2.0 in the parity contract. |
| PyMatching | Apache-2.0 | `qec_decoder_family` | Minimum-weight perfect matching. |
| Qualtran | Apache-2.0 | `algorithm_block_encoding_family` | Faithful and fault-tolerant algorithm mathematics. `logical_resource_estimation` was withdrawn from this row: it landed natively over the compiler's own resource estimate rather than by adaptation, so the licence check below no longer applies to it. |
| cotengra | Apache-2.0 | contraction-path search for the tensor-network core | Contraction ordering is search, not physics; owning it adds no moat. |
| quimb | Apache-2.0 | tensor-network reference implementations |  |
| opt_einsum | MIT | `einsum` path optimisation |  |
| ITensor | Apache-2.0 | tensor-network reference implementations | C++. |
| QuEST | MIT | statevector reference semantics |  |
| PySCF | Apache-2.0 | `chemistry_domain_library` |  |
| QuTiP | BSD-3-Clause | dynamics reference semantics |  |
| Ensmallen | BSD-3-Clause | classical optimiser coverage | Header-only C++. |
| nlohmann/json | MIT | serialisation |  |
| Eigen | MPL-2.0 | dense linear algebra | File-level copyleft. A few bundled modules carry other terms, including GPL-licensed optional external dependencies that must not be pulled in. |
| GMP / MPFR | LGPL-3.0-or-later | exact and arbitrary-precision arithmetic for angle synthesis | Weak copyleft, not permissive. ARCH-012 lists both as vendor-neutral; they are usable under a dynamic-linking arrangement, but they cannot be statically absorbed into a FlagQuantum binary. See rule 4. |
| NLopt | Not a single licence | classical optimiser coverage | Its own terms are permissive, but it bundles algorithms under other terms. Needs per-algorithm review before use. |

Four rules govern this table.

1. **An evaluation algorithm is not reimplemented.** Where a published algorithm
   already exists under a permissive licence, the work is conformance and
   ownership of the result, not a fresh derivation.
2. **A numerical core is not borrowed.** The three cores of § 4 are the moat.
   Depending on a third party for statevector, tensor-network, or dynamics
   execution would recreate the problem the programme exists to solve.
3. **Borrowing a project does not borrow its support model.** An adopted project
   needs a named internal owner, a pinned version in `dependency-policy.toml`, and
   an exit plan, in the same way the existing interop extras have.
4. **A copyleft licence is a different proposition from a permissive one, and the
   table records the difference.** "Vendor-neutral" and "may be used" are not the
   same statement: GMP and MPFR are vendor-neutral and copyleft, Eigen is
   vendor-neutral and file-level copyleft, and NLopt's per-algorithm terms vary.
   None of these is excluded, but each needs a linking and distribution decision
   taken deliberately rather than inherited from a class label.

## 6. What the programme does not do

The following are declined, and the declination is part of the strategy rather
than an omission.

- **Do not depend on CUDA-Q as an execution backend.** It makes CUDA-Q
  capabilities rented rather than owned, and it contradicts the ownership
  constraint. The existing one-way export documented in
  [`docs/development/CUDAQ_EXPORT_CONTRACT.md`](../development/CUDAQ_EXPORT_CONTRACT.md)
  remains available for interoperability, and is not a parity mechanism.
- **Do not adopt, wrap, or ship a proprietary simulation core**, whether from
  CUDA-Q or independently. This includes the `cuStateVec`, `cuTensorNet`, and
  `cuDensityMat` families.
- **Do not reimplement error-correction decoders from scratch.** Decoder work is
  integration over published algorithms, with acceleration later through the
  FlagOS kernel path.
- **Do not build a second pass manager, a second IR, or a second artifact
  boundary.** The internal levels are already designed in
  `docs/architecture/MULTI_LEVEL_IR_ARCHITECTURE.md` and `IR-001` through `IR-006`
  are approved. Parity activates that design; it does not replace it.
- **Do not promise realtime control as a software deliverable.** The software side
  of `realtime_transport` and `realtime_host_api` — a vendor-neutral messaging
  protocol and a device-call abstraction — can be owned and delivered. The
  `realtime_feedback_loop` hardware side cannot. It is delivered, if at all, with a
  hardware partner, and is represented as an owned, blocked row rather than as a
  plan.
- **Do not treat a single-node measurement as scalability evidence.** No parity
  claim may rest on a result whose payload does not carry rank ownership,
  communication, memory, and an explicit statement that a scalability claim is
  not allowed.
- **Do not close a row by editing its status.** The scoreboard is generated, and a
  hand edit to it is a defect.

## 7. How progress is governed

Progress is a property of the generated scoreboard and of the gates around it, not
of a narrative.

- `contracts/cudaq-parity-matrix.toml` is the source of truth; the generated
  [`docs/reference/CUDAQ_PARITY_MATRIX.md`](../reference/CUDAQ_PARITY_MATRIX.md) is
  the scoreboard; `tools/parity_matrix.py --check` fails on drift and is wired into
  CI.
- A probe may claim `measured` only with a landed payload under
  `benchmarks/results/` that `benchmarks/audit_results.py` accepts, together with
  the source revision, container image, environment, and methodology recorded
  beside it.
- Every `supported` row names a `capability-maturity.toml` entry. Traceability runs
  one way: a parity row cites a maturity level and never sets one.
- Capabilities CUDA-Q lacks are recorded in the same matrix rather than in a
  separate document, so that "at parity" and "ahead" cannot drift apart.

## 8. Risks accepted

**Native compilation infrastructure is the largest single uncertainty.** The
programme does not select a compiler implementation ahead of its evidence gate.
If the native route does not clear that gate, the fallback is a vendor-neutral
MLIR route, and the fallback is the reason `B_open_neutral` is an acceptable class
rather than a compromise.

**Domestic-accelerator numerical coverage is a hard constraint, not a target.**
Reduced-precision and unavailable-FP64 behaviour on the target accelerators
constrains the three cores directly. The plan is recorded in
[`DOMESTIC_ACCELERATOR_AND_NUMERICAL_TRUST_PLAN.md`](DOMESTIC_ACCELERATOR_AND_NUMERICAL_TRUST_PLAN.md),
and it binds the cores before it binds anything else.

**The programme is long.** Parity across the simulator matrix, compilation
infrastructure, error correction, and fault-tolerant layers is a multi-quarter
effort. The cost is a property of the requirement, and the mitigation is
sequencing rather than optimism: the `B_open_neutral` rows are closed first
because they are cheapest, while the cores advance under their own replacement
tests.

**Breadth is the failure mode.** 27 `unsupported` rows invite a sprint
across many shallow capabilities. Control-sequence clause 1 is the counterweight: a
round extends a proven vertical path through input, validation, planning,
execution, result, failure, and evidence. Breadth is earned by completing such a
path.

## 9. Maintenance

This document is refreshed when the goal statement changes, when an
`A_nvidia_proprietary` row is added or removed, when a project in § 5 is adopted or
dropped, or when a § 6 declination is revisited. It is an intent document and is
registered as such in [`docs/source_of_truth.json`](../source_of_truth.json);
capability facts belong in the contract, and measured results belong in
`benchmarks/results/`.

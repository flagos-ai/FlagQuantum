# Native fermionic science migration

## Current delivery status

The current checked main baseline is
`bd054faa02f5b3c5de82f97b5b9f7b19ad47ba7b` on November 13, 2026 (UTC+08:00).
General commuting Pauli discovery, its experimental registration and explicit
mixed-axis sector training are merged in #1166, #1167 and #1170. Documentation
closeout #1169 merged before #1170. The runtime materialization repair #1168
merged first. The bounded JW H2 training composition merged in #1171 and passed
actual-main acceptance below. The next three-encoding extension has its scope
recorded before implementation; it is separate from that accepted delivery.
The preceding parity-training baseline was
`e09b68e7add135dbc42f6bf677e00a611cfd463e`.
The preceding Z-discovery registration baseline was
`fc1df0e64da7a8087d5a64b58a439f23f1be515f`.
The earlier spinless training baseline was
`f073a800f040f8b8a667fa19397d1ceb14794200`; its evidence below remains
bound to that revision.
This is selected-workflow coverage, not complete OpenFermion API compatibility.
The earlier merged-baseline audit was
`ffb6e7de71b9b9c9e172b20ecbc49ac04f14a0c5`. The initial selected migration
and EXT01–EXT08 are merged.
The later delivery audit #1133, training/input/object compositions #1137–#1144,
InteractionOperator registration #1146 and frozen-core training proof #1147 are
also merged. The follow-up queue below preserves its earlier checked-revision
status; it is historical and no longer an outstanding merge queue.

Periodic spinful Hubbard assembly merged in [#1150](https://github.com/flagos-ai/FlagQuantum/pull/1150),
with separate experimental capability registration in
[#1151](https://github.com/flagos-ai/FlagQuantum/pull/1151).
That earlier `ffb6e7de` combined main revision passed 143 focused scientific, native
isolation, training, capability and documentation tests, both periodic/open-chain
examples and [Main smoke](https://github.com/flagos-ai/FlagQuantum/actions/runs/43161089138).
These results qualify that revision and bounded CPU correctness only.

Periodic Hubbard UCCSD training merged in
[#1153](https://github.com/flagos-ai/FlagQuantum/pull/1153). Its actual merge
commit `9fb91085d08597458004e4ef9aa2478f9d27f9a5` passed 147 related tests,
five Hubbard CLI/inventory checks and
[Main smoke](https://github.com/flagos-ai/FlagQuantum/actions/runs/43170010993).
This main revision also includes independently merged #1152; these results
qualify the bounded fermionic workflow, not all scientific behavior on main.
The ring training example lowers energy from 3.4853161208 to -2.6945627040
in 60 Adam steps, with each spin population one. It does not reach the analytic
ground energy -3.1231056256 and makes no general convergence claim.

Spinless t-V chains and rings merged in
[#1154](https://github.com/flagos-ai/FlagQuantum/pull/1154), with separate
experimental capability registration in
[#1155](https://github.com/flagos-ai/FlagQuantum/pull/1155).
Each site owns one mode, the unshifted interaction is V*n_i*n_j on each
neighboring bond, and coefficients remain static finite real numbers. The
[spinless golden path](FERMION_SPINLESS.md) covers independent occupation-bit
matrices, free spectra, closing-bond signs, density terms and an actual circuit
angle gradient/optimizer update. Its shortest workflows are:

```bash
python -m examples.algorithms.fermion_spinless
python -m examples.algorithms.fermion_spinless --periodic
```

At three sites, t=1, V=2, mu=0 and N=2, the chain and ring measure
-0.7320508076 and 1.0000000000 respectively through JW, parity and BK.
Dense eigenstate preparation remains an exponential correctness inspection.
The earlier spinless model baseline
`9d1f5f34b9eae002f1c0e74714c2cb8e00bc1080` passed 235 related tests, seven spinless/Hubbard
CLI/inventory checks, both spinless examples and
[Main smoke](https://github.com/flagos-ai/FlagQuantum/actions/runs/43203085504).
The reviewed capability head passed all 233 full Linux CI jobs on
[run 43195279000, attempt 2](https://github.com/flagos-ai/FlagQuantum/actions/runs/43195279000/attempts/2).
Its first attempt failed while resolving an artifact action after the affected
coverage shard passed 210 tests; the targeted rerun completed successfully.
[#1156](https://github.com/flagos-ai/FlagQuantum/pull/1156) separately increased
the native chemistry CI install budget after earlier optional-download timeouts;
those cancelled runs remain failures, not passing scientific evidence.
These results establish bounded CPU correctness, not full mainline scientific
certification or accelerator/distributed support.

The coverage audit [#1158](https://github.com/flagos-ai/FlagQuantum/pull/1158)
and spinless training composition
[#1159](https://github.com/flagos-ai/FlagQuantum/pull/1159) are merged.
The spinless training baseline `f073a800f040f8b8a667fa19397d1ceb14794200`
passed 161 focused scientific, native-isolation,
training, documentation and capability tests, six spinless CLI/inventory checks,
generated-documentation and dependency-policy validation, and
[Main smoke](https://github.com/flagos-ai/FlagQuantum/actions/runs/43217014837).
The training guide records both fixed 160-step CPU runs; only circuit angles
are trainable. No capability promotion or new model API was needed.
The first #1159 CI run failed five dependency-floor jobs on Black 24.1 string
formatting; the corrected final head passed both formatter versions and its
full CI before merging. Earlier failed/cancelled runs retain their own status.

Arbitrary graphs, coefficient gradients, automatic matched encoded/tapered
ansatz construction, accelerator and distributed migration evidence remain
separate possible extensions. They are not unfinished units of the completed
initial migration. No automatic merge approval or capability promotion follows
from this plan.

## Historical delivery audit

Status checked on November 12, 2026, against main revision
`98ea298bd7de9b0e8d0f905adc6b333aa1aadaed` (the merge of #1125). This is a
selected native scientific workflow, not complete OpenFermion API compatibility.
The initial PR01–PR16 units below describe internal responsibilities delivered by one initial PR;
they are not sixteen outstanding pull requests.

| Scope | Delivery | Status at the checked revision | Shortest workflow |
| --- | --- | --- | --- |
| Initial selected migration | [#1109](https://github.com/flagos-ai/FlagQuantum/pull/1109) | Merged | `python -m examples.algorithms.fermion_chemistry` |
| EXT01: spinful Fermi-Hubbard open chain | [#1124](https://github.com/flagos-ai/FlagQuantum/pull/1124) | Merged; static open chains | `python -m examples.algorithms.fermion_hubbard` |
| EXT02: real closed-shell FCIDUMP input | [#1110](https://github.com/flagos-ai/FlagQuantum/pull/1110) | Merged | `python -m examples.algorithms.fermion_fcidump` |
| EXT03: MolecularData HDF5 input | [#1115](https://github.com/flagos-ai/FlagQuantum/pull/1115) | Merged; optional h5py | `python -m examples.algorithms.fermion_hdf5` |
| EXT04: fermionic Fourier circuit | [#1118](https://github.com/flagos-ai/FlagQuantum/pull/1118) | Merged | `python -m examples.algorithms.fermion_fourier` |
| EXT05: optional OpenFermion operator conversion | [#1125](https://github.com/flagos-ai/FlagQuantum/pull/1125) | Merged; separately installed upstream | [Conversion guide](OPENFERMION_OBJECT_INTEROP.md) |
| EXT06: explicit Pauli stabilizer sector | [#1119](https://github.com/flagos-ai/FlagQuantum/pull/1119) | Merged | `python -m examples.algorithms.fermion_tapering` |
| EXT07: controlled low-rank evolution | [#1121](https://github.com/flagos-ai/FlagQuantum/pull/1121) | Merged | `python -m examples.algorithms.fermion_controlled_evolution` |
| EXT08: real orbital-parameter gradients | [#1122](https://github.com/flagos-ai/FlagQuantum/pull/1122) | Merged; real CPU orbital parameters | `python -m examples.algorithms.fermion_orbital_rotation` |

All eight numbered extension slices now have a merged delivery. EXT01 delivers
one spinful open-chain model, not a general model library; EXT05 delivers two
operator types, not the upstream object model. Completion of these slices does
not establish complete OpenFermion compatibility or promote experimental
capabilities to production maturity. The #1123 documentation reconciliation is
also merged; this audit supersedes its checked-revision status table only.

The following initial-batch records and CI counts retain their original scope.
Later extensions and commits require their own evidence. The original branch
was `feat/native-fermion-migration`, based on `2a073ab9d`; its upstream reference
was OpenFermion `34adf3196cd566f4ca53893f868753786797ae2a`.

## Module boundaries and composition

| Responsibility | Owning module | Consumer boundary |
| --- | --- | --- |
| Canonical fermion algebra | `observables.fermion` | Existing public operator type is reused |
| Complex Pauli values | `core.algebra.pauli` | Values contain no execution objects |
| Static molecular data and active spaces | `algorithms.chemistry_domain` | `MolecularProblem` owns static snapshots |
| PySCF, FCIDUMP and HDF5 input | `ecosystem.chemistry` | External data becomes `MolecularProblem` immediately |
| Encodings and operator-sector reduction | `algorithms.fermion.encoding`, `symmetry`, `stabilizer` | Encoded/reduced operator contracts remain explicit |
| Static model Hamiltonians | `algorithms.fermion.hubbard` | Returns existing canonical fermion values; owns no solver |
| Real orbital-parameter gradients | `algorithms.chemistry_domain.orbital_rotation` | Tensor rotation and determinant objective; static snapshots remain separate |
| Optional external operator conversion | `ecosystem.openfermion` | Foreign objects stop at existing native fermion/Pauli values |
| Static orbital mathematics | `algorithms.fermion.decomposition`, `fourier`, `low_rank` | Mathematical plans do not own execution |
| Preparation and evolution circuits | `algorithms.fermion.preparation`, `lowering`, `evolution` | Ordinary circuits are executed by existing `fq.run` |

File readers and the direct driver share the same active-space, encoding and
measurement consumers. Circuit preparation and evolution use full unreduced JW
occupation registers; an encoded or reduced operator does not automatically
provide a matching ansatz. Explicit state lifting is a bounded inspection path.
Routing and low-rank evolution retain separate responsibilities.

Core/native fermionic workflows require neither OpenFermion nor Cirq. The
repository still offers `cirq-core` through optional Cirq interoperability
extras. The separately installed `openfermion==1.7.1` extra is excluded from
core, `all`, and `interop-all`; upstream may bring Cirq transitively when that
explicit extra is installed. This migration removes the native science path's
need for Cirq, not the independently maintained optional Cirq adapter. Optional
PySCF and h5py dependencies stop at their input adapters; they are not native
algebra dependencies. See the local
[fermion domain README](../../flagquantum/algorithms/fermion/README.md),
[chemistry domain README](../../flagquantum/algorithms/chemistry_domain/README.md)
and [input adapter README](../../flagquantum/ecosystem/chemistry/README.md).

## Selected-scope coverage and remaining gaps

This table reconciles the delivered plan at the current checked baseline. It is
not an exhaustive inventory of upstream symbols, a source-line comparison or a
percentage of OpenFermion compatibility. Module paths are relative to
`flagquantum/`; representative tests are relative to `tests/`.

| Responsibility | Authoritative implementation | Representative evidence | Explicit boundary |
| --- | --- | --- | --- |
| Fermion and Pauli values | `observables/fermion.py`, `core/algebra/pauli.py` | `unit/fermion/test_encoding.py`, `unit/test_openfermion_native_boundary.py` | No upstream objects or execution policy in algebra |
| JW, parity and BK encodings | `algorithms/fermion/encoding.py` | `unit/fermion/test_encoding.py`, three-encoding energy examples | Explicit register width and occupation mapping; no automatic matched ansatz |
| Static molecular snapshots and active spaces | `algorithms/chemistry_domain` | `unit/fermion/test_chemistry.py`, `integration/test_fermion_active_space_training.py` | No geometry, SCF or supplied-integral gradients |
| File, driver and populated object input | `ecosystem/chemistry` | FCIDUMP/HDF5 tests and molecular input replacement scenarios | Real closed-shell spatial data; PySCF/h5py remain optional adapters; no implicit solver |
| Optional upstream operator conversion | `ecosystem/openfermion.py` | `unit/test_openfermion_conversion.py`, `integration/test_openfermion_interop.py` | FermionOperator/QubitOperator import/export and InteractionOperator import; no circuit conversion |
| Static spinful and spinless chains/rings | `algorithms/fermion/hubbard.py` | `unit/fermion/test_hubbard*.py`, `unit/fermion/test_spinless.py` | No arbitrary graphs, twisted phases or coefficient gradients |
| Orbital decomposition and preparation | `algorithms/fermion/decomposition.py`, `preparation.py`, `lowering.py` | `unit/fermion/test_preparation.py` | Static CPU Slater determinants and Gaussian vacua; no full excited Gaussian basis transform |
| Fermionic Fourier basis change | `algorithms/fermion/fourier.py`, `lowering.py` | `unit/fermion/test_fourier.py` | Power-of-two widths; adjacent circuit has O(n**2) factors; no FFT speedup claim |
| Low-rank and controlled evolution | `algorithms/fermion/low_rank.py`, `evolution.py` | `unit/fermion/test_evolution.py`, `test_controlled_evolution.py` | Real spatial closed-shell order-1/2 formulas, static factors, scalar-time gradients, unreduced JW |
| Explicit symmetry sectors | `algorithms/fermion/symmetry.py`, `stabilizer.py` | `unit/fermion/test_symmetry.py`, `test_stabilizer.py` | Explicit eigenvalues for supplied or discovered commuting generators; no automatic sector selection or ansatz tapering; lifting is bounded inspection |
| Symmetry discovery | `algorithms/fermion/symmetry.py`, `stabilizer.py` | `unit/fermion/test_parity_discovery.py`, `test_pauli_discovery.py`, independent projector integration tests | Exact static coefficients; general discovery returns a maximum-size commuting subgroup, not the full centralizer; no particle-number or ground-sector inference |
| Real orbital-angle gradients | `algorithms/chemistry_domain/orbital_rotation.py` | `unit/fermion/test_orbital_rotation.py` | Real CPU orbital parameters; static snapshot boundary unchanged |
| Training composition | Existing circuit/ansatz, observable and PyTorch interfaces | Hubbard, periodic Hubbard, spinless, FCIDUMP, active-space, object-replacement, explicit parity-sector and mixed-axis Pauli-sector training scenarios | Circuit-angle optimization only; no separate fermion solver or general convergence guarantee |
| Native dependency isolation | Native modules and explicit ecosystem boundaries | `integration/test_native_fermion_isolation.py`, fresh-process example scenarios | Native workflows need no OpenFermion/Cirq; optional interoperability may install them |

Upstream adaptations retain their license and attribution in
`third_party/openfermion/`, pinned to
`34adf3196cd566f4ca53893f868753786797ae2a`. The provenance record covers the
selected adaptations; it does not certify all upstream code as migrated.

## Proposed next PR sequence

The bounded spinless training sequence below is complete at the current checked
baseline. The heading is retained for existing links; these rows are delivery
records rather than an outstanding queue or approval for additional work.

| Step | Delivery | Acceptance |
| --- | --- | --- |
| Spinless training composition | [#1159](https://github.com/flagos-ai/FlagQuantum/pull/1159), merged | Chain/ring replace only the static producer; independent occupation references, angle gradients, optimizer updates, sector preservation, separate closing-bond checks and native dependency isolation |
| Conditional capability registration | Not needed for this composition | Existing static model constructors remain experimental with unsupported coefficient gradients; ordinary circuit-angle training changes no catalog support boundary |
| Combined-main acceptance | Main `f073a800f040f8b8a667fa19397d1ceb14794200` | 161 related tests, six CLI/inventory checks, documentation/dependency validation and Main smoke passed; see current delivery status |

At the spinless closeout, no further implementation slice was selected. The
later diagonal discovery, explicit parity-sector training and general Pauli
discovery deliveries are recorded at the end of this plan. Each future proposal
must name its user workflow, owning domain, failure boundary and independent
acceptance evidence before implementation.

Keep future arbitrary-graph models, automatic matched encoded ansatz,
coefficient/SCF gradients and accelerator/distributed qualification as separate
proposals with their own requirements and evidence. None is a prerequisite for
the bounded spinless training path. Each proposal must reuse existing native
values and keep adapters, mathematics, circuit construction and execution in
their owning domains. A new horizontal abstraction requires a demonstrated
replacement through an existing boundary, not speculative extension points.

## Historical post-merge boundary audit

This audit inspected package metadata, the dependency policy, scientific module
imports, the stage contracts and existing scenario tests at the checked main
revision. Reproduce the native boundary check without installing upstream:

```bash
python -m pytest tests/integration/test_native_fermion_isolation.py tests/integration/test_openfermion_interop.py tests/unit/test_openfermion_native_boundary.py -q
python tools/check_dependency_policy.py
python -m examples.algorithms.fermion_hubbard
python -m examples.algorithms.fermion_orbital_rotation
```

The first isolation scenario blocks OpenFermion, Cirq, PySCF, h5py, NumPy and
SciPy in a fresh interpreter while executing native chemistry, encodings,
preparation, evolution, Fourier, sector reduction, orbital-gradient and Hubbard
workflows. The adapter scenario also imports the conversion module without
loading upstream and checks the missing-dependency error on explicit calls.
Provider replacement tests cover native conversion semantics and object
ownership. They complement the real OpenFermion 1.7.1 tests in
`.github/workflows/openfermion-interop.yml`; they do not prove compatibility
with arbitrary upstream versions or types.

| Remaining boundary | Delivered evidence | Work still outside that evidence |
| --- | --- | --- |
| Model Hamiltonians | One static spinful open chain, independent occupation-bit and analytic references | Periodic boundaries, arbitrary graphs, spinless variants and trainable coefficients |
| External objects | Explicit FermionOperator/QubitOperator conversion to existing native values | MolecularData object conversion, InteractionOperator and circuit conversion; HDF5 file input already exists separately |
| Differentiation | Native scalar-time evolution gradients and real orbital-angle determinant gradients | Geometry, SCF, supplied-integral and static decomposition gradients |
| Encodings and sectors | Three operator encodings and explicitly supplied stabilizer sectors | Automatic symmetry discovery and automatically matched encoded/tapered ansatz synthesis |
| Execution | Bounded local CPU scientific correctness and native dependency isolation | Migrated-workflow accelerator, distributed training, performance and capacity claims |

## Historical follow-up delivery queue

The following work is implemented on review branches and remains **unmerged**
at the checked main revision. A green branch is not a mainline capability.
The earlier recommendation for a native variational-training composition has
been implemented by the first row; it is no longer a proposal awaiting code.
All rows retain bounded local CPU scope and experimental maturity.

| Delivery | Responsibility | Review artifact and shortest branch-specific verification | Dependency |
| --- | --- | --- | --- |
| Native Hubbard training | Compose existing circuits, observable conversion and PyTorch optimization | [#1137](https://github.com/flagos-ai/FlagQuantum/pull/1137); `python -m examples.algorithms.fermion_hubbard_training` | Existing main APIs |
| FCIDUMP UCCSD training | Replace the chemistry input and reuse the existing UCCSD circuit | [#1138](https://github.com/flagos-ai/FlagQuantum/pull/1138); `python -m examples.algorithms.fermion_fcidump_training` | Existing main input and ansatz APIs |
| Molecular input replacement | Prove native, FCIDUMP and HDF5 inputs preserve training consumers | [#1139](https://github.com/flagos-ai/FlagQuantum/pull/1139); use its focused replacement/isolation commands | Existing main APIs; optional h5py for file comparison |
| InteractionOperator input | Convert an external spin-orbital tensor operator to native fermionic algebra; verify UCCSD replacement | [#1140](https://github.com/flagos-ai/FlagQuantum/pull/1140); `python -m pytest tests/unit/test_openfermion_conversion.py -k interaction -q` | Existing optional OpenFermion boundary |
| Molecular object CI routing | Trigger real upstream qualification when the molecular object adapter changes | [#1141](https://github.com/flagos-ai/FlagQuantum/pull/1141) | Integration-owned prerequisite for #1142 |
| Populated MolecularData input | Convert cached real closed-shell spatial integrals to MolecularProblem without implicit loading | [#1142](https://github.com/flagos-ai/FlagQuantum/pull/1142); `python -m pytest tests/unit/test_openfermion_conversion.py -k molecular_data -q` | #1141 first |
| Molecular object capability registration | Register experimental scope and generate catalog/limitations | [#1143](https://github.com/flagos-ai/FlagQuantum/pull/1143); `python tools/check_capability_maturity.py` and `python tools/docs_source_of_truth.py --check` | Stacked on #1142 |
| Molecular object training replacement | Compare 24 native/imported UCCSD energy, gradient and optimizer-update steps | [#1144](https://github.com/flagos-ai/FlagQuantum/pull/1144); `python -m pytest tests/unit/test_openfermion_conversion.py -k molecular_data_replacement -q` | Stacked on #1142; independent of #1143 |

Commands in this table require the named PR's checkout until it merges.
Do not present the examples or adapters as available on the checked main.
The optional object tests require separately installed OpenFermion 1.7.1;
they belong to the dedicated upstream lane, not native coverage. That upstream
installation may include Cirq, while native chemistry and training remain
independent. MolecularData HDF5 input already on main is a file reader, not the
new in-memory object adapter. Neither adapter calls an upstream solver.

Each review retains its exact head, CI links, scientific scope and failure
history. For example, #1144 corrected an extra integration marker that selected
an upstream-dependent scenario into native coverage; its corrected head is
`614cbe74dd7d7ae0ea0bf50218cc1fbcfb11ff53`. Its full Linux run
[43058554427](https://github.com/flagos-ai/FlagQuantum/actions/runs/43058554427)
and both real upstream endpoints passed. Those branch results do not certify
a later combined tree. Historical initial-batch numerical records below remain
bound to their original revisions.

## Historical acceptance sequence

The next round consolidates these proven paths before adding another external
object type, model family or solver abstraction:

1. Review and merge the independent native training/input compositions and the
   operator converter. Merge #1141 before #1142. After #1142 merges, retarget
   #1143 and #1144 to main; neither sibling requires the other. Recheck conflicts
   and run the checks for the actual target. This plan grants no merge approval.
2. Verify the resulting main revision with the existing native examples,
   dependency-isolation scenarios and replacement tests. Record the actual
   integration commit; do not add passing counts from different branches.
   Linux default, runtime and coverage verification remain CI-owned.
3. Keep the current ownership boundaries: Ecosystem ends at native values;
   chemistry owns static integrals and active spaces; algorithms owns ansatz
   composition; runtime executes ordinary FlagQuantum IR. No solver, registry,
   compatibility contract or second gradient implementation is needed.
4. Reconcile the authoritative capability entries against the combined code.
   Integration-owned capability changes remain separate from domain work;
   #1143 records only MolecularData input and does not silently widen the
   operator-only capability. InteractionOperator scope needs its own accurate
   registration when admitted. Do not change Stable Core snapshots for this.
5. Keep unsupported breadth explicit: arbitrary upstream objects/circuits,
   geometry/SCF/supplied-integral gradients, automatic symmetry discovery,
   matched parity/BK/tapered ansatz synthesis and migrated-workflow accelerator
   or distributed claims require separate requirements and evidence. Do not
   label them unfinished pieces of the already merged selected migration.

The shortest existing mainline scientific path remains
`python -m examples.algorithms.fermion_chemistry`; its native H2 reference
measures -1.1166843871 Ha through three encodings. After the training PRs merge,
their examples extend the same path to parameter updates. Dense reference
matrices establish bounded correctness, not performance, general optimizer
convergence, accelerator support or scalability.

## Initial-batch delivery units

| Unit | Capability | Boundary |
| --- | --- | --- |
| 01 | Scope, provenance and isolation | Development and architecture |
| 02 | Existing canonical fermion algebra | Reuse public type without relocation |
| 03 | Complex Pauli algebra | Core values and explicit observable conversion |
| 04 | Integral and molecular problem data | Static native chemistry |
| 05 | JW, parity and Fenwick BK | Operator encoding |
| 06 | Encoded reference occupations | Shared energy consumer |
| 07 | Explicit diagonal parity sector reduction | Operator and state embedding |
| 08 | Active-space selection and frozen core | Native molecular transformation |
| 09 | Optional direct RHF PySCF driver | Ecosystem |
| 10 | Adjacent complex Givens decomposition | Static mathematical rotations |
| 11 | Slater determinant preparation | Native circuit synthesis |
| 12 | Canonical Gaussian vacuum preparation | Native circuit synthesis |
| 13 | Real chemist-integral low-rank decomposition | Static numerical factors |
| 14 | Fermionic swaps and final layout | Native circuit synthesis |
| 15 | Low-rank first/second-order evolution | Native differentiable time |
| 16 | Compatibility, checks and review draft | Local release preparation |

EXT06 and EXT07 are independent extensions tracked after the selected migration
batch. EXT07 reuses the existing evolution circuit and `Circuit.control`; its
contract and limits are in `FERMION_CONTROLLED_EVOLUTION.md`.

General non-diagonal stabilizer tapering, FFFT and file formats were outside
the initial batch and were delivered separately as shown above. Complete
upstream API compatibility, geometry gradients and distributed training remain
outside the delivered scope. Unit 07 deliberately starts with diagonal parity
generators, an exact and independently verifiable subset.

## Existing workflow preserved

```python
import flagquantum as fq
from flagquantum.observables import jordan_wigner, number

circuit = fq.Circuit(2)
circuit.x(0)
observable = jordan_wigner(number(0), n_modes=2)
result = fq.run(circuit, outputs=[fq.expectation(observable)])
```

New examples and actual validation results are recorded as their implementations
land. This document does not assert that planned interfaces already exist.

## Review rules

Every implementation commit includes focused behavior tests and source metadata
where code is adapted. Core imports no execution or vendor objects. Optional
chemistry absence must not break native algebra or circuit imports. New domains
have a README with a shortest executable example and focused test command.

Integration-owned boundary changes precede their implementations in local
history. Keep machine-readable protected API snapshots unchanged. Stop only the
affected part if an unavoidable stable contract conflict is discovered and
present the concrete compatibility options.

## Implemented modules and support matrix

| Module | Owns | Verified scope |
| --- | --- | --- |
| core.algebra.pauli | Complex canonical Pauli values | Standard library, no circuit or tensor dependency |
| algorithms.fermion.hubbard | Static spinful open-chain model | Explicit interleaved spins, unshifted U, finite real coefficients |
| ecosystem.openfermion | Optional external operator conversion | FermionOperator/QubitOperator only; lazy upstream dependency |
| algorithms.chemistry_domain.orbital_rotation | Differentiable integral rotation and determinant contraction | Real CPU orbital parameters; static source integrals |
| algorithms.fermion.encoding | General operator and occupation encoding | JW, prefix parity, Fenwick BK; explicit idle width |
| algorithms.fermion.observables | Hermitian measurement boundary | Complex values validated before native Observable conversion |
| algorithms.fermion.symmetry | Selected diagonal parity sectors | Independent encoded Z products; affine state embedding |
| algorithms.chemistry_domain | Static molecular data and active selections | Real spatial chemist ERIs, closed shell, interleaved spin, explicit offsets |
| ecosystem.chemistry.pyscf | Optional RHF integral driver | H2 STO-3G/6-31G, LiH STO-3G; native data returned immediately |
| algorithms.fermion.decomposition | Adjacent orbital unitary factors | Complex unitary reconstruction and determinant phase |
| algorithms.fermion.lowering | Small native orbital gates | JW occupation order; one- and two-qubit matrices |
| algorithms.fermion.preparation | Slater and canonical Gaussian vacua | Complex minors, canonical annihilator constraints, even/odd parity |
| algorithms.fermion.low_rank | Signed density-square factors | Real chemist ERIs, explicit rank, full one-body correction |
| algorithms.fermion.swap_network | Adjacent FSWAP and output ordering | Every logical pair once; explicit final layout and fermion signs |
| algorithms.fermion.evolution | First/second-order product formulas | JW full register, exact offset phase, differentiable scalar time |

All execution evidence here is local CPU complex128. Static decompositions own
CPU snapshots and reject autograd inputs. Time differentiation is verified with
finite differences; geometry, supplied integrals, decomposition orbital frames, rank selection and
Gaussian block differentiation are unsupported. EXT08 separately differentiates
real orbital parameters through integral rotation and a closed-shell determinant
objective; it does not differentiate the static decomposition path. Encoding and reduction results
are separate contracts: preparation/evolution circuits do not accept encoded or
reduced values and always use the unreduced JW occupation register.

The swap network is independently usable and tested. Low-rank evolution uses
explicit density-pair gates, which may be nonadjacent, and restores its input
layout after each orbital basis change. It does not insert the swap network or
claim nearest-neighbor connectivity. This avoids an unnecessary dependency
between evolution and routing. A future connectivity pass can consume ordinary
IR and handle this at the compiler boundary.

The low-rank approximation keeps the full contraction correction. Consequently,
a truncated density-square model is not obtained by simply replacing the ERI
and recomputing its correction. discarded_bound bounds omitted density-square
operator norms; it is not a bound on time integration error. The default retains
all eigenpairs. Dense helpers are bounded inspection paths, not scalability
implementations.

## Executable workflow and checks

```bash
python -m examples.algorithms.fermion_chemistry
# All three encodings: HF energy -1.1166843871 Ha.
# The example also differentiates a native evolution amplitude with respect to time.
python -m pytest tests/unit/fermion -q
python -m pytest tests/integration/test_native_fermion_isolation.py -q
python -m pytest tests/integration/test_native_chemistry_driver.py -q
```

The isolation test runs in a fresh interpreter that refuses imports of
OpenFermion, openfermionpyscf, Cirq, PySCF, NumPy, SciPy and h5py. It covers chemistry
data, all encodings, measurement, Slater/Gaussian preparation and differentiable
evolution. The optional PySCF entry raises an actionable missing-extra error in
that environment. Independent references use Fock-space ladder action, complex
Slater minors, annihilator-vacuum residuals and exact quadratic/full Hamiltonian
matrix exponentials. They do not require OpenFermion or Cirq as test oracles.


## Local acceptance evidence

On November 9, 2026, the complete default smoke/unit tier passed with 18,548
passes and 494 skips using four pytest workers, Python 3.12.14 and PyTorch
2.14.1 on local CPU. Focused science, existing fermion, vocabulary and import
isolation checks passed 161 tests; chemistry, molecular and Trotter regressions
including the direct PySCF integrations passed 356 tests. Release configuration
checks passed 62 tests. These are separate runs with overlapping selections;
their counts are not additive.

Ruff and Black pass on the changed Python scope; strict mypy passes on all 18
Python files in the added domains. Architecture, dependency policy, stable API,
capability maturity, team scope and qubit vocabulary gates pass without changes
to the stable API snapshots. The source archive and arm64 wheel build with the
existing isolated PyTorch 2.10 build configuration; all added domain sources and
distributed attribution were checked in the wheel. The local PR description is
in docs/development/OPENFERMION_NATIVE_PR.md. This local acceptance run preceded
the later publication and CI acceptance recorded below.

## Follow-up scientific and compatibility audit

On November 9, 2026, an independent environment installed OpenFermion from the
fixed source revision `34adf3196cd566f4ca53893f868753786797ae2a`.
A reference-only process generated numeric fixtures; native comparison processes
loaded only the arrays and did not import OpenFermion or Cirq. With seed 20261109,
both Python 3.12.14 / PyTorch 2.14.1 and the installed wheel on Python 3.10.22 /
PyTorch 2.10.0 passed the same 347 cases:

| Scope | Cases | Maximum absolute error |
| --- | ---: | ---: |
| Complex operator JW/parity/BK coefficients, widths 1 through 8 | 230 | 4.58e-16 |
| Complex Slater states, all particle counts through 7 modes | 35 | 3.45e-15 |
| Paired Gaussian vacua, including explicit parity flips, 2 through 7 modes | 30 | 6.38e-15 |
| Signed low-rank reconstruction, full contraction and discarded bounds | 40 | 1.71e-13 |
| Frozen-core shifts and reordered active-space integrals | 12 | 1.78e-15 |

Upstream state vectors are compared after aligning their unspecified overall
phase; native determinant-minor and evolution tests continue to check exact
phase separately. Upstream `parity_code(1)` raises a decoder reshape error, so
its ten width-one comparisons are excluded. Native width-one parity remains
covered by the independent Fock-space tests. This audit covers selected native
functions, not all upstream APIs or production-sized systems.

The core wheel was installed outside the source checkout on Python 3.10.22 /
PyTorch 2.10.0. Its native CPU extension loaded and 454 focused science and
existing fermion/chemistry/Trotter tests passed, with no NumPy, SciPy, h5py,
PySCF, OpenFermion or Cirq installed. After adding PySCF 2.8.0, all three direct
RHF tests passed with SciPy 1.14.1 and NumPy 2.2.6. A resolved SciPy 1.15.3 wheel
initially failed dynamic loading on this local macOS host; the successful driver
result applies to the explicitly tested SciPy 1.14.1 combination, not every
transitive dependency combination. No global dependency pin was added for that
local wheel issue.

The direct-driver tests now skip only when PySCF is absent. An installed driver
that fails import raises a test error. Isolated fault-injection probes verify
both behaviors. Final chemistry and native-import integration checks pass four
tests on each tested Python/PyTorch environment. Architecture, dependency,
stable API, ownership and capability gates pass. Review confirmed that Core
contains no execution imports, mathematical decomposition remains independent
of circuits, optional SDK objects stop at the chemistry adapter, existing
chemistry formulas are reused, and routing remains independent of evolution.
The audit made no production source or stable API changes.

## Final scope reconciliation and CI review

The original delivery units describe responsibilities, not sixteen remote PRs
or a commitment to preserve every proposed intermediate type. The implemented
scope makes these deliberate adjustments:

| Units | Final decision | Deferred or excluded work |
| --- | --- | --- |
| 01 | Fixed revision, provenance, import isolation and separate numeric reference process | Complete upstream API catalog and compatibility |
| 02-03 | Reuse FermionOperator in place; add complex Pauli values and explicit measurement conversion | Relocating stable types or rewriting existing Hermitian encodings |
| 04-06 | Native molecular snapshots and three explicit encodings share existing assembly and measurement | Automatic encoded ansatz selection |
| 07 | Exact independent encoded Z-parity sectors and affine embedding | General stabilizer tapering or automatic symmetry discovery |
| 08-09 | Reordered active selections, existing frozen-core formulas and direct optional closed-shell RHF | Automatic active spaces, open shells and geometry gradients |
| 10-12 | Mathematical factors and separate lowering create Slater and canonical Gaussian vacuum circuits | Additional preparation-plan IR, stored decomposition residual field or excited Gaussian basis synthesis |
| 13-15 | Signed density-square factors, independent FSWAP network and first/second-order evolution | Mandatory coupling of evolution to FSWAP or nearest-neighbor routing |
| 16 | Local installation, compatibility, attribution, review and qualified Linux CI | Maintainer review and merge authorization remain outstanding |

A decomposition is verified by reconstruction; it does not carry a separately
stored residual field. This keeps the mathematical boundary small. Circuit
builders consume the mathematical results directly through the existing IR
rather than introducing a second general preparation-plan language. JW/parity
formulas are intentionally retained in the protected Hermitian API while the
new general-complex path has its own small ladder images. Neither becomes an
adapter to an external framework.

The `native-fermion` Ubuntu matrix checks both Python/PyTorch endpoints and the
real optional RHF driver independently from the offline images. Coverage plan
and shard preflight also import PySCF explicitly. The driver skip condition is
visible at module scope, where the existing lane-dependency audit reads it;
missing PySCF skips, while an installed broken driver still fails.
The new matrix depends on source preparation and is required by `ci-complete`.
Existing merge-gate tests verify that failed, cancelled or skipped dependencies
cannot produce a successful final check. The testing manual lists all 27 jobs.

All four immutable CPU images were rebuilt and published. The lock records the
verified registry and offline OCI identities; the offline wrapper still compares
both dependency manifests byte for byte. The 192 coverage worker caches were
qualified against those identities. No manifest comparison, coverage floor or
merge gate was relaxed.

## Historical initial-batch remote CI acceptance

The following results are bound to implementation revision
`901472c81d72f52fa966520de47166557c52ed79`, before the documentation reconciliation:

| Check | Evidence | Result |
| --- | --- | --- |
| Full Linux workflow | CI run `42595072278`, attempt 2 | 233 jobs successful; both native science endpoints passed |
| Runtime selection | Same CI run | 6,215 passed, 403 skipped; real distributed launches passed |
| Coverage | Same CI run | 192 of 192 shards and aggregate passed |
| GPU workflow | Run `42595072337` | Passed |
| Container publication | Run `42595072334` | Passed |

These CI results complement the separately scoped local scientific evidence;
they do not establish native fermionic GPU scalability or full upstream parity.
Linux validation is owned by CI, as requested by the user. Local Docker was
unavailable during the earlier attempt and is not cited as execution evidence.

At that recorded pre-merge point, #1109 was open and #1110 was a stacked
FCIDUMP follow-up. The then-required order was to merge #1109, retarget #1110
to main and validate that target. Both PRs have since merged, as recorded in
the current delivery table. Any later commit, including
documentation changes, needs its own applicable checks; the runs above remain
evidence for their recorded revision.

## Independent extension: fermionic Fourier transform

EXT04 adds `algorithms.fermion.fourier.fourier_decomposition` and
`algorithms.fermion.lowering.fermionic_fourier_transform`. The former returns the
existing OrbitalDecomposition and never imports circuits; the latter reuses the
existing determinant-preserving two-mode lift. The radix-2 Cooley-Tukey
factorization is implemented directly from its mathematical identity; no new
OpenFermion or Cirq source is copied and no production dependency is added.
Forward F[p,q] = exp(-2j*pi*p*q/n)/sqrt(n), inverse F dagger, vacuum phase one,
natural mode order, positive powers of two only. Adjacent fermionic routing
costs O(n**2) factors and storage. This is a static full-register JW workflow,
not a qubit QFT or a claim of OpenFermion API compatibility or FFT speedup.

Run `python -m examples.algorithms.fermion_fourier`: every output mode has
occupation 0.25 and the inverse amplitude residual is below 1e-12. Focused tests
compare one-particle plans with torch.fft, small full Fock matrices and selected
eight-mode particle sectors with Fourier minors, retaining absolute phase.
The fresh-interpreter dependency isolation test also composes the Fourier and
inverse circuits with external scientific SDKs blocked. No chemistry input,
Core, runtime, stable export or dependency manifest changes are needed.

## EXT06: explicit commuting Pauli stabilizer reduction

The next stage extends diagonal Z parity sectors to supplied independent signed
X/Y/Z Pauli words. Its contract and independent acceptance criteria are in
[FERMION_STABILIZER_REDUCTION.md](FERMION_STABILIZER_REDUCTION.md).
`reduce_stabilizers` returns a reduced operator, native Clifford instructions and
a diagonal-frame ReductionMap. The latter is reused rather than expanded to
pretend an affine bit map can express X/Y eigenstate superpositions. Original
state lifting composes the inverse Clifford with that map. Symbolic algebra
imports no circuit, runtime or simulation implementation; lowering owns native
circuit construction and the explicitly bounded statevector inspection path.

Exact CX conjugation has one implementation shared with the original Z-parity
path. New tests check native Clifford matrices, all sectors against independent
dense projectors, the intertwining identity H E = E H_reduced, complex amplitude
norm/phase, signed and randomly Clifford-rotated generators, diagonal replacement
and invalid inputs. The H2 example reduces four qubits to two, measures energy
and particle number with fq.run, and verifies the lifted eigenvector residual.
No upstream source is copied, no dependency or Stable Core schema is changed,
and no full OpenFermion compatibility or automatic symmetry search is claimed.

## Independent extension: controlled low-rank evolution

EXT07 composes the existing `low_rank_evolution` circuit with the existing
`Circuit.control` operation; it adds no public algorithm wrapper or Core
contract. The system remains a full, spin-interleaved JW register and gains one
control wire. The controlled branch retains the energy-offset phase, which is
observable through X/Y interference. The H2 workflow measures the complete
overlap defect against a bounded dense reference for one, two and four second-
order steps. It reports measured error, not an error bound. The feature has no
new dependency and makes no claims for tapered encodings, multiple controls,
hardware routing, GPU or distributed execution. See
`docs/development/FERMION_CONTROLLED_EVOLUTION.md` and
`tests/unit/fermion/test_controlled_evolution.py`.

## EXT08: differentiable orbital parameters

[The stage contract](FERMION_ORBITAL_GRADIENTS.md) defines a local CPU tensor
workflow in `algorithms.chemistry_domain.orbital_rotation`. Static real chemist
integrals and antisymmetric generators produce differentiable rotated tensors.
The separate closed-shell contraction provides an orbital-optimization objective.
Static molecular snapshots, symbolic encodings and circuit lowering remain
independent and reject attempts to detach a trainable integral input silently.

Run `python -m examples.algorithms.fermion_orbital_rotation`. At angle 0.2 radians,
the H2 determinant energy is -1.0286951779 Ha and its gradient is 0.8576738812
Ha/rad, agreeing with central finite differences. One step of size 0.05 lowers
it to -1.0618547836 Ha. This is a determinant objective, not correlated energy,
geometry differentiation or a converged SCF calculation. The fresh-interpreter
isolation test also executes this example with external science SDKs blocked.

## EXT01 and EXT05: delivered narrow extensions

The [Hubbard contract](FERMION_HUBBARD_CHAIN.md) defines static open-chain
assembly in `algorithms.fermion.hubbard`. Its example measures the same two-site
N=2 reference through JW, parity and BK: energy -0.8284271247 with particle
number 2. Dense diagonalization and arbitrary-state preparation are bounded
inspection tools, not a scalable solver or training result.

The [operator conversion contract](OPENFERMION_OBJECT_INTEROP.md) defines the
optional ecosystem boundary. Conversion retains ordered contractions, complex
coefficients, identity and zero terms; exported mutable dictionaries do not
alias native values. It does not add an upstream execution backend. Install
`flagquantum[openfermion]` only when the explicit external-object workflow is
needed. Native algorithms and the isolated core environment require neither
OpenFermion nor Cirq.


## Diagonal symmetry discovery and explicit-sector training

Experimental Z-only discovery merged in #1161 and its separate capability
registration in #1162, after the spinless delivery closeout #1160. The checked
main revision `fc1df0e64da7a8087d5a64b58a439f23f1be515f` passed 134 related
tests, three CLI/inventory checks and Main smoke run 43234849277. These are
bounded CPU correctness results, not complete upstream or scalability evidence.
The [discovery guide](FERMION_PARITY_DISCOVERY.md) records candidate-span
semantics, encoded-qubit conventions and explicit sector ownership. Earlier
statements excluding automatic discovery refer to their original delivery scope.
General X/Y/Z discovery was still a gap at that recorded revision; its later
delivery is recorded below.

The follow-up merged in #1163 and composes that existing static boundary with reduced-angle
training in `python -m examples.algorithms.fermion_parity_training`. A fixed
two-mode pairing model explicitly selects even parity and trains one reduced
qubit. Independent occupation-space energy/gradient checks and JW/parity/BK
replacement tests accompany the example. It changes no Stable Core API,
dependency, runtime, symbolic gradient support or capability maturity level.
Actual main `e09b68e7add135dbc42f6bf677e00a611cfd463e` passed 459 related
tests (one optional-dependency module skipped), three CLI/inventory checks, the
public API baseline and Main smoke run 43247474492. These results remain bounded
local CPU correctness evidence. General matched-ansatz transformation, automatic
sector selection, coefficient
and SCF gradients, and accelerator/distributed qualification remain separate gaps.


## Current stage: general commuting Pauli discovery

[The stage scope](FERMION_PAULI_DISCOVERY.md) extends the existing stabilizer
producer with X/Y/Z discovery in an explicit encoded register. A general
centralizer may anticommute internally; discovery returns a deterministic
maximum-size independent commuting subgroup instead. Candidate spans are binary
modulo scalar phases, and sectors for returned positive words remain explicit.
Algorithms owns implementation, replacement tests and the executable
`python -m examples.algorithms.fermion_pauli_discovery` workflow. Capability
registration merged separately in Integration-owned #1167 after #1166.
Existing reduction and Stable Core contracts remain unchanged; discovery adds
no dependency, result type or runtime implementation.
The earlier general-X/Y discovery gaps refer to their original delivery scope.
Automatic sector selection, matched-ansatz transformation, coefficient/SCF
gradients and accelerator/distributed qualification remain separate gaps.


### Merged delivery and actual-main acceptance

| Delivery | Actual merge revision | Acceptance |
| --- | --- | --- |
| Runtime execution-plan materialization repair [#1168](https://github.com/flagos-ai/FlagQuantum/pull/1168) | `dfa6829f64e185a6c5cfe7e512bcefb6871b0ad6` | Current-head full CI passed; actual main passed 15 materialization/dynamic replay tests and Main smoke run 43288614889 |
| General Pauli discovery [#1166](https://github.com/flagos-ai/FlagQuantum/pull/1166) | `40c3e8a2c7e20d76b0e34052bf9336277fb8782d` | Full CI run 43289088073 passed; actual main passed 57 discovery/runtime tests, the API baseline, runnable example and Main smoke run 43298938014 |
| Experimental capability registration [#1167](https://github.com/flagos-ai/FlagQuantum/pull/1167) | `b9ad91149e7098218250286e9185f774aaf3dbf5` | Current-head CI run 43299388218, attempt 2, passed; final-main acceptance is detailed below |

On actual final main `b9ad91149e7098218250286e9185f774aaf3dbf5`, 81 focused
Pauli discovery, integration, runtime materialization, dynamic replay, capability
maturity and experimental-documentation tests passed. Capability maturity,
documentation source-of-truth `--check`, the public API baseline and
[Main smoke run 43308800001](https://github.com/flagos-ai/FlagQuantum/actions/runs/43308800001)
also passed. The overlapping earlier selections are not additive.

Reproduce the supported user workflow:

```bash
python -m examples.algorithms.fermion_pauli_discovery
```

It discovers Y0 and X1Z2, explicitly chooses +1/-1, reduces three qubits to one
and measures -1.1770329614, matching the analytic selected-sector energy. The
recorded lifted eigenvector residual was 2.776e-17 and may vary by platform.
This example uses bounded dense diagonalization and lifting for inspection; it
does not infer a ground sector or establish scalability.

The first #1167 CI attempt passed every test and coverage partition plus coverage
aggregation, but GitHub could not acquire a hosted macOS ARM runner for the
Python 3.10 wheel job. That job had no executed steps and was cancelled; the
final gate correctly failed. A job-specific retry acquired a runner and passed
wheel construction, tests, content verification and the downstream gate. No
source, test selection, timeout or CI runner configuration was weakened. The
earlier failed attempt remains a failure, rather than scientific acceptance.

This completes the selected general-discovery stage. Its capability remains
experimental with static coefficients and `single_device_fast_path` CPU scope.
Automatic sector selection, general matched-ansatz transformation, coefficient/
SCF gradients, full OpenFermion parity and accelerator/distributed qualification
remain separate work. This closeout introduces no implementation or maturity
promotion and does not itself select or authorize another feature or merge.

## Merged mixed-axis sector training

Documentation closeout #1169 merged as
`01dbf7ddb88e553f68b35b4a5eb490a8d9e3cb15`; explicit mixed-axis training #1170
merged as `2ab457f3067accf7629a4dcac3a60f0bd11b3f59`. Refreshed-head full CI
run 43322279268 passed. Actual final main passed 95 focused scientific,
documentation and capability tests plus six CLI/inventory checks, the public
API baseline, capability checker, documentation consistency and 1,417-file
link audit. Main smoke run 43330343697 passed. These overlapping selections
must not be added to historical counts as distinct scientific coverage.

```bash
python -m examples.algorithms.fermion_pauli_training
```

The fixed example lowers energy from 0.1678214299 to -1.1770329614. Its recorded
gradient residual was 6.528e-12, lifted energy residual 4.441e-16 and joint-sector
residual zero. Residuals may vary by platform. Only the reduced circuit angle
is trained; lifting remains static. See the
[training scope](FERMION_PAULI_TRAINING.md).

## Merged bounded H2 spin-sector training

The [H2 scope](FERMION_H2_SECTOR_TRAINING.md) was committed before implementation.
The merged workflow composes the existing FCIDUMP fixture, native molecular/JW
values, candidate-restricted spin-parity discovery, explicit -1/-1 reduction,
two-qubit RY/CX preparation, native expectations and PyTorch SGD:

```bash
python -m examples.algorithms.fermion_h2_sector_training
```

The bounded run lowers energy from -1.0649609749 to -1.1372701747 Ha and keeps
one electron in each spin block. Independent occupation-space tests verify
gradients, optimizer updates, sector support and file/native producer replacement.
The composition adds
no production API, scientific dependency or maturity promotion. Automatic sector
inference, general transformed ansatzes and coefficient/SCF gradients remain gaps.

Actual main `bd054faa02f5b3c5de82f97b5b9f7b19ad47ba7b` passed 90 focused
scientific/capability/documentation tests and seven CLI/inventory tests, the API
baseline, capability checker, documentation consistency and 1,419-file link
audit. Main smoke 43342082153 passed. Concurrent QEC #1157 changed the combined
tree, so full actual-main CI 43342446759 was run independently: all 236 jobs
passed on attempt one, including coverage aggregation and the final gate.
The original PR head passed full CI 43333357231 and 239 cross-workflow checks
with three conditional skips. These selections are not additive coverage counts.

The next [encoding replacement scope](FERMION_H2_SECTOR_TRAINING.md#encoding-replacement-scope-before-implementation)
extends this same molecular training consumer to parity and BK on a branch cut
from main `e087f51b01f412132d3bf575953d7202f2bc3fc2`. It plans independent
physical-state, gradient and update equivalence, without a production API or
automatic matched-ansatz claim. Its scope is recorded before implementation;
the preceding acceptance belongs to the merged JW workflow only.

# Compiler upstream reuse

This is an implementation audit, not a claim of parity or superiority. The
compiler must reuse useful upstream work while preserving FlagQuantum IR,
dependency direction, layout contracts, and independent correctness checks.
Naming a paper or a corresponding upstream pass is not evidence of a code port.

## Inspected sources

The routing implementation was inspected directly at these release revisions on
2026-11-19. Immutable links identify the exact reviewed source:

| Project | Release | Commit | Source |
| --- | --- | --- | --- |
| Qiskit | 2.6.2 | `0c26a5a158c4d62b2b303a8c2f3c6eb2b14653b4` | [`route.rs`](https://github.com/Qiskit/qiskit/blob/0c26a5a158c4d62b2b303a8c2f3c6eb2b14653b4/crates/transpiler/src/passes/sabre/route.rs) |
| Cirq | 1.7.0 | `de651a7886d1a1383d595f95f36b93ed3ea91685` | [`route_circuit_cqc.py`](https://github.com/quantumlib/Cirq/blob/de651a7886d1a1383d595f95f36b93ed3ea91685/cirq-core/cirq/transformers/routing/route_circuit_cqc.py) |

The respective source SHA-256 digests are
`66a475009562853e676d1cad43dcc5be8e9654e6ef37e3eed407122313998f8f`
and `deeefc23fdf13dedecedd727f78666a4d1e8ee1fa21d667437c9c6717cf42e88`.

Two additional inspected references are Qiskit's
[`two_qubit_decompose.py`](https://github.com/Qiskit/qiskit/blob/0c26a5a158c4d62b2b303a8c2f3c6eb2b14653b4/qiskit/synthesis/two_qubit/two_qubit_decompose.py)
and Cirq's
[`two_qubit_to_cz.py`](https://github.com/quantumlib/Cirq/blob/de651a7886d1a1383d595f95f36b93ed3ea91685/cirq-core/cirq/transformers/analytical_decompositions/two_qubit_to_cz.py).
The modern Qiskit wrapper calls accelerated implementations; reading that wrapper
alone does not constitute an audit of its numerical implementation.

## What is actually reused

| Capability | Upstream mechanism | FlagQuantum implementation | Status and limits |
| --- | --- | --- | --- |
| KAK and supercontrolled basis synthesis | Qiskit Weyl coordinates and zero-to-three-entangler closed forms | `optimization/two_qubit_synthesis.py` | Existing mathematical/implementation adaptation, with native instructions and deterministic Python algebra. The historical originating Qiskit revision has not been established by this audit. This change adds explicit attribution; it does not claim a new synthesis port. |
| Native CX pulse synthesis | Qiskit `get_sx_vz_2cx_efficient_euler` and `get_sx_vz_3cx_efficient_euler` | `optimization/two_qubit_synthesis.py` | Adapted numerical templates with native Euler leaves, preserved CX direction, bounded cost selection and complete-operator verification. See [source, acceptance and limits](COMPILER_CX_PULSE_SYNTHESIS.md). No complete synthesis parity or SDK ranking follows. |
| Bounded search recovery | Qiskit `swap_map_trial`, `State::force_enable_closest_node` | `topology/sabre.py`, recovery branch in `plan_sabre_swaps` | Newly adapted control flow: undo the speculative SWAP suffix, remove its output records, choose the closest blocked operation, and bring its operands together along a shortest path. |
| Repeated-search detection | Cirq `RouteCQC._route`, its `seen` set and forced shortest-path strategy | `topology/sabre.py`, `seen_layouts` | Newly adapted algorithmic mechanism. FQ tracks repeated full layouts between executed instructions; Cirq tracks repeated SWAP choices within a timestep. This is not a port of all RouteCQC. |
| Full-CZ interaction formulas | Cirq's `_xx_interaction_via_full_czs`, `_xx_yy_interaction_via_full_czs`, `_xx_yy_zz_interaction_via_full_czs` | Not enabled | Rejected after both fixed-pulse and arbitrary-RX development probes increased instruction counts. The existing Qiskit-derived two-qubit constructor remains authoritative; see [retained comparisons](COMPILER_CZ_CIRQ_REUSE.md). |
| Local cleanup after CZ synthesis | Cirq's `_merge_single_qubit_gates` and `single_qubit_matrix_to_pauli_rotations` | `optimization/_cz_local_cleanup.py`, called by `optimization/_cz_blocks.py` | Integrated in PR #1229. Native pending-per-wire accumulation emits shorter RX/RZ runs behind the existing complete-operator verifier; symbolic and protected inputs retain their established path. |
| Frontier-relative scoring | Qiskit `State::choose_best_swap` and layer-local score updates | `topology/sabre.py` | Integrated in PR #1232; retained absolute-score tie behavior and the small-circuit path. See [relative scoring](COMPILER_SABRE_RELATIVE_SCORING.md). |
| Multi-trial routing/layout | Qiskit's independent trials and best-result selection | `topology/sabre.py` | Integrated in PR #1233: original route plus three seeded trials, scored including restoration and directed operand repair. See [bounded trials](COMPILER_ROUTING_TRIALS.md). This is not the complete Qiskit layout pipeline. |
| Layer-by-layer lookahead and paired SWAPs | Cirq `_choose_optimal_swap`, `_choose_pair_of_swaps`, `_cost` | `topology/_layer_lookahead.py` and `topology/sabre.py` | Adapted as one additional bounded candidate. See [layer lookahead](COMPILER_LAYER_LOOKAHEAD.md). Integrated in PR #1234 with source-conformance and native routing tests; it is an additional candidate, not all of RouteCQC. |
| Characteristic-polynomial CZ count | Cirq `num_cnots_required` and `_gamma` | Not enabled | Conservative list-algebra proposal tested, then rejected for insufficient complete-workflow benefit, larger QFT allocation peak and extra work on reducing blocks. See [retained source and controls](COMPILER_CIRQ_INVARIANT_PRUNING.md). No production or dependency change. |
| Two-qubit collection and re-synthesis | Upstream collect/consolidate/re-synthesize workflows | `optimization/_cz_blocks.py` | FQ-owned implementation of a shared pattern. It must not be described as copied Qiskit or Cirq code. |

The integrated candidate is commit
`858df9dbd5ba964e8e59b3383304547710a1dee5`. PRs #1229 and #1234 each passed
239 checks with three optional checks skipped before merging. The candidate's
installed-wheel smoke run
[44585780031](https://github.com/flagos-ai/FlagQuantum/actions/runs/44585780031)
also passed. These are source and CI integration facts, not a package-release
or comparative-performance claim. Original failed attempts remain in their PRs.

The prospective SDKBench runs
[ordinary](https://github.com/sdkbench-org/sdkbench/actions/runs/44587635615) and
[target-artifact](https://github.com/sdkbench-org/sdkbench/actions/runs/44587642527)
completed on attempt 1 with this compiler and harness commit
`9e3992d14342601662443b6720dd6757905c3f3b`. Each retained 30 application cases,
90 accepted complete case/participant outcomes, 900 independently replayed blocks
and 2,700 measured compilations. The immutable
[ordinary report](https://github.com/sdkbench-org/sdkbench/blob/89c1448fedc75b785080ccfc9be5f7f145020133/docs/analysis/COMPILER_ORDINARY_RESULTS_20261120.md)
and
[target-artifact report](https://github.com/sdkbench-org/sdkbench/blob/89c1448fedc75b785080ccfc9be5f7f145020133/docs/analysis/COMPILER_TARGET_ARTIFACT_RESULTS_20261120.md)
retain every case and participant pairing, raw archives, source/build identities,
resource evidence and local replay checks. Both reports were integrated through
SDKBench PR #9 after its 14 exact-head CI checks passed.

These are bounded CPU engineering observations, not release certification or a
global SDK ranking. Some cases use fewer FlagQuantum CZ gates; others use more.
The reports also retain compilation-time and process-memory losses to Qiskit.
The ordinary profile does not invoke target-CZ block resynthesis. Target-artifact
time includes FlagQuantum target validation, provenance, artifact construction
and serialization, while the other participants retain compiler workflows.
Final identity placement is required throughout, but the target-artifact profile
uses native restoration for FlagQuantum and the common external restoration for
Qiskit/Cirq. These are complete delivered workflows, not identical internal
algorithms or isolated synthesis timings.

The two profiles use distinct draws and hosts: do not divide their metrics or
attribute between-profile differences to a compiler pass. Six-qubit cases have
complete-operator acceptance; widths 8/12 retain three-probe scope. Neither study
contains an old-FlagQuantum source control. Host/build attestation and publication
qualification remain separate; publication eligibility is false.

The same-revision default-versus-explicit routing-policy control has a separate
[declaration](https://github.com/sdkbench-org/sdkbench/blob/9ce37f85bef327603bc64f63eec7e0fddc9436ef/docs/analysis/COMPILER_ROUTING_POLICY_DECLARATION_20261120.md)
and [retained execution](https://github.com/sdkbench-org/sdkbench/actions/runs/44623673659).
The cross-SDK reports above do not supply that control. The approved default
proposal's original before-merge time/memory requirement was unmet; subsequent
qualification cannot retroactively establish it.

The implementation still lacks several important upstream mechanisms. The
presence of SABRE or KAK alone does not establish equivalent compilation quality.

### Formula conformance and comparative performance

An internal formula-conformance test may disable an upstream optimization to
isolate the particular closed forms being checked. That setting applies only
to the diagnostic. SDKBench comparisons must retain their declared participant
settings, including upstream optimizations that FlagQuantum has not implemented.
Missing pulse optimization is a FlagQuantum gap to measure and improve, not a
reason to constrain a competitor. The published compiler studies do not set
`pulse_optimize=False` in their Qiskit adapter.

## Recovery port and deliberate modifications

Previously, reaching `4 * n_wires` speculative SWAPs was treated as evidence that
the front layer could not be routed. A heuristic can cycle even on a connected
line, so this inference was invalid. The new implementation uses the existing
budget to trigger progress instead of reporting an unroutable circuit.

Qiskit's rollback is adapted directly to FQ's retained SWAP list, placement list
and source anchors. Only SWAPs since the most recent executed instruction are
removed. Earlier placements, quantum dependencies and classical dependencies are
unchanged. The existing source-to-output attribution remains authoritative.

Cirq's repeated-choice guard is adapted to repeated complete layouts. With an
unchanged frontier, repeating a layout repeats the deterministic search state.
The set is cleared on every executed instruction and bounded by the existing
speculation budget. At most `4 * n_wires` layout tuples are retained, using
O(`n_wires ** 2`) additional integer references in the worst case.

The forced path selects the closest blocked instruction as Qiskit does, rather
than Cirq's farthest pair. A path with `d` edges needs exactly `d - 1` SWAPs to
make the operands adjacent. FQ splits those SWAPs between its two ends and stops
one edge short. This uses FQ's existing deterministic `CouplingMap.shortest_path`
and avoids importing an upstream graph or layout object. The next outer
iteration executes at least the selected instruction. Disconnected pairs and
routes requiring physical ancillas outside the IR are still rejected before
search. This mechanism is an explicit compiler routing strategy, not a numerical
backend substitution.

No new public export, routing enum, signature, result field, serialized schema
or dependency is introduced. Existing strategy names and restored-output-layout
contracts remain. Successful routes can change when the previous heuristic
revisited a layout; exact emitted gate sequences are not promised to be frozen.

## Ownership, licensing and dependency decision

Compiler owns the adaptations. Both inspected upstream files carry Apache-2.0
notices. The modified FQ module preserves the IBM and Cirq Developers notices,
names the original source files, and explicitly identifies FQ's modifications.
The repository's `LICENSE` supplies Apache License 2.0. No root `NOTICE` or
`NOTICE.txt` was found at either inspected upstream release. These notices live
in the Python module so they accompany the implementation in built packages.

This change introduces no production dependency, subprocess compiler invocation,
Rust extension, or vendored graph library. Upstream Qiskit DAG/layout objects and
Cirq circuits do not cross into Core. Suitable code is ported behind the
existing compiler boundary rather than invoking another SDK and presenting its
output as a native FQ implementation. The replacement interface is the existing
`plan_sabre_swaps` input/output contract; removal requires no consumer change.
Upstream updates should be reviewed against the pinned sources and the native
conformance tests, rather than copied wholesale.

## Reproducible regression and acceptance

`tests/unit/test_compilation_sabre_progress.py` includes an 11-wire line-topology
regression reduced from 88 CX operations generated with `random.Random(500)`.
The retained 29-operation circuit previously failed after 45 consecutive
speculative SWAPs. The same circuit now completes, respects every coupling edge
and restores the output layout. This was found during development; it is not a
held-out task or a cross-SDK performance result.

The baseline was replayed from commit
`bcbdc9164370da91aba1286158caeb6a8767ba98`; its `sabre.py` SHA-256 is
`9d246bc1016021a910e85fa310effdff1c43afb50ef56c64d0dacc2370a91c30`.
The new plan places all 29 source instructions with 71 forward SWAPs; final
restoration is additional. This is recovery from failure, not evidence that 71
is an efficient or optimal route.

The independent reference computes the complete CX/SWAP operator as a permutation
of all 2,048 computational basis indices, rather than checking only the all-zero
input. Other tests cover odd/even path lengths, removal of rolled-back SWAPs and
anchors, deterministic planning, randomized complete-operator checks, symbolic
parameter identity, classical read-after-measurement order, and disconnected
failure. Existing routing conformance tests cover consumers unchanged.

Run the focused acceptance checks:

```bash
python -m pytest tests/unit/test_compilation_sabre_progress.py \
  tests/unit/test_compilation_sabre_routing.py \
  tests/unit/test_compilation_sabre_layout.py \
  tests/team/compiler/test_routing_conformance.py \
  tests/team/compiler/test_barrier_routing.py \
  tests/unit/test_compilation_default_routing.py \
  tests/unit/test_compilation_routing_estimate.py -q
```

The supported user workflow remains:

```python
import flagquantum as fq
from flagquantum.compiler import CouplingMap, compile

circuit = fq.Circuit(4).h(0).cx(0, 3)
compiled = compile(circuit, coupling_map=CouplingMap.line(4))
print(compiled.metadata["routing"]["mapping_restored"])
# True
```

For a larger executable workflow, use `python -m examples.circuits.compiler.target_aware_compilation`.
The path is `single_device_fast_path`; no distributed capability is added.

This recovery port does not establish lower CZ counts, shorter compiled depth,
lower compilation time or parity with either upstream compiler. Those require
separate unchanged SDKBench runs and prospective held-out tasks. In particular,
source-CZ optimization and restored versus free final permutations remain
separate comparison factors.

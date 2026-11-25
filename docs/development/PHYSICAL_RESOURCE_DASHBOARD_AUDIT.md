# Complete physical-resource dashboard audit

Q-0930/Q-0931 bind the complete frozen `surface_code/ui.py` and `ui_test.py`
at Qualtran v0.7.0 revision `bcb17bb25360ed68ce3bc084645db3e929799389`.
Their Git blobs are `85f4010c42616909458bdb8490b01a9990595469` and
`4fa2944a0cb1c4a74df1ff2a27a4796dcd314787`. The old one-cycle discrepancy is
resolved by the already published shared Simulation/QEC fixes below. Complete
source-file and original-test acceptance now passes. The earlier failed
experiments remain as dated diagnostic evidence, not current acceptance claims.

## Current complete-file acceptance — 2026-11-25

The implementation and tests were an unfinished 2026-11-19 G2 draft. No active
writer was using them at intake. The existing draft is reused rather than
reimplemented; its old precision-failure conclusion had not been updated after
the numerical owners changed.

`5fe9b648e` preserves source sparse Kraus contraction order. `a572a6004`
preserves independent density preparation and optimized H application order.
`905848132` supplies deterministic complex trace reduction. `60ccb5569`
integrates the complete source arithmetic into the shared distillation factory.
Those changes are already in the integration branch. Neither the dashboard nor
this acceptance introduces a new simulator, probability rounding or tolerance.

The unchanged dashboard draft now passes all 34 prior cases, including the
exact figure JSON that previously failed. Two additional tests verify complete
NumPy RNG state after each supported model callback, preventing hidden extra
draws from changing the next request. The complete fourteen source functions
map to fourteen native functions, with the two naming corrections listed below.
The nine-output HTTP callback, original layouts, figure payloads, duration
thresholds, separate rotation depths, factory/QEC combinations and valid input
categories are covered. The 12 unchanged original tests pass.

The first combined revalidation passed 167 dashboard/factory/density cases.
The final run, including RNG state, passes 169 cases without skips. It reuses
full factory-density bitwise comparisons, exact rejection/output probabilities
and cycle demands through 10^18. Local integration passes 145 cases. A fresh
process with Qualtran and Cirq imports blocked executes all nine native callback
outputs and serves the layout through its Flask test client successfully.
Both Python files pass strict typing, Black, Ruff, integration scope, language,
architecture, IR, API and SDK-contract checks. Current receipts are
`g2-dashboard-revalidation-20261125.log`,
`g2-dashboard-original-20261125.log` and
`g2-dashboard-quality-20261125.json` in the reference environment.
Final receipts also include `g2-dashboard-final-regression-20261125.log`,
`g2-dashboard-local-integration-20261125.log` and
`g2-dashboard-native-only-20261125.json`.

Q-0930 and Q-0931 close for the mapped native model/input contract. This is not
a blanket claim of identical behavior for the permissive source's invalid
fractional/negative/nonfinite counts or a correction of the source chart
conventions. Those boundaries remain explicit below. The app computes model
estimates, so primitive-gate, amplitude and measurement-history execution are
not applicable to this presentation module; the shared numerical factory
certificates are reused separately.

## Ownership and source map

`examples/apps/physical_resource_dashboard.py` is a thin optional presentation
application. Existing QEC owners supply all estimates. `ModeledResources` and
`GateCost` supply logical counts, and rotation layers travel separately to the
shared Beverland consumers. Dash, Plotly, pandas and NumPy are imported only at
the example boundary. The application has no execution backend or circuit IR.

| Complete source definitions | Native owner and disposition |
| --- | --- |
| `get_objects` and model dictionaries | Example discovery over existing native rotation models, QEC schemes and three factory presets |
| Algorithm, factory, QEC and rotation input builders | Corresponding example builders, retaining input IDs and order |
| `input_components`, `create_ouputs`, app layout | Example input/output layout, with source spelling corrected to `output_components` |
| `create_qubit_pie_chart` | Native physical search and data layout, retaining source chart arithmetic |
| `format_duration` | Original thresholds and first-element unit selection, including the empty-input failure |
| `create_runtime_plot` | Original random sampling, factory replication, distance labels and complete figure payload |
| `update` | Native model construction and all nine callback outputs, corrected rotation argument spelling |
| `total_magic`, `min_num_factories`, `compute_duration` | Shared native gate categories, Beverland formulas and physical search |
| Callback registration and main entry | Same nine-output Dash dependency graph, loopback-only local serving |
| Complete original test module | Unchanged upstream tests plus actual Flask callback and full JSON comparisons |

The executable example uses Dash 2.18.2 and Plotly 5.24.1 from the isolated
`ftqc-reference-env/dashboard-reference` directory. These are optional local
presentation dependencies. The native package gains no production dependency.
Application title and displayed class representations use FlagQuantum owners.
The source names for the two rotation choices map explicitly to
`BEVERLAND_ROTATION_COST` and `SEVEN_BIT_PHASE_GRADIENT_COST`.

## Historical precision failure — 2026-11-19

The focused suite currently has 33 passing cases and one failing exact-JSON
comparison. The source hash binding, actual HTTP route and callback behavior,
layout field order and output builders, all duration unit thresholds, missing
inputs and independent rotation depths pass. The crossed callback matrix covers
two estimation models, two rotation models, two factory replica counts, two QEC
schemes and three factory presets. All scalar outputs match. The 12 unchanged original UI tests pass. Strict mypy passes
for both new Python files.

The failing configuration uses the Beverland estimate, seven-bit phase gradient
rotation model, one FifteenToOne733 factory and GidneyFowler errors. Each of the
six algorithm inputs is 100,000,000,000. Random seed is 17. Both implementations
produce exactly the same discrete counts, including T=100,000,000,091 and
Toffoli=600,000,000,000, hence T-equivalent demand=2,500,000,000,091.

The density simulator outputs differ by at most 7.981091292474346e-17 per complex
entry. Source rejection probability is 0.0030537435699304893, while the native
probability is 0.0030537435699320437. Source conditional output error is
4.394091968062952e-08, while the native value is 4.394091934756261e-08.
The logical error model values at distances 3 and 7 are exactly equal.

At this demand, the source cycle expression before ceiling is
45137839388430.98, while the native expression is 45137839388431.05. The integer
cycle counts therefore differ by one, 45,137,839,388,431 versus
45,137,839,388,432. The sampled runtime coordinate is 522.4286966253588 days
versus 522.4286966253703 days. This is an actual integer-resource discrepancy
caused by floating-point error amplification at a ceiling boundary.

The exact figure assertion remains failing. No tolerance, rounding, reference
patch, expected-failure marker or substituted factory hides this witness.
The earlier factory density tolerance does not prove integer-cycle agreement
at arbitrarily large demands. Both dashboard parents remain unaccepted pending
resolution of this dependency boundary. Evidence files in the persistent
reference environment are `g2-dashboard-current.log`, `g2-dashboard-difference.log`,
`g2-dashboard-precision.log` and `g2-dashboard-cycle-witness.log`.

## Source defects and admission boundaries

The Beverland pie combines logical data tiles with physical factory qubits.
The runtime graph calculates code distances but omits them from the duration
multiplication and always uses Gidney hardware cycle time. Factory replication
is composed before the extra sampled multiplicity. These formulas and random
candidate selection remain as written in the frozen source. They are retained
presentation/model defects, separate from the native physical estimator.

Native original counts now require six finite nonnegative integer-valued
algorithm inputs and a positive integer-valued factory count before coercion.
Integral JSON floats are accepted. Fractional, negative, nonfinite, boolean and
string inputs are rejected explicitly, while source records are permissive.
This stricter native admission is exercised before model construction. Native hardware validation also
rejects physical inputs that the permissive source records accept. Empty duration
input retains the source IndexError. These boundaries prevent a blanket
input-domain equivalence claim.

This application evaluates analytical models. It emits no primitive circuit,
simulates no algorithm amplitudes, and allocates no quantum work wires. Primitive
counts and measurement-branch cleanup are inapplicable to this presentation file.
The cycle witness concerns a physical model estimate, not measured primitive
execution. Previous gate-level execution certificates remain separate.


## H-rounding isolation experiment

An isolated, uncommitted experiment replaced only the native H matrix payload
with the correctly nearest-rounded sqrt(0.5) coefficient. The baseline native
coefficient is 0.7071067811865475. The nearest value is 0.7071067811865476.
The explicit Cirq unitary protocol returns 0.7071067811865477, while Cirq also
has a separate optimized H application protocol. Reproducing a unitary array
alone does not reproduce that complete simulator execution path.

The nearest-H experiment makes all 34 current dashboard cases pass unchanged.
It yields rejection probability 0.0030537435699309334, which still differs from
the source probability. At T demand 100,000,000,000,000, source production cycles
are 1,805,513,575,471,519 and the nearest-H experiment yields
1,805,513,575,471,520. At demands 10^15 and 10^16, cycle differences are 8 and
64 respectively. This rejects the hypothesis that correcting H rounding alone
closes the complete supported integer-demand boundary.

The experiment is confined to temporary scripts and explicit matrix payloads.
The production H constant and simulator remain unchanged. Exact source assertions
and the open-parent decisions remain in effect. Persistent evidence is in
`g2-dashboard-h-probe.log`, `g2-dashboard-nearest-h.log` and
`g2-dashboard-precision-range.log`. The next safe numerical investigation is
per-operation comparison of the complete density execution protocols, including
Cirq product-state merging, H application and channel accumulation order.


## Per-operation density trace

The complete 100-operation trace has 96 source moments. All 95 non-H operators
have bitwise identical nonzero Kraus payloads. The five explicit H matrices
differ by 2.220446049250313e-16. After the first five H operations, the density
maximum difference is 5.898059818321144e-17 and traces are 1.000000000000001
and 0.9999999999999992. Final traces are 0.9999999999999967 and
0.9999999999999958. Source product-state merging, optimized H and channel
accumulation remain the unresolved execution-order boundary. This evidence
does not establish bitwise simulation or full-range integer-estimate parity.
See `g2-dashboard-density-trace.log` and `g2-dashboard-density-trace.json`.

## Product-state protocol isolation

Replaying the native IR payloads with the source operation protocols, but using
one full initial density matrix, reproduces the source simulator with
`split_untangled_states=False` bit for bit. Its rejection probability is
0.0030537435699311555. The default source configuration uses
`split_untangled_states=True` and yields 0.0030537435699304893. The two source
execution configurations themselves produce different integer-cycle estimates
at the recorded high demands. This isolates initial product-state evolution and
merge order as a required part of default-source parity.

A second diagnostic maintains independent single-wire density states, applies
optimized H separately, and merges the groups in instruction operand order before
channels. With unchanged source channel protocols over native payloads, this
reproduces the entire default-source density matrix bit for bit. The final
projector calculation is another arithmetic boundary. NumPy yields acceptance
0.9969462564300695, while Torch on the identical density and projector yields
0.9969462564300696. Therefore matching the final density alone does not prove
exact probability or arbitrary-demand integer-cycle parity.

These experiments use source protocols only as diagnostics. They introduce no
production dependency or execution-path change. Persistent evidence is in
`g2-dashboard-protocol-isolation.log/json` and
`g2-dashboard-product-isolation.log/json`. The shared Simulation owner must
resolve execution arithmetic before the QEC consumer and dashboard can close.

A Torch-only experiment mirrors independent product-state initialization,
optimized H updates, tensor channel contraction and accumulation. Its final
density still differs from the default source by 3.0117855110836314e-17, and
its NumPy-projected rejection probability is 0.0030537435699302673. Reordering
Torch execution alone has not established parity. The experiment remains outside
production. See `g2-dashboard-torch-product-isolation.log/json`.

These historical isolation results precede the published shared fixes listed
at the start of this document. The current complete callback and numerical
regressions pass without loosening the old exact JSON assertion.

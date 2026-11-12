# Component tensor conversion source disposition

Reference: unchanged Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389`, complete
`qualtran/simulation/tensor/_quimb.py`. Native owners retain Apache-2.0 attribution.
This is a local single-device graph-to-numerics path, not a new simulator.

## Complete source ownership

| Source definition | Native owner and disposition |
| --- | --- |
| `cbloq_to_quimb` | `component_tensor_nodes` wires selected Core SSA calls, invokes explicit suppliers and adds idle identities. Pure discard stops at the first token with native guidance. |
| `_get_placeholder_tensors` | Existing register identity/bit-wiring nodes provide direct unused left-to-right connections. No placeholder foreign Soquet enters IR. |
| `_get_outer_indices` | `register_tensor_indices.register_tensor_outer_indices` provides structured/friendly diagnostic names; numerical integer labels remain authoritative. |
| `DiscardInd` | Original integer trace requests in mixed suppliers; no foreign class or second trace contract is introduced. Pure consumers reject the token. |
| `make_forward_tensor` / `make_backward_tensor` | `conjugate_pairs` creates even/odd labels and elementwise conjugates without transposing data axes. Foreign tags have native metadata counterparts. |
| `cbloq_to_superquimb` | `component_superoperator_nodes` reuses SSA wiring, conjugate pairs and `trace_conjugate_indices`; declared input maps remain intact. |
| `_get_outer_superindices` | The same index diagnostic owner derives lf/lb/rf/rb descriptors; friendly-name collisions do not merge execution edges. |
| `_add_classical_kets` / `initialize_from_zero` | Compiler `initialize_component_from_zero` prepares left quantum words in signature/row-major order using existing clean allocation definitions, then invokes the original function. Simulation supplies those leaves with `register_zero_state_nodes`. |
| Type aliases and logger | Foreign label aliases are represented by existing typed integer labels/port maps. Logging identity is not a numerical or quantum operation; no original package namespace is exported. |

Compiler owns preparation and graph transformation. Simulation owns zero kets,
conjugation, trace and contraction; neither imports the other's implementation
across the prohibited dependency direction. Users retain `circuit.append(op)`.
No new graph, registry, Core field, public export or dependency is needed.

## Classical outputs from zero initialization

The previous guard rejected any classical port, including RIGHT-only results
produced by measurement. Only **left** ports are initialized. The guard now
rejects classical inputs and preserves classical outputs in the declared RIGHT
signature, using the same builder and downstream mixed consumer. It neither
prepares a classical output as a quantum ket nor changes classical input typing.

The new composition initializes a quantum input and idle quantum spectator,
executes H/S then X or Z measurement, and either retains or discards the classical
result. At the exact S phase, the complete native density/effect equals the
unmodified upstream `initialize_from_zero(...).tensor_contract(superoperator=True)`.
A trainable phase perturbation additionally checks complete numerical output and
weighted gradients against an analytic formula; both default and explicit
simplification policies are exercised. This does not modify the reference S gate.

All two physical measurement branches retain their complete complex amplitudes;
the unused clean bit remains zero. These physical checks bind the original graph
with caller-owned zero inputs and exercise `circuit.append(op)`. The initialized
graph's pool-owned destructive measurement still raises the existing retirement
ownership-contract refusal. Numerical zero initialization does not authorize
retiring a physical pool allocation. No ownership bypass is added or claimed.

## Source defects and stricter native boundaries

- Source initialization cannot infer empty output-array dtype and raises
  `IndexError`; native declared-signature behavior is independently tested.
  Zero-width source IntState decomposition has its recorded dtype failure.
- Direct external input `Discard`/`DiscardQ` leaves no producer tensors in the
  source and fails contraction. Native mixed wiring supplies an explicit binary
  trace effect. Existing unchanged Cast-wrapped source graphs provide equivalent
  whole-channel evidence, not identical node counts for the broken direct case.
- Native trace rejects duplicate, missing, internal, external-output and invalid
  dimensions before relabeling. Source raw reindexing does not supply all these
  validations; no exception-equivalence claim is made for malformed inputs.
- Native pure discard uses FlagQuantum guidance, not a byte-identical foreign
  error message. Classical pure ports retain their explicit refusal.
- Arbitrary external supplier discovery and foreign graph objects remain outside
  native IR. Selected providers are explicit and fail closed when unavailable.

Q-0891 remains open under closure rule5: source defects and the physical retirement
boundary remain unresolved. The empty-output negative record Q-0891.04 remains
unchecked. Complete source disposition does not mean those defects were repaired.

## Measured resources and model estimates

Actual caller-owned physical stream for Z: **H1, S1, measurement1**; for X:
**H2, S1, measurement1**. Discard adds no primitive. Two input bits are used;
the third caller-provided clean bit is unused and remains zero. No additional
quantum work is allocated by tensor conversion. One measurement slot is used.

Separately, the upstream initialized graph's QEC model reports **4 Clifford,
1 measurement, 0 T** for both bases. That model includes two IntState preparations
and treats MeasureX coarsely. These estimates are not the emitted gate counts.
Numerical output storage, contraction cost and graph metadata are not gate costs.

Existing tests retain measured-Add per-gate ordering, all small joint measurement
histories, complex amplitudes, clean work and source cost evidence. The historical
default-suite timeout is not rerun unchanged, and this batch makes no full-suite
or GPU certification claim.

Next: complete Q-0895 classical-action tensor source boundary disposition,
especially classical bit axes, empty arrays and phase-free limitations; reuse
existing numerical owners before adding any implementation.

Local acceptance: **8 new scenarios**, **2,632 related tests in 76.68 s**,
**84 CPU/TN tests in 5.82 s**, and **two unchanged original quimb tests**
pass. Strict mypy covers both changed Python files; Black/Ruff, Compiler scope,
architecture, IR, protected API and Bell example pass. Four source blobs match
the frozen inventory. Implementation `32f4e4bd`. Existing warnings and default
timeout remain; this is not whole-repository certification.

The additional direct-input trace-only suite passes **53 tests in 5.36 s**.
Changed zero-initialization owner: **44 statements and 20 branches, 100%**.
The three related initialization/mixed/trace owners total **103 statements
and 50 branches, 99%**, with one preexisting mixed refusal line outside this
concentrated selection. The trace-only suite separately exercises that boundary.

The first concentrated attempt recorded two gradient-assertion roundoff failures
(8.9e-16 with unbounded integer observable weights). The observable is now
normalized to [0,1], retaining the 5e-16 comparison threshold; the final complete
concentrated run above passes. The initial log is retained as
`g2-zero-discard-regression-first.log` in the local reference evidence folder.

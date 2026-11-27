# Kaliski modular inverse source-file audit

The reference is Qualtran v0.7.0 at commit
`bcb17bb25360ed68ce3bc084645db3e929799389`.

| File | Frozen Git blob | Ledger item |
| --- | --- | --- |
| `mod_division.ipynb` | `fd34e136d02bd80d927945123f9a65f428c38e2a` | Q-0572 |
| `mod_division.py` | `947f40eb5d17101ad91ff4d4432ac5287f867bd5` | Q-0573 |
| `mod_division_test.py` | `2b8a7fbca15e4b5e7143f4b50d002334165b4c5d` | Q-0574 |

The [native tutorial](../../examples/notebooks/modular_inverse.ipynb) covers
the concrete and symbolic examples, including the adjoint signatures and
all eight direct operation identities. It takes the history width from the
source signature. The signature exposes four bits per input bit in the
`junk` output, while the original prose describes a two-bit-per-input-bit
register. The forward `circuit.append(op)` transforms the Montgomery word
in place and returns history on explicit output wires. The adjoint consumes
that history. Clean work is private to the emitter.

## Execution and resources

Native tests match the source's gate operands and measurement feedback in
both directions on width/modulus pairs (5, 7) and (6, 15). A width-five
iteration also exercises the source's standalone six-step call sequence.
Across widths five and six and moduli 3, 5, 7, 11, 13 and 15, the tests
compare every unit and the zero sentinel against locked source values
and retained history, then require exact restoration and clear work.
The feedback certificate composes the measurement branches of each block.
Full-support complex-state checks at modulus 15 exercise the actual
forward and inverse channels at both widths.

The independent source model is `26n² + 9n − 1` Toffoli equivalents.
Its default T estimate is four times that expression. The model with
exact controlled swaps is `2n(67n + 13) + 4(n − 1)`. The emitted stream
below is measured through native forward append.

| Width, modulus | Emitted T/T† | Emitted measurements | Clean work | Logical history | Default source T estimate |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5, 7 | 3496 | 454 | 33 | 20 | 2776 |
| 6, 15 | 5000 | 641 | 38 | 24 | 3956 |

## Remaining source-file boundaries

Q-0572 and Q-0574 are accepted. The locked suite passed 55 cases including
its independent notebook, with one Montgomery modulus metadata skip and
one expected failure for the declared counts. Q-0573 retains the source
decomposition's full-width all-ones modulus limit when it loads `p + 1`.
It also retains the nonunit boundary and the divergence in steps five and
six on states outside the reachable iteration domain. Q-0568 package export
has its independent accepted ledger disposition.

## Current stored-graph batch interface — 2026-11-20

Proposal 108 was published at aa23ec1cb. The next complete source-file batch
contains the nine classes below and 48 definitions. Its inventory is retained
in `ftqc-reference-env/g2-modular-inverse-native-inventory.json`.


The complete source-file batch contains nine classes and 48 definitions.
All words and history words are MSB-first. Forward history is a declared
logical output. Temporary conjunction, comparison and arithmetic work returns
zero only on the original promised domain. Specialized inverse dependencies
consume matching forward outputs and preserve the original source constants.

| Frozen class | Native owner and port contract |
| --- | --- |
| `_KaliskiIterationStep1` | Termination owner preserves v/m/f/is_terminal, with zero-test conjunction work owned internally |
| `_KaliskiIterationStep2` | Parity owner preserves u/v/b/a/m/f, applies original two predicates and measured conjunction inverses |
| `_KaliskiIterationStep3` | Ordering owner preserves u/v/b/a/m/f and consumes owned half-comparison outputs after fanout |
| `_KaliskiIterationStep4` | Exchange owner preserves u/v/r/s/a and two exact controlled swaps |
| `_KaliskiIterationStep5` | Arithmetic owner preserves u/v/r/s/b/f with original complements and two controlled additions |
| `_KaliskiIterationStep6` | Scaling owner preserves u/v/r/s/b/a/m/f, original descending controlled shifts, ModDbl alias and parity cleanup |
| `_KaliskiIteration` | Six ordered calls, two owned zero branch flags, retained marker and termination outputs |
| `_KaliskiModInverseImpl` | Through four words, 2n marker/terminal histories and active flag, original initialization and zero correction |
| `KaliskiModInverse` | Through x, RIGHT forward or LEFT inverse 4n junk, inlined algorithm, source allocation/release order |

The existing step, algorithm and public physical emitters retain their accepted
source-stream, inverse, complex-amplitude and resource evidence. Stored graph implementation and normal append replacement were subsequently
validated by the complete batch acceptance below. Source all-ones p+1 loading, nonunits and unreachable step-five/six
callback disagreements remain open. The accepted package export Q-0568 has an
independent ledger disposition.

### Frozen definition inventory

| Class | Source members |
| --- | --- |
| `_KaliskiIterationStep1` | `signature`, `on_classical_vals`, `build_composite_bloq`, `build_call_graph` |
| `_KaliskiIterationStep2` | `signature`, `on_classical_vals`, `build_composite_bloq`, `build_call_graph` |
| `_KaliskiIterationStep3` | `signature`, `on_classical_vals`, `build_composite_bloq`, `build_call_graph` |
| `_KaliskiIterationStep4` | `signature`, `on_classical_vals`, `build_composite_bloq`, `build_call_graph` |
| `_KaliskiIterationStep5` | `signature`, `on_classical_vals`, `build_composite_bloq`, `build_call_graph` |
| `_KaliskiIterationStep6` | `signature`, `on_classical_vals`, `build_composite_bloq`, `build_call_graph` |
| `_KaliskiIteration` | `signature`, `build_composite_bloq`, `build_call_graph`, `on_classical_vals` |
| `_KaliskiModInverseImpl` | `signature`, `_kaliski_iteration`, `build_composite_bloq`, `build_call_graph` |
| `KaliskiModInverse` | `signature`, `build_composite_bloq`, `adjoint`, `build_call_graph`, `on_classical_vals` |

Original example functions are `_kaliskimodinverse_example`, `_kaliskimodinverse_symbolic`.

This inventory records scope and interfaces. It provides no new acceptance
decision for the pending stored graph batch.

### First stored owner in the current batch

`inverse_exchange_component.py` now composes Step4's two actual CSwap calls,
sharing one exact child definition. Forward and adjoint root order match the
source on widths 1, 2, 3, 5 and 8. All primitives, operands and output aliases
match, with no workspace events or classical slots. Complete complex states
and an entangled spectator are checked at widths 1 through 3, including exact
restoration through the append consumer.

Its emitted T count is 14n and CX count is 16n. The separate original default
model is 8n T equivalents, or 14n when the exact seven-T controlled-swap weight
is selected. Physical peak is 4n+1 with zero work and measurements. Existing
Step4 evidence is reused. The focused source/graph regression and strict typing,
format, scope, architecture, IR and API checks pass. Receipt prefix is
`g2-modular-inverse-native-exchange-*`. This owner is locally accepted; the
complete nine-class stored-graph batch remains in progress until the other
owners, normal root execution and concentrated regression close together.

## Complete stored graph closure — 2026-11-20

All nine frozen classes and their 48 definition dispositions now have native
owners. The six original step emitters remain the accepted low-level physical
interfaces. Stored graph owners retain actual child calls, linear register
ownership and the specialized measured adjoints of shared dependencies.

| Frozen class | Stored graph owner | Owned result/work contract |
| --- | --- | --- |
| Step1 | `inversion/inverse_termination_component.py` | v/m/f/is_terminal through; measured zero-test work internal |
| Step2 | `inversion/inverse_parity_component.py` | u/v/b/a/m/f through; second predicate reads updated a |
| Step3 | `inversion/inverse_ordering_component.py` | comparison difference/target consumed by exact measured inverse |
| Step4 | `inversion/inverse_exchange_component.py` | two calls share the exact register-swap definition |
| Step5 | `inversion/inverse_arithmetic_component.py` | original complements and two CAdd calls; finite-width arithmetic |
| Step6 | `inversion/inverse_scaling_component.py` | descending controlled shifts, real ModDbl output aliases, paired exchanges |
| Iteration | `inversion/inverse_iteration_component.py` | six step calls; private a/b restored on reachable inputs |
| Implementation | `inversion/inverse_algorithm_component.py` | 2n calls to one shared iteration; two retained 2n-bit histories |
| Public shell | `modular_inverse_component.py` | source inlining; x through, 4n-bit junk RIGHT forward/LEFT inverse |

The direct dependency `modular_halving_component.py` composes the exact ModDbl
adjoint through existing constant-addition adjoints and the original parity
cleanup. Its allocated high bit becomes data and its input low bit becomes
clean work. The compiler binder receives explicit caller ownership when source
Free consumes an input wire. Consumers follow returned physical aliases.

All words and histories are MSB-first. Marker/terminal history is logical data
and remains live until the public inverse consumes it. Only source-reachable
inputs, zero sentinel, units and matching inverse images carry cleanup promises.

### Source, state and replacement evidence

* Forward and inverse source primitive streams, operands, measurement feedback,
  output aliases and source resource models are checked through eight-bit grids.
  The original Soquet connections bind each physical lifetime. Source And
  generators and original bookkeeping views remain unmodified references.
* Termination, parity, ordering and arithmetic test every small logical column,
  every applicable raw measurement branch, complex phase, spectator and work
  cleanup. Existing larger step/iteration/algorithm certificates are reused.
* Public graph execution checks every unit and zero at (n,p)=(3,5),(4,9),(5,13),
  exact histories and full clean workspace. Complete sparse complex/spectator
  vectors run through all literal primitives for three sampled histories in
  each direction. These numerical trajectories do not enumerate their Cartesian
  product of 176 or more measurement outcomes. The accepted compositional
  measured-And channel certificates cover the full history channel.
* `examples.algorithms.register_modular_inverse` now binds the stored graph on
  the normal `circuit.append(ModularInverse(...))` route. Replacing the previous
  public physical emitter with a raising stub preserves that append workflow.
  A clean-pool permutation retains the example's established target/history
  placement and inverse restoration without inserting routing gates.
  The example constructs and counts IR. It does not allocate a dense 38-bit state.
* Q-0572/Q-0574 notebook/original-suite evidence is reused with matching frozen
  blobs. Original 55 passed, one metadata skip and one expected count failure
  remain independently recorded. Source files have not been modified.

### Measured resources and separate source models

At n=3,p=5 the following are emitted primitive counts, peak live quantum storage
and fresh measurement SSA slots. Reset counts equal measurement counts, and
refer only to erased conjunction work. Retained logical history is 12 bits.

| Module | Emitted T/TDG | Emitted CX | Measurements | Peak quantum | Fresh measurement slots | Source default T | Source exact-swap T |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| step1 | 12 | 21 | 3 | 9 | 3 | 12 | 12 |
| step2 | 12 | 22 | 3 | 12 | 3 | 12 | 12 |
| step3 | 20 | 68 | 5 | 17 | 5 | 20 | 20 |
| step4 | 42 | 48 | 0 | 13 | 0 | 24 | 42 |
| step5 | 44 | 84 | 11 | 20 | 11 | 44 | 44 |
| step6 | 84 | 150 | 7 | 27 | 7 | 60 | 84 |
| iteration | 214 | 393 | 29 | 28 | 29 | 172 | 214 |
| algorithm | 1292 | 2383 | 176 | 38 | 176 | 1040 | 1292 |
| public | 1292 | 2383 | 176 | 38 | 176 | 1040 | 1292 |

The whole forward graph emits 5533 primitives. Its 1292 T/TDG agree with the
source model configured for exact seven-T CSwap, while the source default is
1040 T. Both source and native peak storage are 38 quantum bits. Each source
measurement event owns a fresh static classical SSA identity in the stored
binder. The older explicit physical emitter reuses a smaller caller-supplied
slot pool. Event count, static SSA count and reusable physical slots are
separate resource observations.

### Retained boundaries

* Step1 at n=1 is rejected by the original MultiAnd minimum-control condition.
* Full all-ones p cannot load p+1 in its source n-bit AddK. Neither implementation
  reduces that original constant to conceal the failure.
* Nonunits and arbitrary iteration inputs can violate the source Free promises.
  Step5's unbounded callback differs from its finite-width gates on overflow,
  and Step6's odd active v callback drops a bit while its circuit cyclically rotates.
* Symbolic physical widths, generic foreign/strict dtype admission and unresolved
  source count/autotester interfaces retain their existing independent ledger
  dispositions. This graph closure alone does not close parent Q-0573.
* Earlier whole-repository default/integration failures remain recorded. This
  batch claims its concentrated local regression and strict checks, with no new
  whole-repository success assertion.

The initial new test harness mistakenly sized dense branch states using the
entire reserve pool. Its process exited 137. The harness now derives width from
actual emitted wires before branch simulation, and affected tests passed again.

The initial concentrated run had 883 passed and 12 failures because the append
example changed physical result/history placement. Its caller-compatible pool
binding was corrected, and the unchanged original-grid tests were rerun.

Final batch receipts are `ftqc-reference-env/g2-modular-inverse-native-*` outside
the repository. Final validation passed 899 concentrated cases and 17 local integration cases.
Strict typing passed for 22 files, alongside Black, Ruff, Compiler scope,
architecture, multi-level IR and protected API checks. Six frozen source blobs
match. The dependency scan reports 688 existing repository findings and zero
batch findings.

## Complete-file acceptance after ordered symbolic graph closure

The [inverse/product ordered graph review](INVERSE_PRODUCT_ORDERED_GRAPH_AUDIT.md)
now fills every source-defined symbolic root, including comparison/exchange/
arithmetic layers, iteration, implementation and both public directions. Native
metadata preserves all source parameters and exact inlined connection roles;
retained 4n history remains an output or matching consumed input. Concrete
artifacts use the existing nine physical owners with identical whole circuits,
feedback, output bindings and workspace events. Earlier generic symbolic notes
are superseded by this exact defined/refused disposition.

Fresh original-only all-ones, nonunit, step-five overflow and step-six odd-input
witnesses retain their documented domains and protocol differences. Existing
source streams, complex phases, all-history cleanup, matching inverse and
independent model evidence remain authoritative at unchanged numerical owners.
All 48 definitions are accounted for. Q-0573 is eligible for faithful-source
acceptance after verified publication and final checks. Global constructor/SDK
qualification remains separately tracked; no source failure is silently repaired.

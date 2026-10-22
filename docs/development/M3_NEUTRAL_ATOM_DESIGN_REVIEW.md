# M3 neutral-atom pair: design review and numerical reference

Status: **design review, not an approved public API contract or native atom
implementation**. M2 workflow acceptance is separately reviewed in #686. This
preparation does not merge that PR, approve new signatures or schemas, or promote
capability maturity. No production source file changes here.

The table below records the original inspected baseline. Approved
[proposal 082](API_CHANGE_PROPOSAL_082_REMOTE_PROVIDER_BOUNDARY.md) retires its
historical generic Remote register adapter; Core #700 now owns the native atom
program. The historical table is not a current provider-support claim.

## Existing capabilities and the missing boundary

| Requirement | Reusable implementation | Boundary still missing |
| --- | --- | --- |
| Register declaration and distance-derived connectivity | `remote/qpu/neutral_atom.py` | This is provider discovery/preflight. Its coordinate tuple is not a persistent atom/trap identity model or a transport program. |
| Local internal states, including absorbing loss | Core product spaces and Lindblad dimensions | Assigning a third local basis state does not identify a physical atom, trap or detected event. |
| Position-dependent interaction over intervals | Parameter/ParameterExpression, proposal 073 bindings | Numeric coefficients alone cannot retain a move's endpoints, duration, resource conflicts or physical provenance. |
| Dissipation and unconditional state evolution | Existing Liouvillian, integrators and EvolutionResult | Constant jumps do not establish a calibrated transport-heating law or time-dependent collapse contract. |
| Populations, survival and conditioning | Proposal 080 matrix requests and named expectations | End-time expectation effects are not sampled detection records, feedback or a decoder interface. |
| Program control flow and instruction validation | Existing Program IR and dynamic-program contracts | No atom-transport instruction is admitted merely because general control flow exists. |

Reuse is required, but a matrix-only wrapper is insufficient for the roadmap's
native scenario. The final program must preserve motion and loss meaning before
numerical lowering. A second IR, solver, binder, backend registry or provider-owned
physics model would give that meaning a competing source of truth.

## Concrete native-scenario acceptance requirements

1. **Identity and geometry.** Atom and trap identities are separate and stable.
   Coordinates carry an explicit length unit and reference frame. An atom does
   not become a different atom when moved; loss does not renumber survivors.
   Reusing a vacated trap does not resurrect its previous atom. Coincident
   occupied positions and inconsistent occupancy must be refused.
2. **Explicit transport.** Preserve named source/destination traps, start time,
   duration and the declared path or interpolation law. Concurrent moves read
   the same pre-move occupancy. Validate collisions and shared transport-resource
   conflicts before execution. A list of coefficient values cannot substitute
   for this program. Derive interaction coefficients from the same geometry
   used for validation and visualization.
3. **Physical model.** Declare the selected internal levels, angular-frequency
   convention, laser controls and interaction regime. Never silently infer a
   species, C6 coefficient, blockade radius or conversion between Hz and rad/s.
   Separate imposed classical paths from semiclassical/full motional dynamics;
   the first local implementation may support imposed paths, but must report
   that approximation and refuse unsupported motional requests.
4. **Loss and observation.** Keep physical loss, presence detection and internal
   measurement distinct. Preserve lost-state probability in the unconditional
   state. Report survival and any conditioning denominator explicitly; do not
   silently normalize away loss. A detector's efficiency, false-positive model
   and observation time belong to its declared model, not to an assumed perfect
   herald. Feedback and decoder integration require actual event semantics.
5. **Plans and evidence.** Preserve enough original physical inputs to review
   the program after numerical lowering. Reuse existing numeric plan replay,
   output requests and PyTorch binding. Changes to protected plan schemas,
   Program IR, result/event fields or exports require a specific approved
   proposal and replaceable contract baseline before implementation.

These are acceptance requirements for the native scenario, not a declaration
that the present reference meets them. In particular, the reference below has
no atom/trap identity records, transport instruction, path animation, detection
trajectory, calibrated motion noise or hardware submission.

## Current-API physical reference

Run without hardware or credentials:

```bash
python -m examples.neutral_atom_pair_reference
```

Expected output:

```text
Physical survival: 0.81873075
Unconditional RR: 0.40936538
Survival-conditioned RR: 0.50000000
No-flag-selected RR: 0.47961134
Detected-loss probability: 0.14646434
Distance-dependent coherence: 0.79248111
Exact numeric replay: True
```

Each local space is ordered `(ground, Rydberg, lost)`, with the first atom the
most significant tensor factor. The initial state is `(gg + rr)/sqrt(2)`.
Two intervals of duration 0.25 use separations 2 and 4 with an illustrative
effective angular interaction `V=C6/r^6`, C6=64 and hbar=1. Units are arbitrary
and consistent; these numbers are not an atomic-species or hardware calibration.
The positions are frozen on each interval. The discontinuity between them is
not a simulated physical move, acceleration or heating process.

Independent jumps `sqrt(gamma) |lost><ground|` and
`sqrt(gamma) |lost><Rydberg|` act on each atom, with gamma=0.2. The existing
collapse input takes an unscaled operator and its rate. This state-independent
Markov loss is an explicit toy assumption, not a model of Rydberg decay back to
the ground manifold. The full nine-state density matrix retains all lost sectors.

For one-atom survival `s=exp(-gamma*t)`, pair survival is `S=s^2`, unconditional
RR probability is `S/2`, and the real gg/rr coherence observable is
`S*cos(phi)`, where `phi=sum(V_k*dt_k)`. The independent analytic references
therefore separate loss from interaction phase.

The end-time detector has efficiency eta=0.8 and no false positives, assumed
independent for the two sites. Its no-flag effect is
`diag(1,1,1-eta)` tensor itself. Thus `Q=(1-eta*(1-s))^2` is the no-flag
probability, and detected-loss probability is `1-Q`. Conditioning on actual
survival gives `Prr/S`; selecting no flags gives `Prr/Q`. They differ when loss
can remain undetected. No detector samples or state-update instrument is
implemented by this expectation-only calculation.

`tests/integration/test_neutral_atom_pair_reference.py` checks these analytic
trajectories at absolute error 1e-10, eta=0/0.8/1 and gamma=0/0.2, trace retention,
distance-sensitive coherence, exact numeric replay and coincident-atom refusal.
This commuting phase reference does not replace proposal 073's noncommuting
time-ordering tests or certify a blockade gate.

## Delivery sequence and review decisions

1. Review this scientific reference and the missing semantics above while #686
   awaits its separate merge authorization. This PR grants no implementation or
   merge permission for another PR.
2. Draft one explicit contract for physical identity, transport and loss/event
   meaning at the existing Core/Program IR and plan/result boundaries. State the
   exact new signatures, ownership, schemas, units, refusal paths, gradients and
   compatibility rules. Resolve the model-builder ownership before adding a
   package: Remote owns vendor translation and preflight, not this model; Core
   owns common semantics, Simulation owns numerics, Compiler owns transforms,
   and Runtime owns execution. Do not scatter one ordinary implementation across
   these domains.
3. After explicit contract approval, land a replaceable baseline, then separate
   domain-owned implementations. First prove the local pair's complete path;
   add general array size, calibrated heating, event-driven feedback and QEC
   coupling only against their own acceptance evidence. Do not substitute this
   matrix reference for the native-program gate.

The decisions still required are the public physical-program boundary and its
owner, transport resource/path representation, and event/instrument semantics.
This design review deliberately does not freeze those as guessed signatures.

## Primary references

[Saffman, Walker and Molmer, Quantum information with Rydberg atoms, section II.2](https://arxiv.org/html/0909.4777v3)
describes the interaction regimes behind the effective van der Waals model.
The isotropic C6/r^6 approximation here is chosen explicitly, not inferred from
that review for an unspecified device.

[Bloqade Lanes' instruction set](https://bloqade.quera.com/vendored/lanes/instruction-set/)
makes location/transport resources and simultaneous move semantics explicit.
It is architectural evidence for preserving such meaning, not a proposal to
copy its bytecode or introduce a second FlagQuantum IR.

# M2 shared CPU dynamics workflow review

Status: **acceptance evidence for a bounded CPU workflow**, not completion of
all M2-M4 gates. Proposal 080 is approved and its baseline, Simulation, Remote
and Core implementations landed in #669, #671, #678 and #682. This review adds
no public API, serialized format, numerical implementation or execution target.

## User workflow

Run the two-model example without hardware or credentials:

```bash
python -m examples.shared_dynamics_workflow
```

Expected output:

```text
three-level: axes=(51, 2)
Initial transfer: 0.11757891
Coupling derivative: 0.32210884
Loss decreased: True
Exact numeric replay: True
spin-mode: axes=(51, 2)
Initial transfer: 0.11757891
Coupling derivative: 0.32210884
Loss decreased: True
Exact numeric replay: True
```

The first model transfers population between levels 1 and 2 of a three-level
system. The second exchanges one excitation between a spin and a three-level
motional mode, using the existing boson algebra. In both cases one batch member
is active and the other is a dark vacuum. Frequencies use the inverse of the
time-grid unit. The spin-mode Hamiltonian conserves excitation number, so this
input never reaches the oscillator cutoff. That invariant-sector argument is
specific to this example; it is not a general truncation certificate.

Both models use the same existing `fq.ExecutionOptions`, `fq.expectation`,
`fql.plan`, `fql.run`, `EvolutionResult.expectation` and `torch.optim.SGD` loop.
Only dimensions, Hamiltonian, initial state and measured matrix differ. The
example introduces no trainer, model registry, result envelope or binder.

## Acceptance evidence

| User requirement | Existing owner reused | Reproducible verification |
| --- | --- | --- |
| Named matrix outputs on two physical models | OutputRequest and Simulation observable normalization | `tests/integration/test_shared_dynamics_workflow.py`: full trajectories agree with the independent two-state Rabi law at absolute error 1e-10 |
| Time and batch axes, including a batch of one | Existing EvolutionResult | Same test: `(51, B)` is retained for B=1 and B=2; the dark member remains zero |
| Trainable coupling and optimizer updates | PyTorch and existing dynamics autograd | Analytic derivative T sin(2gT) at absolute error 1e-10; the runnable example rebuilds a live plan each update and reduces its target loss |
| Numeric serialization and exact-plan execution | LindbladPlan JSON and closed-plan run path | Same test: unchanged JSON round trip, exact replay, numeric result without an autograd graph, and refusal of semantic overrides |
| Explicit unsupported paths | Existing Circuit/Remote front doors | `test_matrix_output_frontdoors.py` and `test_remote_matrix_output_refusal.py` refuse dense requests before execution or submission |
| Compatibility and replacement | Existing proposal 080 consumers and fixed fixtures | `test_matrix_expectation_contract.py`: real request factory, three integrators, labelled coordinates, controls, gradients and ten old plan payloads |

Run the focused acceptance with:

```bash
python -m pytest tests/integration/test_shared_dynamics_workflow.py \
  tests/unit/test_matrix_expectation_contract.py \
  tests/unit/test_matrix_output_frontdoors.py \
  tests/unit/test_remote_matrix_output_refusal.py -q
```

The numerical budgets remain those of their individual tests. Exact replay means
the same numeric artifact on the tested runtime, not bitwise equality across
devices or PyTorch versions. JSON does not preserve a training graph. Export a
detached numeric plan for replay; reconstruct a live plan from current parameters
for training. A serialized plan requiring gradients remains subject to the
existing `gradients_not_available` refusal.

## Remaining roadmap boundaries

This slice validates a shared local CPU developer workflow. It does not dispatch
dynamics through root `fq.run`, make dense expectations work on circuit/GPU/QPU
paths, add pulse interpolation or time-dependent jumps, or introduce hardware
motion/loss events. CircuitIR 1.0 and plans 1.3/1.4/1.5/1.6 keep their existing
meanings. No parity or capability maturity is promoted.

The next domain-native scenario remains the moving neutral-atom pair in the
[roadmap](../roadmap/FLAGOS_ALIGNED_RELEASE_TRAIN.md). Before new protected APIs,
make the geometry, motion schedule, loss/detection distinction and unconditional
versus postselected outputs concrete in a separate reviewable contract. Reuse
the existing piecewise controls, Lindblad dissipators, states and output requests;
do not infer atom-motion support from a position-dependent matrix reference.

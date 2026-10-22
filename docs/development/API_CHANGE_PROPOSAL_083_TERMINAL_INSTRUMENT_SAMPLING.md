# Proposal 083: terminal finite-basis instrument sampling

Status: **proposed; not approved for implementation**. The owner requested this
proposal on 2026-10-22 while authorizing the native-control and visualization
examples. That request does not approve the API below. Proposal 081 and merged
#707 remain the authority for native evolution and plan 1.7; proposal 082's
service-owned Remote boundary remains unchanged.

## Five decisions requested

1. Add exactly the Runtime-owned sample_terminal entry specified below. Accept
   ordered full-space tensor Kraus branches and reuse SuperOperator for their
   mathematical action. Add no instrument class, registry, IR hierarchy or solver.
2. Keep this first slice a CPU terminal-state operation. Preserve existing root
   circuit outputs, Lindblad signatures/options/results and all plan versions;
   no Program IR instruction, timestamped event or feedback is added here.
3. Reuse the existing MeasurementResult unchanged, with the precisely specified
   index samples and inspectable metadata below. No second result envelope.
4. Adopt the explicit presence and finite-Fock instrument models, refusal,
   probability, RNG, gradient and capture rules below. Effects alone never select
   a measurement backaction implicitly.
5. Establish an interchangeable two-model contract baseline before separate
   Runtime implementation. Each subsequent PR requires independent merge
   authorization and latest complete applicable CI/GPU success.

## Current owners inspected

| Existing implementation | Reuse and limit |
| --- | --- |
| operators/SuperOperator.from_kraus, apply and adjoint | Already represent each completely positive branch, including trace-decreasing branches. No new channel algebra is needed. |
| simulation/lindblad_request.py density validation | Already validates Hermiticity, unit trace and positivity. Reuse its validation and dtype budget; factor an owned helper if needed rather than writing a second density normalizer. |
| runtime/measurements.py | Owns sampling, private Torch generators and output assembly. Its current density adapter infers a qubit width and cannot be used unchanged for dimension three or non-product labels. Factor only the probability-draw primitive shared by its old and new consumers. |
| runtime/result.py MeasurementResult | Existing kind/qubits/value/shots/metadata/statistics contract. Keep its definition, exports, field order, defaults and schema_version 1.0. |
| core/ir/_encode_value | Existing numeric tensor codec for the output snapshot; no alternate matrix serializer. |
| ExecutionOptions, OutputRequest, EvolutionResult and LindbladPlan | Reused by evolution unchanged. This proposal does not make shots/seed legal on ordinary Lindblad evolution or add a new output kind. |

The initial implementation belongs to Runtime, calling existing mathematical
owners. It does not require moving Runtime contracts into Core, making Simulation
depend on Runtime, or adding a cross-domain feature in one ordinary PR. Core
instruction/result changes for mid-program measurement need a later proposal.

## Exact proposed public entry

```python
# Proposed export in flagquantum.runtime; NOT implemented by this PR.
def sample_terminal(
    density_matrix: torch.Tensor,
    instrument: Mapping[str, Sequence[torch.Tensor]],
    *,
    shots: int,
    seed: int | None = None,
) -> MeasurementResult: ...
```

No root export or fql alias is added. This is a permanent Runtime measurement
primitive that future execution adapters can consume, not a replacement dynamics
entry. Each mapping value is a nonempty finite sequence of Kraus matrices for
one observable outcome. Mapping iteration order defines outcome indices and is
retained exactly; labels are unique nonempty strings with non-whitespace content,
without stripping or inferred bit/atom meaning. Distinct Kraus decompositions of
the same channel need not have equal bytes; no equality/hash identity is invented.

The state is an ordinary CPU complex64 or complex128 strided tensor of shape
(D,D) or (B,D,D), D >= 1, B >= 1. It must be finite, Hermitian, positive and unit
trace within the existing Lindblad dtype validation budget (1e-5 / 1e-10).
Rank two always denotes one density matrix; rank three retains B=1. Raw kets,
implicit tensor factors and logarithmic wire inference are refused. The caller
owns the matrix basis and physical interpretation, as in existing mixed/labelled
state-space execution. The function does not normalize away a missing loss sector.

All Kraus tensors have shape (D,D), the same CPU device and complex dtype as the
state, ordinary strided storage and finite values. Fixed numeric matrices only;
requires_grad on the state or any Kraus tensor is refused before detaching.
The existing DEFAULT_DENSE_MATRIX_BYTES limit bounds the sum of state and Kraus
numeric storage before cloning/capture; it does not claim a total-process RSS
limit. No dense D**4 Choi matrix is needed: tensor Kraus input is CP by construction.

For branch j, J_j(rho) = sum_k K_jk rho K_jk^dagger and
p_j = real(trace(J_j(rho))). Require sum_jk K_jk^dagger K_jk = I at the same dtype
validation budget. A missing residual branch is rejected, never silently treated
as postselection. Probabilities below minus that budget or a sum differing from
one beyond it are rejected. Only roundoff-sized negative entries are clipped;
the resulting vector is divided by its checked near-one sum for Torch sampling.
Keep original branch weights and the maximum correction in result metadata.
These rounding rules do not weaken the separate 1e-10 / 2e-5 physical reference
budgets for the two bounded acceptance models.

Conditional state J_j(rho)/p_j is defined only for p_j > 0; a zero-probability
branch has no conditional state. The helper returns samples, not conditional
state tensors. Users and later execution consumers use the same SuperOperator
branch action to obtain conditional states. It does not modify the input state.
The unread instrument channel is sum_j J_j(rho), which generally differs from
rho because measurement can change coherence. A POVM effect or a no-flag
probability alone is insufficient to choose this channel.

## Result, randomness and capture

Return the existing MeasurementResult with:

- kind = "sample", qubits = (), shots = the requested positive Python integer;
- value = CPU int64 outcome indices, shape (shots,) or (B,shots), including B=1;
- statistics = {}, retaining its existing meaning; no implicit confidence claim;
- metadata has exactly outcome_labels, probabilities, probability_correction and
  terminal_instrument, as described below.

probabilities is the original checked real branch-weight list, shape (M,) or
(B,M); probability_correction is the largest absolute adjustment made for the
draw. outcome_labels is the ordered string list. terminal_instrument contains
exactly schema, version, dimension, dtype, branches, requested_seed,
effective_seed and torch_version. schema is flagquantum.terminal_instrument,
version is 1.0. Each branch contains exactly label and kraus; each matrix uses
the existing Core numeric tensor encoding, captured after validation and before
sampling. Caller mutation after return cannot change these captured values.

This is output metadata in the existing result contract, not a new sealed plan,
signed artifact, public fingerprint, or metadata deserializer. Existing result
serialization retains these JSON-compatible values. Native plans keep exactly
their seven 1.7 fields; old 1.3-1.6 plans and CircuitIR 1.0 remain byte-identical.
No promise is made to recognize arbitrary edited metadata as a valid execution
request. Semantic artifact loading would require its own reviewed boundary.

seed is None or a Python integer in [0, 2**63-1]; booleans are refused. Use one
private CPU torch.Generator per call, never mutate global Torch RNG state. With
None, record the generated effective seed. Batch rows draw in their declared
order. Repeated execution with identical numeric inputs, labels, effective seed,
Torch version and CPU runtime reproduces the samples; no cross-version/device
bitwise guarantee or batch-partition invariance is claimed. Counts are ordinary
bincount reductions of the indices, not a new counts API. Sampling carries no
autograd graph. Existing analytic probabilities/expectations remain the route for
training; a differentiable sampling estimator is not introduced.

No clock or atom identity is inferred from a raw density matrix. Callers retain
the originating evolution time, units and plan alongside the measurement result.
This first mathematical terminal result is not a timestamped detector trajectory.
Feedback, erasure-decoder records and causal identity association remain later
gates; they must never interpret an unobserved lost-state coordinate as a flag.

## Models used to admit the primitive

**Presence detection.** In ordered g/r/lost coordinates, efficiency eta in [0,1]
and zero false positives are explicit assumptions. K_no_flag is
 diag(1,1,sqrt(1-eta)); K_flag is diag(0,0,sqrt(eta)). Thus the effect matches
081's no_flag_operator while this proposal additionally specifies the backaction.
Tensor products define the pair's ordered observed flags; no actor renumbering
or resurrection occurs. Nonzero false positives, internal-state readout and
calibrated motion-dependent detection are not inferred from this model.

**Finite Fock counting.** The three-level number operator from the existing
boson algebra defines the ordered 0/1/2 projectors. This reference is a declared
QND number instrument: K_n = |n><n|. It is not a destructive photodetector,
Gaussian backend, infinite Fock certificate or native photonic hardware frontend.
Both models exercise the same branch algebra, probability sampler and result.

## Failure contract

| Request | Existing error category |
| --- | --- |
| Non-tensor state/Kraus or non-mapping/sequence instrument | TypeError |
| Invalid shape/dtype match, label, shot/seed value, non-finite/non-positive/non-unit state, incomplete instrument or invalid branch probabilities | ValidationError, preserving numeric causes |
| Non-CPU device, gradient-bearing input or existing byte ceiling exceeded | CapabilityError before sampling |
| Unavailable conditional state at zero probability | No state is returned; caller must test p_j > 0 |

No invalid request advances a generator or triggers evolution, provider imports,
submission, fallback, hidden conditioning or a different mathematical model.

## Runnable current-API reference

```bash
python -m examples.terminal_instrument_reference
python -m pytest tests/integration/test_terminal_instrument_reference.py -q
```

This PR contains only current-API mathematical references and this proposal.
It does not implement sample_terminal or update an export/snapshot. Expected
output is recorded after an actual run in the PR and examples README.

Proposed future usage, explicitly unavailable until approval and implementation:

```python
# Future API; this is not a current runnable example.
from flagquantum.runtime import sample_terminal
rho, branches = presence_reference()
record = sample_terminal(rho, branches, shots=10_000, seed=7)
# record.value contains only observed outcome indices, not hidden loss truth.
```

## Delivery and acceptance gates

1. Obtain explicit approval of all five decisions. Document merge is not API
   approval. No candidate/snapshot or public implementation changes before it.
2. Establish replaceable factories for branch construction and terminal sampling
   under the approved signatures. Use presence and Fock consumers and retain
   independent physical/statistical budgets; do not implement a second framework
   in the contract fake.
3. Integrate the approved Runtime export/metadata admission separately from its
   Runtime implementation. Factor the existing draw primitive and preserve
   qubit sample/count shapes, seeded behavior and all old result/plan fixtures.
4. Implement Runtime consumption using existing density checks, SuperOperator
   branch action, Core tensor encoding and Torch RNG. No new solver, source IR,
   registry, trainer or result envelope. Replace factories without editing the
   consumers' assertions. No Simulation-to-Runtime dependency.
5. Verify eta=0/0.8/1, pure/mixture/coherent lost-sector inputs, pair flag order,
   Fock labels, B=1/B=2, zero-probability branches, CP completeness, trace and
   correct unread backaction. Test seeded repeatability, caller mutation/global
   RNG isolation, strict refusals and unchanged old circuit measurement outputs.
   Use analytic branch probabilities/poststates at 1e-10 (complex128) and 2e-5
   (complex64). For fixed-seed empirical draws use independent binomial standard
   errors with a declared six-sigma plus 2/shots bound, not a solver tolerance.
6. Run supported Python minors, appropriate testing tiers, repository gates and
   latest complete applicable CI/GPU. Every PR includes runnable user code and
   expected results and needs its own explicit merge grant. This proposal does
   not authorize later feedback, QEC, hardware or detector-event schemas.

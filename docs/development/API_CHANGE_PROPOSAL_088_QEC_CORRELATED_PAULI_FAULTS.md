# Proposal 088: Correlated Pauli circuit faults

## Status and decision

Proposed for QEC-FUNCTIONAL-004; implementation is not authorized by this
proposal. The user requested this contract PR after #779 merged. Approval to
prepare the proposal is not approval to change accepted noise or DEM behavior.
Core owns QEC and Noise; Integration reviews the protected behavior delta.

Approve or amend the complete scope below before an implementation PR starts.
The target is development version 0.4.0, subject to review and release policy;
merging a proposal neither releases a feature nor freezes an unimplemented API.
No root export, constructor argument, public result field, serialization schema,
decoder protocol, registry or mandatory dependency is added.

## Current behavior and reuse

[#779](https://github.com/flagos-ai/FlagQuantum/pull/779) removes dense-statevector
execution from supported Clifford fault-signature construction. The gate-bound
`DetectorErrorModel.from_memory_circuit(memory, noise=noise)` route still accepts
only its established single-qubit bit-flip and phase-flip grammar. A Pauli
mixture or correlated two-qubit placement is currently refused.

Existing authorities already supply the required representations:

- `flagquantum.noise.KrausChannel.unitary_mixture` derives probabilities and
  normalized unitaries from operators, independently of a channel's name.
- Noise factories already include Y flip, one-qubit depolarization and
  `two_qubit_depolarizing_channel`. `NoiseModel.channels_for` owns matching,
  target ordering and separate application of a one-qubit channel to each gate
  operand. There is no new QEC noise factory or parallel rule matcher.
- Simulation's stabilizer engine recognizes Pauli products up to global phase;
  its private Clifford fault-signature primitive owns propagation. Reuse that
  numerical authority behind a Simulation-owned private boundary. Do not import
  engine internals directly into DEM assembly or duplicate Pauli image tables.
- `DemError.error_id` and `dem/alternatives.py` already express mutually
  exclusive alternatives, exact marginal rates and categorical sampling.
  Keep these types and their existing refusal behavior; add no distribution
  wrapper or second sampler.

The approved adapter-only Tensor batch operation recorded in
[the separate batch proposal](API_CHANGE_PROPOSAL_087_QEC_BATCH_DECODING.md)
is already on the baseline. It is distinct from the still-proposed cross-decoder
operation in [Proposal 087](API_CHANGE_PROPOSAL_087_QEC_SCALABLE_EXPERIMENTS.md).
This proposal changes neither contract nor activates the experiment interface.

## Accepted channel grammar after approval

Extend only the gate-bound route on the supported unconditional Clifford memory
grammar. Admit one- or two-qubit channels whose existing nonzero Kraus branches
are scaled Pauli products, up to global phase. Accept the existing Y-flip and
one-/two-qubit depolarizing factories and explicitly biased mixtures constructed
with the existing `KrausChannel`; the latter use operators as authority rather
than a new string grammar. Each branch is `sqrt(p) * exp(i*theta) * P`, with
finite nonnegative probabilities summing to one within the existing numerical
validation policy. Never infer a distribution from names or parameter metadata.

Retain the original strict checks and results for the existing bit-flip and
phase-flip route, including malformed named channels. Generic admission must not
bypass those existing refusals. All newly admitted branches must pass shared
Noise unitary-mixture validation and Simulation Pauli recognition. Identity
branches and zero-weight Kraus operators are valid; a coherent non-Pauli unitary,
amplitude damping, a nonunitary branch, nonfinite input or a channel acting on
more than two qubits is refused. Do not Pauli-twirl, approximate, renormalize an
invalid distribution, or infer a different Kraus decomposition.

For a two-qubit product `P0 tensor P1`, P0 acts on the first qubit returned by
`channels_for`, P1 on the second. Preserve explicit target order, including
reversed gate operands. A one-qubit channel matching a two-qubit gate remains
two independent placements, one per yielded operand; it is not a correlated
pair. Each matched instruction/rule/target tuple is a separate location.
Multiple matching rules remain sequential channel applications in existing rule
order, with independently sampled location choices. Unmatched rules still place
nothing and retain today's validation timing.

Keep readout/preparation attachment refusals, last-instruction placement
refusals, unsupported instruction/conditional/custom-matrix refusals and their
existing error categories. This tranche does not extend the memory builder,
readout model or round-boundary `PhenomenologicalNoise` grammar.

## Exact distribution and propagation

For one location g, choose exactly one branch k with probability p[g,k],
including identity. Different locations are independent; branches at one
location are alternatives. Each two-qubit Pauli product is one simultaneous
fault with one joint detector/observable signature. Y is one fault, not two
independent X and Z draws.

Example: IX with mass 0.2 and XI with mass 0.3 at one location, identity with
mass 0.5, never produce an XX event at that location. Independent Bernoulli
substitution would incorrectly assign XX mass 0.06. A two-qubit depolarizing
factory with total error p has identity mass 1-p and each of the 15 nonidentity
products mass p/15, not a per-qubit error rate p.

Propagate the product directly using the existing Simulation algebra, or compose
baseline-subtracted responses from that same primitive. Its returned parity
signature may include an odd ideal baseline B: for factors P and Q at the same
position, `S(PQ) = B XOR S(P) XOR S(Q)`. Merely XORing two returned signatures
loses B. Keep the existing detector/observable convention for old inputs and
verify odd baselines, reset erasure and repeated measurement references. Do not
multiply branch probabilities when composing simultaneous Pauli factors.

## DEM assembly and sampling

A location with competing nonidentity effects receives one deterministic
`error_id`; all of its retained alternatives carry that id. Use stable lowered
instruction order, existing rule/target order and lexicographic I/X/Y/Z product
order. Labels are model-local grouping labels, not durable physical identifiers.
Repeated construction must yield identical ordering and serialization.

Within a location, combine branches with identical signatures by adding their
probabilities. Identity and nonidentity branches with an empty signature are
no-effect mass: omit them from `DemError`, whose existing constructor requires a
nonempty signature, and leave their mass in the group's categorical remainder.
A fully silent location emits no mechanism. Never renormalize visible outcomes.
Do not merge equal signatures across exclusive locations using the independent
parity formula, because doing so can destroy their joint distribution. Keep
location groups distinct, including singleton remnants. Independent locations
with ordinary single Bernoulli faults retain the existing merge path where its
premises hold. In particular, the old X/Z-only route must remain byte-for-byte
compatible in mechanism order, ids and seeded outputs.

Reuse the existing group sampler: one categorical choice per group per shot,
then XOR the chosen signatures across independent locations. Preserve same-seed
repeatability on a fixed supported software/platform profile. Newly admitted
mixtures have no promised shot-by-shot equivalence with another engine's RNG;
validate their exact law independently. Existing seed behavior is frozen for
previously accepted inputs.

Existing matrix projections remain incidence/rate views. Their columns alone
are not an independent Bernoulli model when ids are present. Existing operations
that cannot preserve exclusivity, including affected graph/text/decoder adapter
routes, must continue to refuse rather than strip ids or silently approximate.
An exact correlated decoder or export format is a separately reviewed scope;
this tranche promises exact construction, marginals and declared-model sampling.
Keep the gate-bound `decompose_composite_faults=True` refusal. The existing
phenomenological decomposition option is not a license to split these faults.

## Failure and compatibility contract

Invalid probabilities, malformed channels and invalid targets retain established
Noise validation and error categories. A mathematically valid but unsupported
channel or operation raises `CapabilityError` at the earliest supported
validation boundary. Preserve current `channels_for` placement errors. A
refused construction returns no partial model and does not run a fallback.

Keep the ordered 108 root exports, surviving class and pickle identities,
single registered `RotatedSurfaceCode`, registry replacement policy, existing
results and original 15 restructuring fixtures unchanged. Old DEM constructors,
text semantics, error-id schema and ungrouped sampling remain unchanged. Use new
separately named tests for the intentional extension of previously refused
inputs; do not regenerate protected snapshots or frozen evidence. No old module
is restored. Existing AI decoding work is outside scope. Do not modify or merge
DynamicCircuit #731 or merge this PR without user authorization.

## Acceptance before implementation delivery

1. Independently enumerate identity and every nonzero Pauli outcome of bounded
   one-/two-qubit channels. Inject each simultaneous product into small executed
   circuits; compare every detector/observable signature and final grouped DEM.
   Cover repetition, rotated, Steane and mixed ZXXZ checks with supported X/Z
   readout, multiple rounds, reversed targets, resets and odd ideal parities.
2. Compare complete joint outcome probabilities by Cartesian enumeration for
   small multiple-location examples. Check identical/empty signatures, duplicate
   Pauli Kraus branches, global phases, zero/one error rates, biased mixtures,
   concurrent matching rules and mass conservation. Include the IX/XI witness
   above and all 15 depolarizing products. Marginals alone are insufficient.
3. Test categorical exclusivity without relying only on sampled frequencies;
   independently check fixed-seed samples and circuit/model joint statistics
   with predeclared statistical bounds. Treat external engines as optional
   additional references, and run their acceptance environment with extras
   installed. No new mandatory dependency or skipped required check.
4. Keep all existing compatibility, plugin, numerical, serialization and
   refusal tests. Verify grouped-model consumers fail closed where required;
   do not silently select another decoder. Add runnable public examples showing
   the actual new constructor route and its distribution after implementation.
5. Check source and built sdist/wheel installation, existing user examples and
   full required CI including single-/two-GPU gates at the reviewed head. No
   threshold reduction, check suppression or new skip. Larger construction is
   capacity evidence only, not a decoding-quality, threshold or speed claim.

## Runnable current-user example

The extension above is proposed, not implemented by this PR. With an installed
current distribution, or `PYTHONPATH=.` from a source checkout, run:

```bash
python examples/qec/circuit_noise_dem.py
```

This existing public example constructs a distance-five, three-round memory
with `NoiseModel().add("cx", bit_flip_channel(0.01))`, builds its DEM and samples
four declared-model shots with seed zero. It requires only core Torch.
Expected output:

```text
Physical qubits: 49
Detectors: 72
Logical observables: 1
Sample shapes: (4, 72) (4, 1)
```

This verifies the supported user path and its compatibility, not correlated
noise support or dense execution of 49 qubits. The implementation PR must supply
its own working mixture example and expected joint outcomes.

## Delivery boundary

This PR records the exact proposed grammar and acceptance. After explicit owner
approval, create a separate implementation PR directly from then-current main.
If reuse requires a public Noise or Simulation contract addition, stop that
portion and review a separate delta before implementing it. Keep B/C, correlated
readout, non-Pauli approximations, decoder expansion and performance claims out
of this tranche.

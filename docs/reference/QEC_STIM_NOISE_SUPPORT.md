# QEC Stim noise support and evidence

This is a support inventory for the existing APIs, not approval of a wider
noise grammar or a change to numerical defaults. Core owns the QEC noise
semantics; Simulation owns propagation, SDK objects and numerical execution.
The approved [correlated Pauli contract](../development/API_CHANGE_PROPOSAL_088_QEC_CORRELATED_PAULI_FAULTS.md)
and [Stim bridge proposal](../development/QEC_STIM_EXPERIMENT_BRIDGE_PROPOSAL.md)
remain authoritative. The [delivery plan](../roadmap/QEC_STIM_PYMATCHING_DELIVERY_PLAN.md)
defines the remaining phase-one exit gates.

## Supported locations

| Entrance | Placement and law | Boundary |
| --- | --- | --- |
| Native `PhenomenologicalNoise` | Data faults precede each round's first gate; a syndrome fault precedes the check measurement it changes. Independent X/Z channels remain independent placements. | This is the existing round-boundary grammar, not arbitrary preparation, terminal readout or duration-based idle noise. |
| Native gate-bound `NoiseModel` | A matching rule applies after its matched gate. `channels_for` determines operand order. A one-qubit channel on a two-qubit gate applies independently to each operand; a two-qubit Pauli branch is one simultaneous product. | Direct attachment to `measure` or `reset`, unsupported terminal placements and unsupported memory programs retain their existing refusals. An unmatched rule places nothing. |
| Multiple matching native rules | Channels apply sequentially in existing rule order; each location has its own independent choice. | Branches inside one channel are alternatives, not independent locations. |
| External Stim circuit text | `compile_stim_sampler` executes the original circuit, including supported explicit preparation/readout faults, conditional error alternatives and explicit idle faults. | The adapter does not synthesize timing, T1/T2 or missing locations. SDK acceptance alone does not establish exact matching-DEM convertibility. |

Native gate-bound channels admit the approved one-/two-qubit scaled Pauli
mixtures, including Y and depolarization. Operators determine their law;
names and parameter metadata do not override operators. Y is a single branch,
not independent X/Z draws. Two-qubit depolarization has 15 competing nonidentity
branches. Non-Pauli channels and unsupported arities fail closed; no twirling
or silent approximation is provided by this route.

Native memory lowering still validates its established repeated-round structure.
The more general external-text entrance does not expand that native grammar.
Detector/observable flips use the noiseless reference; a nonzero ideal parity
must not be mistaken for an error. Existing measurement maps, code classes,
`MemorySampler` and `DemSample` remain the authorities.

## Sampling and matching boundaries

An owned DEM can retain mutually exclusive alternatives through `error_id` and
categorical sampling. For one location with IX mass 0.2, XI mass 0.3 and identity
mass 0.5, the joint XX alternative has mass zero. Independent Bernoulli
substitution would incorrectly create it with mass 0.06. Separate locations
compose independently; equal visible signatures do not justify discarding
location grouping.

Consumers that cannot preserve that law retain their explicit refusals.
`stim_circuit_to_dem_text` disables gauge detectors and disjoint-error
approximation, and does not ignore decomposition failures. Some circuits can
therefore be sampled through `compile_stim_sampler` while their strict DEM
conversion is refused. Optional graphlike decomposition prepares a decoder
input; it never replaces the original circuit used to generate samples.
PyMatching acceptance is an additional boundary after exact DEM conversion.

## Probability precision

Three quantities must be recorded separately:

1. The configured decimal probability supplied to a channel factory.
2. The finite-precision law derived from its actual Kraus operators and the
   existing execution path.
3. The law reconstructed from exported circuit text.

On the recorded Python 3.12.14/Torch 2.13.0/Stim 1.16.0/macOS arm64 profile,
`bit_flip_channel(0.005)` has nonidentity weight `0.004999999422580004`.
The default factory precision is not changed here. A Python float can pass
through an initial float32 conversion even when complex128 output is requested;
three bounded float64-Tensor/complex128 witnesses were exact, which is not a
general accuracy guarantee or a new factory contract.

[#813](https://github.com/flagos-ai/FlagQuantum/pull/813) fixes a separate
transport defect: the SDK circuit printer shortened the executed value to
`0.005`. Native export now preserves the SDK's actual binary64 gate arguments
when its returned source is parsed. Reprinting it with another pretty printer
can again lose precision. The text has no stable byte-identity promise.

The earlier nominal-rate `1e-12` comparisons still fail and remain evidence of
an unresolved configured-versus-executed distinction. Actual-law comparisons
do not make those nominal comparisons pass. No operators, default dtype, native
seed stream or scientific tolerance were changed. Altering those protected
behaviors needs a concrete approved delta before implementation.

## Runnable public inspection

Install `python -m pip install -e '.[stim]'` from the checkout, then run:

```python
import stim
from flagquantum.noise import bit_flip_channel
from flagquantum.qec import PhenomenologicalNoise, RepetitionCode, build_memory_circuit
from flagquantum.qec.adapters.stim import memory_to_stim_text

memory = build_memory_circuit(RepetitionCode(distance=3), rounds=3)
mixture = bit_flip_channel(0.005).unitary_mixture
assert mixture is not None
executed = float(mixture.probabilities[-1])
source = memory_to_stim_text(memory, noise=PhenomenologicalNoise(data_flip=0.005))
reparsed = next(
    instruction.gate_args_copy()[0]
    for instruction in stim.Circuit(source)
    if instruction.name == "PAULI_CHANNEL_1"
)
print(executed)
print(reparsed)
print(executed == reparsed)
```

Observed output on the profile above is:

```text
0.004999999422580004
0.004999999422580004
True
```

This checks transport equality to the executed weight; it does not certify
agreement with the nominal decimal. PyMatching is not required for inspection.
For both experiment entrances and decoding, use the existing runnable
`examples/qec/stim_experiment_bridge.py` with the `stim,pymatching` extras.

## Evidence and remaining acceptance

| Concern | Existing executable evidence | Limit |
| --- | --- | --- |
| Exact gate placement | `tests/qec/test_gate_bound_noise.py` | Includes absent rules, per-round placement, readout/preparation refusals and round-boundary compatibility. |
| Exclusive products and target order | `tests/qec/test_correlated_circuit_noise.py` | Independent product injection, Cartesian joint laws, reversed operands, multiple rules, silent/duplicate branches and all 15 products; do not replace these with marginal sampling alone. |
| Exported executed law | `tests/qec/test_noise_law_bridge_evidence.py` and `tests/team/simulation/test_stim_annotated_engine.py` | Repetition d=3 Z and Surface d=3/5 X/Z, three rounds; detector/logical and selected joint differences were zero at unchanged absolute `1e-12`, with exact argument boundary tests. This does not close the nominal-rate gap. |
| External/source semantics | `tests/qec/test_stim_experiment_bridge.py` | Reference normalization, shape, repeat/control, coordinates, resource and lifetime boundaries; retain strict conversion failures. |

The retained fixed-budget exclusive-law audit used 40,000 shots per entrance,
seed 53, and a predeclared simultaneous Hoeffding bound `0.014274285099328213`.
Owned detector/logical rates were `0.201225, 0.299575, 0.5008`; original external
circuit rates were `0.202225, 0.300725, 0.50295`. Both had zero forbidden joint
alternatives. This is bounded distribution evidence, not a decoding-quality,
threshold or performance claim.

The complete route remains open. Reconcile the numerical-policy decision,
remaining placement/distribution cases and limitations before marking F/G/H
complete. I additionally requires four real optional-dependency installations,
wheel and sdist workflows outside the checkout, current-head CI/GPU, normal
merges and actual latest-main acceptance. Historical installation evidence for
contract 088 does not certify the newer bridge.

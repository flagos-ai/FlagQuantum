# QEC-FUNCTIONAL-004: Exclusive Pauli circuit noise

## Approved scope and ownership

The API owner explicitly approved [Proposal 088](API_CHANGE_PROPOSAL_088_QEC_CORRELATED_PAULI_FAULTS.md)
after [#783](https://github.com/flagos-ai/FlagQuantum/pull/783) merged on
2026-10-24. This implementation starts from main
`a5fc00898af53ab85037bec10c2db7579c68823b`. It includes the separately approved
reusable sampler from #784: preserve its two appended exports as well as the
ordered historical 108 exports. No public signature, class identity, pickle
path, registry, original fixture, dependency or serialized schema changes.

The path is single-device/local CPU construction and sampling. It carries no
accelerator, distributed, latency, logical-suppression or threshold claim.

- Simulation's private `_fault_signatures.py` reuses its engine's existing Pauli
  channel recognition and Clifford images. A simultaneous product has one
  anticommutation response against each backward-propagated parity. Baseline
  parity signs are added once, including odd ideal outcomes, not once per factor.
- QEC `dem/circuit_noise.py` shares validated instruction/rule/target locations
  with execution. Noise owns rule matching, target order and Kraus probabilities.
  Locations store the actual channel and its classified outcomes because both
  construction and circuit execution consume them; there is no public wrapper.
- DEM assembly adds probabilities only among identical signatures at one
  exclusive location. Each such location retains a deterministic model-local
  `error_id`; equal signatures at independent exclusive locations stay separate.
  Identity and silent products remain no-effect mass, without renormalization.
- Execution places the entire original mixture as one positioned channel. The
  existing stabilizer sampler makes one categorical choice per location. Its
  compiled memory handle retains the same lifecycle, snapshot and seed contract.

Single-qubit channels on a two-qubit gate still apply independently to each
operand. A two-qubit product uses the declared target order, including reversed
operands. Separate rules remain independent channel applications in rule order.

The old bit-flip/phase-flip validation, probability extraction and second factory
conversion in the sampler are retained. Their merged mechanisms, error ids,
positioned channel matrices and fixed-seed streams remain unchanged. The
phenomenological route and its decomposition option are unchanged. Gate-bound
`decompose_composite_faults=True` remains refused.

## Supported boundary and refusals

The gate-bound factory now accepts one-/two-qubit Kraus mixtures whose nonzero
branches are Pauli products up to global phase, including Y flip, biased mixtures
and the existing depolarizing factories. Operator-derived probabilities and
existing tolerances are reused unchanged. Duplicate Kraus products combine;
zero branches and identity-only channels contribute no visible mechanism.

Non-Pauli unitaries, nonunitary branches, more-than-two-qubit channels, invalid
target arity, preparation/readout attachment, last-instruction placement,
unsupported/conditional/custom-matrix instructions and nondeterministic parity
requests retain fail-closed errors. No alternative Kraus decomposition is inferred
and no non-Pauli channel is approximated.

The model's incidence/rate matrices do not imply independent Bernoulli columns
when ids are present. Existing text, graph, component-projection and decoder
operations that cannot preserve alternatives retain their refusals. This is
exact noise construction and sampling, not an implementation of a correlated
maximum-likelihood decoder or an automatic approximation for PyMatching.

## Runnable user workflow

From an installed package, or with `PYTHONPATH=.` in a source checkout:

```bash
python examples/qec/correlated_circuit_noise.py
```

Only core Torch is required. The example applies a two-qubit depolarizing channel
of total nonidentity probability 0.15 to one CNOT pair in each of two syndrome
rounds. Identity has mass 0.85; each of the 15 nonidentity Pauli products has
mass 0.01 before signatures are combined. Expected stdout:

```text
Detectors, observables: 6 1
Independent circuit locations: 2
Exclusive groups: 2
Sample shapes: (4, 6) (4, 1)
Independent-only text export: refused
```

This samples the declared DEM. To execute the corresponding physical circuit,
install the existing Stim extra and use the unchanged public
`sample_memory_circuit(memory, noise=noise, shots=..., seed=...)` or
`compile_memory_sampler(memory, noise=noise, seed=...)` workflow. Random engines
need not produce identical shots; their joint law is the comparison target.

## Independent numerical and compatibility acceptance

`tests/team/simulation/test_correlated_fault_signatures.py` checks every two-qubit
Pauli product at every fault position around all supported Clifford gates against
dense evolution, with reversed targets and an odd ideal baseline. Measurement
record cancellation and reset erasure are separate identities. Classification
checks all 15 depolarizing terms, p=0/1, duplicate/global-phase Kraus branches,
non-Pauli and excessive-arity refusals.

`tests/qec/test_correlated_circuit_noise.py` independently injects each biased
product at every matching location for repetition, rotated, Steane and mixed
ZXXZ memories, including supported X/Z readout, then compares the entire grouped
DEM. It injects all 15 depolarizing products on a bounded repetition memory and
compares a two-location experiment by Cartesian enumeration of all nine joint
outcomes. Silent/duplicate signatures and reversed targets have direct checks.

The IX=0.2, XI=0.3 exclusivity witness forbids simultaneous alternatives, whereas
independent substitution produces joint mass 0.06. Existing DEM sampling is
checked for that categorical invariant and seed repeatability. Physical circuit
sampling checks the exact enumerated joint law with 40,000 shots, fixed seed 47,
and a predeclared union-bound Hoeffding tolerance at familywise alpha=1e-5 over
the entire detector/observable bit space. The test does not relax its bound after
observing results. Core-only environments exercise the actionable missing-extra
error; installed Stim environments must execute the physical cross-check.

Old X/Z sampler instructions and Kraus matrices are compared bit-for-bit with
the retained construction, and fixed-seed engine output is compared when Stim
is installed. Existing root/pickle/plugin/retirement tests and the original 15
fixtures remain in place. Only the old depolarizing-refusal test case changes:
that input is now intentionally accepted by the approved contract and receives
stronger separately named signature, joint-law and sampling coverage. Damping
refusals and their channel-name assertions remain active.

Reproduce focused acceptance:

```bash
PYTHONPATH=. python -m pytest tests/team/simulation/test_clifford_fault_signatures.py tests/team/simulation/test_correlated_fault_signatures.py tests/qec/test_gate_fault_propagation.py tests/qec/test_gate_bound_noise.py tests/qec/test_correlated_circuit_noise.py -q
PYTHONPATH=. python -m pytest tests/qec tests/api_contract/test_qec_retirement_contract.py -q
```

Source, installed wheel/sdist, runnable examples and complete required CI results
are recorded at the reviewed PR head before delivery. Optional manual runner-pool
checks are distinct from required single-/two-GPU gates. No checks, thresholds,
fixtures or protected snapshots are weakened. Do not modify or merge #731,
activate the broad batch/experiment proposals, or merge this PR automatically.

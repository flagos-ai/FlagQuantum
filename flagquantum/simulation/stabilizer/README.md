# Stabilizer sampling and Pauli readout

This package owns Clifford stabilizer sampling: circuits whose gates all
normalize the Pauli group have a tableau representation whose storage grows
quadratically with the qubit count, so the qubit counts this package reaches are
the ones no amplitude store can hold.

- Start in `engine.py`. It owns the five things this package does: translating
  validated Circuit IR into the engine's circuit form, sampling measurement
  outcomes for a requested qubit list, expanding a Pauli string through the
  circuit and reading it against the zero state, sampling measurement outcomes
  from a circuit that already carries noise channels at explicit positions, and
  reporting what that second sampling route would make of a program without
  executing it.
- Call `sample_stabilizer(program, shots=..., qubits=..., seed=...)`. It accepts
  a `Circuit` or a validated `CircuitIR` and returns an `int64` tensor of shape
  `(shots, len(qubits))`.
- Call `pauli_readout(program, x=..., y=..., z=...)` to read a measured Pauli
  instead of drawing outcomes. It conjugates the string through the whole circuit
  and reads the result against the all-zero state, which is exact and is one of
  `-1`, `0`, and `1`; the returned `PauliReadout` also names the conjugated string
  and its `i`-power, which is the observable a run of that circuit would have
  flipped. This entry point imports no external engine, because expending a Pauli
  through a Clifford circuit is Pauli algebra rather than tableau simulation.
- The Core measurement kinds this representation answers are
  `STABILIZER_MEASUREMENT_KINDS`, split into `STABILIZER_SAMPLING_KINDS` (drawn
  outcomes) and the expectation kinds (exact values). Runtime reads both sets
  from here rather than keeping its own list, so a request the representation
  cannot answer is refused by the same answer the executor reads.
- Call `sample_noisy_measurements(program, shots=..., terminal_qubits=..., seed=...)`
  when the caller has placed noise itself. It executes a channel at the position
  it finds it, which means the *caller* owns the Pauli-frame attribution and this
  entry point owns only the execution; the recorded measurement columns come back
  in program order and the terminal qubits are appended in the order the caller
  names them. The routed `mode='stabilizer'` reaches this entry point for a
  program that carries a channel and `sample_stabilizer` for one that does not,
  and it asks `survey_stabilizer_program` which of the two applies rather than
  being told, so the route cannot hold a program and a claim about it that
  disagree. Which entry point a routed run takes is therefore read from the
  program, and an inline channel and an equivalent `NoiseModel` rule reach the
  same one.
- Call `survey_stabilizer_program(program)` to ask that same question without
  sampling. It returns a `StabilizerSurvey`: how many gates, resets, recorded
  measurements, and noise instructions the program has, whether every channel in
  it is a mixture of Pauli frames, and one blocker per instruction the sampling
  route would refuse. It reads the IR and never imports the engine, so a caller
  can decide between the stabilizer regime and a dense one before the optional
  distribution is installed. It is a census rather than a promise: an empty
  blocker list says every instruction translated, not that the samples are right.
- Do not import `stim` anywhere but `engine.py`. It is the single seam a
  replacement Clifford kernel replaces, and keeping it in one place is what
  makes that replacement a local change.
- The accepted gates are `CLIFFORD_GATE_NAMES`. Everything else - a
  parameterized rotation, `t`, `ccx`, `cswap` - is refused with `CapabilityError`
  naming the accepted set. Nothing is approximated and nothing falls back. Noise
  is the one exception and it is a class rather than a list: `sample_stabilizer`
  still refuses every channel, while `sample_noisy_measurements` accepts a
  channel whose Kraus operators are a mixture of Pauli operators up to a global
  phase, on as many qubits as the channel acts on, and refuses every other channel,
  every lowered measurement node, and every non-Clifford opcode by name.
- The engine names no channel instruction past two qubits, so a frame wider than
  that is spelled as a chain of correlated-error instructions instead - one term
  per non-identity frame, with the identity being the chain falling through. A
  chain term is conditional on no earlier term having fired, which is exactly the
  shape of a Kraus mixture, so the translation divides each weight by the
  probability the frames before it did not fire. Handing the engine the caller's
  own weights would make every frame after the first too rare.
- That class is decided by the operators, not by the instruction's name, and
  `flagquantum.noise.KrausChannel.unitary_mixture` is what decides it. A channel
  the engine cannot express is refused rather than approximated: a depolarizing
  channel is four frames and samples as four frames, a two-qubit depolarizing
  channel is sixteen, an amplitude damping or thermal relaxation channel is not a
  mixture of unitaries at all, and a coherent over-rotation is a mixture of
  unitaries whose one branch is not a Pauli. The last two cases are separate
  refusals because they are separate facts.
- Run `python -m pytest tests/team/simulation/test_stabilizer_engine.py tests/team/simulation/test_stabilizer_positioned_noise.py -q`
  after a typical local change.

The private `_fault_signatures.py` primitive reuses the engine's Clifford images
to propagate recorded-measurement and terminal-Z parities backwards through gates,
measurements and resets. It returns exact post-instruction Pauli responses without
Stim, amplitudes or sampling, and refuses nondeterministic parities. QEC's
execution bridge supplies its layouts; this primitive owns no codes, noise-rule
placement or detector numbering. Run
`python -m pytest tests/team/simulation/test_clifford_fault_signatures.py -q`
after changing it. Its local conjugation tests use independent dense evolution.

## What this package must not own

Device selection, backend dispatch, shot policy, seed streams, job lifecycle,
result assembly, and public execution evidence. Runtime owns those; this package
is a numerical primitive. It also must not import Runtime, and it does not read
lowered `MeasurementNode`s, because reconciling a caller's requested qubits with
an already-lowered output request is a Runtime decision.

## Why the engine is an external distribution

The dependency policy requires a new production dependency to record a
documented need, an ownership boundary, a licence and supply-chain review, a
replacement interface, and an exit plan. Here they are:

- Need: the thousand-qubit Clifford regime, where the reference implementation
  of stabilizer simulation is already published and already the implementation
  behind the comparison framework's own stabilizer target. Re-deriving it would
  be research spent on a solved algorithm.
- Ownership: this package owns the conversion and the sampling call. Core, the
  import-time path, and every other domain are free of the dependency.
- Licence and supply chain: Stim is Apache-2.0 and imports no accelerator
  vendor component, so it belongs to the open-neutral dependency class rather
  than the class this project exists to replace.
- Replacement interface and exit plan: `sample_stabilizer` and
  `sample_noisy_measurements` are the seam. `CLIFFORD_GATE_NAMES` is the contract
  a replacement satisfies, and the positioned-noise entry point adds only the
  channel placement, which a replacement kernel would inherit unchanged. A future
  first-party or FlagOS Clifford kernel replaces the body without changing a
  caller, and the extra can then be dropped. That work is scheduled separately
  rather than scaffolded here. `pauli_readout` is not part of that seam, because
  it is already first-party: the Clifford phase bookkeeping this module would
  otherwise owe the engine is what it performs, and it performs it without
  importing the engine.

## Determinism

QEC repeated sampling uses the private `engine._prepare_noisy_measurements`
boundary. It accepts the same positioned-noise grammar as the one-shot entry,
compiles once, and returns a private numerical stream with `sample` and `close`.
SDK circuit and sampler objects stay in `engine.py`. Draws return owned CPU
integer measurement records with validated width and binary values; the stream
retains no drawn batches. Close drops the compiled-state reference. QEC owns
process/thread restrictions, RNG call sequencing, poisoning and public lifecycle;
this primitive adds no Runtime scheduling policy or public Simulation export.

The existing one-shot entry uses the same preparation implementation and closes
it after its draw. Its seed and result behavior remain protected by existing
fixtures. `test_prepared_stabilizer_sampling.py` checks one compilation, release,
invalid records and independently authored measurement distributions. A future
Clifford kernel can replace this private boundary without changing the QEC
consumer, as the controlled-record replacement test demonstrates.

A seed reproduces a run on one engine version and one machine instruction set
and does not pin a bit pattern. That is the engine's documented behaviour, so a
seeded comparison across hosts is a comparison of distributions, not of bits.
Sampling is not differentiable and no gradient is claimed for it - a
differentiable stabilizer objective is a separate capability with its own
evidence.

Private `_fault_signatures.py` also classifies bounded Pauli channels through
the existing engine authority and propagates simultaneous products for QEC.
It does not sample or import Stim. Run
`python -m pytest tests/team/simulation/test_correlated_fault_signatures.py -q`
for dense-reference, global-phase, reset and odd-parity checks.

The approved QEC Stim experiment adapter additionally uses private
`_annotated_stim_text`, `_prepare_stim_experiment` and
`_stim_experiment_dem_text`. These keep all SDK parsing, serialization,
reference-normalized detector sampling and strict DEM conversion here. QEC
supplies positioned IR and parity columns and reuses its existing sampler
lifecycle. Original noisy circuits generate samples independently of whether
exact noisy DEM analysis succeeds. Normalization supplies only missing shape
declarations and preserves effective coordinates through bounded repeats.
Run `python -m pytest tests/team/simulation/test_stim_annotated_engine.py -q`
for independent SDK, joint-law, reference, bounds and failure evidence.
Pure scalar and Pauli-support checks live in `_validation.py`; their validation
behavior is unchanged and they hold no SDK state.

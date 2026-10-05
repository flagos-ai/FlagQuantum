# Stabilizer sampling

This package owns Clifford stabilizer sampling: circuits whose gates all
normalize the Pauli group have a tableau representation whose storage grows
quadratically with the qubit count, so the qubit counts this package reaches are
the ones no amplitude store can hold.

- Start in `engine.py`. It owns the three things this package does: translating
  validated Circuit IR into the engine's circuit form, sampling measurement
  outcomes for a requested qubit list, and sampling measurement outcomes from a
  circuit that already carries noise channels at explicit positions.
- Call `sample_stabilizer(program, shots=..., qubits=..., seed=...)`. It accepts
  a `Circuit` or a validated `CircuitIR` and returns an `int64` tensor of shape
  `(shots, len(qubits))`.
- Call `sample_noisy_measurements(program, shots=..., terminal_qubits=..., seed=...)`
  when the caller has placed noise itself. It executes a bit-flip channel at the
  position it finds it, which means the *caller* owns the Pauli-frame attribution
  and this entry point owns only the execution; the recorded measurement columns
  come back in program order and the terminal qubits are appended in the order the
  caller names them. It is deliberately not reachable from the planner or the
  executor, so `mode='stabilizer'` keeps refusing a noisy program.
- Do not import `stim` anywhere but `engine.py`. It is the single seam a
  replacement Clifford kernel replaces, and keeping it in one place is what
  makes that replacement a local change.
- The accepted gates are `CLIFFORD_GATE_NAMES`. Everything else - a
  parameterized rotation, `t`, `ccx`, `cswap` - is refused with `CapabilityError`
  naming the accepted set. Nothing is approximated and nothing falls back. Noise
  is the one exception and it is a narrow one: `sample_stabilizer` still refuses
  every channel, while `sample_noisy_measurements` accepts exactly the two-element
  bit-flip pair and refuses any other channel, any lowered measurement node, and
  any non-Clifford opcode by name.
- Run `python -m pytest tests/team/simulation/test_stabilizer_engine.py tests/team/simulation/test_stabilizer_positioned_noise.py -q`
  after a typical local change.

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
  rather than scaffolded here.

## Determinism

A seed reproduces a run on one engine version and one machine instruction set
and does not pin a bit pattern. That is the engine's documented behaviour, so a
seeded comparison across hosts is a comparison of distributions, not of bits.
Sampling is not differentiable and no gradient is claimed for it - a
differentiable stabilizer objective is a separate capability with its own
evidence.

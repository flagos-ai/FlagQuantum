# Stabilizer sampling

This package owns Clifford stabilizer sampling: circuits whose gates all
normalize the Pauli group have a tableau representation whose storage grows
quadratically with the wire count, so the wire counts this package reaches are
the ones no amplitude store can hold.

- Start in `engine.py`. It owns the two things this package does: translating
  validated Circuit IR into the engine's circuit form, and sampling measurement
  outcomes for a requested wire list.
- Call `sample_stabilizer(program, shots=..., wires=..., seed=...)`. It accepts
  a `Circuit` or a validated `CircuitIR` and returns an `int64` tensor of shape
  `(shots, len(wires))`.
- Do not import `stim` anywhere but `engine.py`. It is the single seam a
  replacement Clifford kernel replaces, and keeping it in one place is what
  makes that replacement a local change.
- The accepted gates are `CLIFFORD_GATE_NAMES`. Everything else - a
  parameterized rotation, a noise channel, `t`, `ccx`, `cswap` - is refused with
  `CapabilityError` naming the accepted set. Nothing is approximated and nothing
  falls back.
- Run `python -m pytest tests/team/simulation/test_stabilizer_engine.py -q`
  after a typical local change.

## What this package must not own

Device selection, backend dispatch, shot policy, seed streams, job lifecycle,
result assembly, and public execution evidence. Runtime owns those; this package
is a numerical primitive. It also must not import Runtime, and it does not read
lowered `MeasurementNode`s, because reconciling a caller's requested wires with
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
- Replacement interface and exit plan: `sample_stabilizer` is the seam.
  `CLIFFORD_GATE_NAMES` is the contract a replacement satisfies. A future
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

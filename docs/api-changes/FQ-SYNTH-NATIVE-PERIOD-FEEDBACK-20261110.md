# Native period finding and private-feedback equivalence

Status: successful native probes; independent acceptance pending.

## Reuse and boundaries

The frozen Shor period-finding program can already be expressed using existing
native allocation, Hadamard transform, modular exponentiation and inverse QFT.
Keep its21modulus,11base,5residue bits and10all-fractional phase bits. The modular
operation reads the phase counter's stored integer bit pattern as its exponent;
its final numeric output interpretation remains phase/1024. This composition
matches the source's powered modular QPE exactly, without inventing a new public
flexible-QPE marker or generating1023 callback copies. The initialized private
residue starts at1 and is dropped only after inverse QFT. Its state remains an
environment; dropping it cannot erase it or certify clean work. This is the
quantum period-finding stage, not a complete classical factoring workflow.

The teleportation source exposes only Bob's qubit and drops both measured Alice
qubits. Its private computational-basis outcomes control only disjoint Pauli
corrections. This specific output channel admits the deferred-measurement
principle: retain the two outcome qubits coherently, apply controlled corrections
and trace them out at the output boundary. The source adapter already checks
and uses this equivalence. The native example must explicitly call itself the
deferred-measurement implementation, preserve the retained environment, and
prove the measured output channel independently. It does not execute physical
mid-circuit measurements, deliver classical messages, expose outcomes or add a
public measure API. Do not generalize this equivalence to exposed measurement
results, reset, other feedforward protocols or active reuse of measured storage.

No production Compiler or numerical runtime change is needed by these probes.
Native examples call only existing public APIs; original Qmod stays test-side.

## Acceptance

- Shor: original/native/independent absolute full-state comparison including
  all five dirty residue bits and every clean work amplitude. Independent
  Circuit constructs controlled modular permutation matrices with identity on
  invalid residues and an analytic inverse DFT, without calling synthesis
  modular permutation or Fourier lowering. Verify two additional nonzero
  coprime modulus/base pairs, raw bit order, phase decoding, coprimality and
  resource refusals. Preserve the original10bit counter and source hashes.
- Teleportation: compare original/native/independent purified full states for
  the frozen input, two additional nonzero complex input states and a Bell
  input entangled with an external reference. Independently enumerate actual
  measurement branches, apply classical corrections, sum branch density
  matrices and verify the identity channel including the external reference.
  Check branch probabilities, environment state and all work amplitudes.
- Re-run relevant native storage/control/transform regressions, all216 originals
  and six approved revisions after final production changes. Keep raw resource
  baselines intact; update catalog, provenance, five-category inventory,
  immutable evidence and delivery ledger only after acceptance.

Starting coverage is193/216 native quantum programs,24/213 Notebook workflows
and14/19 revision-nine merged scopes. No integration PR,push or full remote CI
before all216 native quantum programs pass independent acceptance.

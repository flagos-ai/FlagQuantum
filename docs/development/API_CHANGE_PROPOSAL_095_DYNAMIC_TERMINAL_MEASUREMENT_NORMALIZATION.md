# API Change Proposal 095: DynamicCircuit terminal-measurement normalization

## Status

**Proposal and implementation approved on 2026-10-26; implementation merged in
PR #892 and the local performance closure is recorded.**

The API owner authorized preparation of this proposal with:

```text
approve API-CHANGE-095-DYNAMIC-TERMINAL-MEASUREMENT-NORMALIZATION-PROPOSAL
```

This approval covers the contract and implementation plan below. It does not
authorize a Stable Core behavior change, credential access, discovery, network
request, task submission, charging, or a claim that a remote QPU supports
mid-circuit measurement or feed-forward.

## User outcome

Users may express ordinary shot-based execution with the same builder they use
for dynamic programs:

```python
import flagquantum as fq
from flagquantum.dynamic import DynamicCircuit

circuit = DynamicCircuit(2).h(0).cx(0, 1)
circuit.measure(0, classical_bit=0)
circuit.measure(1, classical_bit=1)

result = fq.run(circuit, target="quafu:Baihua", shots=1024)
```

Expected result after implementation: the terminal measurements are normalized
into the existing static counts workflow, and `result.counts` uses the declared
classical-bit order. The call requires only the same provider prerequisites as
an equivalent static `fq.Circuit`; it does not require a dynamic-QPU capability
snapshot.

API Change 094 already admits a `DynamicCircuit` containing only unitary
instructions. This proposal defines the next safe subset: explicit measurements
that occur only after all quantum evolution has finished.

## Why this is not remote dynamic execution

A terminal measurement does not influence a later operation. Removing its IR
instruction from the submitted unitary prefix and asking the existing static
provider path for final counts preserves the program's observable behavior,
provided that FlagQuantum also preserves the declared qubit-to-classical-bit
mapping.

This equivalence does not apply to:

- a measurement followed by a gate or another non-measurement instruction;
- reset;
- a condition or feed-forward operation;
- repeated measurement of one qubit;
- writes to one classical bit from more than one measurement; or
- a result contract that cannot preserve the declared classical ordering.

Those cases remain genuinely dynamic or ambiguous and continue to fail closed.

## V1 admissible subset

The first implementation accepts a `DynamicCircuit` only when all of the
following are true:

1. the circuit contains at least one explicit `measure` instruction;
2. all measurement instructions form one terminal suffix;
3. no `reset`, condition, disjunction, or other classical-control operation is
   present;
4. every logical qubit is measured exactly once;
5. every classical bit in `range(circuit.n_qubits)` is written exactly once;
6. no other classical bit is read or written;
7. `outputs` is absent or requests one full-register `fq.counts()` result; and
8. the selected static provider route can preserve and report the measurement
   mapping and final bit order; and
9. the Quafu V1 bridge uses SDK service compilation (`compiler=None`), whose
   raw `c0_rightmost` result is already normalized into FlagQuantum's public
   qubit order before the declared classical permutation is applied.

The qubit-to-classical-bit mapping may be a permutation. For example,
`measure(1, classical_bit=0)` followed by `measure(0, classical_bit=1)` is
admissible only if the result normalizer deterministically remaps provider bits
to the declared classical register. The mapping must never be ignored merely
because all qubits were measured.

Partial measurement, sparse classical registers, `fq.samples()`, observable
expectations, and repeated measurement are deliberately outside V1. Each needs
an independently specified result shape or semantic rule rather than an
inferred fallback.

## Normalization contract

Normalization produces one internal, immutable record containing:

- the validated source `CircuitIR` identity;
- the static unitary-prefix `CircuitIR`;
- the ordered `(qubit, classical_bit)` measurement mapping;
- the classical register width;
- the requested output identity; and
- a normalization fingerprint binding all of the above.

The source program is not mutated. The unitary prefix is submitted through the
already selected static route. Provider counts are accepted only when their bit
width, shot total, provider bit-order declaration, and measurement mapping are
consistent with the normalization record. FlagQuantum then projects the counts
into public left-to-right classical order `c[0]...c[n-1]`, matching the existing
counts convention after each provider's raw bit order has been normalized.

An absent, conflicting, or unverifiable mapping is a deterministic refusal. It
is not permission to assume identity order, reverse bits heuristically, submit
the original dynamic IR, switch providers, or run a simulator.

## Admission and refusal order

The implementation must perform these steps before any provider-owned work:

1. validate the source IR and classical-control metadata;
2. classify the program as unitary-only, V1 terminal-measurement, or genuinely
   dynamic;
3. validate the terminal suffix and full-register bijection;
4. validate the output request;
5. create the normalization record; and
6. enter the existing static provider workflow.

Malformed or unsupported programs fail before provider SDK import,
construction, credential lookup, discovery, compilation, network access, task
creation, charging, or random-number consumption. Genuine dynamic programs
continue through the existing dynamic remote preflight and remain fail closed
when exact device facts are unavailable.

## Provider boundary

The normalization classifier is provider-neutral and belongs beside Runtime's
dynamic semantic classification. Each Remote integration remains responsible
for proving that its static request and result contracts preserve the mapping.

For Quafu, API Change 093 confirms that the reviewed `quafu-sdk` revision
supports terminal measurement metadata and final counts but not true dynamic
execution. The implementation must use offline SDK fixtures to prove the exact
request and result mapping before enabling the positive Quafu path. This
proposal does not authorize a live request.

The first Quafu bridge does not admit `compiler="qsteed"`. Its physical result
path binds a separate logical-to-physical measurement receipt and therefore
needs its own three-way logical, physical, and classical mapping contract.
Until that contract exists, an explicit compiler keeps the program on dynamic
preflight and fail-closed behavior.

No provider is enabled merely because it returns bit strings. Azure or another
provider requires its own mapping fixture in the implementation PR before the
generic normalization path may dispatch to it.

## Implementation batches

### Batch A: pure normalization model

Add an internal classifier and immutable normalization record. Prove acceptance
of identity and permuted full-register mappings and refusal of partial, sparse,
repeated, mid-circuit, reset, conditional, and incompatible-output programs.

### Batch B: Quafu offline bridge

Connect the normalization record to the existing static Quafu path with mocked
SDK fixtures. Prove exactly one static submission, no dynamic preflight for the
admissible subset, deterministic classical-bit remapping, and side-effect-free
refusal.

### Batch C: stable-facade activation

Activate the V1 subset through the existing `fq.run` signature. Lock the user
example, return contract, refusal ordering, and unchanged behavior for ordinary
`Circuit`, unitary-only `DynamicCircuit`, and genuinely dynamic programs.

Batch C requires the same API-owner approval as the implementation. A future
provider bridge or wider measurement subset requires a separate proposal or an
explicit amendment to this one.

## Compatibility and non-goals

- no new public class, function, argument, output type, or serialized IR field;
- no change to local dynamic execution semantics;
- no implicit `fq.samples()` support;
- no partial-register or observable measurement lowering in V1;
- no mid-circuit measurement, reset, repeated-measurement, or feed-forward
  activation;
- no simulator, compiler, provider, backend, or target fallback; and
- no weakening of API Changes 086, 089, 090, or 093.

## Acceptance criteria

- [x] Classification depends on validated IR semantics, not Python type alone.
- [x] The V1 admissible subset is enforced exactly and fails closed otherwise.
- [x] Identity and permuted measurement mappings preserve classical bit order.
- [x] The source IR is not mutated and normalization identity is inspectable.
- [x] Quafu offline fixtures prove one static submission and correct counts.
- [x] Refusal occurs before every external or irreversible boundary.
- [x] Existing true-dynamic preflight and blocker behavior remain unchanged.
- [x] The implementation PR includes the runnable user example and expected
      result.
- [x] Public API snapshots and signatures remain unchanged.

## Implementation approval requested

```text
approve API-CHANGE-095-DYNAMIC-TERMINAL-MEASUREMENT-NORMALIZATION-IMPLEMENTATION
```

That approval authorizes Batches A-C with offline provider fixtures. It does not
authorize credentials, network access, live discovery, task submission,
charging, another provider integration, or a live-QPU support claim.

**Implementation approval recorded:**

```text
approve API-CHANGE-095-DYNAMIC-TERMINAL-MEASUREMENT-NORMALIZATION-IMPLEMENTATION
```

## Closure evidence

PR #892 merged Batches A-C without changing the public API snapshot. The
maintained local performance runner and its measured budget are documented in
[`API_CHANGE_IMPLEMENTATION_095_DYNAMIC_TERMINAL_MEASUREMENT_PERFORMANCE_BUDGET.md`](API_CHANGE_IMPLEMENTATION_095_DYNAMIC_TERMINAL_MEASUREMENT_PERFORMANCE_BUDGET.md).

API Change 095 is complete at the V1 boundary. Qsteed physical mapping, partial
measurement, sparse classical registers, samples, and genuine remote dynamic
execution require separate contracts.

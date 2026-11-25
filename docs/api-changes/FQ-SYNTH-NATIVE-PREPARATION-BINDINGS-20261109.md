# Native preparation from bound classical vectors

The user authorized completing all cases in the five native example categories,
implementing missing native operations where required, and continuing subsequent
batches after verification. This narrow unreleased extension connects the existing
classical real-array binder to the existing probability/amplitude preparation
operations. It introduces no export, signature, result field, opcode, serialization
schema or executor. Existing protected contracts and historical admission records
remain unchanged; release authorization remains false.

## User journey and semantics

The state-and-amplitude preparation Notebook constructs a real amplitude vector
on the host, normalizes it, synthesizes a quantum preparation and reads its state.
The native equivalent passes a `list[float]` through the already admitted
`classical_inputs` mapping. A captured `prepare_state(probabilities, bound, out)`
or `prepare_amplitudes(amplitudes, bound, out)` may now use a named bound array.
The existing root binder snapshots a finite real list/tuple; capture never evaluates
arbitrary Python expressions, reads an unbound global list, or calls host code.

The existing vector rules apply unchanged: power-of-two length 2..256, finite
values, normalized real amplitudes or nonnegative probabilities summing to one
within 1e-12, and a finite nonnegative bound. Normalization is not repaired.
Preparation owns fresh storage, infers width and respects Const, alias, resource
and repeated-initialization checks. Scalar binding, unsupported vector expressions,
invalid length and out-of-budget circuits fail before execution. Existing literal
syntax remains supported. Backend selection and gate construction remain owned by
the existing Compiler pipeline.

Parameterized helpers retain their existing gate-only borrowed-port profile;
Output/Const parameterized helpers remain refused. This extension admits bound
vectors at the program root and does not broaden helper ownership or escape rules.

## Acceptance

- Two different normalized nonuniform probability arrays and two signed amplitude
  arrays match independent dense-reflection Circuits in every complex amplitude,
  including absolute phase, bit order and zero hidden workspace.
- Explicit array binding and an equivalent literal program produce the same
  resources and state; later host-list mutation does not change the artifact.
- Reject malformed length, nonfinite/bool entries, negative or unnormalized
  probabilities, unnormalized amplitudes, unbound globals, repeated initialization,
  Const mutation and resource ceilings.
- Complete the source Notebook's probability sampling and signed amplitude
  reconstruction. Phase alignment is only a presentation step; verification uses
  raw complex amplitudes. Local sampling/state access replace cloud calls.
- Preserve all original sources and resource baselines. Run native regressions,
  strict typing and the public API checker. Keep quantum-program and complete
  Notebook coverage separate.

# Typed classical word reads and feedback

## Approved scope and source

Proposal 108 was explicitly approved on 2026-11-20. Its Core and Compiler
implementation admits canonical classical bit, unsigned, signed,
ones'-complement, bounded unsigned, Montgomery and fixed-point integer storage.
User execution continues through `circuit.append(op)`.

The unchanged reference is Qualtran v0.7.0 revision
`bcb17bb25360ed68ce3bc084645db3e929799389`. The prior complete controlled-interface
inventory and original source test evidence remain in
`CONTROLLED_INTERFACE_SOURCE_AUDIT.md` and
`CLASSICAL_COMPONENT_CONTROL_AUDIT.md`. This batch closes the approved integer
word dependency. Those source parents retain their other recorded boundaries.

| Reference file | Verified Git blob |
| --- | --- |
| `_infra/controlled.py` | `6fabe760e793f1b39f825d76f80e92b6e5fa5c28` |
| `bloqs/mcmt/classically_controlled.py` | `a4236872f616527a1dbc92b5453cb647b656650f` |

The datatype and tensor-indexing direct dependency hashes are checked against
the frozen intake CSV in `g2-classical-word-evidence.json`. Qualtran 0.7.0,
Kickmix 0.1 and its native Simulator are checked before recording acceptance.

## Ownership and execution

| Owner | Completed responsibility |
| --- | --- |
| Core `register_verifier.py` | Canonical classical descriptors, zero quantum width, concrete execution validation, identical-type read successor and width boolean snapshots |
| Compiler `classical_word_storage.py` | Physical MSB slot values through the existing `ControlRegister.bits` encoder |
| Compiler `entry_bindings.py` and `register_binding.py` | Atomic entry validation, row-major word elements, slot initialization, call transport, shared read snapshots and per-element classical outputs |
| Compiler `component_schedule.py` | Classical edges with zero physical quantum width |
| Compiler `component_feedback.py` and `component_classical_control.py` | Shared stored definitions, numerical value guards, original child conditions and unconditional work lifetimes |
| Simulation | Existing codecs and callbacks reused and differentially verified |
| Ecosystem | Existing descriptor metadata reused; source physical classical-wrapper decomposition remains unavailable |

An owned read consumes one register handle, returns its same-typed successor and
then MSB-first boolean snapshots. Repeated reads alias the original slots.
Snapshots permit read-only fanout. Calls carry one owned element regardless of
its bit width. Quantum outputs and classical slot outputs retain their separate
existing representations. The full input tuple is validated before binding.

External CBit values retain their built-in zero/one and boolean contract. Wider
words require built-in integers in their stored domain. Fixed-point values are
raw integers. Montgomery metadata retains stored width-domain values, including
values above the modulus. Bounded values satisfy their exclusive bound.
Symbolic declarations retain metadata until concrete execution is requested.

Scalar CUInt(0) owns an empty slot tuple and returns a successor with no snapshots.
Empty arrays own no elements. Positive-width scalar, shaped and mixed-family
ports forward through repeated stored calls without allocating quantum wires.
Other zero-width encodings remain outside the proved physical input domain.

## Source disagreements retained

The frozen `active_space_for_ctrl_spec` uses `int(cv)` directly as a tensor index.
For CIntOnesComp(2), activation -1 therefore selects raw 11. Its numerical callback
and `to_bits(-1)` activate raw 10. Raw 11 decodes to zero. Tests explicitly assert
this disagreement without modifying the reference. Native feedback follows the
approved numerical equality contract. Ones'-complement zero guards include raw
00 and raw 11, while integer entry initialization canonically emits raw 00.
A test substitutes only initializer constants to exercise all four raw patterns.

The unchanged CUInt(0) scalar codec returns `[0]` despite its declared zero width.
Its scalar identity graph returns integer zero. An empty classical-only identity
call raises `KeyError(RightDangle)`. These source observations are separately
asserted against the native empty physical ownership contract.

The source ClassicallyControlled wrapper has no physical decomposition or QEC
model. Native child primitives, operands, phases and conditions are compared at
the existing comparable boundary. The wrapper's numerical action and tensor are
checked separately, with the ones'-complement tensor defect above explicitly
excluded from an equality claim.

## Numerical and replacement evidence

The inherited nested-child witness uses two shared bodies, each allocating a
clean work bit, applying CX, phase(pi/4), CX and H, then releasing work. Every
small signed, unsigned, ones'-complement, Montgomery and fixed-point activation
and input is compared on complete complex states entangled with a spectator.
The two-bit source wrapper tensor is used where it agrees with numerical
activation. Ones'-complement uses original activation and the original child
tensor, retaining the incompatible source wrapper tensor as a counterexample.
All final nonzero-work amplitudes are below `4e-14`.

Mixed scalar and shaped words verify row-major/MSB conditions. A measured inner
CBit combines with signed outer-word controls for every X-basis history under
negative and zero activations. Complete unnormalized branch amplitudes retain
measurement history and spectator information. Zero-width and empty controls
verify the original vacuous activation and direct tensor.

The exact old CBit function representation, bound CircuitIR JSON, content hash,
quantum outputs and classical slots match the pre-extension fixture. Its hash is
`cd42e7974c0e3c6c87fd81fdba2291dc66a5eafcded73785d567e64e35b08787`.
Parity, destructive measurement and discard retain their existing admission.

## Emitted resources and independent model

| Witness | Emitted primitives | Quantum workspace | Classical storage | Source estimate |
| --- | --- | --- | --- | --- |
| Two shared child bodies, one two-bit word | 4 CX, 2 phase, 2 H, 2 classical initializers, zero measurements | One reusable clean bit, four allocate/free events | Two initial slots, zero additional read slots | Child 2 T, 6 Clifford, zero measurements; wrapper refuses a model |
| Signed word and measured inner control | 1 H, 1 measurement, 1 guarded X, 2 classical initializers | Zero work bits | Two entry slots plus one measured CBit | Numerical wrapper reference; no inferred wrapper model |
| Runnable signed-X example | 1 guarded X, 2 classical initializers, zero measurements | Zero work bits | Two slots | Wrapper refuses a model |

Disabled gates remain emitted conditional instructions. Typed reads emit zero
primitive gates and zero new slots. These physical counts are kept separate from
the analytical child model and the absent wrapper estimate.

## Validation and remaining boundaries

Concentrated regression passes 2685 cases without skips, with 38 existing paired
source/native warnings. Local SDK, coherent dynamic and construction integration
passes 128 cases with two existing warnings. The original SDK launcher omitted
the frozen source's notebook marker and failed five source imports. The final
run uses the existing reference marker plugin and passes all 128 cases. Strict
typing passes nine Python files. Black, Ruff, classified team scope, architecture,
multi-level IR and public API checks pass. Dependency policy reports 688 existing
repository findings and zero batch findings.

Full integration was interrupted after 157.41 seconds with 829 passed, 47 failed,
299 skipped and 20 errors. It encountered the QEC historical-export contract,
sandbox-denied distributed TCPStore sockets and other failures retained in the
raw log. The failed test files and QEC checker/root files are byte-identical to
the published batch base. A separate QEC contract run has 36 passed and one
historical-export failure. The default smoke/unit fail-fast run has 434 passed,
51 skipped and the established nine-module public-docstring coverage failure.
These runs do not establish whole-repository success.

Finite-field classical Program admission, unresolved symbolic physical values,
arbitrary external wrappers and conditional measurement/reset/classical writes
remain separately tracked. Parent Q-0158/Q-0159/Q-0547 acceptance is unchanged.
Persistent evidence uses `ftqc-reference-env/g2-classical-word-*`.

## Contributor path

```bash
python -m examples.algorithms.typed_classical_feedback
python -m pytest tests/unit/test_classical_word_binding.py tests/unit/test_classical_word_feedback.py tests/unit/test_classical_word_feedback_contract.py -q
```

The example verifies all two-bit signed inputs, full complex amplitudes and
spectator entanglement, and prints emitted resources independently from the
source wrapper model refusal. The source-reference tests require the pinned
reference environment and its existing notebook marker plugin.

The classical binary field/poly admission boundary above is superseded by
explicitly approved Proposal 110. `CLASSICAL_FIELD_FEEDBACK_AUDIT.md` records
its concrete typed execution acceptance, retained source model refusal and
remaining nonbinary/symbolic/conditional-write boundaries.

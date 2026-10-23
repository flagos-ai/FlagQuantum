# QEC-RESTRUCTURE-006: decoder algorithms and optional adapters

Status: implemented for review under the approved package plan.
Owner: Core; physical evidence paths precede implementation in a separate
Integration commit. Classification: `single_device_fast_path`.

## Stack and ownership

This branch targets `refactor/qec-restructure-005`, PR #744, at `671f1b456`.
Implementation began only after that exact head passed all 38 full-CI jobs and
both required hardware checks. It inherits the complete stacked-PR CI trigger,
umask fixture fixes and corrected existing-file mutation proof. No PR is merged.
Core, Compiler and Runtime are outside this diff; #731 is not edited or merged.

The [decoder README](../../flagquantum/qec/decoders/README.md) assigns repetition
protocols, matching, belief propagation, ordered statistics, sliding windows and
the sole registry to their existing responsibilities. The
[adapter README](../../flagquantum/qec/adapters/README.md) assigns optional
PyMatching import/translation to its existing boundary. `decoders.py` and
`adapters.py` become compatibility package initializers atomically. Other former
flat modules retain explicit aliases. No algorithm, protocol, registry,
dependency, selection policy or numerical threshold is added.

## Compatibility and plugin behavior

The unchanged 001 baseline still protects all 108 ordered root exports,
signatures, defaults, fields and registration policy. Both rotated surface
classes retain their independent identities. The code registry does not move.

There is one authoritative decoder table in `decoders/registry.py`. Historical
`qec.registry`, canonical imports and root entry points reference its exact
objects. Registration returns the original candidate; duplicates, explicit
replacement, graph carriers, default names and lazy window registration retain
their behavior. A plugin test registers through the old path and constructs
through all three paths against that one table.

The existing metadata helper retains old class, method and function module/name
identifiers without wrappers. The project-owned `_DecoderT` explicitly keeps
`flagquantum.qec.registry` as its pickle module. The returned registration
decorator keeps the same module as well; this is the sole metadata assignment
added inside an existing function. Mathematical expressions, branches and
registration control flow remain unchanged.

The frozen pre-move 006 fixtures retain 186 pickle globals, 204 metadata entries,
seven historical namespaces, nine instances and exact result scenarios. They
include a mid-stream checkpoint resumed through its remaining detector rounds,
model/text/graph carriers, hyperedges, repetition feedback, refusal behavior,
adapter answers and the returned decorator. Fixtures were verified before
relocation across Python 3.10–3.12 and four SDK profiles, then left unchanged.
The 001–005 fixtures are unchanged.

Importing QEC and listing decoder names still works without PyMatching. Selecting
it without the SDK raises the existing named dependency error; there is no
fallback. The verified PyMatching 2.4.0 native graph remains non-pickleable and
raises its existing TypeError category. The adapter's own class/exception/global
pickle identifiers remain available; no new reducer or serialized schema is
introduced for the SDK object.

Physical source filenames change. Live documentation, contract evidence and the
existing public BPOSD classmethod's source-coverage check follow the actual
owner. The same doctest entry still executes. The BPOSD dependency guard reads
the actual implementation file and keeps its original standard-library/Torch
allowlist; internal dependencies use relative imports. The old registry plugin test now
isolates the actual private table rather than rebinding a facade variable; its
assertions are unchanged. Public plugin imports and registry behavior remain
supported. Private global rebinding and class source inspection must follow the
physical owner instead of treating retained pickle metadata as a source file.

Core owns compatibility aliases. No removal release is approved; removal needs
an approved deprecation and artifact migration plan. No library-owned safe-global
allowlist is added.

## Runnable user example

No optional SDK, credential or GPU is needed:

```bash
PYTHONPATH=. python examples/qec/decoder_registry_and_pickle.py
```

Expected output (`(0,)` names flipped observable index zero):

```text
minimum_weight_matching (0,)
belief_propagation (0,)
belief_propagation_osd (0,)
Historical pickle class: flagquantum.qec.matching
Same class: True
Registered decoders: 5
```

The example is registered in the examples index. The PR includes its complete
Python code, plus a standalone optional-PyMatching example with expected output.

## Acceptance evidence

The full QEC suite passes with **1,820 passed, one existing no-applicable-gate-fault
skip**. All twelve supported-Python/SDK source profiles pass: 46 compatibility
cases with Stim, or 45 plus the original Stim-only sampling skip without it.

- All 64 original top-level class/function ASTs match after excluding imports
  and the single returned-decorator metadata assignment.
- Strict mypy checks all 827 package files. Minimum/current tool versions,
  architecture boundaries, Stable Core snapshot, complete vocabulary ledgers,
  parity rendering, public doctests and the 499-check QEC alignment gate are
  checked without new suppressions or relaxed policy.
- The contract mutation, maturity, documentation, hygiene and example-inventory
  suites include the unchanged 35 alignment-checker mutation assertions.
- Exact full-QEC, supported-Python/SDK profile and current-head CI results are
  recorded in the PR. Installed wheel/sdist closure remains the final batch.

CPU checks are compatibility evidence, not a new GPU scalability claim.
